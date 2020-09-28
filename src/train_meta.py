import os
import sys
import os.path as osp

import dill
import time
from tqdm import tqdm
from argparse import ArgumentParser
from easydict import EasyDict as edict

import numpy as onp

import jax
from jax.random import split
from jax import (
    jit,
    pmap,
    vmap,
    random,
    partial,
    tree_map,
    numpy as jnp,
    value_and_grad,
)

import optax as ox
import haiku as hk

from config import rsetattr
from lib import (
    flatten,
    outer_loop,
    setup_device,
    fsl_inner_loop,
    batched_outer_loop,
    parse_and_build_cfg,
    mean_xe_and_acc_dict,
    meta_step,
    reset_by_idxs,
    reset_all,
    delayed_cosine_decay_schedule,
    # outer_loop_reset_per_task,
)
from data import prepare_data, augment as augment_fn, preprocess_images
from experiment import Experiment, Logger
from data.sampling import fsl_sample, fsl_build, BatchSampler

# from models.maml_conv import miniimagenet_cnn_argparse, prepare_model, make_params
from models import make_params, prepare_model
from test_utils import test_fsl_maml, test_fsl_embeddings
from test_sup import test_sup_cosine


def parse_args(parser=None):
    parser = Experiment.add_args(parser)
    # parser = miniimagenet_cnn_argparse(parser)
    # Training arguments
    parser.add_argument("--pool_lr", type=int, default=0)
    # parser.add_argument("--n_jobs_lr", type=int, default=None)

    parser.add_argument("--train.num_outer_steps", type=int, default=30000)
    parser.add_argument(
        "--train.batch_size", help="Number of FSL tasks", default=4, type=int
    )
    parser.add_argument(
        "--train.way", help="Number of classes per task", default=5, type=int
    )
    parser.add_argument(
        "--train.shot", help="Number of samples per class", default=5, type=int
    )
    parser.add_argument(
        "--train.qry_shot",
        type=int,
        help="Number of quried samples per class",
        default=10,
    )
    parser.add_argument("--train.cl_qry_way", default=64, type=int)
    parser.add_argument("--train.cl_qry_shot", default=1, type=int)
    parser.add_argument("--train.inner_lr", type=float, default=1e-2)
    parser.add_argument("--train.outer_lr", type=float, default=1e-3)
    parser.add_argument("--train.num_inner_steps", type=int, default=5)
    parser.add_argument("--train.train_inner_lr", default=False, action="store_true")

    parser.add_argument("--train.prefetch", default=0, type=int)
    parser.add_argument("--train.weight_decay", default=0.0, type=float)
    parser.add_argument("--train.apply_every", default=1, type=int)
    parser.add_argument("--train.scheduler", choices=["none", "step", "cosine"])

    parser.add_argument(
        "--train.piecewise_constant_schedule",
        nargs="*",
        type=int,
        default=[10000, 25000],
    )
    parser.add_argument(
        "--train.piecewise_constant_alpha",
        default=0.1,
        type=float,
    )

    parser.add_argument("--train.cosine_alpha", type=float, default=0.01)
    parser.add_argument("--train.cosine_decay_steps", type=float, default=10000)
    parser.add_argument("--train.cosine_transition_begin", type=float, default=5000)

    parser.add_argument(
        "--train.augment", default="none", choices=["none", "all", "spt", "qry"]
    )
    parser.add_argument("--train.num_prefetch", default=10, type=int)

    parser.add_argument("--train.val_interval", type=int, default=1000)
    parser.add_argument("--train.method", default="fsl", choices=["fsl", "cl"])
    parser.add_argument(
        "--train.reset_head",
        type=str,
        default="none",
        choices=[
            "none",
            "all-zero",
            "all-glorot",
            "all-kaiming",
            "cls-zero",
            "cls-glorot",
            "cls-kaiming",
        ],
    )

    # parser.add_argument("--val.pool", type=int, default=4)
    parser.add_argument(
        "--val.fsl.batch_size", help="Number of FSL tasks", default=25, type=int
    )
    parser.add_argument(
        "--val.fsl.qry_shot",
        type=int,
        help="Number of quried samples per class",
        default=15,
    )
    parser.add_argument("--val.fsl.num_inner_steps", type=int, default=10)
    parser.add_argument("--val.fsl.num_tasks", type=int, default=300)

    parser.add_argument("--model.name", choices=["resnet12", "convnet4"])
    parser.add_argument("--model.output_size", type=int)
    parser.add_argument("--model.hidden_size", default=0, type=int)
    parser.add_argument("--model.activation", default="relu", type=str)
    parser.add_argument(
        "--model.initializer",
        default="glorot_uniform",
        type=str,
        choices=["kaiming_normal", "glorot_uniform"],
    )
    parser.add_argument("--model.avg_pool", default=False, action="store_true")
    parser.add_argument("--model.head_bias", default=False, action="store_true")
    parser.add_argument("--model.norm_before_act", default=1, type=int, choices=[0, 1])
    parser.add_argument(
        "--model.final_norm", default="none", choices=["bn", "gn", "in", "ln", "none"]
    )
    parser.add_argument(
        "--model.normalize",
        default="bn",
        type=str,
        choices=["bn", "affine", "gn", "in", "ln", "custom", "none"],
    )
    parser.add_argument(
        "--model.track_stats",
        default="none",
        type=str,
        choices=["none", "inner", "outer", "inner", "inner-outer"],
    )

    args = parser.parse_args()
    cfg = edict(train=edict(cl=edict()), val=edict(fsl=edict()), model=edict())
    for argname, argval in vars(args).items():
        rsetattr(cfg, argname, argval)

    return args, cfg


# def step(
#     rng,
#     step_num,
#     outer_opt_state,
#     slow_params,
#     fast_params,
#     slow_state,
#     fast_state,
#     inner_lr,
#     x_spt,
#     y_spt,
#     x_qry,
#     y_qry,
#     spt_classes,
#     # inner_opt_init,
#     inner_opt,
#     outer_opt_update,
#     batched_outer_loop_ins,
#     reset_fast_params_fn=None,
#     preprocess_images_fn=None,
#     learn_lr=False,
# ):
#     rng, rng_preprocess = split(rng)
#     if preprocess_images_fn:
#         x_spt, x_qry = preprocess_images_fn(rng_preprocess, x_spt, x_qry)
#     if reset_fast_params_fn:
#         rng, rng_reset = split(rng)
#         tree_flat, tree_struct = jax.tree_flatten(fast_params)
#         rng_tree = jax.tree_unflatten(tree_struct, split(rng_reset, len(tree_flat)))
#         fast_params = jax.tree_multimap(
#             partial(reset_fast_params_fn, spt_classes), rng_tree, fast_params
#         )
#     # inner_opt_state = inner_opt_init(fast_params)

#     (outer_loss, (slow_state, fast_state, info)), grads = value_and_grad(
#         batched_outer_loop_ins, (0, 1), has_aux=True
#     )(
#         slow_params,
#         fast_params,
#         slow_state,
#         fast_state,
#         inner_opt_state,
#         split(rng, x_spt.shape[0]),
#         x_spt,
#         y_spt,
#         x_qry,
#         y_qry,
#         spt_classes,
#     )
#     grads = jax.tree_map(lambda v: jax.lax.pmean(v, axis_name="i"), grads)
#     updates, outer_opt_state = outer_opt_update(
#         grads, outer_opt_state, (slow_params, fast_params)
#     )
#     slow_params, fast_params = ox.apply_updates((slow_params, fast_params), updates)

#     return outer_opt_state, slow_params, fast_params, slow_state, fast_state, info


# def step_reset(
#     rng,
#     step_num,
#     outer_opt_state,
#     slow_params,
#     fast_params,
#     slow_state,
#     fast_state,
#     x_spt,
#     y_spt,
#     x_qry,
#     y_qry,
#     spt_classes,
#     inner_opt_init,
#     outer_opt_update,
#     batched_outer_loop_ins,
#     train_method,
# ):
#     if train_method == "fsl-reset-zero":
#         print("\nReseting head params to zero")
#         fast_params = hk.data_structures.merge(
#             {
#                 "mini_imagenet_cnn_head/linear": {
#                     "w": jnp.zeros(
#                         fast_params["mini_imagenet_cnn_head/linear"]["w"].shape
#                     ),
#                 }
#             }
#         )
#     else:
#         raise NameError(f"Unkwown train method `{train_method}`")

#     inner_opt_state = inner_opt_init(fast_params)

#     (outer_loss, (slow_state, _, info)), grads = value_and_grad(
#         batched_outer_loop_ins, 0, has_aux=True
#     )(
#         slow_params,
#         fast_params,
#         slow_state,
#         fast_state,
#         inner_opt_state,
#         split(rng, x_spt.shape[0]),
#         x_spt,
#         y_spt,
#         x_qry,
#         y_qry,
#         spt_classes,
#     )
#     updates, outer_opt_state = outer_opt_update(
#         grads,
#         outer_opt_state,
#         slow_params,
#     )
#     slow_params = ox.apply_updates(slow_params, updates)

#     return outer_opt_state, slow_params, fast_params, slow_state, fast_state, info


# def preprocess_images(rng, x_spt, x_qry, normalize_fn, augment="none", augment_fn=None):
#     x_spt = x_spt / 255
#     x_qry = x_qry / 255
#     if augment == "all":
#         print("\nAugmenting support and query")
#         rng_spt, rng_qry = split(rng)
#         x_spt = augment_fn(rng_spt, flatten(x_spt, (0, 1))).reshape(*x_spt.shape)
#         x_qry = augment_fn(rng_qry, flatten(x_qry, (0, 1))).reshape(*x_qry.shape)
#     elif augment == "spt":
#         print("\nAugmenting support only")
#         x_spt = augment_fn(rng, flatten(x_spt, (0, 1))).reshape(*x_spt.shape)
#     elif augment == "qry":
#         print("\nAugmenting query only")
#         x_qry = augment_fn(rng, flatten(x_qry, (0, 1))).reshape(*x_qry.shape)
#     elif augment == "none":
#         print("\nNo augmentation")
#     else:
#         raise NameError(f"Unkwown augmentation {augment}")

#     x_spt = normalize_fn(x_spt)
#     x_qry = normalize_fn(x_qry)

#     return x_spt, x_qry


# def reset_by_idxs(w_make_fn, idxs, rng, array):
#     print("Resetting head by indexes")
#     return jax.ops.index_update(
#         array,
#         jax.ops.index[:, idxs],
#         w_make_fn(dtype=array.dtype)(rng, (array.shape[0], idxs.shape[0])),
#     )


# def reset_all(w_make_fn, idxs, rng, array):
#     print("Resetting all head")
#     return w_make_fn(dtype=array.dtype)(rng, array.shape)


def main(args, cfg):
    # cfg = parse_and_build_cfg(args)
    # The Experiment class creates a directory for the experiment,
    # copies source files and creates configurations files
    # It also creates a logfile.log which can be written to with exp.log
    # that is a wrapper of the print method
    exp = Experiment(cfg, args)
    if not cfg.no_log:
        exp.logfile_init(
            [sys.stdout]
        )  # Send logged stuff also to stdout (but not all stdout to log)
        exp.loggers_init()
        sys.stderr = Logger(exp.logfile, [sys.stderr])  # Send stderr to log

    if cfg.debug:  # Debugging creates experiments folders in experiments/debug dir
        exp.log("Debugging ...")

    jit_enabled = not cfg.disable_jit
    # Temporarily hard-code default default_platform as cpu
    # it recommended for any big (bigger than omniglot) dataset
    # cpu, device = setup_device(cfg.gpus, default_platform="cpu")
    rng = random.PRNGKey(cfg.seed)  # Default seed is 0
    # exp.log(f"Running on {device} with seed: {cfg.seed}")
    exp.log(f"JAX available CPUS {jax.devices('cpu')}")
    try:
        exp.log(f"JAX available GPUS {jax.devices('gpu')}")
    except RuntimeError:
        pass
    num_devices = max(cfg.gpus, 1)
    exp.log(f"Using {num_devices} devices")
    cpu = jax.devices("cpu")[0]

    # Data
    train_images, train_labels, normalize_fn = prepare_data(
        cfg.dataset,
        osp.join(
            cfg.data_dir,
            "miniImageNet_category_split_train_phase_train_ordered.pickle",
        ),
        # device,
    )
    val_images, _val_labels, _ = prepare_data(
        cfg.dataset,
        osp.join(
            cfg.data_dir,
            "miniImageNet_category_split_val_ordered.pickle",
        ),
        # device,
    )
    val_labels = (
        _val_labels - 64
    )  # make val labels start at 0 (they originally begin are at 64-79)

    exp.log("Train data:", train_images.shape, train_labels.shape)
    exp.log(
        "Validation data:",
        val_images.shape,
        val_images.shape,
    )

    # Model

    body, head = prepare_model(
        cfg.model.name,
        cfg.dataset,
        cfg.model.output_size,
        hidden_size=cfg.model.hidden_size,
        activation=cfg.model.activation,
        track_stats=cfg.model.track_stats != "none",
        initializer=cfg.model.initializer,
        avg_pool=cfg.model.avg_pool,
        head_bias=cfg.model.head_bias,
        norm_before_act=cfg.model.norm_before_act,
        final_norm=cfg.model.final_norm,
        normalize=cfg.model.normalize,
    )

    @jit
    def embeddings_fn(slow_params, slow_state, inputs):
        return body.apply(slow_params, slow_state, None, inputs, None)[0][0]

    effective_batch_size, ragged = divmod(cfg.train.batch_size, cfg.train.apply_every)
    if ragged:
        raise ValueError(
            f"Global batch size {cfg.train.batch_size} must be divisible by "
            f"apply_every {cfg.train.apply_every}"
        )
    exp.log(f"Effective batch size: {effective_batch_size}")
    per_device_batch_size, ragged = divmod(effective_batch_size, num_devices)
    if ragged:
        raise ValueError(
            f"Effective batch size {effective_batch_size} must be divisible by "
            f"num devices {num_devices}"
        )
    exp.log(f"Per device batch size: {per_device_batch_size}")

    # Optimizers
    if cfg.train.scheduler == "cosine":
        schedule = delayed_cosine_decay_schedule(
            -cfg.train.outer_lr,
            cfg.train.cosine_transition_begin * cfg.train.apply_every,
            cfg.train.cosine_decay_steps * cfg.train.apply_every, 
            cfg.train.cosine_alpha,
        )
    elif cfg.train.scheduler == "step":
        schedule = ox.piecewise_constant_schedule(
            -cfg.train.outer_lr, {e: 0.1 for e in cfg.train.piecewise_constant_schedule}
        )
    outer_opt = ox.chain(
        ox.clip(10),
        ox.scale(1 / cfg.train.apply_every),
        ox.apply_every(cfg.train.apply_every),
        ox.additive_weight_decay(cfg.train.weight_decay),
        ox.scale_by_adam(),
        ox.scale_by_schedule(schedule),
    )

    train_sample_fn_kwargs = {
        "images": train_images,
        "labels": train_labels,
        "num_tasks": effective_batch_size,
        "way": cfg.train.way,
        "spt_shot": cfg.train.shot,
        "qry_shot": cfg.train.qry_shot,
        "shuffled_labels": cfg.train.method == "fsl",
        "disjoint": False,  # tasks can share classes
    }
    train_sample_fn = partial(
        fsl_sample,
        **train_sample_fn_kwargs,
    )
    fsl_build_ins = jit(
        partial(
            fsl_build,
            batch_size=effective_batch_size,
            way=cfg.train.way,
            shot=cfg.train.shot,
            qry_shot=cfg.train.qry_shot,
        )
    )
    if cfg.train.method == "cl":
        exp.log("Training for continual learning")
        train_cl_sample_fn_kwargs = {
            "images": train_images,
            "labels": train_labels,
            "num_tasks": effective_batch_size,
            "way": cfg.train.cl_qry_way,
            "spt_shot": 0,
            "qry_shot": cfg.train.cl_qry_shot,
            "shuffled_labels": False,
            "disjoint": False,  # tasks can share classes
        }
        train_cl_sample_fn = partial(
            fsl_sample,
            **train_cl_sample_fn_kwargs,
        )
        cl_build_ins = jit(
            partial(
                fsl_build,
                batch_size=effective_batch_size,
                way=cfg.train.cl_qry_way,
                shot=0,
                qry_shot=cfg.train.cl_qry_shot,
            )
        )

    def iterator(rng, sample_fn):
        rng, rng_sample = split(rng)
        while True:
            yield sample_fn(rng_sample)
            rng, rng_sample = split(rng)

    if cfg.train.prefetch > 0:
        # Move here because I cannot install it on local computer
        from acme.jax import utils as acme_utils

        exp.log(f"Using ACME prefetch {cfg.train.prefetch}")
        rng, rng_sampler = split(rng)
        train_input = acme_utils.prefetch(
            iterator(rng_sampler, train_sample_fn), buffer_size=cfg.train.prefetch
        )
        if cfg.train.method == "cl":
            rng, rng_sampler = split(rng)
            train_input_cl = acme_utils.prefetch(
                iterator(rng_sampler, train_cl_sample_fn),
                buffer_size=cfg.train.prefetch,
            )

    if "zero" in cfg.train.reset_head:
        exp.log("Using Zeros to reset head")
        head_initializer = lambda dtype: lambda rng, shape: jax.nn.initializers.zeros(
            rng,
            shape,
            dtype=dtype,
        )
    elif "glorot" in cfg.train.reset_head:
        exp.log("Using Glorot Uniform to reset head")
        head_initializer = jax.nn.initializers.glorot_uniform
    elif "kaiming" in cfg.train.reset_head:
        exp.log("Using Kaiming Normal to reset head")
        head_initializer = jax.nn.initializers.he_normal
    else:
        head_initializer = None

    # For ILR optimization
    inner_lr = jnp.ones([]) * cfg.train.inner_lr
    inner_opt = ox.sgd(inner_lr)

    def inner_opt_update_fn(lr, updates, state, params):
        inner_opt = ox.sgd(lr)
        return inner_opt.update(updates, state, params)

    train_inner_loop_ins = partial(
        fsl_inner_loop,
        is_training="inner" in cfg.model.track_stats,
        num_steps=cfg.train.num_inner_steps,
        slow_apply=body.apply,
        fast_apply=head.apply,
        loss_fn=mean_xe_and_acc_dict,
        opt_update_fn=inner_opt_update_fn,
    )
    if "cls" in cfg.train.reset_head:
        reset_fast_params_fn_outer = partial(reset_by_idxs, head_initializer)
    else:
        reset_fast_params_fn_outer = None
    train_outer_loop_ins = partial(
        outer_loop,
        is_training="outer" in cfg.model.track_stats,
        inner_loop=train_inner_loop_ins,
        slow_apply=body.apply,
        fast_apply=head.apply,
        loss_fn=mean_xe_and_acc_dict,
        # train_method=cfg.train.reset_head,
        track_slow_state=cfg.model.track_stats,
        reset_fast_params_fn=reset_fast_params_fn_outer,
    )
    train_batched_outer_loop_ins = partial(
        batched_outer_loop, outer_loop=train_outer_loop_ins
    )
    # if (cfg.train.reset_head == "fsl") or (
    #     cfg.train.reset_head == "fsl-reset-per-task"
    # ):
    #     step_ins = step
    # elif cfg.train.reset_head == "fsl-reset-zero":
    #     step_ins = step_reset
    step_fn = meta_step
    if "all" in cfg.train.reset_head:
        reset_fast_params_fn_step = partial(reset_all, head_initializer)
    else:
        reset_fast_params_fn_step = None
    step_pins = pmap(
        partial(
            step_fn,
            inner_opt_init=inner_opt.init,
            outer_opt_update=outer_opt.update,
            batched_outer_loop_ins=train_batched_outer_loop_ins,
            # train_method=cfg.train.reset_head,
            reset_fast_params_fn=reset_fast_params_fn_step,
            preprocess_images_fn=partial(
                preprocess_images,
                normalize_fn=normalize_fn,
                augment=cfg.train.augment,
                augment_fn=augment_fn,
            ),
            train_inner_lr=cfg.train.train_inner_lr,
        ),
        axis_name="i",
    )

    # Val data sampling
    test_sample_fn_kwargs = {
        "images": val_images,
        "labels": val_labels,
        "num_tasks": cfg.val.fsl.batch_size,
        # "way": 5,
        # "spt_shot": 1,
        "qry_shot": 15,
        "shuffled_labels": True,
        "disjoint": False,  # tasks can share classes
    }

    test_preprocess_images_jins = jit(
        partial(preprocess_images, normalize_fn=normalize_fn)
    )

    def sample_fn(rng, way, shot):
        rng_sample, rng_augment = split(rng)
        x, y = fsl_sample(rng_sample, way=way, spt_shot=shot, **test_sample_fn_kwargs)
        x = jax.device_put(x, jax.devices()[0])
        y = jax.device_put(y, jax.devices()[0])
        x_spt, y_spt, x_qry, y_qry = fsl_build(
            x, y, batch_size=cfg.val.fsl.batch_size, way=way, shot=shot, qry_shot=15
        )
        x_spt, x_qry = test_preprocess_images_jins(rng_augment, x_spt, x_qry)
        return x_spt, y_spt, x_qry, y_qry

    # For testing with Multinomial Rgression
    test_sample_fn_5_w_1_s = partial(sample_fn, way=5, shot=1)
    test_sample_fn_5_w_5_s = partial(sample_fn, way=5, shot=5)
    # For MAML style test
    if cfg.train.way > val_labels.shape[0]:
        exp.log(
            f"Training with {cfg.train.way}-way, but validation only has {val_labels.shape[0]} classes, reducing val way to {val_labels.shape[0]}"
        )
        val_way = val_labels.shape[0]
    else:
        val_way = cfg.train.way
    test_sample_fn_1_shot = partial(
        fsl_sample,
        spt_shot=1,
        way=val_way,
        **test_sample_fn_kwargs,
    )
    test_sample_fn_5_shot = partial(
        fsl_sample,
        spt_shot=5,
        way=val_way,
        **test_sample_fn_kwargs,
    )
    # Val loops
    test_inner_loop_ins = partial(
        fsl_inner_loop,
        is_training=False,
        num_steps=cfg.val.fsl.num_inner_steps,
        slow_apply=body.apply,
        fast_apply=head.apply,
        loss_fn=mean_xe_and_acc_dict,
        opt_update_fn=inner_opt_update_fn,
    )
    test_outer_loop_ins = partial(
        outer_loop,
        is_training=False,
        inner_loop=test_inner_loop_ins,
        slow_apply=body.apply,
        fast_apply=head.apply,
        loss_fn=mean_xe_and_acc_dict,
        track_slow_state="none",
    )
    test_batched_outer_loop_ins = partial(
        batched_outer_loop,
        outer_loop=test_outer_loop_ins,
        bspt_classes=None,
    )
    test_batched_outer_loop_ins = jit(test_batched_outer_loop_ins)
    test_fn_ins = partial(
        test_fsl_maml,
        inner_opt_init=inner_opt.init,
        #  sample_fn=test_sample_fn,
        batched_outer_loop=test_batched_outer_loop_ins,
        normalize_fn=normalize_fn,
        # build_fn=jit(
        #     partial(
        #         fsl_build,
        #         batch_size=cfg.val.batch_size,
        #         way=cfg.way,
        #         shot=shot,
        #         qry_shot=15,
        #     )
        # ),
        augment_fn=None,
        device=jax.devices()[0],
    )

    rng, rng_params = split(rng)
    (slow_params, fast_params, slow_state, fast_state,) = make_params(
        rng_params,
        cfg.dataset,
        body.init,
        body.apply,
        head.init,
    )

    # if (cfg.train.reset_head == "fsl") or (
    #     cfg.train.reset_head == "fsl-reset-per-task"
    # ):
    #     outer_opt_state = outer_opt.init((slow_params, fast_params))
    # elif cfg.train.reset_head == "fsl-reset-zero":
    #     outer_opt_state = outer_opt.init(slow_params)
    if cfg.train.train_inner_lr:
        outer_opt_state = outer_opt.init((slow_params, fast_params, inner_lr))
    else:
        outer_opt_state = outer_opt.init((slow_params, fast_params))

    # (slow_params, fast_params, slow_state, fast_state, outer_opt_state) = [
    #     jax.device_put(t, device)
    #     for t in (slow_params, fast_params, slow_state, fast_state, outer_opt_state)
    # ]
    replicate_array = lambda x: jnp.broadcast_to(x, (num_devices,) + x.shape)
    replicate_array_test = lambda x: jnp.broadcast_to(
        x, (cfg.val.fsl.batch_size,) + x.shape
    )
    (rep_slow_params, rep_fast_params, rep_inner_lr, rep_outer_opt_state,) = tree_map(
        replicate_array,
        (
            slow_params,
            fast_params,
            inner_lr,
            outer_opt_state,
        ),
    )
    (rep_slow_state, rep_fast_state,) = tree_map(
        lambda x: jnp.broadcast_to(x, (num_devices, per_device_batch_size) + x.shape),
        (
            slow_state,
            fast_state,
        ),
    )

    # augment = jit(augment)
    preprocess_images_jins = jit(
        partial(
            preprocess_images,
            normalize_fn=normalize_fn,
            augment=cfg.train.augment,
            augment_fn=augment_fn,
        )
    )

    pbar = tqdm(
        range(1, 1 + cfg.train.num_outer_steps),
        file=sys.stdout,
        miniters=25,
        mininterval=10,
        maxinterval=30,
        ncols=0,
    )
    best_val_acc = 0.0
    counter = 0
    for i in range(1, cfg.train.num_outer_steps * cfg.train.apply_every + 1):
        if (i % cfg.train.apply_every) == 0:
            pbar.update()
            counter += 1

        rng, rng_step, rng_sample,pudate rng_augment, rng_sample_cl = split(rng, 5)
        if cfg.train.prefetch > 0:
            x, y = next(train_input)
        else:
            x, y = train_sample_fn(rng_sample)
        # x = jax.device_put(x, device)
        # y = jax.device_put(y, device)
        x_spt, y_spt, x_qry, y_qry = fsl_build_ins(x, y)
        if cfg.train.method == "cl":
            # (rng_sample_cl,) = split(rng_sample, 1)
            if cfg.train.prefetch > 0:
                x_cl, y_cl = next(train_input_cl)
            else:
                x_cl, y_cl = train_cl_sample_fn(rng_sample_cl)
            # x_cl = jax.device_put(x_cl, device)
            # y_cl = jax.device_put(y_cl, device)
            _, _, x_cl_qry, y_cl_qry = cl_build_ins(x_cl, y_cl)
            x_qry = onp.concatenate((x_spt, x_qry, x_cl_qry), 1)
            y_qry = onp.concatenate((y_spt, y_qry, y_cl_qry), 1)

        spt_classes = onp.unique(y_spt, axis=1)
        # x_spt, x_qry = preprocess_images_jins(rng_augment, x_spt, x_qry)
        x_spt = x_spt.reshape(num_devices, per_device_batch_size, *x_spt.shape[1:])
        x_qry = x_qry.reshape(num_devices, per_device_batch_size, *x_qry.shape[1:])
        y_spt = y_spt.reshape(num_devices, per_device_batch_size, *y_spt.shape[1:])
        y_qry = y_qry.reshape(num_devices, per_device_batch_size, *y_qry.shape[1:])
        spt_classes = spt_classes.reshape(
            num_devices, per_device_batch_size, *spt_classes.shape[1:]
        )
        # x = x / 255
        # x = augment(rng, flatten(x, (0, 2))).reshape(*x.shape)
        # x = normalize_fn(x)

        (
            rep_outer_opt_state,
            rep_slow_params,
            rep_fast_params,
            rep_inner_lr,
            rep_slow_state,
            rep_fast_state,
            info,
        ) = step_pins(
            split(rng_step, num_devices),
            tree_map(replicate_array, jnp.array(i)),
            rep_outer_opt_state,
            rep_slow_params,
            rep_fast_params,
            rep_inner_lr,
            rep_slow_state,
            rep_fast_state,
            x_spt,
            y_spt,
            x_qry,
            y_qry,
            spt_classes,
        )

        if ((counter == 1) and (i == cfg.train.apply_every)) or (
            ((i % cfg.train.apply_every) == 0)
            and ((counter) % cfg.train.val_interval) == 0
        ):
            exp.log("\nEvaluating MAML")
            slow_state = jax.tree_map(lambda xs: xs[0, 0], rep_slow_state)
            fast_state = jax.tree_map(lambda xs: xs[0, 0], rep_fast_state)
            test_slow_state = tree_map(replicate_array_test, slow_state)
            test_fast_state = tree_map(replicate_array_test, fast_state)
            rng, rng_reset = split(rng)
            tree_flat, tree_struct = jax.tree_flatten(fast_params)
            rng_tree = jax.tree_unflatten(tree_struct, split(rng_reset, len(tree_flat)))
            slow_params = jax.tree_map(lambda xs: xs[0], rep_slow_params)
            fast_params = jax.tree_map(lambda xs: xs[0], rep_fast_params)
            inner_lr = jax.tree_map(lambda xs: xs[0], rep_inner_lr)
            test_fast_params = jax.tree_multimap(
                partial(reset_all, head_initializer, None), rng_tree, fast_params
            )
            now = time.time()
            rng, rng_test_1, rng_test_5, rng_test_lr_1, rng_test_lr_5 = split(rng, 5)
            fsl_maml_1_res = test_fn_ins(
                rng_test_1,
                slow_params,
                test_fast_params,
                inner_lr,
                test_slow_state,
                test_fast_state,
                num_batches=cfg.val.fsl.num_tasks // cfg.val.fsl.batch_size,
                sample_fn=test_sample_fn_1_shot,
                build_fn=partial(
                    fsl_build,
                    batch_size=cfg.val.fsl.batch_size,
                    way=val_way,
                    shot=1,
                    qry_shot=15,
                ),
            )
            fsl_maml_5_res = test_fn_ins(
                rng_test_5,
                slow_params,
                test_fast_params,
                inner_lr,
                test_slow_state,
                test_fast_state,
                num_batches=cfg.val.fsl.num_tasks // cfg.val.fsl.batch_size,
                sample_fn=test_sample_fn_5_shot,
                build_fn=partial(
                    fsl_build,
                    batch_size=cfg.val.fsl.batch_size,
                    way=val_way,
                    shot=5,
                    qry_shot=15,
                ),
            )

            exp.log("Fitting Multinomial Regression")
            fsl_lr_1_preds, fsl_lr_1_targets = test_fsl_embeddings(
                rng_test_lr_1,
                partial(embeddings_fn, slow_params, slow_state),
                test_sample_fn_5_w_1_s,
                cfg.val.fsl.num_tasks // cfg.val.fsl.batch_size,
                pool=0,
                # n_jobs=cfg.n_jobs_lr,
            )
            fsl_lr_5_preds, fsl_lr_5_targets = test_fsl_embeddings(
                rng_test_lr_5,
                partial(embeddings_fn, slow_params, slow_state),
                test_sample_fn_5_w_5_s,
                cfg.val.fsl.num_tasks // cfg.val.fsl.batch_size,
                pool=0,
                # n_jobs=cfg.n_jobs_lr,
            )

            fsl_maml_loss_1 = fsl_maml_1_res[0].mean()
            fsl_maml_acc_1 = fsl_maml_1_res[1]["outer"]["final"]["aux"][0]["acc"].mean()
            fsl_maml_loss_5 = fsl_maml_5_res[0].mean()
            fsl_maml_acc_5 = fsl_maml_5_res[1]["outer"]["final"]["aux"][0]["acc"].mean()

            fsl_lr_1_acc = (fsl_lr_1_preds == fsl_lr_1_targets).astype(onp.float).mean()
            fsl_lr_5_acc = (fsl_lr_5_preds == fsl_lr_5_targets).astype(onp.float).mean()

            exp.log(f"\nValidation step {counter} results:")
            exp.log(f"Multinomial Regression 5-way-1-shot acc: {fsl_lr_1_acc}")
            exp.log(f"Multinomial Regression 5-way-5-shot acc: {fsl_lr_5_acc}")
            exp.log(
                f"{val_way}-way-1-shot acc: {fsl_maml_acc_1}, loss: {fsl_maml_loss_1}"
            )
            exp.log(
                f"{val_way}-way-5-shot acc: {fsl_maml_acc_5}, loss: {fsl_maml_loss_5}"
            )

            exp.log_metrics(
                {
                    "5w5s-lr-acc": fsl_lr_5_acc,
                    "5w1s-lr-acc": fsl_lr_1_acc,
                    "acc_5": fsl_maml_acc_5,
                    "acc_1": fsl_maml_acc_1,
                    "loss_5": fsl_maml_loss_5,
                    "loss_1": fsl_maml_loss_1,
                    "inner_lr": inner_lr,
                },
                step=counter,
                prefix="val",
            )

            if fsl_lr_5_acc > best_val_acc:
                # if fsl_maml_acc_5 > best_val_acc:
                best_val_acc = fsl_lr_5_acc
                outer_opt_state = jax.tree_map(lambda xs: xs[0, 0], rep_outer_opt_state)
                # best_val_acc = fsl_maml_acc_5
                exp.log(
                    f"\  New best {val_way}-way-5-shot validation accuracy: {best_val_acc}"
                )
                if not cfg.no_log:
                    exp.log("Saving checkpoint\n")
                    with open(osp.join(exp.exp_dir, "checkpoints/best.ckpt"), "wb") as f:
                        dill.dump(
                            {
                                "val_acc_1": fsl_maml_acc_1,
                                "val_loss_1": fsl_maml_loss_1,
                                "val_acc_5": fsl_maml_acc_5,
                                "val_loss_5": fsl_maml_loss_5,
                                # "val_lr_5w5s_acc": fsl_lr_5_acc,
                                # "val_lr_5w1s_acc": fsl_lr_1_acc,
                                "optimizer_state": outer_opt_state,
                                "slow_params": slow_params,
                                "fast_params": fast_params,
                                "slow_state": slow_state,
                                "fast_state": fast_state,
                                "rng": rng,
                                "counter": counter,
                            },
                            f,
                            protocol=3,
                        )

        # print(counter, i)
        if ((counter == 1) and (i == cfg.train.apply_every)) or (
            ((i % cfg.train.apply_every) == 0)
            and (
                (((counter) % cfg.progress_bar_refresh_rate) == 0)
                or (((counter) % cfg.train.val_interval) == 0)
            )
        ):
            train_loss = info["outer"]["final"]["loss"].mean()
            train_final_outer_acc = info["outer"]["final"]["aux"][0]["acc"].mean()
            exp.log_metrics(
                {
                    "foa": train_final_outer_acc,
                    "loss": train_loss,
                },
                step=counter,
                prefix="train",
            )

            inner_lr = jax.tree_map(lambda xs: xs[0], rep_inner_lr)
            outer_opt_state = jax.tree_map(lambda xs: xs[0, 0], rep_outer_opt_state)
            current_lr = schedule(outer_opt_state[-1].count)
            pbar.set_postfix(
                lr=f"{current_lr:.4f}",
                inner_lr=f"{inner_lr:.4f}",
                loss=f"{train_loss:.2f}",
                foa=f"{train_final_outer_acc:.2f}",
                va1=f"{fsl_maml_acc_1:.2f}",
                va5=f"{fsl_maml_acc_5:.2f}",
                # vlr5=f"{fsl_lr_5_acc:.2f}",
                # vlr1=f"{fsl_lr_1_acc:.2f}",
                refresh=False,
            )


if __name__ == "__main__":
    # parser = parse_args()
    args, cfg = parse_args()
    main(args, cfg)