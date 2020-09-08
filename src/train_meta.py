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
    # outer_loop_reset_per_task,
)
from data import prepare_data, augment
from experiment import Experiment, Logger
from data.sampling import fsl_sample, fsl_build_single, BatchSampler, FSLSampler
from models.maml_conv import miniimagenet_cnn_argparse, prepare_model, make_params
from test_utils import test_fsl_maml, test_fsl_embeddings
from test_sup import test_sup_cosine

TRAIN_SIZE = 500


def parse_args(parser=None):
    parser = Experiment.add_args(parser)
    # parser = miniimagenet_cnn_argparse(parser)
    # Training arguments
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
    parser.add_argument("--train.inner_lr", type=float, default=1e-2)
    parser.add_argument("--train.outer_lr", type=float, default=1e-3)
    parser.add_argument("--train.num_inner_steps", type=int, default=5)

    parser.add_argument("--train.cosine_schedule", action="store_true", default=False)
    parser.add_argument("--train.cosine_alpha", type=float, default=0.01)
    parser.add_argument(
        "--train.piecewise_constant_schedule",
        nargs="*",
        type=int,
        default=[10000, 25000],
    )

    parser.add_argument(
        "--train.augment", default="none", choices=["none", "all", "spt", "qry"]
    )

    parser.add_argument("--model.output_size", type=int)
    parser.add_argument("--model.hidden_size", default=32, type=int)
    parser.add_argument("--model.activation", default="relu", type=str)
    parser.add_argument(
        "--model.initializer",
        default="glorot_uniform",
        type=str,
        choices=["kaiming_normal", "glorot_uniform"],
    )
    parser.add_argument("--model.avg_pool", default=False, action="store_true")
    parser.add_argument("--model.head_bias", default=False, action="store_true")

    args = parser.parse_args()
    cfg = edict(train=edict(), val=edict(), model=edict())
    for argname, argval in vars(args).items():
        rsetattr(cfg, argname, argval)

    return args, cfg


def step(
    rng,
    step_num,
    outer_opt_state,
    slow_params,
    fast_params,
    slow_state,
    fast_state,
    x_spt,
    y_spt,
    x_qry,
    y_qry,
    spt_classes,
    inner_opt_init,
    outer_opt_update,
    batched_outer_loop_ins,
    train_method=None,  # Just for compatibility
):
    inner_opt_state = inner_opt_init(fast_params)

    (outer_loss, (states, info)), grads = value_and_grad(
        batched_outer_loop_ins, (0, 1), has_aux=True
    )(
        slow_params,
        fast_params,
        slow_state,
        fast_state,
        inner_opt_state,
        split(rng, x_spt.shape[0]),
        x_spt,
        y_spt,
        x_qry,
        y_qry,
        spt_classes,
    )
    updates, outer_opt_state = outer_opt_update(
        grads, outer_opt_state, (slow_params, fast_params)
    )
    slow_params, fast_params = ox.apply_updates((slow_params, fast_params), updates)

    return outer_opt_state, slow_params, fast_params, *states, info


def step_reset(
    rng,
    step_num,
    outer_opt_state,
    slow_params,
    fast_params,
    slow_state,
    fast_state,
    x_spt,
    y_spt,
    x_qry,
    y_qry,
    spt_classes,
    inner_opt_init,
    outer_opt_update,
    batched_outer_loop_ins,
    train_method,
):
    if train_method == "fsl-reset-zero":
        print("Reseting head params to zero")
        fast_params = hk.data_structures.merge(
            {
                "mini_imagenet_cnn_head/linear": {
                    "w": jnp.zeros(
                        fast_params["mini_imagenet_cnn_head/linear"]["w"].shape
                    ),
                }
            }
        )
    else:
        raise NameError(f"Unkwown train method `{train_method}`")

    inner_opt_state = inner_opt_init(fast_params)

    (outer_loss, (slow_state, _, info)), grads = value_and_grad(
        batched_outer_loop_ins, 0, has_aux=True
    )(
        slow_params,
        fast_params,
        slow_state,
        fast_state,
        inner_opt_state,
        split(rng, x_spt.shape[0]),
        x_spt,
        y_spt,
        x_qry,
        y_qry,
        spt_classes,
    )
    updates, outer_opt_state = outer_opt_update(grads, outer_opt_state, slow_params,)
    slow_params = ox.apply_updates(slow_params, updates)

    return outer_opt_state, slow_params, fast_params, slow_state, fast_state, info


if __name__ == "__main__":
    # parser = parse_args()
    args, cfg = parse_args()
    # cfg = parse_and_build_cfg(args)
    # The Experiment class creates a directory for the experiment,
    # copies source files and creates configurations files
    # It also creates a logfile.log which can be written to with exp.log
    # that is a wrapper of the print method
    exp = Experiment(cfg, args)
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
    cpu, device = setup_device(cfg.gpus, default_platform="cpu")
    rng = random.PRNGKey(cfg.seed)  # Default seed is 0
    exp.log(f"Seed {cfg.seed}")
    exp.log(f"JAX available devices {jax.devices()}")
    num_devices = max(cfg.gpus, 1)
    exp.log(f"Using {num_devices} devices")

    # Data
    train_images, train_labels, normalize_fn, (mean, std) = prepare_data(
        cfg.dataset,
        osp.join(
            cfg.data_dir,
            "miniImageNet_category_split_train_phase_train_ordered.pickle",
        ),
    )
    val_images, _val_labels, _, _ = prepare_data(
        cfg.dataset,
        osp.join(cfg.data_dir, "miniImageNet_category_split_val_ordered.pickle",),
    )
    val_labels = (
        _val_labels - 64
    )  # make val labels start at 0 (they originally begin are at 64-79)

    exp.log("Train data:", train_images.shape, train_labels.shape)
    exp.log(
        "Validation data:", val_images.shape, val_images.shape,
    )

    # Model
    body, head = prepare_model(
        cfg.dataset,
        cfg.model.output_size,
        cfg.model.hidden_size,
        cfg.model.activation,
        track_stats=False,
        initializer=cfg.model.initializer,
        avg_pool=cfg.model.avg_pool,
        head_bias=cfg.model.head_bias,
    )

    # Optimizers
    inner_opt = ox.sgd(cfg.train.inner_lr)
    if cfg.train.cosine_schedule:
        schedule = ox.cosine_decay_schedule(
            -cfg.train.outer_lr, cfg.train.num_outer_steps, cfg.train.cosine_alpha
        )
    else:
        schedule = ox.piecewise_constant_schedule(
            -cfg.train.outer_lr, {e: 0.1 for e in cfg.train.piecewise_constant_schedule}
        )
    outer_opt = ox.chain(
        ox.clip(10), ox.scale_by_adam(), ox.scale_by_schedule(schedule),
    )

    # Train data sampling

    # train_sample_fn_kwargs = {
    #     "images": train_images,
    #     "labels": train_labels,
    #     "num_tasks": cfg.train.batch_size,
    #     "way": cfg.train.way,
    #     "spt_shot": cfg.train.shot,
    #     "qry_shot": cfg.train.qry_shot,
    #     "shuffled_labels": True,
    #     "disjoint": False,  # tasks can share classes
    # }
    # train_sample_fn = jit(partial(fsl_sample, **train_sample_fn_kwargs,), backend="cpu")

    sampler = FSLSampler(train_images, train_labels, shuffle_labels=True,)  # TODO
    train_sample_jins = jit(
        partial(
            sampler.sample,
            batch_size=cfg.train.batch_size,
            way=cfg.train.way,
            shot=cfg.train.shot + cfg.train.qry_shot,
        ),
        static_argnums=(0,),
        backend="cpu",
    )

    # Train loops
    train_inner_loop_ins = partial(
        fsl_inner_loop,
        body.apply,
        head.apply,
        mean_xe_and_acc_dict,
        inner_opt.update,
        is_training=True,
        num_steps=cfg.train.num_inner_steps,
    )
    train_outer_loop_ins = partial(
        outer_loop,
        body.apply,
        head.apply,
        mean_xe_and_acc_dict,
        train_inner_loop_ins,
        is_training=True,
        # train_method=cfg.train_method,
    )
    train_batched_outer_loop_ins = partial(batched_outer_loop, train_outer_loop_ins)
    step_ins = step
    step_ins = jit(
        partial(
            step_ins,
            inner_opt_init=inner_opt.init,
            outer_opt_update=outer_opt.update,
            batched_outer_loop_ins=train_batched_outer_loop_ins,
            # train_method=cfg.train_method,
        ),
    )
    augment = jit(augment)
    fsl_build_ins = jit(
        partial(
            fsl_build_single,
            # num_devices=1,
            batch_size=cfg.train.batch_size,
            way=cfg.train.way,
            shot=cfg.train.shot,
            qry_shot=cfg.train.qry_shot,
        ),
        backend="cpu",
    )
    # Val data sampling
    # test_sample_fn_kwargs = {
    #     "images": val_images,
    #     "labels": val_labels,
    #     "num_tasks": cfg.val.batch_size,
    #     "way": cfg.way,
    #     # "spt_shot": 1,
    #     "qry_shot": 15,
    #     "shuffled_labels": True,
    #     "disjoint": False,  # tasks can share classes
    # }
    # test_sample_fn_1 = partial(fsl_sample, spt_shot=1, **test_sample_fn_kwargs,)
    # test_sample_fn_5 = partial(fsl_sample, spt_shot=5, **test_sample_fn_kwargs,)
    # # Val loops
    # test_inner_loop_ins = partial(
    #     fsl_inner_loop,
    #     is_training=False,
    #     num_steps=cfg.num_inner_steps_test,
    #     slow_apply=body.apply,
    #     fast_apply=head.apply,
    #     loss_fn=mean_xe_and_acc_dict,
    #     opt_update_fn=inner_opt.update,
    # )
    # test_outer_loop_ins = partial(
    #     outer_loop,
    #     is_training=False,
    #     inner_loop=test_inner_loop_ins,
    #     slow_apply=body.apply,
    #     fast_apply=head.apply,
    #     loss_fn=mean_xe_and_acc_dict,
    # )
    # test_batched_outer_loop_ins = partial(
    #     batched_outer_loop, outer_loop=test_outer_loop_ins, spt_classes=None,
    # )
    # test_batched_outer_loop_ins = jit(test_batched_outer_loop_ins)
    # test_fn_ins = partial(
    #     test_fsl_maml,
    #     inner_opt_init=inner_opt.init,
    #     #  sample_fn=test_sample_fn,
    #     batched_outer_loop=test_batched_outer_loop_ins,
    #     normalize_fn=normalize_fn,
    #     # build_fn=jit(
    #     #     partial(
    #     #         fsl_build,
    #     #         batch_size=cfg.val.batch_size,
    #     #         way=cfg.way,
    #     #         shot=shot,
    #     #         qry_shot=15,
    #     #     )
    #     # ),
    #     augment_fn=augment,
    #     device=device,
    # )
    rng, rng_params = split(rng)
    (slow_params, fast_params, slow_state, fast_state,) = [
        jax.device_put(p, device)
        for p in make_params(rng_params, cfg.dataset, body.init, body.apply, head.init)
    ]
    outer_opt_state = outer_opt.init((slow_params, fast_params))
    mean, std = jax.device_put(mean, device), jax.device_put(std, device)

    pbar = tqdm(
        range(cfg.train.num_outer_steps),
        file=sys.stdout,
        miniters=25,
        mininterval=10,
        maxinterval=30,
    )
    best_val_acc = 0.0
    for i in pbar:
        rng, rng_step, rng_sample, rng_augment = split(rng, 4)
        # x, y = train_sample_fn(rng_sample)
        x, y = train_sample_jins(rng_sample)
        x = jax.device_put(x, device)
        y = jax.device_put(y, device)
        x = x / 255
        x = augment(rng, flatten(x, (0, 2))).reshape(*x.shape)
        # x = normalize_fn(x)
        x = (x - mean) / std
        x_spt, y_spt, x_qry, y_qry = fsl_build_ins(x, y)

        spt_classes = jax.device_put(onp.unique(y_spt, axis=1), device)
        (
            outer_opt_state,
            slow_params,
            fast_params,
            slow_state,
            fast_state,
            info,
        ) = step_ins(
            rng_step,
            i,
            outer_opt_state,
            slow_params,
            fast_params,
            slow_state,
            fast_state,
            x_spt,
            y_spt,
            x_qry,
            y_qry,
            spt_classes,
        )

        # if (i == 0) or (((i + 1) % cfg.val_interval) == 0):
        #     now = time.time()
        #     rng, rng_test_1, rng_test_5 = split(rng, 3)
        #     fsl_maml_1_res = test_fn_ins(
        #         rng_test_1,
        #         slow_params,
        #         fast_params,
        #         slow_state,
        #         fast_state,
        #         num_batches=cfg.fsl_num_tasks_test // cfg.val.batch_size,
        #         sample_fn=test_sample_fn_1,
        #         build_fn=jit(
        #             partial(
        #                 fsl_build,
        #                 batch_size=cfg.val.batch_size,
        #                 way=cfg.way,
        #                 shot=1,
        #                 qry_shot=15,
        #             )
        #         ),
        #     )
        #     fsl_maml_5_res = test_fn_ins(
        #         rng_test_5,
        #         slow_params,
        #         fast_params,
        #         slow_state,
        #         fast_state,
        #         num_batches=cfg.fsl_num_tasks_test // cfg.val.batch_size,
        #         sample_fn=test_sample_fn_5,
        #         build_fn=jit(
        #             partial(
        #                 fsl_build,
        #                 batch_size=cfg.val.batch_size,
        #                 way=cfg.way,
        #                 shot=5,
        #                 qry_shot=15,
        #             )
        #         ),
        #     )

        #     fsl_maml_loss_1 = fsl_maml_1_res[0].mean()
        #     fsl_maml_acc_1 = fsl_maml_1_res[1]["outer"]["final"]["aux"][0]["acc"].mean()
        #     fsl_maml_loss_5 = fsl_maml_5_res[0].mean()
        #     fsl_maml_acc_5 = fsl_maml_5_res[1]["outer"]["final"]["aux"][0]["acc"].mean()

        #     exp.log(f"\nValidation step {i} results:")
        #     exp.log(f"5-way-1-shot acc: {fsl_maml_acc_1}, loss: {fsl_maml_loss_1}")
        #     exp.log(f"5-way-5-shot acc: {fsl_maml_acc_5}, loss: {fsl_maml_loss_5}")

        #     exp.log_metrics(
        #         {
        #             "acc_5": fsl_maml_acc_5,
        #             "acc_1": fsl_maml_acc_1,
        #             "loss_5": fsl_maml_loss_5,
        #             "loss_1": fsl_maml_loss_1,
        #         },
        #         step=i + 1,
        #         prefix="val",
        #     )

        #     if fsl_maml_acc_5 > best_val_acc:
        #         best_val_acc = fsl_maml_acc_5
        #         exp.log(f"\  New best 5-way-5-shot validation accuracy: {best_val_acc}")
        #         exp.log("Saving checkpoint\n")
        #         with open(osp.join(exp.exp_dir, "checkpoints/best.ckpt"), "wb") as f:
        #             dill.dump(
        #                 {
        #                     "val_acc_1": fsl_maml_acc_1,
        #                     "val_loss_1": fsl_maml_loss_1,
        #                     "val_acc_5": fsl_maml_acc_5,
        #                     "val_loss_5": fsl_maml_loss_5,
        #                     "optimizer_state": outer_opt_state,
        #                     "slow_params": slow_params,
        #                     "fast_params": fast_params,
        #                     "slow_state": slow_state,
        #                     "fast_state": fast_state,
        #                     "rng": rng,
        #                     "i": i,
        #                 },
        #                 f,
        #                 protocol=3,
        # )

        if (
            (i == 0)
            or (((i + 1) % cfg.progress_bar_refresh_rate) == 0)
            or (((i + 1) % cfg.val_interval) == 0)
        ):
            # exp.log_metrics(
            #     {
            #         "foa": info["outer"]["final"]["aux"][0]["acc"].mean(),
            #         "loss": info["outer"]["final"]["loss"].mean(),
            #     },
            #     step=i + 1,
            #     prefix="train",
            # )

            current_lr = schedule(outer_opt_state[-1].count)
            pbar.set_postfix(
                lr=f"{current_lr:.4f}",
                loss=f"{info['outer']['final']['loss'].mean():.2f}",
                foa=f"{info['outer']['final']['aux'][0]['acc'].mean():.2f}",
                # va1=f"{fsl_maml_acc_1:.2f}",
                # va5=f"{fsl_maml_acc_5:.2f}",
            )
