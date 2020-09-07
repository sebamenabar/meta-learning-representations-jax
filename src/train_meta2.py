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
    grad,
    vmap,
    pmap,
    random,
    partial,
    tree_map,
    numpy as jnp,
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
    mean_xe_and_acc_dict,
)
from data import prepare_data, augment
from experiment import Experiment, Logger
from data.sampling import FSLSampler, fsl_build
from models.maml_conv import prepare_model, make_params

def parse_args(parser=None):
    parser = Experiment.add_args(parser)
    # parser = miniimagenet_cnn_argparse(parser)
    # Training arguments

    parser.add_argument("--train.num_outer_steps", type=int, default=30000)
    parser.add_argument("--train.batch_size", help="Number of FSL tasks", default=4, type=int)
    parser.add_argument("--train.way", help="Number of classes per task", default=5, type=int)
    parser.add_argument(
        "--train.shot", help="Number of samples per class", default=5, type=int
    )
    parser.add_argument(
        "--train.qry_shot", type=int, help="Number of quried samples per class", default=10,
    )
    parser.add_argument("--train.inner_lr", type=float, default=1e-2)
    parser.add_argument("--train.outer_lr", type=float, default=1e-3)
    parser.add_argument("--train.num_inner_steps", type=int, default=5)

    parser.add_argument("--train.cosine_schedule", action="store_true", default=False)
    parser.add_argument("--train.cosine_alpha", type=float, default=0.01)
    parser.add_argument("--train.piecewise_constant_schedule", nargs="*", type=int, default=[10000, 25000])

    parser.add_argument("--train.augment", default="none", choices=["none", "all", "spt", "qry"])

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
    preprocess_images_fn=None,
):
    rng_preprocess, rng_step = split(rng)
    if preprocess_images_fn:
        x_spt, x_qry = preprocess_images_fn(rng_preprocess, x_spt, x_qry)

    inner_opt_state = inner_opt_init(fast_params)
    
    grad_fn = grad(batched_outer_loop_ins, argnums=(0, 1), has_aux=True)
    grads, (states, logs) = grad_fn(
        slow_params,
        fast_params,
        slow_state,
        fast_state,
        inner_opt_state,
        split(rng_step, x_spt.shape[0]),
        x_spt,
        y_spt,
        x_qry,
        y_qry,
        spt_classes,
    )
    
    grads = jax.tree_map(lambda v: jax.lax.pmean(v, axis_name="i"), grads)
    updates, outer_opt_state = outer_opt_update(
        grads, outer_opt_state, (slow_params, fast_params)
    )
    slow_params, fast_params = ox.apply_updates((slow_params, fast_params), updates)

    return outer_opt_state, slow_params, fast_params, *states, logs

def preprocess_images(rng, x_spt, x_qry, normalize_fn, augment="none", augment_fn=None):
    if augment == "all":
        rng_spt, rng_qry = split(rng)
        x_spt = augment_fn(rng_spt, x_spt)
        x_qry = augment_fn(rng_qry, x_qry)
    elif augment == "spt":
        x_spt = augment_fn(rng, x_spt)
    elif augment == "qry":
        x_qry = augment_fn(rng, x_qry)
        
    x_spt = normalize_fn(x_spt / 255)
    x_qry = normalize_fn(x_qry / 255)
    
    return x_spt, x_qry

if __name__ == "__main__":
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
    rng = random.PRNGKey(cfg.seed)  # Default seed is 0
    exp.log(f"Seed {cfg.seed}")
    exp.log(f"JAX available devices {jax.devices()}")
    num_devices = max(cfg.gpus, 1)
    exp.log(f"Using {num_devices} devices")

    # Data
    train_images, train_labels, normalize_fn = prepare_data(
        cfg.dataset,
        osp.join(
            cfg.data_dir,
            "miniImageNet_category_split_train_phase_train_ordered.pickle",
        ),
    )
    val_images, _val_labels, _ = prepare_data(
        cfg.dataset,
        osp.join(cfg.data_dir, "miniImageNet_category_split_val_ordered.pickle",),
    )
    val_labels = (
        _val_labels - 64 # TEMP
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
        track_stats=False, # TODO
        initializer=cfg.model.initializer,
        avg_pool=cfg.model.avg_pool,
        head_bias=cfg.model.head_bias,
    )

    # Optimizers
    inner_opt = ox.sgd(cfg.train.inner_lr)
    if cfg.train.cosine_schedule:
        schedule = ox.cosine_decay_schedule(-cfg.train.outer_lr, cfg.train.num_outer_steps, cfg.train.cosine_alpha)
    else:
        schedule = ox.piecewise_constant_schedule(
            -cfg.train.outer_lr, {e: 0.1 for e in cfg.train.piecewise_constant_schedule}
        )
    outer_opt = ox.chain(
        ox.clip(10), ox.scale_by_adam(), ox.scale_by_schedule(schedule),
    )

    # SAMPLING
    per_device_batch_size, ragged = divmod(cfg.train.batch_size, num_devices)
    if ragged:
      raise ValueError(
          f'Global batch size {cfg.train.batch_size} must be divisible by '
          f'num devices {num_devices}')
    exp.log(f"Per device batch size: {per_device_batch_size}")
    sampler = FSLSampler(
        train_images,
        train_labels,
        shuffle_labels=True, # TODO
    )
    train_sample_jins = jit(partial(sampler.sample, batch_size=cfg.train.batch_size, way=cfg.train.way, shot=cfg.train.shot + cfg.train.qry_shot), static_argnums=(0,))
    fsl_build_jins = jit(
        partial(
            fsl_build,
            num_devices=num_devices,
            batch_size=per_device_batch_size,
            way=cfg.train.way,
            shot=cfg.train.shot,
            qry_shot=cfg.train.qry_shot,
        )
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
    )
    train_batched_outer_loop_ins = partial(
        batched_outer_loop, train_outer_loop_ins
    )
    step_ins = step
    step_pins = pmap(
        partial(
            step_ins,
            inner_opt_init=inner_opt.init,
            outer_opt_update=outer_opt.update,
            batched_outer_loop_ins=train_batched_outer_loop_ins,
            spt_classes=None,
            preprocess_images_fn=partial(preprocess_images, normalize_fn=normalize_fn, augment=cfg.train.augment, augment_fn=augment)
        ),
        axis_name="i",
    )

    rng, rng_params = split(rng)
    replicate_array = lambda x: jnp.broadcast_to(x, (num_devices,) + x.shape)

    (slow_params, fast_params, slow_state, fast_state,) = make_params(
        rng_params, cfg.dataset, body.init, body.apply, head.init,
    )
    outer_opt_state = outer_opt.init((slow_params, fast_params))

    (replicated_slow_params, replicated_fast_params, replicated_slow_state, replicated_fast_state, replicated_outer_opt_state) = [
        tree_map(replicate_array, tree) for tree in [slow_params, fast_params, slow_state, fast_state, outer_opt_state]
    ]

    pbar = tqdm(
        range(cfg.train.num_outer_steps),
        file=sys.stdout,
        miniters=25,
        mininterval=5,
        maxinterval=20,
    )
    best_val_acc = 0.0
    for i in pbar:
        rng, rng_step, rng_sample = split(rng, 3)
        x, y = train_sample_jins(rng_sample)
        x_spt, y_spt, x_qry, y_qry = fsl_build_jins(x, y)

        # spt_classes = jax.device_put(onp.unique(y_spt, axis=1), device)
        (
            replicated_outer_opt_state, replicated_slow_params, replicated_fast_params, replicated_slow_state, replicated_fast_state,
            info,
        ) = step_pins(
            split(rng_step, num_devices),
            tree_map(replicate_array, jnp.array(i)),
            replicated_outer_opt_state,
            replicated_slow_params,
            replicated_fast_params,
            replicated_slow_state,
            replicated_fast_state,
            x_spt,
            y_spt,
            x_qry,
            y_qry,
            # spt_classes,
        )

        if (
            (i == 0)
            or (((i + 1) % cfg.progress_bar_refresh_rate) == 0)
            or (((i + 1) % cfg.val_interval) == 0)
        ):
            outer_opt_state = jax.tree_map(lambda xs: xs[0], replicated_outer_opt_state)
            current_lr = schedule(outer_opt_state[-1].count)
            pbar.set_postfix(
                lr=f"{current_lr:.4f}",
                loss=f"{info['outer']['final']['loss'].mean():.2f}",
                foa=f"{info['outer']['final']['aux'][0]['acc'].mean():.2f}",
            )
