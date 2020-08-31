import os
import sys
import os.path as osp

import dill
import time
from tqdm import tqdm
from argparse import ArgumentParser

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

from lib import (
    flatten,
    outer_loop,
    setup_device,
    fsl_inner_loop,
    batched_outer_loop,
    parse_and_build_cfg,
    mean_xe_and_acc_dict,
)
from data import prepare_data
from experiment import Experiment, Logger
from data.sampling import fsl_sample_transfer_build, BatchSampler
from models.maml_conv import miniimagenet_cnn_argparse, prepare_model, make_params
from test_utils import test_fsl_maml, test_fsl_embeddings
from test_sup import test_sup_cosine

TRAIN_SIZE = 500


def parse_args(parser=None):
    parser = Experiment.add_args(parser)
    parser = miniimagenet_cnn_argparse(parser)
    # Training arguments
    parser.add_argument(
        "--meta_batch_size", help="Number of FSL tasks", default=20, type=int
    )
    parser.add_argument(
        "--meta_batch_size_test", help="Number of FSL tasks", default=25, type=int
    )
    parser.add_argument("--sup_batch_size_test", default=512, type=int)
    parser.add_argument("--way", help="Number of classes per task", default=5, type=int)
    parser.add_argument(
        "--shot", help="Number of samples per class", default=5, type=int
    )
    parser.add_argument(
        "--qry_shot", type=int, help="Number of quried samples per class", default=10,
    )

    # Optimization arguments
    parser.add_argument("--train_method", type=str, default="fsl", choices=["fsl"])
    parser.add_argument("--inner_lr", type=float, default=1e-2)
    parser.add_argument("--outer_lr", type=float, default=1e-3)
    parser.add_argument("--num_inner_steps_train", type=int, default=5)
    parser.add_argument("--num_inner_steps_test", type=int, default=10)
    parser.add_argument("--num_outer_steps", type=int, default=30000)
    parser.add_argument(
        "--disjoint_tasks",
        action="store_true",
        help="Classes between tasks do not repeat",
        default=False,
    )

    parser.add_argument("--pool", type=int, default=4)
    parser.add_argument("--fsl_num_tasks_test", type=int, default=500)

    return parser


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
    inner_opt_init,
    outer_opt_update,
    batched_outer_loop_ins,
):
    inner_opt_state = inner_opt_init(fast_params)

    (outer_loss, (slow_state, fast_state, info)), grads = value_and_grad(
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
    )
    updates, outer_opt_state = outer_opt_update(
        grads, outer_opt_state, (slow_params, fast_params)
    )
    slow_params, fast_params = ox.apply_updates((slow_params, fast_params), updates)

    return outer_opt_state, slow_params, fast_params, slow_state, fast_state, info


if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    cfg = parse_and_build_cfg(args)
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
    exp.log(f"Running on {device} with seed: {cfg.seed}")

    # Data
    train_images, train_labels, val_images, val_labels, preprocess_fn = prepare_data(
        cfg.dataset, cfg.data_dir, device,
    )
    sup_train_images = train_images[:, :TRAIN_SIZE]
    sup_train_labels = train_labels[:, :TRAIN_SIZE]
    # These are for supervised learning validation
    sup_val_images = train_images[:, TRAIN_SIZE:]
    sup_val_labels = train_labels[:, TRAIN_SIZE:]
    exp.log(
        "Supervised validation data:", sup_val_images.shape, sup_val_labels.shape,
    )
    exp.log(
        "FSL and Transfer learning data:", val_images.shape, val_labels.shape,
    )

    # Model
    body, head = prepare_model(cfg.dataset, cfg.way, cfg.hidden_size, cfg.activation)
    rng, rng_params = split(rng)
    (slow_params, fast_params, slow_state, fast_state,) = make_params(
        rng, cfg.dataset, body.init, body.apply, head.init, device,
    )

    # Optimizers
    inner_opt = ox.sgd(cfg.inner_lr)
    lr_schedule = ox.cosine_decay_schedule(-cfg.outer_lr, cfg.num_outer_steps, 0.1)
    outer_opt = ox.chain(
        ox.clip(10), ox.scale_by_adam(), ox.scale_by_schedule(lr_schedule),
    )
    outer_opt_state = outer_opt.init((slow_params, fast_params))

    # Train data sampling
    train_sample_fn_kwargs = {
        "images": sup_train_images,
        "labels": sup_train_labels,
        "batch_size": cfg.meta_batch_size,
        "way": cfg.way,
        "shot": cfg.shot,
        "qry_shot": cfg.qry_shot,
        "preprocess_fn": preprocess_fn,
        "device": device,
    }
    train_sample_fn = partial(fsl_sample_transfer_build, **train_sample_fn_kwargs,)
    # Train loops
    train_inner_loop_ins = partial(
        fsl_inner_loop,
        is_training=True,
        num_steps=cfg.num_inner_steps_train,
        slow_apply=body.apply,
        fast_apply=head.apply,
        loss_fn=mean_xe_and_acc_dict,
        opt_update_fn=inner_opt.update,
    )
    train_outer_loop_ins = partial(
        outer_loop,
        is_training=True,
        inner_loop=train_inner_loop_ins,
        slow_apply=body.apply,
        fast_apply=head.apply,
        loss_fn=mean_xe_and_acc_dict,
    )
    train_batched_outer_loop_ins = partial(
        batched_outer_loop, outer_loop=train_outer_loop_ins
    )
    step_ins = jit(
        partial(
            step,
            inner_opt_init=inner_opt.init,
            outer_opt_update=outer_opt.update,
            batched_outer_loop_ins=train_batched_outer_loop_ins,
        ),
    )
    # Val data sampling
    test_fsl_sample_fn_kwargs = {
        "images": val_images,
        "labels": val_labels,
        "batch_size": cfg.meta_batch_size_test,
        "way": cfg.way,
        "shot": cfg.shot,
        "qry_shot": 15,  # Standard
        "preprocess_fn": preprocess_fn,
        "device": device,
    }
    test_fsl_sample_fn = partial(
        fsl_sample_transfer_build, **test_fsl_sample_fn_kwargs,
    )
    test_sup_spt_sampler = BatchSampler(
        rng,
        flatten(sup_train_images, 1),
        flatten(sup_train_labels),
        cfg.sup_batch_size_test,
        shuffle=False,
        keep_last=True,
    )
    test_sup_qry_sampler = BatchSampler(
        rng,
        flatten(sup_val_images, 1),
        flatten(sup_val_labels),
        cfg.sup_batch_size_test,
        shuffle=False,
        keep_last=True,
    )
    # Val loops
    test_inner_loop_ins = partial(
        fsl_inner_loop,
        is_training=False,
        num_steps=cfg.num_inner_steps_test,
        slow_apply=body.apply,
        fast_apply=head.apply,
        loss_fn=mean_xe_and_acc_dict,
        opt_update_fn=inner_opt.update,
    )
    test_outer_loop_ins = partial(
        outer_loop,
        is_training=False,
        inner_loop=test_inner_loop_ins,
        slow_apply=body.apply,
        fast_apply=head.apply,
        loss_fn=mean_xe_and_acc_dict,
    )
    test_batched_outer_loop_ins = partial(
        batched_outer_loop, outer_loop=test_outer_loop_ins
    )
    test_batched_outer_loop_ins = jit(test_batched_outer_loop_ins)
    # Test embeddings
    embeddings_fn = lambda slow_params, slow_state, inputs: body.apply(
        slow_params, slow_state, None, inputs, False
    )[0][0]
    embeddings_fn = jit(embeddings_fn)
    ##

    pbar = tqdm(
        range(cfg.num_outer_steps),
        file=sys.stdout,
        miniters=25,
        mininterval=10,
        maxinterval=30,
    )
    for i in pbar:
        rng, rng_step, rng_sample = split(rng, 3)
        x_spt, y_spt, x_qry, y_qry = train_sample_fn(rng_sample)

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
        )

        if (i == 0) or (((i + 1) % cfg.val_every_k_steps) == 0):
            now = time.time()
            rng, rng_test = split(rng)
            fsl_maml_results = test_fsl_maml(
                rng,
                slow_params,
                fast_params,
                slow_state,
                fast_state,
                cfg.fsl_num_tasks_test // cfg.meta_batch_size_test,
                inner_opt.init,
                test_fsl_sample_fn,
                test_batched_outer_loop_ins,
            )
            fsl_maml_loss = fsl_maml_results[0].mean()
            fsl_maml_acc = fsl_maml_results[1]["outer"]["final"]["aux"][0]["acc"].mean()

            fsl_embeddings_preds, fsl_embeddings_targets = test_fsl_embeddings(
                rng_test,
                partial(embeddings_fn, slow_params, slow_state),
                test_fsl_sample_fn,
                cfg.fsl_num_tasks_test // cfg.meta_batch_size_test,
                device=device,
                pool=cfg.pool,
            )
            fsl_embeddings_acc = (
                (fsl_embeddings_preds == fsl_embeddings_targets)
                .astype(onp.float)
                .mean()
            )

            sup_preds, sup_targets = test_sup_cosine(
                partial(embeddings_fn, slow_params, slow_state),
                test_sup_spt_sampler,
                test_sup_qry_sampler,
                device,
                preprocess_fn,
            )
            sup_acc = (sup_preds == sup_targets).astype(onp.float).mean()

            test_time = time.time() - now

        if (
            (i == 0)
            or (((i + 1) % cfg.progress_bar_refresh_rate) == 0)
            or (((i + 1) % cfg.val_every_k_steps) == 0)
        ):
            current_lr = lr_schedule(outer_opt_state[-1].count)
            pbar.set_postfix(
                lr=f"{current_lr:.4f}",
                ttime=f"{test_time:.1f}",
                loss=f"{info['outer']['final']['loss'].mean():.2f}",
                foa=f"{info['outer']['final']['aux'][0]['acc'].mean():.2f}",
                vam=f"{fsl_maml_acc:.2f}",
                vae=f"{fsl_embeddings_acc:.2f}",
                vas=f"{sup_acc:.2f}",
                # vfol=f"{vfol:.3f}",
                # vioa=f"{vioa:.3f}",
                # vfoa=f"{vfoa:.3f}",
                # bfoa=f"{best_val_acc:.3f}",
                # fia=f"{info['inner']['auxs'][0]['acc'][:, -1].mean():.3f}",
                # iil=f"{info['inner']['losses'][:, 0].mean():.3f}",
                # fil=f"{info['inner']['losses'][:, -1].mean():.3f}",
            )
