import os
import sys

import pickle
import functools
from tqdm import tqdm
from argparse import ArgumentParser

import numpy as onp

import jax
from jax.random import split
from jax.image import resize as im_resize
from jax.tree_util import Partial as partial
from jax import ops, nn, jit, grad, value_and_grad, lax, vmap, random, numpy as jnp

from jax.experimental import stax
from jax.experimental import optix
from jax.experimental import optimizers

import haiku as hk

from lib import (
    setup_device,
    mean_xe_and_acc,
    make_fsl_inner_outer_loop,
    make_batched_outer_loop,
)
from models.maml_conv import MiniImagenetCNNMaker
from data import prepare_data, statistics
from data.sampling import fsl_sample_transfer_and_build


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress_bar_refresh_rate", type=int, default=50)
    parser.add_argument("--val_every_k_steps", type=int, default=500)
    parser.add_argument("--disable-jit", action="store_true", default=False)

    # Training config settings
    parser.add_argument("--way", type=int, required=True)
    parser.add_argument("--shot", type=int, required=True)
    parser.add_argument("--qry-shot", type=int, required=True)

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--val_batch_size", type=int, default=25)
    parser.add_argument("--inner_lr", type=float, default=5e-1)
    parser.add_argument("--outer_lr", type=float, default=1e-2)
    parser.add_argument("--num_outer_steps", type=int)
    parser.add_argument("--num_inner_steps", type=int, required=True)
    parser.add_argument(
        "--disjoint_tasks",
        action="store_true",
        help="Classes between tasks do not repeat",
        default=False,
    )

    # Data settings
    parser.add_argument(
        "--data_dir", type=str, default="/workspace1/samenabar/data/mini-imagenet/"
    )
    parser.add_argument("--dataset", type=str, choices=["miniimagenet"], required=True)

    # Device settings
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--prefetch_data_gpu", action="store_true", default=False)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    print(args)
    jit_enabled = not args.disable_jit

    cpu, device = setup_device(args.gpus)  # gpu is None if args.gpus == 0
    rng = random.PRNGKey(args.seed)
    # miniimagenet_data_dir = "/workspace1/samenabar/data/mini-imagenet/"
    # num_tasks = 25
    # way = 5
    # spt_shot = 1
    # qry_shot = 15
    # inner_lr = 5e-1
    # outer_lr = 1e-2
    # num_inner_steps = 5
    # num_outer_steps = 50000
    # disjoint_tasks = False

    ### DATA

    train_images, train_labels, val_images, val_labels, mean, std = prepare_data(
        args.dataset, args.data_dir, cpu, device, args.prefetch_data_gpu,
    )

    print("Train data:", train_images.shape, train_labels.shape)
    print("Val data:", val_images.shape, val_labels.shape)
    val_way = args.way
    if args.way > val_images.shape[0]:
        print(
            f"Training with {args.way}-way but validation only has {val_images.shape[0]}-classes"
        )
        val_way = val_images.shape[0]

    ### MODEL
    def loss_fn(logits, targets):
        loss, acc = mean_xe_and_acc(logits, targets)
        return loss, {"acc": acc}

    MiniImagenetCNN, apply_and_loss_fn = MiniImagenetCNNMaker(args.way, loss_fn)

    inner_opt = optix.chain(optix.sgd(args.inner_lr))
    inner_loop, outer_loop = make_fsl_inner_outer_loop(
        apply_and_loss_fn, inner_opt.update, args.num_inner_steps, update_state=False
    )
    batched_outer_loop = make_batched_outer_loop(outer_loop)

    ### PREPARE PARAMETERS
    params, state = MiniImagenetCNN.init(rng, jnp.zeros((2, 84, 84, 3)), False)
    params = jax.tree_map(lambda x: jax.device_put(x, device), params)
    state = jax.tree_map(lambda x: jax.device_put(x, device), state)
    predicate = lambda m, n, v: m == "mini_imagenet_cnn/linear"

    outer_opt_init, outer_opt_update, outer_get_params = optimizers.adam(
        step_size=args.outer_lr,
    )
    outer_opt_state = outer_opt_init(params)

    ### TRAIN FUNCTIONS
    def step(step_num, outer_opt_state, state, x_spt, y_spt, x_qry, y_qry):
        params = outer_get_params(outer_opt_state)
        fast_params, slow_params = hk.data_structures.partition(predicate, params)
        inner_opt_state = inner_opt.init(fast_params)

        (outer_loss, (state, info)), grads = value_and_grad(
            batched_outer_loop, (1, 2), has_aux=True
        )(
            None,
            slow_params,
            fast_params,
            state,
            inner_opt_state,
            True,
            x_spt,
            y_spt,
            x_qry,
            y_qry,
        )

        grads = hk.data_structures.merge(*grads)
        outer_opt_state = outer_opt_update(i, grads, outer_opt_state)

        return outer_opt_state, state, info

    if jit_enabled:
        step = jit(step)
    pbar = tqdm(range(args.num_outer_steps))
    val_outer_loss = 0
    vfol = 0
    vioa = 0
    vfoa = 0
    for i in pbar:
        rng, rng_sample = split(rng)
        x_spt, y_spt, x_qry, y_qry = fsl_sample_transfer_and_build(
            rng_sample,
            mean,
            std,
            train_images,
            train_labels,
            args.batch_size,
            args.way,
            args.shot,
            args.qry_shot,
            device,
            args.disjoint_tasks,
        )

        outer_opt_state, state, info = step(
            i, outer_opt_state, state, x_spt, y_spt, x_qry, y_qry
        )

        # print("train info")
        # print(info)

        if (((i + 1) % args.progress_bar_refresh_rate) == 0) or (i == 0):
            if (((i + 1) % args.val_every_k_steps) == 0) or (i == 0):
                rng, rng_sample = split(rng)
                x_spt, y_spt, x_qry, y_qry = fsl_sample_transfer_and_build(
                    rng_sample,
                    mean,
                    std,
                    val_images,
                    val_labels,
                    args.val_batch_size,
                    val_way,
                    args.shot,
                    args.qry_shot,
                    device,
                    False,
                )

                params = outer_get_params(outer_opt_state)
                fast_params, slow_params = hk.data_structures.partition(
                    predicate, params
                )
                inner_opt_state = inner_opt.init(fast_params)
                val_outer_loss, (val_state, val_info) = batched_outer_loop(
                    None,
                    slow_params,
                    fast_params,
                    state,
                    inner_opt_state,
                    False,
                    x_spt,
                    y_spt,
                    x_qry,
                    y_qry,
                )
                # print("val_info")
                # print(val_info)
                vioa = val_info["outer"]["initial_aux"][0]["acc"].mean()
                vfoa = val_info["outer"]["final_aux"][0]["acc"].mean()

            pbar.set_postfix(
                # iol=f"{info['outer']['initial_loss'].mean():.3f}",
                loss=f"{info['outer']['final_loss'].mean():.3f}",
                # iil=f"{info['inner']['initial_loss'].mean():.3f}",
                # fil=f"{info['inner']['final_loss'].mean():.3f}",
                # iia=f"{info['inner']['initial_aux'][0]['acc'].mean():.3f}",
                # fia=f"{info['inner']['final_aux'][0]['acc'].mean():.3f}",
                # ioa=f"{info['outer']['initial_aux'][0]['acc'].mean():.3f}",
                foa=f"{info['outer']['final_aux'][0]['acc'].mean():.3f}",
                vfol=f"{val_outer_loss:.3f}",
                vioa=f"{vioa:.3f}",
                vfoa=f"{vfoa:.3f}",
            )
