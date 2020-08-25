import os
import sys

import pickle
import functools
from tqdm import tqdm
from argparse import ArgumentParser

import numpy as onp

import jax
from jax.random import split
from jax.tree_util import Partial as partial
from jax import (
    ops,
    # nn,
    jit,
    grad,
    value_and_grad,
    # lax,
    vmap,
    random,
    numpy as jnp,
)

# from jax.experimental import stax
from jax.experimental import optix
from jax.experimental import optimizers

import haiku as hk

from lib import (
    setup_device,
    mean_xe_and_acc,
    make_fsl_inner_outer_loop,
    make_batched_outer_loop,
)
from data import prepare_data, statistics, preprocess_data
from data.sampling import fsl_sample_transfer_and_build
from models.activations import activations
from models.maml_conv import MiniImagenetCNNMaker


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress_bar_refresh_rate", type=int, default=50)
    parser.add_argument("--val_every_k_steps", type=int, default=500)
    parser.add_argument("--disable-jit", action="store_true", default=False)

    # Model settings
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument(
        "--activation", type=str, default="relu", choices=list(activations.keys())
    )

    # Training config settings
    parser.add_argument("--way", type=int, required=True)
    parser.add_argument("--shot", type=int, required=True)
    parser.add_argument("--qry-shot", type=int, required=True)

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--val_batch_size", type=int, default=25)
    parser.add_argument("--val_num_tasks", type=int, default=1000)
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
    parser.add_argument(
        "--dataset", type=str, choices=["miniimagenet", "omniglot"], required=True
    )

    # Device settings
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--prefetch_data_gpu", action="store_true", default=False)

    return parser.parse_args()


def loss_fn(logits, targets):
    loss, acc = mean_xe_and_acc(logits, targets)
    return loss, {"acc": acc}


def mm_fn(x, *xs):
    return jnp.stack(jnp.array(xs)).reshape(-1)


def validate(rng, loss_acc_fn, sample_fn, val_num_tasks, val_batch_size):
    results = []
    for i in range(val_num_tasks // val_batch_size):
        rng, rng_sample = split(rng)
        x_spt, y_spt, x_qry, y_qry = sample_fn(rng)
        results.append(loss_acc_fn(x_spt, y_spt, x_qry, y_qry))

    results = jax.tree_util.tree_multimap(mm_fn, results[0], *results)
    return results


def prepare_model(dataset, way, hidden_size, activation):
    if dataset == "miniimagenet":
        max_pool = True
        spatial_dims = 25
    elif dataset == "omniglot":
        max_pool = False
        spatial_dims = 4

    return MiniImagenetCNNMaker(
        loss_fn,
        output_size=way,
        hidden_size=hidden_size,
        spatial_dims=spatial_dims,
        max_pool=max_pool,
        activation=activation,
    )


def make_params(rng, dataset, slow_init, slow_apply, fast_init, device):
    if dataset == "miniimagenet":
        setup_tensor = jnp.zeros((2, 84, 84, 3))
    elif dataset == "omniglot":
        setup_tensor = jnp.zeros((2, 28, 28, 1))
    slow_params, slow_state = slow_init(rng, setup_tensor, False)
    slow_outputs, _ = slow_apply(rng, slow_params, slow_state, False, setup_tensor,)
    fast_params, fast_state = fast_init(rng, *slow_outputs, False)
    move_to_device = lambda x: jax.device_put(x, device)
    slow_params = jax.tree_map(move_to_device, slow_params)
    fast_params = jax.tree_map(move_to_device, fast_params)
    slow_state = jax.tree_map(move_to_device, slow_state)
    fast_state = jax.tree_map(move_to_device, fast_state)

    return slow_params, fast_params, slow_state, fast_state


if __name__ == "__main__":
    args = parse_args()
    print(args)
    jit_enabled = not args.disable_jit

    if args.dataset == "omniglot" and args.prefetch_data_gpu:
        default_platform = "gpu"
    else:
        default_platform = "cpu"
    cpu, device = setup_device(
        args.gpus, default_platform
    )  # gpu is None if args.gpus == 0
    rng = random.PRNGKey(args.seed)
    # miniimagenet_data_dir = "/workspace1/samenabar/data/mini-imagenet/"

    ### DATA
    ### TEMP
    if args.dataset == "miniimagenet":
        args.data_dir = "/workspace1/samenabar/data/mini-imagenet/"
    elif args.dataset == "omniglot":
        args.data_dir = "/workspace1/samenabar/data/omniglot/"

    train_images, train_labels, val_images, val_labels, preprocess_fn = prepare_data(
        args.dataset, args.data_dir, cpu, device, args.prefetch_data_gpu,
    )

    print("Train data:", train_images.shape, train_labels.shape)
    print("Val data:", val_images.shape, val_labels.shape)
    val_way = args.way
    if args.way > val_images.shape[0]:
        print(
            f"Training with {args.way}-way but validation only has {val_images.shape[0]} classes"
        )
        val_way = val_images.shape[0]

    (
        MiniImagenetCNNBody,
        MiniImagenetCNNHead,
        slow_apply,
        fast_apply_and_loss_fn,
    ) = prepare_model(args.dataset, args.way, args.hidden_size, args.activation)
    slow_params, fast_params, slow_state, fast_state = make_params(
        rng,
        args.dataset,
        MiniImagenetCNNBody.init,
        slow_apply,
        MiniImagenetCNNHead.init,
        device,
    )

    inner_opt = optix.chain(optix.sgd(args.inner_lr))
    inner_loop, outer_loop = make_fsl_inner_outer_loop(
        slow_apply,
        fast_apply_and_loss_fn,
        inner_opt.update,
        args.num_inner_steps,
        update_state=False,
    )
    batched_outer_loop = make_batched_outer_loop(outer_loop)

    outer_opt_init, outer_opt_update, outer_get_params = optimizers.adam(
        step_size=args.outer_lr,
    )
    outer_opt_state = outer_opt_init((slow_params, fast_params))

    ### TRAIN FUNCTIONS
    def step(
        step_num, outer_opt_state, slow_state, fast_state, x_spt, y_spt, x_qry, y_qry
    ):
        slow_params, fast_params = outer_get_params(outer_opt_state)
        # fast_params, slow_params = hk.data_structures.partition(predicate, params)
        inner_opt_state = inner_opt.init(fast_params)

        (outer_loss, (slow_state, fast_state, info)), grads = value_and_grad(
            batched_outer_loop, (1, 2), has_aux=True
        )(
            None,  # rng
            slow_params,
            fast_params,
            slow_state,
            fast_state,
            True,  # is_training
            inner_opt_state,
            x_spt,
            y_spt,
            x_qry,
            y_qry,
        )

        # grads = hk.data_structures.merge(*grads)
        outer_opt_state = outer_opt_update(i, grads, outer_opt_state)

        return outer_opt_state, slow_state, fast_state, info

    pbar = tqdm(range(args.num_outer_steps))
    val_outer_loss = 0
    vfol = 0
    vioa = 0
    vfoa = 0

    def validation_loss_acc_fn(
        slow_params, fast_params, slow_state, fast_state, x_spt, y_spt, x_qry, y_qry
    ):
        inner_opt_state = inner_opt.init(fast_params)
        val_outer_loss, (_, _, val_info) = batched_outer_loop(
            None,
            slow_params,
            fast_params,
            slow_state,
            fast_state,
            False,
            inner_opt_state,
            x_spt,
            y_spt,
            x_qry,
            y_qry,
        )
        return val_outer_loss, val_info

    def val_sample_fn(rng):
        return fsl_sample_transfer_and_build(
            rng,
            preprocess_fn,
            val_images,
            val_labels,
            args.val_batch_size,
            val_way,
            args.shot,
            args.qry_shot,
            device,
            False,
        )

    if jit_enabled:
        step = jit(step)
        # validate = jit(validate, static_argnums=(2, 3, 4))
        validation_loss_acc_fn = jit(validation_loss_acc_fn)

    for i in pbar:
        rng, rng_sample = split(rng)
        x_spt, y_spt, x_qry, y_qry = fsl_sample_transfer_and_build(
            rng_sample,
            preprocess_fn,
            train_images,
            train_labels,
            args.batch_size,
            args.way,
            args.shot,
            args.qry_shot,
            device,
            args.disjoint_tasks,
        )

        outer_opt_state, slow_state, fast_state, info = step(
            i, outer_opt_state, slow_state, fast_state, x_spt, y_spt, x_qry, y_qry
        )

        # print("train info")
        # print(info)

        if (((i + 1) % args.progress_bar_refresh_rate) == 0) or (i == 0):
            if (((i + 1) % args.val_every_k_steps) == 0) or (i == 0):
                rng, rng_validation = split(rng)

                slow_params, fast_params = outer_get_params(outer_opt_state)

                val_outer_loss, val_info = validate(
                    rng_validation,
                    partial(
                        validation_loss_acc_fn,
                        slow_params,
                        fast_params,
                        slow_state,
                        fast_state,
                    ),
                    val_sample_fn,
                    args.val_num_tasks,
                    args.val_batch_size,
                )

                vfol = val_info["outer"]["final_loss"].mean()
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
                vfol=f"{vfol:.3f}",
                vioa=f"{vioa:.3f}",
                vfoa=f"{vfoa:.3f}",
            )
