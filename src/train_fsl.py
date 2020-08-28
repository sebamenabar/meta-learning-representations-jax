import os
import sys

# import atexit
import os.path as osp
import pprint as pp

import pickle
import functools
from tqdm import tqdm
from argparse import ArgumentParser
from omegaconf import OmegaConf
import dill

import numpy as onp

import jax
from jax.random import split
from jax.tree_util import Partial as partial
from jax import (
    # nn,
    # lax,
    ops,
    jit,
    grad,
    vmap,
    random,
    numpy as jnp,
    value_and_grad,
)

# from jax.experimental import stax
from jax.experimental import optix

# from jax.experimental import optimizers
import optax

import haiku as hk

import config
from lib import (
    setup_device,
    mean_xe_and_acc,
    make_fsl_inner_outer_loop,
    make_batched_outer_loop,
)
from experiment import Experiment, Logger
from data import prepare_data, statistics, preprocess_data
from data.sampling import fsl_sample_transfer_and_build
from models.activations import activations
from models.maml_conv import MiniImagenetCNNMaker


def loss_fn(logits, targets):
    loss, acc = mean_xe_and_acc(logits, targets)
    return loss, {"acc": acc}


def mm_fn(x, *xs):
    return jnp.stack(xs)


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
    args, cfg = config.parse_args()
    exp = Experiment(cfg, args)
    exp.log_init([sys.stdout])
    exp.comet_init()
    sys.stderr = Logger(exp.logfile, [sys.stderr])
    if cfg.debug:
        exp.log("Debugging ...")

    # Moved this to exp.log_init
    # exp.log("\nCLI arguments")
    # exp.log(pp.pformat(vars(args)))
    # exp.log("\nConfiguration")
    # exp.log(pp.pformat(OmegaConf.to_container(cfg)))
    # exp.log()

    jit_enabled = not args.disable_jit

    if args.dataset == "omniglot" and args.prefetch_data_gpu:
        default_platform = "gpu"
    else:
        default_platform = "cpu"
    cpu, device = setup_device(
        args.gpus, default_platform
    )  # gpu is None if args.gpus == 0
    rng = random.PRNGKey(args.seed)

    ### DATA
    ### TEMP
    if args.data_dir is None:
        if args.dataset == "miniimagenet":
            args.data_dir = "/workspace1/samenabar/data/mini-imagenet/"
        elif args.dataset == "omniglot":
            args.data_dir = "/workspace1/samenabar/data/omniglot/"

    train_images, train_labels, val_images, val_labels, preprocess_fn = prepare_data(
        args.dataset, args.data_dir, cpu, device, args.prefetch_data_gpu,
    )

    exp.log("Train data:", train_images.shape, train_labels.shape)
    exp.log("Val data:", val_images.shape, val_labels.shape)
    val_way = args.way
    if args.way > val_images.shape[0]:
        exp.log(
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

    inner_opt = optix.chain(optix.sgd(cfg.inner_lr))
    inner_loop, outer_loop = make_fsl_inner_outer_loop(
        slow_apply,
        fast_apply_and_loss_fn,
        inner_opt.update,
        args.num_inner_steps,
        update_state=False,
    )
    batched_outer_loop = make_batched_outer_loop(outer_loop)

    outer_opt = optax.chain(
        optax.clip(10),
        optax.scale_by_adam(),
        optax.scale_by_schedule(
            optax.cosine_decay_schedule(-cfg.outer_lr, cfg.num_outer_steps, 0.1)
        ),
    )
    outer_opt_state = outer_opt.init((slow_params, fast_params))

    ### TRAIN FUNCTIONS
    def step(
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
    ):
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
            args.num_inner_steps,
        )
        updates, outer_opt_state = outer_opt.update(
            grads, outer_opt_state, (slow_params, fast_params)
        )
        slow_params, fast_params = optax.apply_updates(
            (slow_params, fast_params), updates
        )

        return outer_opt_state, slow_params, fast_params, slow_state, fast_state, info

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
            10,
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

    pbar = tqdm(
        range(args.num_outer_steps),
        file=sys.stdout,
        miniters=25,
        mininterval=10,
        maxinterval=30,
    )
    val_outer_loss = 0.0
    vfol = 0.0
    vioa = 0.0
    vfoa = 0.0
    best_val_acc = 0.0
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

        outer_opt_state, slow_params, fast_params, slow_state, fast_state, info = step(
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

        if (((i + 1) % args.progress_bar_refresh_rate) == 0) or (i == 0):
            train_metrics = jax.tree_util.tree_map(
                lambda x: {"mean": x.mean(), "std": x.std()}, info
            )
            str_train_metrics = jax.tree_util.tree_map(
                lambda x: x.item(), train_metrics
            )
            current_lr = optax.cosine_decay_schedule(
                cfg.outer_lr, min(20000, cfg.num_outer_steps), 0.01
            )(outer_opt_state[-1].count)
            if (((i + 1) % args.val_every_k_steps) == 0) or (i == 0):
                rng, rng_validation = split(rng)
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
                vfoa = val_info["outer"]["final_aux"][0]["acc"].mean()
                vioa = val_info["outer"]["initial_aux"][0]["acc"].mean()

                exp.log("\n")
                exp.log(f"---------- Step {i + 1} ----------")
                exp.log(pbar.format_meter(**pbar.format_dict))
                exp.log(f"\nCurrent learning rate: {current_lr}")

                val_metrics = jax.tree_util.tree_map(
                    lambda x: {"mean": x.mean(), "std": x.std()}, val_info
                )
                str_val_metrics = jax.tree_util.tree_map(
                    lambda x: x.item(), val_metrics
                )
                exp.log()
                exp.log(f"       Step {i + 1} validation metrics ------")
                exp.log(pp.pformat(str_val_metrics, indent=4))

                exp.log()
                exp.log(f"       Step {i + 1} last training batch metrics ------")
                exp.log(pp.pformat(str_train_metrics, indent=4))
                exp.log()
                # Send to comet

                new_val_acc = val_metrics["outer"]["final_aux"][0]["acc"]["mean"]
                if new_val_acc > best_val_acc:
                    best_val_acc = new_val_acc
                    exp.log(f"New best validation accuracy: {new_val_acc}")
                    exp.log("Saving checkpoint\n")

                    with open(
                        osp.join(exp.exp_dir, "checkpoints/best.ckpt"), "wb"
                    ) as f:
                        dill.dump(
                            {
                                "val_metrics": val_metrics,
                                "train_metrics": train_metrics,
                                "optimizer_state": outer_opt_state,
                                "slow_params": slow_params,
                                "fast_params": fast_params,
                                "slow_state": slow_state,
                                "fast_state": fast_state,
                                "rng": rng,
                                "i": i,
                            },
                            f,
                            protocol=3,
                        )

                exp.comet.log_metrics(
                    {
                        "inner_final_loss": val_metrics["inner"]["final_loss"]["mean"],
                        "inner_final_acc": val_metrics["inner"]["final_aux"][0]["acc"][
                            "mean"
                        ],
                        "outer_final_loss": val_metrics["outer"]["final_loss"]["mean"],
                        "outer_final_acc": val_metrics["outer"]["final_aux"][0]["acc"][
                            "mean"
                        ],
                        "best_outer_final_acc": best_val_acc,
                    },
                    step=i + 1,
                    prefix="val",
                )

            exp.comet.log_metrics(
                {
                    "inner_final_loss": train_metrics["inner"]["final_loss"]["mean"],
                    "inner_final_acc": train_metrics["inner"]["final_aux"][0]["acc"][
                        "mean"
                    ],
                    "outer_final_loss": train_metrics["outer"]["final_loss"]["mean"],
                    "outer_final_acc": train_metrics["outer"]["final_aux"][0]["acc"][
                        "mean"
                    ],
                    "learning_rate": current_lr,
                },
                step=i + 1,
                prefix="train",
            )

            pbar.set_postfix(
                lr=f"{current_lr:.4f}",
                vfol=f"{vfol:.3f}",
                vioa=f"{vioa:.3f}",
                vfoa=f"{vfoa:.3f}",
                loss=f"{info['outer']['final_loss'].mean():.3f}",
                foa=f"{info['outer']['final_aux'][0]['acc'].mean():.3f}",
                bfoa=f"{best_val_acc:.3f}",
            )
