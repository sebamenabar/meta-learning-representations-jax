import os
import sys
import os.path as osp

import pickle
import functools
import pprint as pp
from argparse import ArgumentParser

import dill
from tqdm import tqdm
from easydict import EasyDict as edict

# from omegaconf import OmegaConf

import time
import numpy as onp

import jax
from jax.random import split
from jax.tree_util import Partial as partial
from jax import jit, grad, vmap, random, value_and_grad, numpy as jnp

import optax as ox
import haiku as hk

from config import rsetattr
from data.sampling import BatchSampler, fsl_sample_transfer_build
from data import prepare_data
from experiment import Experiment, Logger
from models.maml_conv import make_params, prepare_model
from lib import (
    setup_device,
    mean_xe_and_acc_dict,
    flatten,
    apply_and_loss_fn,
)
from models.activations import activations
from test_utils import SupervisedStandardTester, SupervisedCosineTester


def step(rng, params, state, inputs, targets, opt_state, loss_fn, opt_update_fn):
    rng, rng_step = split(rng)
    (loss, (state, aux)), grads = value_and_grad(loss_fn, (0, 1), has_aux=True)(
        *params, *state, rng_step, inputs, targets,
    )
    # print(grads)
    updates, opt_state = opt_update_fn(grads, opt_state, params)
    params = ox.apply_updates(params, updates)

    return params, state, opt_state, loss, aux


def pred_fn(
    slow_params,
    fast_params,
    slow_state,
    fast_state,
    rng,
    inputs,
    is_training,
    slow_apply,
    fast_apply,
):
    rng_slow, rng_fast = split(rng)
    slow_outputs, _ = slow_apply(slow_params, slow_state, rng_slow, inputs, is_training)
    fast_outputs, _ = fast_apply(
        fast_params, fast_state, rng_fast, *slow_outputs, is_training
    )
    return fast_outputs


def embeddings_fn(slow_params, slow_state, rng, inputs, is_training, slow_apply):
    return slow_apply(slow_params, slow_state, rng, inputs, is_training)[0][0]


def parse_args():
    parser = Experiment.add_args()
    # Network hyperparameters
    parser.add_argument("--model.hidden_size", default=32, type=int)
    parser.add_argument("--model.no_track_bn_stats", default=False, action="store_true")
    parser.add_argument(
        "--model.activation", type=str, default="relu", choices=list(activations.keys())
    )

    # FSL evaluation arguments
    parser.add_argument("--val.fsl.way", type=int, default=5)
    parser.add_argument("--val.fsl.spt_shot", type=int, default=5)
    parser.add_argument("--val.fsl.qry_shot", type=int, default=15)
    parser.add_argument(
        "--val.fsl.batch_size", type=int, default=25, help="Number of tasks per batch"
    )
    parser.add_argument("--val.fsl.total_num_tasks", type=int, default=500)
    parser.add_argument("--val.pool", type=int, default=4)
    parser.add_argument("--val.sup.batch_size", type=int, default=256)

    # Training hyperparameters
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--lr_schedule", nargs="*", type=int, default=[50, 80])
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=5e-4, type=float)
    parser.add_argument("--val_interval", default=10, type=int)  # In epochs

    args = parser.parse_args()
    cfg = edict(model=edict(), val=edict(fsl=edict(), sup=edict()))
    for argname, argval in vars(args).items():
        rsetattr(cfg, argname, argval)

    return args, cfg


if __name__ == "__main__":
    args, cfg = parse_args()
    exp = Experiment(cfg, args)
    exp.logfile_init(
        [sys.stdout]
    )  # Send logged stuff also to stdout (but not all stdout to log)
    exp.loggers_init()
    sys.stderr = Logger(exp.logfile, [sys.stderr])  # Send stderr to log

    if cfg.debug:  # Debugging creates experiments folders in experiments/debug dir
        exp.log("Debugging ...")

    jit_enabled = not cfg.disable_jit

    if cfg.dataset == "omniglot" and cfg.prefetch_data_gpu:
        default_platform = "gpu"
    else:
        default_platform = "cpu"
    cpu, device = setup_device(
        cfg.gpus, default_platform
    )  # gpu is None if cfg.gpus == 0
    rng = random.PRNGKey(cfg.seed)

    if cfg.data_dir is None:
        if cfg.dataset == "miniimagenet":
            # cfg.data_dir = "/workspace1/samenabar/data/mini-imagenet/"
            cfg.data_dir = "/mnt/ialabnas/homes/samenabar/data/FSL/mini-imagenet"
        elif cfg.dataset == "omniglot":
            cfg.data_dir = "/workspace1/samenabar/data/omniglot/"

    TRAIN_SIZE = 500
    train_images, train_labels, val_images, val_labels, preprocess_fn = prepare_data(
        cfg.dataset, cfg.data_dir, device,
    )
    sup_train_images = flatten(train_images[:, :TRAIN_SIZE], 1)
    sup_train_labels = flatten(train_labels[:, :TRAIN_SIZE], 1)
    # These are for supervised learning validation
    sup_val_images = flatten(train_images[:, TRAIN_SIZE:], 1)
    sup_val_labels = flatten(train_labels[:, TRAIN_SIZE:], 1)

    transfer_spt_images = val_images[:, :TRAIN_SIZE]
    transfer_spt_labels = val_labels[:, :TRAIN_SIZE]
    transfer_qry_images = val_images[:, TRAIN_SIZE:]
    transfer_qry_labels = val_labels[:, TRAIN_SIZE:]

    exp.log("Sup train data:", sup_train_images.shape, sup_train_labels.shape)
    exp.log("Sup val data:", sup_val_images.shape, sup_val_labels.shape)
    exp.log("Other tasks val data:", val_images.shape, val_labels.shape)

    # output_size = sup_train_images.shape[0]
    output_size = 64
    body, head = prepare_model(
        cfg.dataset,
        output_size,
        cfg.model.hidden_size,
        cfg.model.activation,
        track_stats=not cfg.model.no_track_bn_stats,
    )
    rng, rng_params = split(rng)
    (slow_params, fast_params, slow_state, fast_state,) = make_params(
        rng, cfg.dataset, body.init, body.apply, head.init, device,
    )
    params = (slow_params, fast_params)
    state = (slow_state, fast_state)

    # SGD  with momentum, wd and schedule
    # schedule = ox.piecewise_constant_schedule(
    #     -cfg.lr, {e: 0.1 for e in cfg.lr_schedule}
    # )
    schedule = ox.cosine_decay_schedule(-cfg.lr, cfg.epochs, 0.01)
    opt = ox.chain(
        ox.trace(decay=cfg.momentum, nesterov=False),
        ox.additive_weight_decay(cfg.weight_decay),
        ox.scale_by_schedule(schedule),
    )
    opt_state = opt.init(params)
    schedule_state = opt_state[-1]

    train_apply_fn_kwargs = {
        "slow_apply": body.apply,
        "fast_apply": head.apply,
        "loss_fn": mean_xe_and_acc_dict,
        "is_training": True,
    }
    train_apply_and_loss_fn = partial(apply_and_loss_fn, **train_apply_fn_kwargs)
    rng, rng_sampler = split(rng)
    train_sampler = BatchSampler(
        rng_sampler, sup_train_images, sup_train_labels, cfg.batch_size,
    )

    # 4. Test supervised learning functions
    test_pred_fn = jit(partial(pred_fn, is_training=False, slow_apply=body.apply, fast_apply=head.apply))
    test_embeddings_fn = jit(partial(embeddings_fn, is_training=False, slow_apply=body.apply))

    rng, rng_std_tester, rng_cosine_tester = split(rng, 3)
    supervised_std_tester = SupervisedStandardTester(
        rng_std_tester,
        sup_val_images,
        sup_val_labels,
        cfg.val.sup.batch_size,
        # pred_fn_test,
        preprocess_fn,
        device,
    )
    supervised_cosine_tester = SupervisedCosineTester(
        rng_cosine_tester,
        sup_train_images,
        sup_train_labels,
        sup_val_images,
        sup_val_labels,
        cfg.val.sup.batch_size,
        preprocess_fn,
        device,
    )

    step_ins = jit(
        partial(step, loss_fn=train_apply_and_loss_fn, opt_update_fn=opt.update)
    )
    # fsl_test_apply_fn = jit(fsl_test_apply_fn)
    # transfer_test_apply_fn = jit(transfer_test_apply_fn)
    # sup_test_pred_fn = jit(sup_test_pred_fn)

    pbar = tqdm(
        range(1),
        total=(((sup_train_images.shape[0] // cfg.batch_size) - 1) * cfg.epochs),
        file=sys.stdout,
        miniters=25,
        mininterval=5,
        maxinterval=30,
    )
    curr_step = 0
    best_sup_val_acc = 0
    for epoch in range(1, cfg.epochs + 1):
        # sampler = batch_sampler(rng, train_images, train_labels, cfg.batch_size)
        pbar.set_description(f"E:{epoch}")
        for j, (X, y) in enumerate(train_sampler):
            X = jax.device_put(X, device)
            y = jax.device_put(y, device)
            X = preprocess_fn(X)

            opt_state[-1] = schedule_state
            params, state, opt_state, loss, aux = step_ins(
                rng, params, state, X, y, opt_state
            )
            opt_state[
                -1
            ] = schedule_state  # Before and after for checkpointing and safety

            curr_step += 1
            if (
                (j == 0) and (((epoch % cfg.val_interval) == 0) or epoch == 1)
            ) or (
                (epoch == cfg.epochs) and (j == len(train_sampler) - 1)  # Last step
            ):
                # Test supervised learning
                sup_std_loss, sup_std_acc = supervised_std_tester(partial(test_pred_fn, *params, *state, rng))
                sup_cosine_acc = supervised_cosine_tester(partial(test_embeddings_fn, params[0], state[0], rng))

                exp.log(f"\nValidation epoch {epoch} results:")
                exp.log(f"Supervised standard accuracy: {sup_std_acc}, loss: {sup_std_loss}")
                exp.log(f"Supervised cosine accuracy: {sup_cosine_acc}")
                if sup_std_acc > best_sup_val_acc:
                    best_sup_val_acc = sup_std_acc
                    exp.log(
                        f"\nNew best supervised validation accuracy: {best_sup_val_acc}"
                    )
                    exp.log("Saving checkpoint\n")

                    with open(
                        osp.join(exp.exp_dir, "checkpoints/best.ckpt"), "wb"
                    ) as f:
                        dill.dump(
                            {
                                "sup_std_val_acc": sup_std_acc,
                                "sup_std_val_loss": sup_std_loss,
                                "sup_cosine_val_acc": sup_cosine_acc,
                                # "transfer_val_acc": transfer_acc,
                                # "fsl_val_acc": val_acc,
                                "optimizer_state": opt_state,
                                "slow_params": params[0],
                                "fast_params": params[1],
                                "slow_state": state[0],
                                "fast_state": state[1],
                                "rng": rng,
                                "i": epoch,
                                "curr_step": curr_step,
                            },
                            f,
                            protocol=3,
                        )

                pbar.set_postfix(
                    loss=f"{loss:.3f}",
                    acc=f"{aux[0]['acc'].mean():.2f}",
                    lr=f"{schedule(opt_state[-1].count):.4f}",
                    # fsla=f"{val_acc:.2f}",
                    # fslt=f"{end - start:.2f}",
                    supsa=f"{sup_std_acc:.3f}",
                    supca=f"{sup_cosine_acc:.3f}",
                    # transfera=f"{transfer_acc:.3f}",
                )

            elif (curr_step % cfg.progress_bar_refresh_rate) == 0:
                pbar.set_postfix(
                    loss=f"{loss:.3f}",
                    acc=f"{aux[0]['acc'].mean():.2f}",
                    lr=f"{schedule(opt_state[-1].count):.4f}",
                    # fsla=f"{val_acc:.2f}",
                    # fslt=f"{end - start:.2f}",
                    supsa=f"{sup_std_acc:.3f}",
                    supca=f"{sup_cosine_acc:.3f}",
                    # transfera=f"{transfer_acc:.3f}",
                )

            pbar.update()

        schedule_state = ox.ScaleByScheduleState(
            count=schedule_state.count + 1,
        )  # Unsafe for max int

