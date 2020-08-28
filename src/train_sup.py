import os
import sys
import os.path as osp

import pickle
import functools
import pprint as pp
from argparse import ArgumentParser

import dill
from tqdm import tqdm
from omegaconf import OmegaConf

import time
import numpy as onp

import jax
from jax.random import split
from jax.tree_util import Partial as partial
from jax import jit, grad, vmap, random, value_and_grad, numpy as jnp

import optax as ox
import haiku as hk

from config import rsetattr
from test_fsl import meta_test
from test_sup import test_sup, test_sup_cosine
from data.sampling import BatchSampler, fsl_sample_transfer_and_build
from optimizers import sgd
from data import prepare_data
from experiment import Experiment, Logger
from models.maml_conv import MiniImagenetCNNMaker, make_params, prepare_model
from lib import setup_device, mean_xe_and_acc_dict as xe_and_acc, flatten
from models.activations import activations


def apply_and_loss_fn(
    rng, params, state, inputs, targets, is_training, slow_apply, fast_apply_and_loss_fn
):
    slow_params, fast_params = params
    slow_state, fast_state = state
    slow_rng, fast_rng = split(rng)

    slow_outputs, slow_state = slow_apply(
        slow_rng, slow_params, slow_state, is_training, inputs
    )
    loss, (fast_state, *aux) = fast_apply_and_loss_fn(
        fast_rng, fast_params, fast_state, is_training, *slow_outputs, targets
    )

    return loss, ((slow_state, fast_state), aux)


# @jit
def step(rng, params, state, inputs, targets, opt_state, loss_fn, opt_update_fn):
    rng, rng_step = split(rng)
    (loss, (state, aux)), grads = value_and_grad(loss_fn, 1, has_aux=True)(
        rng_step, params, state, inputs, targets,
    )
    # print(grads)
    updates, opt_state = opt_update_fn(grads, opt_state, params)
    params = ox.apply_updates(params, updates)

    return params, state, opt_state, loss, aux


# @jit
def slow_extract_feats_fn(params, state, inputs, slow_apply_fn):
    return slow_apply_fn(None, params, state, False, inputs)[0][0]


def preprocess_and_extract_feats(params, state, inputs, slow_apply_fn, preprocess_fn):
    inputs = preprocess_fn(inputs)
    return slow_apply_fn(None, params, state, False, inputs)[0][0]


def preprocess_and_pred_fn(
    params, state, inputs, slow_apply, fast_apply, preprocess_fn
):
    inputs = preprocess_fn(inputs)
    slow_params, fast_params = params
    slow_state, fast_state = state
    slow_outputs, _ = slow_apply(None, slow_params, slow_state, False, inputs)
    fast_outputs = MiniImagenetCNNHead.apply(
        fast_params, fast_state, None, *slow_outputs, False
    )
    return fast_outputs[0].argmax(-1)


def parse_args():
    parser = Experiment.add_args()
    # Network hyperparameters
    parser.add_argument("--hidden_size", default=32, type=int)
    parser.add_argument("--no_track_bn_stats", default=False, action="store_true")
    parser.add_argument(
        "--activation", type=str, default="relu", choices=list(activations.keys())
    )

    # FSL evaluation arguments
    parser.add_argument("--way", type=int, default=5)
    parser.add_argument("--spt_shot", type=int, default=5)
    parser.add_argument("--qry_shot", type=int, default=15)
    parser.add_argument("--fsl_val_batch_size", type=int, default=25)
    parser.add_argument("--fsl_num_val_tasks", type=int, default=500)
    parser.add_argument("--pool", type=int, default=4)

    # Training hyperparameters
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--lr_schedule", nargs="*", type=int, default=[50, 80])
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=5e-4, type=float)
    parser.add_argument("--val_every_k_steps", default=10, type=int)

    args = parser.parse_args()
    cfg = OmegaConf.create()
    for argname, argval in vars(args).items():
        rsetattr(cfg, argname, argval)

    return args, cfg


if __name__ == "__main__":
    args, cfg = parse_args()
    exp = Experiment(cfg, args)
    exp.log_init(
        [sys.stdout]
    )  # Send logged stuff also to stdout (but not all stdout to log)
    exp.comet_init()
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

    train_images, train_labels, val_images, val_labels, preprocess_fn = prepare_data(
        cfg.dataset, cfg.data_dir, cpu, device, cfg.prefetch_data_gpu,
    )
    if cfg.dataset == "miniimagenet":
        sup_train_images = train_images[:, :500]
        sup_train_labels = train_labels[:, :500]
        sup_val_images = train_images[:, 500:]
        sup_val_labels = train_labels[:, 500:]

        transfer_spt_images = val_images[:, :500]
        transfer_spt_labels = val_labels[:, :500]
        transfer_qry_images = val_images[:, 500:]
        transfer_qry_labels = val_labels[:, 500:]

    sup_train_images = flatten(sup_train_images, 1)
    sup_train_labels = flatten(sup_train_labels, 1)
    sup_val_images = flatten(sup_val_images, 1)
    sup_val_labels = flatten(sup_val_labels, 1)
    print("Sup train data:", sup_train_images.shape, sup_train_labels.shape)
    print("Sup val data:", sup_val_images.shape, sup_val_labels.shape)
    print("Other tasks val data:", val_images.shape, val_labels.shape)

    (
        MiniImagenetCNNBody,
        MiniImagenetCNNHead,
        slow_apply,
        fast_apply_and_loss_fn,
    ) = prepare_model(
        xe_and_acc,
        cfg.dataset,
        train_images.shape[1],
        cfg.hidden_size,
        cfg.activation,
        track_stats=not cfg.no_track_bn_stats,
    )
    slow_params, fast_params, slow_state, fast_state = make_params(
        rng,
        cfg.dataset,
        MiniImagenetCNNBody.init,
        slow_apply,
        MiniImagenetCNNHead.init,
        device,
    )
    params = (slow_params, fast_params)
    state = (slow_state, fast_state)

    # SGD  with momentum, wd and schedule
    schedule = ox.piecewise_constant_schedule(
        -cfg.lr, {e: 0.1 for e in cfg.lr_schedule}
    )
    opt = ox.chain(
        ox.trace(decay=cfg.momentum, nesterov=False),
        ox.additive_weight_decay(cfg.weight_decay),
        ox.scale_by_schedule(schedule),
    )
    opt_state = opt.init(params)
    schedule_state = opt_state[-1]

    # 1. Train functions
    train_apply_fn_kwargs = {
        "slow_apply": slow_apply,
        "fast_apply_and_loss_fn": fast_apply_and_loss_fn,
        "is_training": True,
    }
    train_apply_and_loss_fn = partial(apply_and_loss_fn, **train_apply_fn_kwargs)
    train_step = partial(
        step, loss_fn=train_apply_and_loss_fn, opt_update_fn=opt.update
    )
    rng, rng_sampler = split(rng)
    train_sampler = BatchSampler(
        rng_sampler, sup_train_images, sup_train_labels, cfg.batch_size,
    )

    # 2. Test FSL functions
    fsl_test_sample_fn_kwargs = {
        "preprocess_fn": preprocess_fn,
        "images": val_images,
        "labels": val_labels,
        "num_tasks": cfg.fsl_val_batch_size,
        "way": cfg.way,
        "spt_shot": cfg.spt_shot,
        "qry_shot": cfg.qry_shot,
        "device": device,
        "disjoint": False,
    }
    fsl_test_sample_fn = partial(
        fsl_sample_transfer_and_build, **fsl_test_sample_fn_kwargs
    )
    fsl_test_apply_fn = partial(slow_extract_feats_fn, slow_apply_fn=slow_apply)

    # 3. Test transfer functions
    rng, rng_sampler = split(rng)
    transfer_test_spt_sampler = BatchSampler(
        rng_sampler,
        flatten(transfer_spt_images, 1),
        flatten(transfer_spt_labels, 1),
        cfg.batch_size,
        shuffle=False,
        keep_last=True,
    )
    transfer_test_qry_sampler = BatchSampler(
        rng_sampler,
        flatten(transfer_qry_images, 1),
        flatten(transfer_qry_labels, 1),
        cfg.batch_size,
        shuffle=False,
        keep_last=True,
    )
    transfer_test_apply_fn = partial(
        preprocess_and_extract_feats,
        slow_apply_fn=slow_apply,
        preprocess_fn=preprocess_fn,
    )

    # 4. Test supervised learning functions
    sup_test_pred_fn = partial(
        preprocess_and_pred_fn,
        slow_apply=slow_apply,
        fast_apply=MiniImagenetCNNHead.apply,
        preprocess_fn=preprocess_fn,
    )
    rng, rng_sampler = split(rng)
    sup_val_sampler = BatchSampler(
        rng_sampler, sup_val_images, sup_val_labels, cfg.batch_size,
    )

    if jit_enabled:
        train_step = jit(train_step)
        fsl_test_apply_fn = jit(fsl_test_apply_fn)
        transfer_test_apply_fn = jit(transfer_test_apply_fn)
        sup_test_pred_fn = jit(sup_test_pred_fn)

    pbar = tqdm(range(1), total=((sup_train_images.shape[0] // cfg.batch_size) * cfg.epochs))
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
            params, state, opt_state, loss, aux = train_step(
                rng, params, state, X, y, opt_state
            )
            opt_state[-1] = schedule_state # Before and after for checkpointing and safety

            curr_step += 1
            if ((j == 0) and (((epoch % cfg.val_every_k_steps) == 0) or epoch == 1)) or (
                (epoch == cfg.epochs) and (j == len(train_sampler) - 1)  # Last step
            ):
                # Test FSL
                rng, rng_val = split(rng)
                start = time.time()
                fsl_preds, fsl_targets = meta_test(
                    rng,
                    partial(fsl_test_apply_fn, params[0], state[0]),
                    fsl_test_sample_fn,
                    cfg.fsl_num_val_tasks // cfg.fsl_val_batch_size,
                    device=device,
                    pool=cfg.pool,
                )
                end = time.time()
                val_acc = (fsl_preds == fsl_targets).astype(onp.float).mean()


                # Test supervised learning
                sup_preds, sup_targets = test_sup(
                    partial(sup_test_pred_fn, params, state), sup_val_sampler, device,
                )
                sup_acc = (sup_preds == sup_targets).astype(onp.float).mean()
                    

                # Test transfer learning
                transfer_preds, transfer_targets = test_sup_cosine(
                    partial(transfer_test_apply_fn, params[0], state[0]),
                    transfer_test_spt_sampler,
                    transfer_test_qry_sampler,
                    device,
                )
                transfer_acc = (
                    (transfer_preds == transfer_targets).astype(onp.float).mean()
                )

                if sup_acc > best_sup_val_acc:
                    best_sup_val_acc = sup_acc
                    exp.log(f"\nNew best validation accuracy: {best_sup_val_acc}")
                    exp.log("Saving checkpoint\n")

                    with open(
                        osp.join(exp.exp_dir, "checkpoints/best.ckpt"), "wb"
                    ) as f:
                        dill.dump(
                            {
                                "sup_val_acc": best_sup_val_acc,
                                "transfer_val_acc": transfer_acc,
                                "fsl_val_acc": val_acc,
                                "optimizer_state": opt_state,
                                "slow_params": slow_params,
                                "fast_params": fast_params,
                                "slow_state": slow_state,
                                "fast_state": fast_state,
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
                    fsla=f"{val_acc:.2f}",
                    fslt=f"{end - start:.2f}",
                    supa=f"{sup_acc:.3f}",
                    transfera=f"{transfer_acc:.3f}",
                )

            elif (curr_step % cfg.progress_bar_refresh_rate) == 0:
                pbar.set_postfix(
                    loss=f"{loss:.3f}",
                    acc=f"{aux[0]['acc'].mean():.2f}",
                    lr=f"{schedule(opt_state[-1].count):.4f}",
                    fsla=f"{val_acc:.2f}",
                    fslt=f"{end - start:.2f}",
                    supa=f"{sup_acc:.3f}",
                    transfera=f"{transfer_acc:.3f}",
                )

            pbar.update()

        schedule_state = ox.ScaleByScheduleState(
            count=schedule_state.count + 1,
        )  # Unsafe for max int

