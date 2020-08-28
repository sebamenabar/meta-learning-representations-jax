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

from test_fsl import meta_test
from data.sampling import BatchSampler, fsl_sample_transfer_and_build
from optimizers import sgd
from data import prepare_data
from experiment import Experiment, Logger
from models.maml_conv import MiniImagenetCNNMaker, make_params, prepare_model
from lib import setup_device, mean_xe_and_acc_dict as xe_and_acc, flatten


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


if __name__ == "__main__":
    cfg = OmegaConf.create(
        dict(
            gpus=1,
            dataset="miniimagenet",
            data_dir=None,
            batch_size=64,
            epochs=100,
            prefetch_data_gpu=False,
            hidden_size=32,
            activation="relu",
            learning_rate=0.05,
            momentum=0.9,
            weight_decay=5e-4,
            pool=4,
            fsl_val_batch_size=25,
            fsl_val_way=5,
            fsl_val_spt_shot=5,
            fsl_val_qry_shot=15,
        ),
    )
    cpu, device = setup_device(cfg.gpus)
    rng = random.PRNGKey(0)
    if cfg.data_dir is None:
        if cfg.dataset == "miniimagenet":
            # cfg.data_dir = "/workspace1/samenabar/data/mini-imagenet/"
            cfg.data_dir = "/mnt/ialabnas/homes/samenabar/data/FSL/mini-imagenet"
        elif cfg.dataset == "omniglot":
            cfg.data_dir = "/workspace1/samenabar/data/omniglot/"

    train_images, train_labels, val_images, val_labels, preprocess_fn = prepare_data(
        cfg.dataset, cfg.data_dir, cpu, device, cfg.prefetch_data_gpu,
    )
    train_images = flatten(train_images, 1)
    train_labels = flatten(train_labels, 1)
    print("Train data:", train_images.shape, train_labels.shape)
    print("Val data:", val_images.shape, val_labels.shape)

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
        track_stats=True,
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
    schedule = ox.piecewise_constant_schedule(-cfg.learning_rate, {60: 0.1, 80: 0.1})
    opt = ox.chain(
        ox.trace(decay=cfg.momentum, nesterov=False),
        ox.additive_weight_decay(cfg.weight_decay),
        ox.scale_by_schedule(schedule),
    )
    opt_state = opt.init(params)
    schedule_state = opt_state[-1]

    fn_kwargs = {
        "slow_apply": slow_apply,
        "fast_apply_and_loss_fn": fast_apply_and_loss_fn,
        "is_training": True,
    }
    papply_and_loss_fn = partial(apply_and_loss_fn, **fn_kwargs)
    papply_and_loss_fn = jit(papply_and_loss_fn)

    @jit
    def step(rng, params, state, inputs, targets, opt_state):
        rng, rng_fwd = split(rng)
        (loss, (state, aux)), grads = value_and_grad(
            papply_and_loss_fn, 1, has_aux=True
        )(rng_fwd, params, state, inputs, targets,)
        # print(grads)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = ox.apply_updates(params, updates)

        return params, state, opt_state, loss, aux

    test_sample_fn_kwargs = {
        "preprocess_fn": preprocess_fn,
        "images": val_images,
        "labels": val_labels,
        "num_tasks": cfg.fsl_val_batch_size,
        "way": cfg.fsl_val_way,
        "spt_shot": cfg.fsl_val_spt_shot,
        "qry_shot": cfg.fsl_val_qry_shot,
        "device": device,
        "disjoint": False,
    }
    test_sample_fn = partial(fsl_sample_transfer_and_build, **test_sample_fn_kwargs)

    @jit
    def test_apply_fn(params, state, inputs):
        return slow_apply(None, params, state, False, inputs)[0][0]

    pbar = tqdm(range((train_images.shape[0] // cfg.batch_size) * cfg.epochs))
    curr_step = 0
    rng, rng_sampler = split(rng)
    sampler = BatchSampler(rng_sampler, train_images, train_labels, cfg.batch_size)
    for epoch in range(1, cfg.epochs + 1):
        # sampler = batch_sampler(rng, train_images, train_labels, cfg.batch_size)
        pbar.set_description(f"E:{epoch}")
        for j, (X, y) in enumerate(sampler):

            X = jax.device_put(X, device)
            y = jax.device_put(y, device)
            X = preprocess_fn(X)

            opt_state[-1] = schedule_state
            params, state, opt_state, loss, aux = step(
                rng, params, state, X, y, opt_state
            )

            curr_step += 1
            if ((j == 0) and (((epoch % 10) == 0) or epoch == 1)) or (
                (epoch == cfg.epochs) and (j == len(sampler) - 1)  # Last step
            ):
                rng, rng_val = split(rng)
                start = time.time()
                val_preds, val_targets = meta_test(
                    rng,
                    partial(test_apply_fn, params[0], state[0]),
                    test_sample_fn,
                    20,
                    device=device,
                    pool=cfg.pool,
                )

                end = time.time()
                val_acc = (val_preds == val_targets).astype(onp.float).mean()
                pbar.set_postfix(
                    loss=f"{loss:.3f}",
                    acc=f"{aux[0]['acc'].mean():.2f}",
                    lr=f"{schedule(opt_state[-1].count):.4f}",
                    val_acc=f"{val_acc:.2f}",
                    val_time=f"{end - start:.2f}",
                )

            elif (curr_step % 300) == 0:
                pbar.set_postfix(
                    loss=f"{loss:.3f}",
                    acc=f"{aux[0]['acc'].mean():.2f}",
                    lr=f"{schedule(opt_state[-1].count):.4f}",
                    val_acc=f"{val_acc:.2f}",
                    val_time=f"{end - start:.2f}",
                )

            pbar.update()

        schedule_state = ox.ScaleByScheduleState(
            count=schedule_state.count + 1,
        )  # Unsafe for max int

