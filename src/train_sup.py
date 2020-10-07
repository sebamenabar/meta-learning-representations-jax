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
from models import make_params, prepare_model
from lib import (
    setup_device,
    mean_xe_and_acc_dict,
    flatten,
    apply_and_loss_fn,
)
from models.activations import activations
from test_utils import SupervisedStandardTester, SupervisedCosineTester
from data import augment
from main_loop import Evaluator
from data.miniimagenet import MiniImageNetDataset

# import utils.augmentations as augmentations


def step(
    rng,
    params,
    state,
    inputs,
    targets,
    opt_state,
    loss_fn,
    opt_update_fn,
    normalize_fn=None,
    augment_fn=None,
):
    rng, rng_step, rng_augment = split(rng, 3)
    inputs = inputs / 255
    if augment_fn:
        inputs = augment_fn(rng_augment, inputs)
    if normalize_fn:
        inputs = normalize_fn(inputs)
    (loss, (state, aux)), grads = value_and_grad(loss_fn, (0, 1), has_aux=True)(
        *params,
        *state,
        rng_step,
        inputs,
        targets,
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
    parser.add_argument(
        "--model.name", default="resnet12", choices=["convnet4", "resnet12"]
    )
    parser.add_argument("--model.hidden_size", default=64, type=int)
    parser.add_argument(
        "--model.no_track_bn_stats",
        default=True,
        action="store_false",
        dest="model.track_bn_stats",
    )
    parser.add_argument(
        "--model.normalization",
        default="bn",
        choices=["bn", "custom", "gn", "in", "ln"],
    )
    parser.add_argument(
        "--model.no_head_bias", default=True, action="store_false", dest="model.head_bias"
    )
    parser.add_argument(
        "--model.initializer",
        default="kaiming_normal",
        choices=["kaiming_normal", "glorot_uniform"],
    )
    parser.add_argument(
        "--model.activation",
        type=str,
        default="leaky_relu",
        choices=list(activations.keys()),
    )

    # FSL evaluation arguments
    parser.add_argument("--val.batch_size", type=int, default=16)
    parser.add_argument("--val.num_tasks", type=int, default=600)
    parser.add_argument(
        "--no_eval_aug", action="store_false", default=True, dest="eval_aug"
    )
    # parser.add_argument("--val.fsl_batch_size", type=int, default=16)

    # Training hyperparameters
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--train.batch_size", default=64, type=int)
    parser.add_argument("--lr", default=5e-2, type=float)
    parser.add_argument("--lr_schedule", nargs="*", type=int, default=[60, 80])
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=5e-4, type=float)
    parser.add_argument("--val_interval", default=1, type=int)  # In epochs
    parser.add_argument(
        "--train.augment", default="all", choices=["none", "all"]
    )  # In epochs

    args = parser.parse_args()
    cfg = edict(train=edict(), model=edict(), val=edict(fsl=edict(), sup=edict()))
    for argname, argval in vars(args).items():
        rsetattr(cfg, argname, argval)

    return args, cfg


# def augment(rng, imgs, color_jitter_prob=1.0):
#     rng_crop, rng_color, rng_flip = split(rng, 3)
#     imgs = augmentations.random_crop(imgs, rng_crop, 84, ((8, 8), (8, 8), (0, 0)))
#     imgs = augmentations.color_transform(
#         imgs,
#         rng_color,
#         brightness=0.4,
#         contrast=0.4,
#         saturation=0.4,
#         hue=0.0,
#         color_jitter_prob=color_jitter_prob,
#         to_grayscale_prob=0.0,
#     )
#     imgs = augmentations.random_flip(imgs, rng_flip)
#     # imgs = normalize_fn(imgs)
#     return imgs


if __name__ == "__main__":
    args, cfg = parse_args()
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

    exp.log(f"JAX available CPUS {jax.devices('cpu')}")
    try:
        exp.log(f"JAX available GPUS {jax.devices('gpu')}")
    except RuntimeError:
        pass
    rng = random.PRNGKey(cfg.seed)

    """
    for miniimagenet the shapes are:
        train_images: [64, 600, 84, 84, 3]
        train_labels: [64, 600]
        val_images: [18748, 84, 84, 3]
        val_labels: [18748]
    """

    train_dataset = MiniImageNetDataset("train", cfg.data_dir)
    # val_dataset = MiniImageNetDataset("train_val", cfg.data_dir)
    rng, rng_sampler = split(rng)
    train_loader = BatchSampler(
        rng_sampler,
        train_dataset._images,
        train_dataset._labels,
        cfg.train.batch_size,
        shuffle=True,
        keep_last=False,
    )

    exp.log("Train data:", train_dataset._images.shape, train_dataset._labels.shape)
    #  exp.log("Validation data:", val_dataset._images.shape, val_dataset._labels.shape)

    output_size = 64  # TEMP
    body, head = prepare_model(
        cfg.dataset,
        cfg.model.name,
        output_size,
        hidden_size=cfg.model.hidden_size,
        avg_pool=True,
        activation=cfg.model.activation,
        initializer=cfg.model.initializer,
        track_stats=cfg.model.track_bn_stats,
        head_bias=cfg.model.head_bias,
        normalize=cfg.model.normalization,
    )
    rng, rng_params = split(rng)
    (slow_params, fast_params, slow_state, fast_state,) = make_params(
        rng,
        cfg.dataset,
        body.init,
        body.apply,
        head.init,
        train_dataset._normalize(next(iter(train_loader))[0] / 255),
    )
    params = (slow_params, fast_params)
    state = (slow_state, fast_state)

    # SGD  with momentum, wd and schedule
    schedule = ox.piecewise_constant_schedule(
        -cfg.lr, {e: 0.1 for e in cfg.lr_schedule}
    )
    # schedule = ox.cosine_decay_schedule(-cfg.lr, cfg.epochs, 0.01)
    opt = ox.chain(
        ox.additive_weight_decay(cfg.weight_decay),
        ox.trace(decay=cfg.momentum, nesterov=False),
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
    # rng, rng_sampler = split(rng)

    # test_pred_fn = jit(
    #     partial(
    #         pred_fn, is_training=False, slow_apply=body.apply, fast_apply=head.apply
    #     )
    # )

    # rng, rng_std_tester, rng_cosine_tester = split(rng, 3)
    # supervised_std_tester = SupervisedStandardTester(
    #     rng_std_tester,
    #     val_images,
    #     val_labels,
    #     cfg.val.batch_size,
    #     normalize_fn,
    #     device,
    # )

    if cfg.train.augment == "all":
        augment_fn = augment
        exp.log("Using data augment")
    else:
        augment_fn = None

    step_ins = jit(
        partial(
            step,
            loss_fn=train_apply_and_loss_fn,
            opt_update_fn=opt.update,
            normalize_fn=train_dataset._normalize,
            augment_fn=augment_fn,
        )
    )
    #  augment = jit(augment)
    #  normalize_fn = jit(normalize_fn)

    evaluator = Evaluator(
        cfg.data_dir,
        cfg.val,
        body.apply,
        head.apply,
    )

    pbar = tqdm(
        range(1),
        total=(((train_dataset._images.shape[0] // cfg.train.batch_size)) * cfg.epochs),
        file=sys.stdout,
        miniters=25,
        mininterval=5,
        maxinterval=30,
    )
    curr_step = 0
    best_sup_val_acc = 0
    for epoch in range(1, cfg.epochs + 1):
        schedule_state = ox.ScaleByScheduleState(
            count=schedule_state.count + 1,
        )  # Unsafe for max int
        # sampler = batch_sampler(rng, train_images, train_labels, train.cfg.batch_size)
        pbar.set_description(f"E:{epoch}")
        for j, (X, y) in enumerate(train_loader):
            rng_augment, rng_step, rng = split(rng, 3)
            # X = jax.device_put(X, device)
            # y = jax.device_put(y, device)
            # X = X / 255
            # if cfg.data_augment:
            #     X = augment(rng_augment, X)
            # X = normalize_fn(X)

            opt_state[-1] = schedule_state
            params, state, opt_state, loss, aux = step_ins(
                rng_step, params, state, X, y, opt_state
            )
            opt_state[
                -1
            ] = schedule_state  # Before and after for checkpointing and safety

            curr_step += 1
            if ((epoch == 1) and (j == 0)) or (
                (j == (len(train_loader) - 1))
                and ((epoch == cfg.epochs) or ((epoch % cfg.val_interval) == 0))
            ):
                # rng, rng_test = split(rng)
                # Test supervised learning
                # sup_std_loss, sup_std_acc = supervised_std_tester(
                #     partial(test_pred_fn, *params, *state, rng_test)
                # )

                # exp.log_metrics(
                #     {"sup_acc": sup_std_acc, "sup_loss": sup_std_loss},
                #     step=curr_step,
                #     prefix="val",
                # )

                val_metrics = evaluator.eval(
                    exp,
                    *params,
                    *state,
                    inner_lr=None,
                    reset_head=None,
                    eval_aug=cfg.eval_aug,
                    maml_eval=False,
                    sup_eval=True,
                )

                exp.log(f"\nValidation epoch {epoch} results:")
                exp.log(
                    f"Logistic Regression No-Aug Acc: {val_metrics.lr_no_aug_acc}±{val_metrics.lr_no_aug_std}"
                )
                if cfg.eval_aug:
                    exp.log(
                        f"Logistic Regression Aug Acc: {val_metrics.lr_aug_acc}±{val_metrics.lr_aug_std}"
                    )
                exp.log(f"Supervised Acc: {val_metrics.sup_acc}")
                if val_metrics.sup_acc > best_sup_val_acc:
                    best_sup_val_acc = val_metrics.sup_acc
                    exp.log(
                        f"\nNew best supervised validation accuracy: {best_sup_val_acc}"
                    )
                    exp.log("Saving checkpoint\n")

                    with open(
                        osp.join(exp.exp_dir, "checkpoints/best.ckpt"), "wb"
                    ) as f:
                        dill.dump(
                            {
                                # "val_acc": sup_std_acc,
                                # "val_loss": sup_std_loss,
                                **val_metrics,
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
                    # val_acc=f"{sup_std_acc:.3f}",
                    # val_loss=f"{sup_std_loss:.3f}",
                )

            elif (curr_step % cfg.progress_bar_refresh_rate) == 0:
                exp.log_metrics(
                    {"acc": aux[0]["acc"].mean(), "sup_loss": loss},
                    step=curr_step,
                    prefix="train",
                )
                pbar.set_postfix(
                    loss=f"{loss:.3f}",
                    acc=f"{aux[0]['acc'].mean():.2f}",
                    lr=f"{schedule(opt_state[-1].count):.4f}",
                    # val_acc=f"{sup_std_acc:.3f}",
                    # val_loss=f"{sup_std_loss:.3f}",
                )

            pbar.update()

        exp.log("\n")
        exp.log(f"---------- Epoch {epoch} ----------")
        exp.log(pbar.format_meter(**pbar.format_dict))
        exp.log(f"\nCurrent learning rate: {schedule(schedule_state.count)}")

    with open(osp.join(exp.exp_dir, "checkpoints/last.ckpt"), "wb") as f:
        dill.dump(
            {
                # "val_acc": sup_std_acc,
                # "val_loss": sup_std_loss,
                **val_metrics,
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
