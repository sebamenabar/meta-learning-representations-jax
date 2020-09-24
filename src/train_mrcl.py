import os
import sys
import os.path as osp

import dill
import time
from tqdm import tqdm
from argparse import ArgumentParser, Action
from easydict import EasyDict as edict

import numpy as onp

import jax
from jax.random import split
from jax import (
    jit,
    pmap,
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
    meta_step,
    delayed_cosine_decay_schedule,
)
from data import prepare_data, augment as augment_fn
from experiment import Experiment, Logger
from data.sampling import fsl_sample, fsl_build, BatchSampler
from models import make_params, prepare_model
from test_utils import test_fsl_maml, test_fsl_embeddings
from test_sup import test_sup_cosine


class ParseKwargs(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, edict())
        for value in values:
            key, value = value.split("=")
            getattr(namespace, self.dest)[key] = value


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
    parser.add_argument("--train.cl_qry_way", default=64, type=int)
    parser.add_argument("--train.cl_qry_shot", default=1, type=int)
    parser.add_argument("--train.inner_lr", type=float, default=1e-2)
    parser.add_argument("--train.outer_lr", type=float, default=1e-3)
    parser.add_argument("--train.num_inner_steps", type=int, default=5)
    parser.add_argument("--train.learn_inner_lr", default=False, action="store_true")

    parser.add_argument("--train.prefetch", default=0, type=int)
    # parser.add_argument("--train.weight_decay", default=0.0, type=float)
    parser.add_argument("--train.apply_every", default=1, type=int)
    parser.add_argument("--train.scheduler", choices=["none", "step", "cosine"])

    parser.add_argument(
        "--train.piecewise_constant_schedule",
        nargs="*",
        type=int,
        default=[10000, 25000],
    )
    parser.add_argument(
        "--train.piecewise_constant_alpha",
        default=0.1,
        type=float,
    )

    parser.add_argument("--train.cosine_alpha", type=float, default=0.01)
    parser.add_argument("--train.cosine_decay_steps", type=float, default=10000)
    parser.add_argument("--train.cosine_transition_begin", type=float, default=5000)

    parser.add_argument(
        "--train.augment", default="none", choices=["none", "all", "spt", "qry"]
    )
    parser.add_argument("--train.num_prefetch", default=10, type=int)

    parser.add_argument("--train.val_interval", type=int, default=1000)
    parser.add_argument(
        "--train.reset_head",
        type=str,
        choices=["zero", "kaiming", "glorot"],
    )

    # parser.add_argument("--val.pool", type=int, default=4)
    parser.add_argument(
        "--val.fsl.batch_size", help="Number of FSL tasks", default=20, type=int
    )
    parser.add_argument(
        "--val.fsl.qry_shot",
        type=int,
        help="Number of queried samples per class",
        default=15,
    )
    parser.add_argument("--val.fsl.num_inner_steps", type=int, default=10)
    parser.add_argument("--val.fsl.num_tasks", type=int, default=300)

    parser.add_argument("--model.name", choices=["resnet12", "convnet4"])
    parser.add_argument("--model.output_size", type=int)
    parser.add_argument("--model.hidden_size", default=0, type=int)
    parser.add_argument("--model.activation", default="relu", type=str)
    parser.add_argument(
        "--model.initializer",
        default="glorot_uniform",
        type=str,
        choices=["kaiming_normal", "glorot_uniform"],
    )
    parser.add_argument(
        "--model.no_avg_pool", default=True, action="store_false", dest="model.avg_pool"
    )
    parser.add_argument("--model.head_bias", default=False, action="store_true")
    # parser.add_argument("--model.norm_before_act", default=1, type=int, choices=[0, 1])
    # parser.add_argument(
    #     "--model.final_norm", default="none", choices=["bn", "gn", "in", "ln", "none"]
    # )
    parser.add_argument(
        "--model.normalize",
        default="bn",
        type=str,
        choices=["bn", "affine", "gn", "in", "ln", "custom", "none"],
    )
    parser.add_argument(
        "--model.track_stats",
        default="none",
        type=str,
        choices=["none", "inner", "outer", "inner", "inner-outer"],
    )

    args = parser.parse_args()
    cfg = edict(train=edict(cl=edict()), val=edict(fsl=edict()), model=edict())
    for argname, argval in vars(args).items():
        rsetattr(cfg, argname, argval)

    return args, cfg


def main(args, cfg):
    # args, cfg = parse_args()
    # cfg = parse_and_build_cfg(args)
    # The Experiment class creates a directory for the experiment,
    # copies source files and creates configurations files
    # It also creates a logfile.log which can be written to with exp.log
    # that is a wrapper of the print method
    exp = Experiment(cfg, args)
    if not cfg.no_log:
        exp.logfile_init(
            [sys.stdout]
        )  # Send logged stuff also to stdout (but not all stdout to log)
        exp.loggers_init()
        sys.stderr = Logger(exp.logfile, [sys.stderr])  # Send stderr to log

    if cfg.debug:  # Debugging creates experiments folders in experiments/debug dir
        exp.log("Debugging ...")

    rng = random.PRNGKey(cfg.seed)  # Default seed is 0
    # exp.log(f"Running on {device} with seed: {cfg.seed}")
    exp.log(f"JAX available CPUS {jax.devices('cpu')}")
    try:
        exp.log(f"JAX available GPUS {jax.devices('gpu')}")
    except RuntimeError:
        pass
    num_devices = max(cfg.gpus, 1)
    exp.log(f"Using {num_devices} devices")
    cpu = jax.devices("cpu")[0]

    # Data
    train_images, train_labels, normalize_fn = prepare_data(
        cfg.dataset,
        osp.join(
            cfg.data_dir,
            "miniImageNet_category_split_train_phase_train_ordered.pickle",
        ),
        # device,
    )
    val_images, _val_labels, _ = prepare_data(
        cfg.dataset,
        osp.join(
            cfg.data_dir,
            "miniImageNet_category_split_val_ordered.pickle",
        ),
        # device,
    )
    val_labels = (
        _val_labels - 64
    )  # make val labels start at 0 (they originally begin are at 64-79)

    exp.log("Train data:", train_images.shape, train_labels.shape)
    exp.log(
        "Validation data:",
        val_images.shape,
        val_images.shape,
    )

    body, head = prepare_model(
        cfg.model.name,
        cfg.dataset,
        cfg.model.output_size,
        hidden_size=cfg.model.hidden_size,
        activation=cfg.model.activation,
        track_stats=cfg.model.track_stats != "none",
        initializer=cfg.model.initializer,
        avg_pool=cfg.model.avg_pool,
        head_bias=cfg.model.head_bias,
        normalize=cfg.model.normalize,
    )

    @jit
    def embeddings_fn(slow_params, slow_state, inputs):
        return body.apply(slow_params, slow_state, None, inputs, None)[0][0]

    effective_batch_size, ragged = divmod(cfg.train.batch_size, cfg.train.apply_every)
    if ragged:
        raise ValueError(
            f"Global batch size {cfg.train.batch_size} must be divisible by "
            f"apply_every {cfg.train.apply_every}"
        )
    exp.log(f"Effective batch size: {effective_batch_size}")
    per_device_batch_size, ragged = divmod(effective_batch_size, num_devices)
    if ragged:
        raise ValueError(
            f"Effective batch size {effective_batch_size} must be divisible by "
            f"num devices {num_devices}"
        )
    exp.log(f"Per device batch size: {per_device_batch_size}")

    inner_lr = jnp.ones([]) * cfg.train.inner_lr

    opt_transforms = [
        ox.clip(10),
        ox.scale(1 / cfg.train.apply_every),
        ox.apply_every(cfg.train.apply_every),
        # ox.additive_weight_decay(cfg.train.weight_decay),
        ox.scale_by_adam(),
    ]
    if cfg.train.scheduler == "cosine":
        schedule = delayed_cosine_decay_schedule(
            init_value=-cfg.train.outer_lr,
            transition_begin=cfg.train.cosine_transition_begin,
            decay_steps=cfg.train.cosine_decay_steps,
            alpha=cfg.train.cosine_alpha,
        )
        opt_transforms.append(ox.scale_by_schedule(schedule))
        opt_transforms.append(ox.scale_by_schedule(schedule))
    elif cfg.train.scheduler == "step":
        schedule = ox.piecewise_constant_schedule(
            -cfg.train.outer_lr,
            {
                e: cfg.train.piecewise_constant_alpha
                for e in cfg.train.piecewise_constant_schedule
            },
        )
        opt_transforms.append(ox.scale_by_schedule(schedule))
    else:
        schedule = None

    outer_opt = ox.chain(*opt_transforms)

    train_sample_fn_kwargs = {
        "images": train_images,
        "labels": train_labels,
        "num_tasks": effective_batch_size,
        "way": cfg.train.way,
        "spt_shot": cfg.train.shot,
        "qry_shot": cfg.train.qry_shot,
        "shuffled_labels": False,
        "disjoint": False,  # tasks can share classes
    }
    train_sample_fn = partial(
        fsl_sample,
        **train_sample_fn_kwargs,
    )
    fsl_build_ins = jit(
        partial(
            fsl_build,
            batch_size=effective_batch_size,
            way=cfg.train.way,
            shot=cfg.train.shot,
            qry_shot=cfg.train.qry_shot,
        )
    )

    if cfg.train.cl_qry_way > 0:
        train_cl_sample_fn_kwargs = {
            "images": train_images,
            "labels": train_labels,
            "num_tasks": effective_batch_size,
            "way": cfg.train.cl_qry_way,
            "spt_shot": 0,
            "qry_shot": cfg.train.cl_qry_shot,
            "shuffled_labels": False,
            "disjoint": False,  # tasks can share classes
        }
        train_cl_sample_fn = partial(
            fsl_sample,
            **train_cl_sample_fn_kwargs,
        )
        cl_build_ins = jit(
            partial(
                fsl_build,
                batch_size=effective_batch_size,
                way=cfg.train.cl_qry_way,
                shot=0,
                qry_shot=cfg.train.cl_qry_shot,
            )
        )

if __name__ == "__main__":
    args, cfg = parse_args()
    main(args, cfg)