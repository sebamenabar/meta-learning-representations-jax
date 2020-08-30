import os
import sys
import os.path as osp

import dill
from tqdm import tqdm
from argparse import ArgumentParser

import numpy as onp

import jax
from jax.random import split
from jax import (
    jit,
    vmap,
    random,
    numpy as jnp,
    value_and_grad,
)

import optax as ox
import haiku as hk

from experiment import Experiment, Logger
from trainers.meta_trainer import MetaTrainer
from lib import parse_and_build_cfg, setup_device
from models.maml_conv import miniimagenet_cnn_argparse


def parse_args(parser=None):
    parser = Experiment.add_args(parser)
    parser = miniimagenet_cnn_argparse(parser)
    # Training arguments
    parser.add_argument(
        "--meta_batch_size", help="Number of FSL tasks", default=20, type=int
    )
    parser.add_argument("--way", help="Number of classes per task", default=5, type=int)
    parser.add_argument(
        "--shot", help="Number of samples per class", default=5, type=int
    )
    parser.add_argument(
        "--qry_shot", type=int, help="Number of quried samples per class", default=15
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

    return parser


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
    rng, rng_trainer = split(rng)
    trainer = MetaTrainer(rng_trainer, cfg, exp, cpu, device)

    if jit_enabled:
        step = jit(trainer.step, static_argnums=0)

