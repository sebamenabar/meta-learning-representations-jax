import os
import sys
import os.path as osp

import dill
import time
from tqdm import tqdm
from argparse import ArgumentParser
from easydict import EasyDict as edict

import jax

from config import rsetattr
from mrcl_experiment import MetaLearner
from experiment import Experiment, Logger


def parse_args(parser=None):
    parser = Experiment.add_args(parser)

    parser.add_argument("--train.num_outer_steps", type=int, default=30000)
    parser.add_argument(
        "--train.batch_size", help="Number of FSL tasks", default=4, type=int
    )
    parser.add_argument(
        "--train.sub_batch_size", help="Number of FSL tasks", default=None, type=int
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
    parser.add_argument("--train.weight_decay", default=0.0, type=float)
    #  parser.add_argument("--train.apply_every", default=1, type=int)
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
    parser.add_argument("--train.prefetch", default=10, type=int)

    parser.add_argument("--train.val_interval", type=int, default=1000)
    parser.add_argument("--train.method", default="maml", choices=["maml", "mrcl"])
    parser.add_argument(
        "--train.reset_head",
        type=str,
        default="none",
        choices=[
            "none",
            "all-zero",
            "all-glorot",
            "all-kaiming",
            "cls-zero",
            "cls-glorot",
            "cls-kaiming",
        ],
    )

    # parser.add_argument("--val.pool", type=int, default=4)
    parser.add_argument(
        "--val.fsl.batch_size", help="Number of FSL tasks", default=25, type=int
    )
    parser.add_argument(
        "--val.fsl.qry_shot",
        type=int,
        help="Number of quried samples per class",
        default=15,
    )
    parser.add_argument("--val.fsl.num_inner_steps", type=int, default=10)
    parser.add_argument("--val.fsl.num_tasks", type=int, default=300)

    parser.add_argument("--model.model_name", default="convnet4", choices=["resnet12", "convnet4"])
    parser.add_argument("--model.output_size", type=int)
    parser.add_argument("--model.hidden_size", default=0, type=int)
    parser.add_argument("--model.activation", default="relu", type=str)
    parser.add_argument(
        "--model.initializer",
        default="glorot_uniform",
        type=str,
        choices=["kaiming_normal", "glorot_uniform"],
    )
    parser.add_argument("--model.avg_pool", default=True, action="store_true")
    parser.add_argument("--model.head_bias", default=False, action="store_true")
    parser.add_argument("--model.norm_before_act", default=1, type=int, choices=[0, 1])
    parser.add_argument(
        "--model.final_norm", default="none", choices=["bn", "gn", "in", "ln", "none"]
    )
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
    exp = Experiment(cfg, args)
    if not cfg.no_log:
        exp.logfile_init(
            [sys.stdout]
        )  # Send logged stuff also to stdout (but not all stdout to log)
        exp.loggers_init()
        sys.stderr = Logger(exp.logfile, [sys.stderr])  # Send stderr to log

    if cfg.debug:  # Debugging creates experiments folders in experiments/debug dir
        exp.log("Debugging ...")

    exp.log(f"JAX available CPUS {jax.devices('cpu')}")
    try:
        exp.log(f"JAX available GPUS {jax.devices('gpu')}")
    except RuntimeError:
        pass

    meta_learner = MetaLearner(
        cfg.seed,
        cfg.dataset,
        cfg.data_dir,
        cfg.model,
        cfg.train,
    )

    counter = 0
    for i in range(1, cfg.train.num_outer_steps * meta_learner._apply_every + 1):
        
        meta_learner.step(global_step=0, rng=None)
        
        if (i % cfg.train.apply_every) == 0:
            pbar.update()
            counter += 1

        break


if __name__ == "__main__":
    main(*parse_args())
