from models.activations import activations
from argparse import ArgumentParser
import functools
from omegaconf import OmegaConf
from experiment import Experiment


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--val_every_k_steps", type=int, default=500)
    parser.add_argument("--disable_jit", action="store_true", default=False)
    parser.add_argument("--progress_bar_refresh_rate", type=int, default=50)
    # Device settings
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--prefetch_data_gpu", action="store_true", default=False)

    # Data settings
    parser.add_argument(
        "--data_dir", type=str, default=None,
    )
    parser.add_argument(
        "--dataset", type=str, choices=["miniimagenet", "omniglot"], required=True
    )
    # Model settings
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument(
        "--activation", type=str, default="relu", choices=list(activations.keys())
    )

    # Training config settings
    parser.add_argument("--way", type=int, required=True)
    parser.add_argument("--shot", type=int, required=True)
    parser.add_argument("--qry_shot", type=int, required=True)

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--val_batch_size", type=int, default=25)
    parser.add_argument("--val_num_tasks", type=int, default=1000)
    parser.add_argument("--inner_lr", type=float, default=1e-2)
    parser.add_argument("--outer_lr", type=float, default=1e-3)
    parser.add_argument("--num_outer_steps", type=int)
    parser.add_argument("--num_inner_steps", type=int, required=True)
    parser.add_argument(
        "--disjoint_tasks",
        action="store_true",
        help="Classes between tasks do not repeat",
        default=False,
    )
    Experiment.add_args(parser)

    args = parser.parse_args()
    cfg = OmegaConf.create()

    for argname, argval in vars(args).items():
        rsetattr(cfg, argname, argval)

    return args, cfg
