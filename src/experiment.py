import os
import sys
import os.path as osp

import json
import yaml
import errno
import atexit
import shutil
import pprint as pp
from dateutil import tz
from datetime import datetime as dt

from omegaconf import OmegaConf
from argparse import ArgumentParser

# import wandb
# import tensorflow as tf
from tensorboardX import SummaryWriter
from comet_ml import Experiment as CometExperiment


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class Logger(object):
    def __init__(self, logfile, backends=None):
        if backends is None:
            self.backends = []
        else:
            self.backends = backends
        self.log = logfile

    def write(self, message):
        self.log.write(message)
        for b in self.backends:
            b.write(message)
        self.flush()

    def flush(self):
        self.log.flush()
        for b in self.backends:
            b.flush()


class Experiment:
    def __init__(self, cfg, args=None):
        self.cfg = cfg
        self.args = args
        self.comet_set = False
        self.run_name = cfg.run_name
        self.exp_dir = ""
        self.exp_name = cfg.exp_name
        if cfg.work_dir:
            self.work_dir = cfg.work_dir
        else:
            self.work_dir = osp.dirname(osp.dirname(osp.realpath(__file__)))

        if not cfg.no_log:
            self.on_create()

        self.comet = None
        self.tensorboard = None
        self.wandb = None
        self.logfile = None
        self.logging = None

    def logfile_init(self, backends=None):
        self.logfile = open(osp.join(self.exp_dir, "logfile.log"), "a")
        self.logging = Logger(self.logfile, backends)
        # atexit.register(self.logfile.close)
        self.log(f"Experiment directory: {self.exp_dir}")
        self.log("\nCLI arguments")
        self.log(pp.pformat(vars(self.args)))
        self.log("\nConfiguration")
        if isinstance(self.cfg, OmegaConf):
            self.log(pp.pformat(OmegaConf.to_container(self.cfg)))
        else:
            self.log(pp.pformat(self.cfg))
        self.log()

    def log(self, *args, **kwargs):
        if self.logging:
            return print(*args, file=self.logging, **kwargs)
        else:
            return print(*args, **kwargs)
        # return print(*args, **kwargs)

    @staticmethod
    def add_args(parser=None):
        if parser is None:
            parser = ArgumentParser(conflict_handler="resolve")

        parser.add_argument("--no_log", action="store_true", default=False)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--debug", action="store_true", default=False)
        # parser.add_argument("--val_interval", type=int, default=500)
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
        parser.add_argument("--work_dir", type=str, default="")
        parser.add_argument("--run_name", type=str, default="")
        parser.add_argument("--exp_name", type=str, default="")
        parser.add_argument("--logcomet", action="store_true", default=False)
        parser.add_argument("--project", type=str, default="debug")
        parser.add_argument("--workspace", type=str, default="debug")

        return parser

    def on_create(self):
        self.exp_init()
        # self.comet_init()
        # self.wandb_init()

    @property
    def logcomet(self):
        return self.cfg.logcomet

    def log_metrics(self, metrics, step=None, prefix=""):
        if self.comet:
            self.comet.log_metrics(metrics, step=step, prefix=prefix)
        if self.tensorboard:
            for name, val in metrics.items():
                if prefix:
                    tag = f"{prefix}_{name}"
                else:
                    tag = name
                self.tensorboard.add_scalar(
                    tag=tag, scalar_value=val, global_step=step,
                )

    def loggers_init(self):
        self.comet_init()
        self.tensorboard_init()

    def tensorboard_init(self):
        # self.tensorboard = tf.summary.create_file_writer(self.exp_dir)
        # self.tensorboard.set_as_default()
        self.tensorboard = SummaryWriter(logdir=self.exp_dir,)

    def comet_init(self):
        # if not self.logcomet:
        #     return
        self.comet = CometExperiment(
            api_key=os.environ["COMET_API_KEY"] if self.logcomet else "",
            workspace=self.cfg.workspace,
            project_name=self.cfg.project,
            disabled=not self.logcomet,
        )
        self.comet.set_name(self.run_name)
        if len(self.exp_dir):
            self.comet.log_asset(osp.join(self.exp_dir, "cfg.json"))
            self.comet.log_asset(osp.join(self.exp_dir, "cfg.yml"))
            if os.path.exists(osp.join(self.exp_dir, "args.json")):
                self.comet.log_asset(osp.join(self.exp_dir, "args.json"))
            self.log("Sent config to comet")  # TODO replace with logger
            self.comet.log_asset_folder(
                osp.join(self.work_dir, "src"), recursive=True, log_file_name=True
            )
            self.log("Sent source code to comet")

    def exp_init(self):
        now = dt.now(tz.tzlocal())
        now_str = now.strftime("%b%d-%Y")  # e.x: Aug25-2020
        now = now.strftime("%m-%d-%Y-%H-%M-%S")

        log_dir = self.exp_name
        _run_name = self.cfg.run_name
        self.exp_name = self.cfg.exp_name
        run_name = _run_name
        if _run_name == "" or _run_name is None:
            _run_name = now
        debug_str = "debug" if self.cfg.debug else ""
        _exp_dir = osp.join(
            self.work_dir, "experiments", debug_str, f"{now_str}-{log_dir}", _run_name
        )
        exp_dir = _exp_dir

        i = 2
        while os.path.exists(exp_dir):
            exp_dir = f"{_exp_dir}-{i}"
            run_name = f"{_run_name}-{i}"
            i += 1
        self.run_name = run_name
        self.exp_dir = exp_dir
        # mkdir_p(self.exp_dir)
        mkdir_p(osp.join(self.exp_dir, "checkpoints"))

        # self.logfile = open(osp.join(self.exp_dir, "logfile.log"), "a")
        # self.logger = Logger(self.logfile, [sys.stdout])
        # sys.stderr = Logger(self.logfile, [sys.stderr])
        #  atexit.register(self.logfile.close)

        shutil.copytree(
            # osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), "src"),
            osp.dirname(osp.realpath(__file__)),
            osp.join(self.exp_dir, "src"),
            ignore=shutil.ignore_patterns(".*", "__pycache__", ".DS_Store"),
        )

        with open(osp.join(self.exp_dir, "cfg.json"), "w") as f:
            if isinstance(self.cfg, OmegaConf):
                json.dump(OmegaConf.to_container(self.cfg), f, indent=4)
            else:
                json.dump(self.cfg, f, indent=4)
        with open(osp.join(self.exp_dir, "cfg.yml"), "w") as f:
            if isinstance(self.cfg, OmegaConf):
                yaml.dump(
                    json.loads(json.dumps(OmegaConf.to_container(self.cfg))),
                    f,
                    indent=4,
                )
            else:
                yaml.dump(json.loads(json.dumps(self.cfg)), f, indent=4)
        if self.args is not None:
            with open(osp.join(self.exp_dir, "args.json"), "w") as f:
                json.dump(vars(self.args), f, indent=4)

