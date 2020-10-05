import os
import sys
import os.path as osp

import dill
import time
from tqdm import tqdm
from argparse import ArgumentParser
from easydict import EasyDict as edict

import jax
import jax.numpy as jnp
from jax.random import split

import haiku as hk

# import chex

from config import rsetattr
from mrcl_experiment import MetaLearner, replicate_array, MetaMiniImageNet
from eval_experiment import GPUMultinomialRegression, MAMLTester
from experiment import Experiment, Logger


def parse_args(parser=None):
    parser = Experiment.add_args(parser)

    parser.add_argument("--worker_tpu_driver", default="")

    parser.add_argument("--train.num_outer_steps", type=int, default=30000)
    parser.add_argument(
        "--train.batch_size", help="Number of FSL tasks", default=8, type=int
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
        default=15,
    )
    parser.add_argument("--train.cl_qry_way", default=64, type=int)
    parser.add_argument("--train.cl_qry_shot", default=1, type=int)
    parser.add_argument("--train.inner_lr", type=float, default=1e-2)
    parser.add_argument("--train.outer_lr", type=float, default=1e-3)
    parser.add_argument("--train.num_inner_steps", type=int, default=5)
    parser.add_argument("--train.learn_inner_lr", default=False, action="store_true")

    parser.add_argument("--train.prefetch", default=0, type=int)
    parser.add_argument("--train.weight_decay", default=0.0, type=float)
    # parser.add_argument("--train.apply_every", default=1, type=int)
    parser.add_argument(
        "--train.scheduler", default="none", choices=["none", "step", "cosine"]
    )

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
    parser.add_argument("--train.cosine_steps", type=float, default=10000)
    parser.add_argument("--train.cosine_delay", type=float, default=5000)

    parser.add_argument(
        "--train.augment", default="none", choices=["none", "all", "spt", "qry"]
    )
    # parser.add_argument("--train.prefetch", default=10, type=int)

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
        "--val.batch_size", help="Number of FSL tasks", default=32, type=int
    )
    # parser.add_argument(
    #     "--val.fsl.qry_shot",
    #     type=int,
    #     help="Number of quried samples per class",
    #     default=15,
    # )
    parser.add_argument("--val.num_inner_steps", type=int, default=10)
    parser.add_argument("--val.num_tasks", type=int, default=600)

    parser.add_argument(
        "--model.model_name", default="convnet4", choices=["resnet12", "convnet4"]
    )
    parser.add_argument("--model.output_size", type=int, required=True)
    parser.add_argument("--model.hidden_size", default=64, type=int)
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

    parser.add_argument("--fake_pmap_jit", action="store_true", default=False)
    parser.add_argument(
        "--no_eval_aug", action="store_false", default=True, dest="eval_aug"
    )

    args = parser.parse_args()
    cfg = edict(train=edict(cl=edict()), val=edict(fsl=edict()), model=edict())
    for argname, argval in vars(args).items():
        rsetattr(cfg, argname, argval)

    return args, cfg


class Evaluator:
    def __init__(self, data_dir, val_cfg, slow_apply, fast_apply):
        self.slow_apply = slow_apply
        self.fast_apply = fast_apply
        self.val_dataset1 = MetaMiniImageNet(
            jax.random.PRNGKey(0),
            "val",
            data_dir,
            val_cfg.batch_size,
            way=5,
            shot=5,
            qry_shot=15,
        )
        # For validation with augmented samples we need
        # another dataset with reduced batch_size
        self.val_dataset2 = MetaMiniImageNet(
            jax.random.PRNGKey(0),
            "val",
            data_dir,
            max(val_cfg.batch_size // 4, 1),
            way=5,
            shot=5,
            qry_shot=15,
        )
        self.lr_no_aug_tester = GPUMultinomialRegression(
            5,
            slow_apply,
            val_cfg.num_tasks,
            self.val_dataset1,
            0,
            self.val_dataset1._normalize,
        )
        self.lr_aug_tester = GPUMultinomialRegression(
            5,
            slow_apply,
            val_cfg.num_tasks,
            self.val_dataset2,
            5,
            self.val_dataset2._normalize,
            keep_orig_aug=False,
        )
        # self.maml_tester_no_aug = MAMLTester(
        #     slow_apply,
        #     fast_apply,
        #     val_cfg.num_tasks,
        #     # cfg.val.batch_size,
        #     self.val_dataset,
        #     # val_cfg.num_inner_steps,
        #     5,
        #     0,
        #     self.val_dataset1._normalize,
        # )
        self.maml_tester_no_aug_10_steps = MAMLTester(
            slow_apply,
            fast_apply,
            val_cfg.num_tasks,
            # cfg.val.batch_size,
            self.val_dataset1,
            # cfg.val.num_inner_steps,
            10,
            0,
            self.val_dataset1._normalize,
        )
        self.maml_tester_aug_10_steps = MAMLTester(
            slow_apply,
            fast_apply,
            val_cfg.num_tasks,
            # cfg.val.batch_size,
            self.val_dataset2,
            # cfg.val.num_inner_steps,
            10,
            5,
            self.val_dataset2._normalize,
            keep_orig_aug=False,
        )

    def eval(
        self,
        exp,
        slow_params,
        fast_params,
        slow_state,
        fast_state,
        inner_lr,
        reset_head,
        eval_aug,
    ):
        if reset_head != "none":
            fast_params = jax.tree_map(jnp.zeros_like, fast_params)
        exp.log()
        exp.log("Evaluating LR No-Aug")

        out = edict()

        out.lr_no_aug_acc, out.lr_no_aug_std = self.lr_no_aug_tester.eval(
            slow_params,
            slow_state,
        )
        if eval_aug:
            exp.log("Evaluating LR Aug")
            out.lr_aug_acc, out.lr_aug_std = self.lr_aug_tester.eval(
                slow_params,
                slow_state,
            )
        exp.log("Evaluating MAML No-Aug")
        (
            out.maml_acc_no_aug,
            out.maml_std_no_aug,
        ) = self.maml_tester_no_aug_10_steps.eval(
            slow_params,
            fast_params,
            slow_state,
            fast_state,
            inner_lr,
        )
        if eval_aug:
            exp.log("Evaluating MAML Aug")
            out.maml_acc_aug, out.maml_std_aug = self.maml_tester_aug_10_steps.eval(
                slow_params,
                fast_params,
                slow_state,
                fast_state,
                inner_lr,
            )

        return out

        # exp.log("Evaluating MAML Aug")


def main(args, cfg):
    if cfg.worker_tpu_driver:
        jax.config.update("jax_xla_backend", "tpu_driver")
        jax.config.update("jax_backend_target", cfg.worker_tpu_driver)
        print("Backend: %s %r", cfg.worker_tpu_driver, jax.devices())

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
    try:
        exp.log(f"JAX available TPUS {jax.devices('tpu')}")
    except RuntimeError:
        pass

    # if cfg.fake_pmap_jit:
    #     chex.fake_pmap().context.start()
    #     chex.fake_jit().context.start()

    meta_learner = MetaLearner(
        cfg.seed,
        cfg.dataset,
        cfg.data_dir,
        cfg.model,
        cfg.train,
        cfg.val,
    )

    evaluator = Evaluator(
        cfg.data_dir,
        cfg.val,
        meta_learner._encoder.apply,
        meta_learner._classifier.apply,
    )

    scalars_ema = None
    rng = rng_step = jax.random.PRNGKey(0)  # Get's replaced in first step with seed
    counter = 0
    pbar = tqdm(range(cfg.train.num_outer_steps), ncols=0)
    best_val_acc = 0.0
    for global_step in pbar:

        if rng is not None:
            rng, rng_step = split(rng)
        rng, scalars = meta_learner.step(global_step=global_step, rng=rng_step)

        if scalars_ema is None or global_step < 50:
            scalars_ema = jax.tree_map(
                lambda x: x,
                scalars,
            )
        else:
            scalars_ema = jax.tree_multimap(
                lambda ema, x: ema * 0.99 + x * 0.01, scalars_ema, scalars
            )

        if (
            (global_step == 0)
            or (
                (((global_step + 1) % cfg.train.val_interval) == 0)
                and (global_step != 1)
            )
            or (global_step == (cfg.train.num_outer_steps - 1))
        ):
            learner_state = meta_learner.get_first_state()

            val_metrics = evaluator.eval(
                exp,
                learner_state.slow_params,
                learner_state.fast_params,
                learner_state.slow_state,
                learner_state.fast_state,
                learner_state.inner_lr,
                cfg.train.reset_head,
                cfg.eval_aug,
            )

            exp.log(f"\nStep {global_step + 1} statistics:")
            # exp.log(
            #     f"MAML {cfg.val.num_inner_steps}-steps No-Aug Acc: {maml_acc_no_aug}±{maml_std_no_aug}"
            # )
            exp.log(
                f"MAML 10-steps No-Aug Acc: {val_metrics.maml_acc_no_aug}±{val_metrics.maml_std_no_aug}"
            )
            # exp.log(
            #     f"MAML 20-steps No-Aug Acc: {maml_acc_no_aug_10_steps}±{maml_std_no_aug_10_steps}"
            # )
            # exp.log(f"MAML Aug Acc: {maml_acc_aug}±{maml_std_aug}")
            exp.log(
                f"Logistic Regression No-Aug Acc: {val_metrics.lr_no_aug_acc}±{val_metrics.lr_no_aug_std}"
            )

            if cfg.eval_aug:
                exp.log(
                    f"Logistic Regression Aug Acc: {val_metrics.lr_aug_acc}±{val_metrics.lr_aug_std}"
                )
                exp.log(
                    f"MAML 10-steps Aug Acc: {val_metrics.maml_acc_aug}±{val_metrics.maml_std_aug}"
                )

            exp.log_metrics(
                dict(
                    ilr=meta_learner._learner_state.inner_lr[0],
                    **val_metrics,
                ),
                step=global_step,
                prefix="val",
            )
            exp.log()

            if val_metrics.lr_no_aug_acc > best_val_acc:
                best_val_acc = val_metrics.lr_no_aug_acc
                #  outer_opt_state = jax.tree_map(lambda xs: xs[0, 0], rep_outer_opt_state)
                # best_val_acc = fsl_maml_acc_5
                exp.log(f"\  New best 5-way-5-shot validation accuracy: {best_val_acc}")
                if not cfg.no_log:
                    exp.log("Saving checkpoint\n")
                    with open(
                        osp.join(exp.exp_dir, "checkpoints/best.ckpt"), "wb"
                    ) as f:
                        dill.dump(
                            {
                                **val_metrics,
                                **learner_state,
                                "rng": rng,
                                "counter": counter,
                            },
                            f,
                            protocol=3,
                        )

        if (
            ((global_step % cfg.progress_bar_refresh_rate) == 0)
            or (global_step == (cfg.train.num_outer_steps - 1))
            or ((global_step % cfg.train.val_interval) == 0)
        ):
            inner_scalars = jax.tree_map(
                lambda x: jnp.mean(x, (0, 1)), scalars_ema["inner"]
            )
            outer_scalars = jax.tree_map(jnp.mean, scalars_ema["outer"])

            pbar.update()
            # counter += 1

            pbar.set_postfix(
                foa=outer_scalars["final"]["aux"][0]["acc"].item(),
                fol=outer_scalars["final"]["loss"].item(),
                ioa=outer_scalars["initial"]["aux"][0]["acc"].item(),
                iol=outer_scalars["initial"]["loss"].item(),
                iia=inner_scalars["auxs"][0]["acc"][0].item(),
                fia=inner_scalars["auxs"][0]["acc"][-1].item(),
                ilr=meta_learner._learner_state.inner_lr[0],
                olr=meta_learner._scheduler(global_step),
                refresh=False,
            )

            exp.log_metrics(
                dict(
                    foa=outer_scalars["final"]["aux"][0]["acc"].item(),
                    fol=outer_scalars["final"]["loss"].item(),
                    ioa=outer_scalars["initial"]["aux"][0]["acc"].item(),
                    iol=outer_scalars["initial"]["loss"].item(),
                    iia=inner_scalars["auxs"][0]["acc"][0].item(),
                    fia=inner_scalars["auxs"][0]["acc"][-1].item(),
                    #  ilr=meta_learner._learner_state.inner_lr[0],
                    olr=meta_learner._scheduler(global_step),
                ),
                step=global_step,
                prefix="train",
            )


if __name__ == "__main__":
    main(*parse_args())
