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
from eval_experiment import LRTester, MAMLTester
from experiment import Experiment, Logger


def parse_args(parser=None):
    parser = Experiment.add_args(parser)

    parser.add_argument("--worker_tpu_driver", default="")

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
        "--val.batch_size", help="Number of FSL tasks", default=4, type=int
    )
    # parser.add_argument(
    #     "--val.fsl.qry_shot",
    #     type=int,
    #     help="Number of quried samples per class",
    #     default=15,
    # )
    parser.add_argument("--val.num_inner_steps", type=int, default=5)
    parser.add_argument("--val.num_tasks", type=int, default=500)

    parser.add_argument(
        "--model.model_name", default="convnet4", choices=["resnet12", "convnet4"]
    )
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

    parser.add_argument("--fake_pmap_jit", action="store_true", default=False)

    args = parser.parse_args()
    cfg = edict(train=edict(cl=edict()), val=edict(fsl=edict()), model=edict())
    for argname, argval in vars(args).items():
        rsetattr(cfg, argname, argval)

    return args, cfg


def main(args, cfg):
    if cfg.worker_tpu_driver:
        jax.config.update("jax_xla_backend", "tpu_driver")
        jax.config.update("jax_backend_target", cfg.worker_tpu_driver)
        print("Backend: %s %r", FLAGS.worker_tpu_driver, jax.devices())

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

    val_dataset = MetaMiniImageNet(
        jax.random.PRNGKey(0),
        "val",
        cfg.data_dir,
        cfg.val.batch_size,
        5,
        5,
        15,
        # cfg.train.way,
        # cfg.train.shot,
        # cfg.train.qry_shot,
    )
    val_dataset2 = MetaMiniImageNet(
        jax.random.PRNGKey(0),
        "val",
        cfg.data_dir,
        # cfg.val.batch_size,
        5,  # For augmented testing
        5,
        5,
        15,
        # cfg.train.way,
        # cfg.train.shot,
        # cfg.train.qry_shot,
    )

    lr_tester_no_aug = LRTester(
        meta_learner._encoder.apply,
        cfg.val.num_tasks,
        # cfg.val.batch_size,
        # 1,
        val_dataset,
        0,
        val_dataset._normalize,
    )
    # lr_tester_aug = LRTester(
    #     meta_learner._encoder.apply,
    #     cfg.val.num_tasks,
    #     # cfg.val.batch_size,
    #     val_dataset2,
    #     5,
    #     val_dataset2._normalize,
    #     keep_orig_aug=False,
    # )
    maml_tester_no_aug = MAMLTester(
        meta_learner._encoder.apply,
        meta_learner._classifier.apply,
        cfg.val.num_tasks,
        #  cfg.val.batch_size,
        val_dataset,
        cfg.val.num_inner_steps,
        0,
        val_dataset._normalize,
    )
    maml_tester_no_aug_10_steps = MAMLTester(
        meta_learner._encoder.apply,
        meta_learner._classifier.apply,
        cfg.val.num_tasks,
        #  cfg.val.batch_size,
        val_dataset,
        # cfg.val.num_inner_steps,
        10,
        0,
        val_dataset._normalize,
    )
    maml_tester_no_aug_20_steps = MAMLTester(
        meta_learner._encoder.apply,
        meta_learner._classifier.apply,
        cfg.val.num_tasks,
        #  cfg.val.batch_size,
        val_dataset,
        # cfg.val.num_inner_steps,
        20,
        0,
        val_dataset._normalize,
    )
    # maml_tester_aug = MAMLTester(
    #     meta_learner._encoder.apply,
    #     meta_learner._classifier.apply,
    #     cfg.val.num_tasks,
    #     # cfg.val.batch_size,
    #     # 1,
    #     val_dataset,
    #     cfg.val.num_inner_steps,
    #     5,
    #     val_dataset._normalize,
    #     keep_orig_aug=False,
    # )

    scalars_ema = None
    rng = rng_step = jax.random.PRNGKey(0)  # Get's replaced in first step with seed
    counter = 0
    pbar = tqdm(range(cfg.train.num_outer_steps), ncols=0)
    # for i in range(1, cfg.train.num_outer_steps * meta_learner._apply_every + 1):
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

        # if (i % meta_learner._apply_every) == 0:

        if (
            (global_step == 0)
            or (
                (((global_step + 1) % cfg.train.val_interval) == 0)
                and (global_step != 1)
            )
            or (global_step == (cfg.train.num_outer_steps - 1))
        ):
            learner_state = meta_learner.get_first_state()

            exp.log()
            exp.log("Evaluation Logistic Regression No-Aug")
            lr_no_aug_acc, lr_no_aug_std = lr_tester_no_aug.eval(
                learner_state.slow_params, learner_state.slow_state
            )
            # exp.log("Evaluation Logistic Regression Aug")
            # lr_aug_acc, lr_aug_std = lr_tester_aug.eval(
            #     learner_state.slow_params, learner_state.slow_state
            # )

            zero_fast_params = jax.tree_map(jnp.zeros_like, learner_state.fast_params)
            exp.log("Evaluation MAML No-Aug")
            maml_acc_no_aug, maml_std_no_aug = maml_tester_no_aug.eval(
                learner_state.slow_params,
                # learner_state.fast_params,
                zero_fast_params,
                learner_state.slow_state,
                learner_state.fast_state,
                learner_state.inner_lr,
            )
            (
                maml_acc_no_aug_10_steps,
                maml_std_no_aug_10_steps,
            ) = maml_tester_no_aug_10_steps.eval(
                learner_state.slow_params,
                # learner_state.fast_params,
                zero_fast_params,
                learner_state.slow_state,
                learner_state.fast_state,
                learner_state.inner_lr,
            )
            (
                maml_acc_no_aug_20_steps,
                maml_std_no_aug_20_steps,
            ) = maml_tester_no_aug_20_steps.eval(
                learner_state.slow_params,
                # learner_state.fast_params,
                zero_fast_params,
                learner_state.slow_state,
                learner_state.fast_state,
                learner_state.inner_lr,
            )
            # exp.log("Evaluation MAML Aug")
            # maml_acc_aug, maml_std_aug = maml_tester_aug.eval(
            #     learner_state.slow_params,
            #     learner_state.fast_params,
            #     jax.tree_map(
            #         jax.partial(replicate_array, num_devices=cfg.val.batch_size),
            #         learner_state.slow_state,
            #     ),
            #     jax.tree_map(
            #         jax.partial(replicate_array, num_devices=cfg.val.batch_size),
            #         learner_state.fast_state,
            #     ),
            #     learner_state.inner_lr,
            # )

            exp.log(f"\nStep {global_step + 1} statistics:")
            exp.log(
                f"MAML {cfg.val.num_inner_steps}-steps No-Aug Acc: {maml_acc_no_aug}±{maml_std_no_aug}"
            )
            exp.log(
                f"MAML 10-steps No-Aug Acc: {maml_acc_no_aug_10_steps}±{maml_std_no_aug_10_steps}"
            )
            exp.log(
                f"MAML 20-steps No-Aug Acc: {maml_acc_no_aug_10_steps}±{maml_std_no_aug_10_steps}"
            )
            # exp.log(f"MAML Aug Acc: {maml_acc_aug}±{maml_std_aug}")
            exp.log(f"Logistic Regression No-Aug Acc: {lr_no_aug_acc}±{lr_no_aug_std}")
            # exp.log(f"Logistic Regression Aug Acc: {lr_aug_acc}±{lr_aug_std}")

            exp.log_metrics(
                dict(
                    lr_no_aug_acc=lr_no_aug_acc,
                    lr_no_aug_std=lr_no_aug_std,
                    # lr_aug_acc=lr_aug_acc,
                    # lr_aug_std=lr_aug_std,
                    maml_acc_no_aug=maml_acc_no_aug,
                    maml_std_no_aug=maml_std_no_aug,
                    maml_acc_no_aug_10_steps=maml_acc_no_aug_10_steps,
                    maml_std_no_aug_10_steps=maml_std_no_aug_10_steps,
                    maml_acc_no_aug_20_steps=maml_acc_no_aug_20_steps,
                    maml_std_no_aug_20_steps=maml_std_no_aug_20_steps,
                    # maml_acc_aug=maml_acc_aug,
                    # maml_std_aug=maml_std_aug,
                ),
                step=global_step,
                prefix="val",
            )
            exp.log()

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


if __name__ == "__main__":
    main(*parse_args())
