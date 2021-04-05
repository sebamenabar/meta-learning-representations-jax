import os

import json
import dill

from tqdm import tqdm
from easydict import EasyDict as edict
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as onp

from comet_ml import Experiment

import jax
from jax import numpy as jnp
from jax.random import split, PRNGKey
import haiku as hk
from haiku.data_structures import merge, partition
import optax as ox
from acme.jax.utils import prefetch

# from models.utils import kaiming_normal
from models.oml import OMLConvnet
from meta.trainers import MetaTrainerB
from meta.testing import ContinualTesterB
from meta.wrappers import ContinualLearnerB

from datasets import get_dataset
from utils import (
    pmap_init,
    get_sharded_array_first,
    tree_flatten_array,
    resize_batch_dim,
    flatten_dims,
    is_sorted,
)
from utils.data import MetaDatasetArray
from utils.data.sampling import make_test_iterators
from utils.data.augmentation import crop_only

from dotenv import load_dotenv
import logging
from logging import handlers
from torch.utils.tensorboard import SummaryWriter

from utils.utils import tree_shape

load_dotenv()

logger = logging.getLogger("experiment")


MODELS = {
    "oml": OMLConvnet,
}


def get_trainset(
    rng,
    dataset_name,
    dataset_split,
    all,
    batch_size,
    way,
    shot,
    qry_shot,
    cl_qry_shot,
    image_size=None,
    disjoint_outer_set=False,
):
    trainset = get_dataset(
        dataset_name, dataset_split, all=all, train=True, image_size=image_size
    )
    unique_classes = sorted(onp.unique(trainset.targets))
    if disjoint_outer_set:
        outer_trainset = trainset.get_classes_subset(
            unique_classes[: len(unique_classes) // 2]
        )
        inner_trainset = trainset.get_classes_subset(
            unique_classes[len(unique_classes) // 2 :]
        )
    else:
        inner_trainset = outer_trainset = trainset

    meta_inner_trainset = MetaDatasetArray(
        inner_trainset,
        batch_size,
        way,
        shot,
        qry_shot,
        cl_qry_shot=0,
        disjoint=False,  # different trajeectories can repeat classes
    )
    meta_outer_trainset = MetaDatasetArray(
        outer_trainset,
        batch_size,
        0,
        0,
        0,
        cl_qry_shot=cl_qry_shot,
        disjoint=False,  # different trajeectories can repeat classes
    )

    rng_inner, rng_outer = split(rng)
    # return meta_inner_trainset, meta_outer_trainset

    inner_trainsampler = meta_inner_trainset.get_sampler(rng_inner)
    outer_trainsampler = meta_outer_trainset.get_sampler(rng_outer)

    def yielder():
        while True:
            x_spt, y_spt, x_qry, y_qry, *_ = next(inner_trainsampler)
            *_, x_qry_cl, y_qry_cl = next(outer_trainsampler)
            yield x_spt, y_spt, x_qry, y_qry, x_qry_cl, y_qry_cl

    return inner_trainset, outer_trainset, yielder(), trainset.normalize

    # return meta_inner_trainset.get_sampler(rng_inner), meta_outer_trainset.get_sampler(
    #     rng_outer
    # )


def get_test_iterators(
    rng,
    dataset_name,
    dataset_split,
    image_size=None,
    all=True,
    n=5,
    batch_size=256,
):
    testset = get_dataset(
        dataset_name, dataset_split, all=all, train=True, image_size=image_size
    )
    targets = onp.unique(testset.targets)
    testset = testset.get_classes_subset(targets, sort_by_class=True)
    traj_length = len(targets)
    test_trainloaders, test_testloaders = make_test_iterators(
        rng,
        testset,
        testset.targets,
        n,
        traj_length,
        sort=False,
        shuffle=True,
        batch_size=batch_size,
        dataset_is_array=True,
    )
    return testset, test_trainloaders, test_testloaders


def init_model(
    rng,
    model_name,
    image_size,
    cross_replica_axis,
    dummy_input,
    preprocess_fn=lambda x: x,
):
    _model = MODELS[model_name]
    model = hk.transform_with_state(
        lambda x, phase, training: _model(
            image_size=image_size,
            cross_replica_axis=cross_replica_axis,
        )(preprocess_fn(x), phase=phase, training=training)
    )
    dummy_input = tree_flatten_array(dummy_input / 255)
    if cross_replica_axis is not None:
        params, state = pmap_init(
            model,
            (rng,),
            dict(phase="all", training=True),
            jax.device_put_replicated(dummy_input, jax.local_devices()),
        )
    else:
        params, state = model.init(rng, dummy_input, "all", True)
    return model, _model, params, state


def init_loggers(args):
    args.exp_name = args.exp_name + "_" + str(args.seed)
    logdir = os.path.join(args.logdir, args.exp_name)
    if logdir:
        os.makedirs(logdir)
        with open(os.path.join(logdir, "args.json"), "w") as f:
            json.dump(vars(args), f)

    fh = logging.FileHandler(os.path.join(logdir, "log.txt"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        # logging.Formatter('rank:' + str(args['rank']) + ' ' + name + ' %(levelname)-8s %(message)s'))
        logging.Formatter("%(levelname)-8s %(message)s")
    )
    logger.addHandler(fh)

    ch = logging.handlers.logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(
        # logging.Formatter('rank:' + str(args['rank']) + ' ' + name + ' %(levelname)-8s %(message)s'))
        logging.Formatter("%(levelname)-8s %(message)s")
    )
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    logger.info(str(vars(args)))

    if args.logcomet:
        comet_api_key = os.environ["COMET_API_KEY"]
    else:
        comet_api_key = os.environ.get("COMET_API_KEY", default="")
    comet = Experiment(
        api_key=comet_api_key,
        workspace="the-thesis",
        project_name=args.comet_project,
        disabled=not args.logcomet,
    )
    comet.set_name(args.exp_name)
    comet.log_parameters(vars(args))

    writer = SummaryWriter(os.path.join(logdir, "tensorboard"))

    return logger, comet, writer, logdir


def reset_params(w_make_fn, w_get, rng, params, spt_classes=None):
    ws = w_get(params)
    leaves, layout = jax.tree_flatten(ws)
    rng = split(rng, len(leaves))
    if spt_classes is None:
        leaves = [
            w_make_fn(dtype=w.dtype)(_rng, w.shape)
            for _rng, w in zip(rng, leaves)
        ]
    else:
        leaves = [
            jax.ops.index_update(
                w,
                jax.ops.index[:, spt_classes],
                w_make_fn(dtype=w.dtype)(_rng, (w.shape[0], len(spt_classes))),
            )
            for _rng, w in zip(rng, leaves)
        ]

    return merge(params, jax.tree_unflatten(layout, leaves))


def main(args: edict):
    if jax.local_device_count() > 1:
        cross_replica_axis = "i"
        args.multi_device = True
    else:
        cross_replica_axis = None
        args.multi_device = False

    logger, comet, writer, logdir = init_loggers(args)

    logger.info(f"Using {jax.local_device_count()} devices")

    rng = PRNGKey(args.seed)
    rng, rng_trainset, rng_testset, rng_model = split(rng, 4)

    inner_trainset, outer_trainset, trainsampler, normalize = get_trainset(
        rng_trainset,
        args.dataset,
        args.train_split,
        args.all,
        args.batch_size,
        args.way,
        args.shot,
        args.qry_shot,
        args.cl_qry_shot,
        args.image_size,
        args.disjoint_outer_set,
    )
    testset, test_trainloaders, test_testloaders = get_test_iterators(
        rng_testset,
        args.dataset,
        args.test_split,
        args.image_size,
        args.num_test_trajs,
    )

    logger.info(f"Inner trainset length: {len(inner_trainset)}")
    logger.info(f"Outer trainset length: {len(outer_trainset)}")
    logger.info(f"Testset length: {len(testset)}")

    if args.prefetch > 0:
        trainsampler = prefetch(trainsampler, args.prefetch)
        logger.info(f"Prefetching {args.prefetch} samples")

    if args.normalize_input:
        preprocess_fn = normalize
        logger.info(f"Using input normalization")
    else:
        preprocess_fn = lambda x: x
        logger.info(f"No input normalization")

    model, model_class, params, state = init_model(
        rng_model,
        args.model_name,
        args.image_size,
        cross_replica_axis,
        next(trainsampler)[-2],
        preprocess_fn,
    )

    continual_learner = ContinualLearnerB(
        model.apply,
        params=None,
        state=None,
        slow_phase=model_class.inner_slow_train_phase,
        fast_phase=model_class.inner_fast_train_phase,
        get_slow_params=model_class.get_train_slow_params,
        get_fast_params=model_class.get_train_fast_params,
        get_slow_state=model_class.get_train_slow_state,
        get_fast_state=model_class.get_train_fast_state,
    )

    if args.reset_fast_params == "none":
        reset_fast_params = None
    elif args.reset_fast_params == "zero":
        zero_init = lambda dtype: lambda rng, shape: jax.nn.initializers.zeros(
            rng, dtype=dtype, shape=shape
        )
        reset_fast_params = jax.partial(
            reset_params, zero_init, model_class.get_classifier_w
        )
    elif args.reset_fast_params == "kaiming":
        # zero_init = lambda dtype: lambda rng, shape: jax.nn.initializers.zeros(rng, dtype=dtype, shape=shape)
        reset_fast_params = jax.partial(
            reset_params, jax.nn.initializers.he_normal, model_class.get_classifier_w
        )

    scheduler = None
    meta_trainer = MetaTrainerB(
        continual_learner,
        model.apply,
        params,
        state,
        model_class.outer_slow_train_phase,
        model_class.outer_fast_train_phase,
        model_class.get_train_slow_params,
        model_class.get_train_fast_params,
        model_class.get_train_slow_state,
        model_class.get_train_fast_state,
        inner_lr=args.inner_lr,
        train_lr=args.train_lr,
        optimizer=ox.adam(args.lr),
        scheduler=scheduler,
        cross_replica_axis=cross_replica_axis,
        reset_fast_params=reset_fast_params,
        reset_before_outer_loop=args.reset_before_outer_loop,
        include_spt=args.include_spt,
        augmentation=args.augment,
        augmentation_fn=jax.partial(crop_only, out_size=args.image_size),
    )

    meta_trainer.replicate_state().initialize_opt_state()

    tester = ContinualTesterB(
        model.apply,
        params=None,
        state=None,
        slow_phase=model_class.slow_test_phase,
        fast_phase=model_class.fast_test_phase,
        get_slow_params=model_class.get_test_slow_params,
        get_fast_params=model_class.get_test_fast_params,
        get_slow_state=model_class.get_test_slow_state,
        get_fast_state=model_class.get_test_fast_state,
    )

    histories = {
        "train": {
            "iil": [],
            "fil": [],
            "iia": [],
            "fia": [],
            "iol": [],
            "fol": [],
            "ioa": [],
            "foa": [],
            "lr": [],
            "ilr": [],
            "steps": [],
        },
        "train_special": {
            "inner_loss_progress": [],
            "inner_acc_progress": [],
        },
        "val": {
            "step": [],
            "zero": {
                "loss_train": {lr: [] for lr in args.lr_sweep},
                "acc_train": {lr: [] for lr in args.lr_sweep},
                "loss_test": {lr: [] for lr in args.lr_sweep},
                "acc_test": {lr: [] for lr in args.lr_sweep},
            },
            "random": {
                "loss_train": {lr: [] for lr in args.lr_sweep},
                "acc_train": {lr: [] for lr in args.lr_sweep},
                "loss_test": {lr: [] for lr in args.lr_sweep},
                "acc_test": {lr: [] for lr in args.lr_sweep},
            },
        },
    }
    loss_ema = aux_ema = None

    best_test_acc_all = 0
    train_pbar = tqdm(range(1, args.num_steps + 1), ncols=0, mininterval=2.5)
    for step_num in train_pbar:
        rng, rng_step = split(rng)
        loss, aux, inner_lr = train_step(
            step_num - 1,
            rng_step,
            trainsampler,
            meta_trainer,
        )
        if loss_ema is None:
            loss_ema, aux_ema = loss, aux
        else:
            loss_ema, aux_ema = jax.tree_multimap(
                lambda ema, x: ema * 0.9 + x * 0.1,
                (loss_ema, aux_ema),
                (loss, aux),
            )
        if (
            (((step_num) % 100) == 0)
            or ((step_num) == args.num_steps)
            or (step_num == 1)
            or ((step_num % args.test_interval) == 0)
        ):
            loss = fol = loss_ema.mean().item()

            iil = aux_ema["iil"].mean().item()
            fil = aux_ema["fil"].mean().item()
            iia = aux_ema["iia"].mean().item()
            fia = aux_ema["fia"].mean().item()

            iol = aux_ema["iol"].mean().item()
            # fol = aux_ema["outer"]["final"]["loss"].item()
            ioa = aux_ema["ioa"].mean().item()
            foa = aux_ema["foa"].mean().item()

            # print(aux_ema)
            # print(aux_ema["inner_progress"]["acc"].shape)
            # print(aux_ema["inner_progress"]["acc"])

            inner_loss_progress = (
                aux_ema["inner_progress"]["loss"].mean((0, 1, 3)).tolist()
            )
            inner_acc_progress = (
                aux_ema["inner_progress"]["acc"].mean((0, 2)).tolist()
            )

            inner_lr = meta_trainer.inner_lr
            if len(inner_lr.shape) > 1:
                inner_lr = inner_lr[0]
            inner_lr = inner_lr.item()

            train_pbar.set_postfix(
                loss=loss,
                iil=iil,
                fil=fil,
                iia=iia,
                fia=fia,
                iol=iol,
                ioa=ioa,
                foa=foa,
                # ilr=f"{inner_lr:.5f}",
                ilr=inner_lr,
            )

            histories["train"]["iil"].append(iil)
            histories["train"]["fil"].append(fil)
            histories["train"]["iia"].append(iia)
            histories["train"]["fia"].append(fia)
            histories["train"]["iol"].append(iol)
            histories["train"]["fol"].append(fol)
            histories["train"]["ioa"].append(ioa)
            histories["train"]["foa"].append(foa)

            histories["train_special"]["inner_loss_progress"].append(
                inner_loss_progress
            )
            histories["train_special"]["inner_acc_progress"].append(inner_acc_progress)

            histories["train"]["steps"].append(step_num)
            if scheduler is not None:
                pass
                # step_lr = get_step_lr(
                #     step_num - 1,
                #     lr=args.lr,
                #     sch_dict=sch_dict,
                # ).item()
            else:
                step_lr = args.lr
            histories["train"]["lr"].append(step_lr)
            histories["train"]["ilr"].append(inner_lr)

            train_metrics = {
                "inner_lr": inner_lr,
                "lr": step_lr,
                "loss": loss,
                "iil": iil,
                "fil": fil,
                "iia": iia,
                "fia": fia,
                "iol": iol,
                "ioa": ioa,
                "foa": foa,
            }

            for tag, metric in train_metrics.items():
                writer.add_scalar(
                    "train/" + tag,
                    metric,
                    global_step=step_num,
                )

            comet.log_metrics(
                train_metrics,
                prefix="train",
                step=step_num,
            )

        if (
            ((step_num % args.test_interval) == 0)
            or (step_num == 1)
            or (step_num == args.num_steps)
        ):
            rng, rng_test = split(rng)
            histories["val"]["step"].append(step_num)
            logger.info("\nTrain metrics")
            logger.info(str(train_metrics))
            logger.info(f"TEST STEP {step_num}")

            params, state, *_ = meta_trainer.get_status_first()
            cls_params = model_class.get_classifier_params(params)

            zero_test_results = test_loop(
                hk.data_structures.merge(params, reset_classifier_zero(cls_params)),
                state,
                tester,
                test_trainloaders,
                test_testloaders,
                args.lr_sweep,
            )

            random_test_results = test_loop(
                hk.data_structures.merge(params, reset_classifier_random(rng_test, cls_params)),
                state,
                tester,
                test_trainloaders,
                test_testloaders,
                args.lr_sweep,
            )

            logger.info(f"RESULTS ZERO")
            for lr in args.lr_sweep:
                histories["val"]["zero"]["loss_train"][lr].append(
                    zero_test_results[0][lr]
                )
                histories["val"]["zero"]["acc_train"][lr].append(
                    zero_test_results[1][lr]
                )
                histories["val"]["zero"]["loss_test"][lr].append(
                    zero_test_results[2][lr]
                )
                histories["val"]["zero"]["acc_test"][lr].append(
                    zero_test_results[3][lr]
                )

                logger.info(f"Test lr: {lr}")
                logger.info("Test train loss: %f" % (zero_test_results[0][lr]))
                logger.info("Test train acc: %f" % (zero_test_results[1][lr]))
                logger.info("Test test loss: %f" % (zero_test_results[2][lr]))
                logger.info("Test test acc: %f" % (zero_test_results[3][lr]))
                
            logger.info(f"RESULTS RANDOM")
            for lr in args.lr_sweep:
                histories["val"]["random"]["loss_train"][lr].append(
                    random_test_results[0][lr]
                )
                histories["val"]["random"]["acc_train"][lr].append(
                    random_test_results[1][lr]
                )
                histories["val"]["random"]["loss_test"][lr].append(
                    random_test_results[2][lr]
                )
                histories["val"]["random"]["acc_test"][lr].append(
                    random_test_results[3][lr]
                )
                logger.info(f"Test lr: {lr}")
                logger.info("Test train loss: %f" % (random_test_results[0][lr]))
                logger.info("Test train acc: %f" % (random_test_results[1][lr]))
                logger.info("Test test loss: %f" % (random_test_results[2][lr]))
                logger.info("Test test acc: %f" % (random_test_results[3][lr]))

            best_lr_zero, best_test_acc_zero = max(
                zero_test_results[3].items(), key=lambda v: v[1]
            )
            best_train_acc_zero = zero_test_results[1][best_lr_zero]
            best_lr_random, best_test_acc_random = max(
                random_test_results[3].items(), key=lambda v: v[1]
            )
            best_train_acc_random = random_test_results[1][best_lr_random]
            # val_acc = onp.mean(test_accs[best_lr])

            logger.info(
                f"Best test_acc zero: {best_test_acc_zero:.2f}, lr {best_lr_zero}, train_acc: {best_train_acc_zero:.2f}"
            )
            logger.info(
                f"Best test_acc random: {best_test_acc_random:.2f}, lr {best_lr_random}, train_acc: {best_train_acc_random:.2f}"
            )

            writer.add_scalar(
                "test/best_zero_acc_test", best_test_acc_zero, global_step=step_num
            )
            writer.add_scalar(
                "test/best_zero_acc_train", best_train_acc_zero, global_step=step_num
            )
            writer.add_scalar("test/best_zero_lr", best_lr_zero, global_step=step_num)

            writer.add_scalar(
                "test/best_random_acc_test", best_test_acc_random, global_step=step_num
            )
            writer.add_scalar(
                "test/best_random_acc_train",
                best_train_acc_random,
                global_step=step_num,
            )
            writer.add_scalar(
                "test/best_random_lr", best_lr_random, global_step=step_num
            )

            best_lr, best_test_acc, best_train_acc = max(
                [
                    (best_lr_zero, best_test_acc_zero, best_train_acc_zero),
                    (best_lr_random, best_test_acc_random, best_train_acc_random),
                ],
                key=lambda x: x[1],
            )

            if best_test_acc > best_test_acc_all:
                best_test_acc_all = best_test_acc
                logger.info(
                    f"\nNEW BEST TEST ACC: {best_test_acc_all}, lr: {best_lr}, train acc: {best_train_acc}\n"
                )
                dump(logdir, "best", histories, meta_trainer)

            writer.add_scalar(
                "test/best_acc_train", best_train_acc, global_step=step_num
            )
            writer.add_scalar("test/best_acc_test", best_test_acc, global_step=step_num)
            writer.add_scalar("test/best_lr", best_lr, global_step=step_num)

            comet.log_metrics(
                {
                    "best_zero_train_acc": best_train_acc_zero,
                    "best_zero_test_acc": best_test_acc_zero,
                    "best_zero_lr": best_lr_zero,
                    "best_random_train_acc": best_train_acc_random,
                    "best_random_test_acc": best_test_acc_random,
                    "best_random_lr": best_lr_random,
                    "best_train_acc": best_train_acc,
                    "best_test_acc": best_test_acc,
                    "best_lr": best_lr,
                },
                prefix="test",
                step=step_num,
            )

            for sth in ["zero", "random"]:
                writer.add_scalars(
                    "test/loss_train/" + sth,
                    {
                        str(lr): onp.mean(v[-1])
                        for lr, v in histories["val"][sth]["loss_train"].items()
                    },
                    global_step=step_num,
                )
                writer.add_scalars(
                    "test/loss_test/" + sth,
                    {
                        str(lr): onp.mean(v[-1])
                        for lr, v in histories["val"][sth]["loss_test"].items()
                    },
                    global_step=step_num,
                )
                writer.add_scalars(
                    "test/acc_train/" + sth,
                    {
                        str(lr): onp.mean(v[-1])
                        for lr, v in histories["val"][sth]["acc_train"].items()
                    },
                    global_step=step_num,
                )
                writer.add_scalars(
                    "test/acc_test/" + sth,
                    {
                        str(lr): onp.mean(v[-1])
                        for lr, v in histories["val"][sth]["acc_test"].items()
                    },
                    global_step=step_num,
                )

                test_metrics = {}
                test_metrics.update(
                    {
                        f"loss_train_{lr}_{sth}": onp.mean(v[-1])
                        for lr, v in histories["val"][sth]["loss_train"].items()
                    }
                )
                test_metrics.update(
                    {
                        f"loss_test_{lr}_{sth}": onp.mean(v[-1])
                        for lr, v in histories["val"][sth]["loss_test"].items()
                    }
                )
                test_metrics.update(
                    {
                        f"acc_train_{lr}_{sth}": onp.mean(v[-1])
                        for lr, v in histories["val"][sth]["acc_train"].items()
                    }
                )
                test_metrics.update(
                    {
                        f"acc_test_{lr}_{sth}": onp.mean(v[-1])
                        for lr, v in histories["val"][sth]["acc_test"].items()
                    }
                )

                comet.log_metrics(
                    test_metrics,
                    prefix="test",
                    step=step_num,
                )

        if (step_num % 5000) == 0:
            dump(logdir, "", histories, meta_trainer)
        if (step_num % 20000) == 0:
            dump(logdir, "20k", histories, meta_trainer)
        if (step_num % 200000) == 0:
            dump(logdir, "200k", histories, meta_trainer)
        if (step_num % 500000) == 0:
            dump(logdir, "500k", histories, meta_trainer)
        if (step_num % 700000) == 0:
            dump(logdir, "700k", histories, meta_trainer)

    dump(logdir, "", histories, meta_trainer)


def reset_classifier_zero(params):
    return jax.tree_map(jnp.zeros_like, params)


def reset_classifier_random(rng, params):
    w, b = partition(lambda module_name, name, value: name == "w", params)
    return merge(
        jax.tree_map(lambda x: jax.nn.initializers.he_normal(dtype=x.dtype)(rng, x.shape), w),
        jax.tree_map(jnp.zeros_like, b),
    )


def train_step(i, rng, loader, trainer, cross_replica_axis=None):
    # rng, rng_step = split(rng)
    x_spt, y_spt, x_qry, y_qry, x_qry_cl, y_qry_cl = next(loader)

    x_spt, y_spt, x_qry, y_qry = flatten_dims((x_spt, y_spt, x_qry, y_qry))
    # x_qry_cl, y_qry_cl = ((x_qry_cl, y_qry_cl))
    if cross_replica_axis is not None:
        x_spt, y_spt, x_qry, y_qry = resize_batch_dim((x_spt, y_spt, x_qry, y_spt))
        x_qry_cl, y_qry_cl = resize_batch_dim((x_qry_cl, y_qry_cl))

    loss, aux, inner_lr = trainer.step(
        rng,
        i,
        x_spt,
        y_spt,
        x_qry,
        y_qry,
        x_qry_cl,
        y_qry_cl,
    )
    return loss, aux, inner_lr


def test_loop(params, state, tester, train_loaders, test_loaders, lr_sweep):
    train_accs = {}
    test_accs = {}
    train_losses = {}
    test_losses = {}

    for inner_lr in lr_sweep:
        train_accs[inner_lr] = []
        test_accs[inner_lr] = []
        train_losses[inner_lr] = []
        test_losses[inner_lr] = []

        for test_train_iterator, test_test_iterator in zip(train_loaders, test_loaders):
            (
                test_train_acc,
                test_train_loss,
                test_test_acc,
                test_test_loss,
            ) = tester.test(
                test_train_iterator,
                test_test_iterator,
                params,
                state,
                inner_lr,
            )
            # print(test_train_loss, test_train_acc, test_test_loss, test_test_acc)
            train_accs[inner_lr].append(test_train_acc.item())
            test_accs[inner_lr].append(test_test_acc.item())
            train_losses[inner_lr].append(test_train_loss.item())
            test_losses[inner_lr].append(test_test_loss.item())
        train_accs[inner_lr] = onp.mean(train_accs[inner_lr])
        test_accs[inner_lr] = onp.mean(test_accs[inner_lr])
        train_losses[inner_lr] = onp.mean(train_losses[inner_lr])
        test_losses[inner_lr] = onp.mean(test_losses[inner_lr])

    return train_losses, train_accs, test_losses, test_accs

    # return jax.tree_map(
    #     lambda x: onp.mean(x).item(), (train_losses, train_accs, test_losses, test_accs)
    # )


def dump(logdir, postfix, histories, trainer, best_val_acc=None):
    with open(os.path.join(logdir, f"histories_{postfix}.json"), "w") as f:
        json.dump(histories, f)

    params, state, fast_state, opt_state, inner_lr = trainer.get_status_first()
    with open(os.path.join(logdir, f"model_{postfix}.dill"), "wb") as f:
        dill.dump(
            {
                "histories": histories,
                "params": params,
                "state": state,
                "fast_state": fast_state,
                "opt_state": opt_state,
                "inner_lr": inner_lr,
                # "best_val_acc": best_val_acc,
            },
            f,
        )


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--logcomet", default=False, action="store_true")
    parser.add_argument("--comet_project", default="continual_learning3", type=str)

    parser.add_argument("--prefetch", type=int, default=0)
    parser.add_argument("--num_steps", type=int, required=True)
    parser.add_argument("--test_interval", type=int, required=True)

    parser.add_argument(
        "--augment", choices=["none", "spt", "qry", "all"], required=True
    )
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--way", type=int, required=True)
    parser.add_argument("--shot", type=int, required=True)
    parser.add_argument("--qry_shot", type=int, required=True)
    parser.add_argument("--cl_qry_shot", type=int, required=True)
    parser.add_argument("--disjoint_outer_set", default=False, action="store_true")
    parser.add_argument("--include_spt", default=False, action="store_true")

    parser.add_argument("--dataset", default="omniglot", choices=["omniglot"])
    parser.add_argument("--image_size", choices=[28, 84], type=int, required=True)
    parser.add_argument("--all", choices=[0, 1], required=True, type=int)
    parser.add_argument(
        "--train_split", default="train", choices=["train", "train+val"]
    )
    parser.add_argument("--test_split", default="val", choices=["val", "test"])
    parser.add_argument("--num_test_trajs", default=5, type=int)

    parser.add_argument("--model_name", choices=["oml"], required=True)
    parser.add_argument("--normalize_input", default=False, action="store_true")
    parser.add_argument("--inner_lr", type=float, required=True)
    parser.add_argument("--train_lr", default=False, action="store_true")

    parser.add_argument(
        "--reset_fast_params", choices=["none", "zero", "kaiming"], required=True
    )
    parser.add_argument(
        "--reset_before_outer_loop", type=int, choices=[0, 1], required=True
    )

    parser.add_argument("--sorted_test", type=int, default=0, choices=[0, 1])
    parser.add_argument("--shuffle_test", type=int, default=1, choices=[0, 1])
    parser.add_argument(
        "--lr_sweep",
        type=float,
        nargs="+",
        default=[
            # 0.01,
            0.0085,
            0.005,
            0.003,
            0.001,
            0.00085,
            0.0005,
            0.0003,
            0.0001,
            # 0.000085,
            # 0.00005,
            # 0.00003,
        ],
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(edict(vars(parse_args())))