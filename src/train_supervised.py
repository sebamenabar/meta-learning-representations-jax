import argparse
from argparse import ArgumentParser
from tqdm import tqdm

import numpy as onp

import jax
from jax import random, numpy as jnp
from jax.random import split
import haiku as hk
from numpy.lib.histograms import _search_sorted_inclusive
import optax as ox

from lib import flatten, zero_initializer, evaluate_supervised_accuracy
from losses import mean_xe_and_acc_dict

from data import augment, TensorDataset
from data.omniglot import OmniglotDataset
from data.sampling import BatchSampler, shuffle_along_axis

from models.mrcl import OMLConvnet
from trainer.meta import CLWrapper
from trainer.supervised import ModelWrapper


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, choices=[28, 84], default=84)
    parser.add_argument(
        "--data_dir", type=str, default="/home/samenabar/storage/data/FSL/omniglot/"
    )
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--schedule", nargs="*", action=keyvalue, default={200: 0.1, 250: 0.1},
    )

    args = parser.parse_args()
    args.schedule = {int(k): float(v) for k, v in args.schedule.items()}

    return args


class keyvalue(argparse.Action):
    # Constructor calling
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())

        for value in values:
            # split it into key and value
            key, value = value.split("=")
            # assign into dictionary
            getattr(namespace, self.dest)[key] = value


def prepare_data(rng, data_dir, size):
    omniglot_train_dataset = OmniglotDataset("train", data_dir, size=size)

    print(omniglot_train_dataset.images.shape)

    train_dataset = TensorDataset(
        flatten(omniglot_train_dataset.images[:, :15], 1),
        flatten(omniglot_train_dataset.labels[:, :15], 1),
    )
    val_dataset = TensorDataset(
        flatten(omniglot_train_dataset.images[:, 15:], 1),
        flatten(omniglot_train_dataset.labels[:, 15:], 1),
    )

    normalize_fn = (
        lambda x: (x - omniglot_train_dataset.mean) / omniglot_train_dataset.std
    )

    train_sampler = BatchSampler(
        rng,
        train_dataset,
        64,
        collate_fn=lambda batch: tuple(map(onp.stack, zip(*batch))),
        keep_last=False,
    )
    val_sampler = BatchSampler(
        None,
        val_dataset,
        256,
        collate_fn=lambda batch: tuple(map(onp.stack, zip(*batch))),
        shuffle=False,
        keep_last=True,
    )

    omniglot_val_dataset = OmniglotDataset("val", data_dir, size=size,)
    shape = omniglot_val_dataset.images.shape

    # omniglot_val_dataset.images.shape

    total_classes = 300
    # rng_test = random.PRNGKey(1)
    # rng_classes, rng_shuffle = random.split(rng_test)

    sampled_classes = onp.arange(shape[0])
    # sampled_classes = random.choice(rng_classes, jnp.arange(omniglot_val_dataset.images.shape[0]), (total_classes,), replace=False)

    # shuffled_order = shuffle_along_axis(
    #     rng_shuffle, onp.arange(omniglot_val_dataset.images.shape[1])[None, :].repeat(total_classes, 0), 1
    # )
    # sampled_images = omniglot_val_dataset.images[sampled_classes[:, None].repeat(20, 1), shuffled_order]
    # sampled_labels = omniglot_val_dataset.labels[sampled_classes[:, None].repeat(20, 1), shuffled_order]

    sampled_images = omniglot_val_dataset.images
    sampled_labels = omniglot_val_dataset.labels
    test_train_images = sampled_images[:, :15]
    test_train_labels = sampled_labels[:, :15]
    test_test_images = sampled_images[:, 15:]
    test_test_labels = sampled_labels[:, 15:]

    test_train_dataset = TensorDataset(
        flatten(test_train_images, 1), flatten(test_train_labels, 1),
    )
    test_test_dataset = TensorDataset(
        flatten(test_test_images, 1), flatten(test_test_labels, 1),
    )
    test_train_sampler = BatchSampler(
        None,
        test_train_dataset,
        128,
        collate_fn=lambda batch: tuple(map(onp.stack, zip(*batch))),
        shuffle=False,
        keep_last=True,
    )
    test_test_sampler = BatchSampler(
        None,
        test_test_dataset,
        128,
        collate_fn=lambda batch: tuple(map(onp.stack, zip(*batch))),
        shuffle=False,
        keep_last=True,
    )

    return (
        train_sampler,
        val_sampler,
        normalize_fn,
        test_train_sampler,
        test_test_sampler,
    )


def main(args, cfg=None):
    print(args)

    rng, rng_data, rng_model = split(random.PRNGKey(args.seed), 3)

    (
        train_sampler,
        val_sampler,
        normalize_fn,
        test_train_sampler,
        test_test_sampler,
    ) = prepare_data(rng_data, args.data_dir, args.image_size)

    if args.image_size == 84:
        strides = (2, 1, 2, 1, 2, 2)
    else:
        strides = (1, 1, 1, 2, 1, 2)
    model = hk.transform_with_state(
        lambda x, is_training, phase="all": OMLConvnet(
            output_size=1000,
            spatial_dims=9,
            num_fc_layers=1,
            num_adaptation_fc=1,
            head_bias=True,
            normalization="none",
            strides=strides,
        )(x, is_training, phase)
    )
    dummy_input, _ = next(iter(train_sampler))
    print(dummy_input.shape)
    params, state = model.init(rng_model, normalize_fn(dummy_input / 255), True)

    trainer = ModelWrapper(model.apply, params, state,)

    optimizer = ox.chain(
        ox.additive_weight_decay(5e-4), ox.trace(decay=0.9, nesterov=False),
    )

    def schedule(step_num, updates, lr, sch_dict):
        return ox.scale(ox.piecewise_constant_schedule(lr, sch_dict)(step_num)).update(
            updates, None
        )[0]

    trainer.init_opt_state(optimizer).set_step_fn(
        jax.jit(
            trainer.make_step_fn(
                optimizer,
                jax.partial(schedule, lr=-args.lr, sch_dict=args.schedule),
                mean_xe_and_acc_dict,
                lambda rng, inputs: normalize_fn(augment(rng, inputs / 255, out_size=args.image_size)),
            )
        )
    )

    test_optimizer = ox.adam(0)

    def test_inner_opt_update_fn(lr, updates, state, params):
        inner_opt = ox.adam(lr)
        return inner_opt.update(updates, state, params)

    test_sup_model = CLWrapper(
        slow_apply=lambda *args, **kwargs: model.apply(
            *args, phase="encoder", **kwargs
        ),
        fast_apply=lambda *args, **kwargs: model.apply(
            *args, phase="adaptation", **kwargs
        ),
        slow_params=trainer.params,
        fast_params=jax.tree_map(jnp.zeros_like, trainer.params),
        slow_state=trainer.state,
        fast_state=trainer.state,
        training=False,
        loss_fn=mean_xe_and_acc_dict,
        test_init_inner_opt_state_fn=test_optimizer.init,
        test_inner_opt_update_fn=test_inner_opt_update_fn,
        preprocess_test_fn=jax.jit(lambda x: normalize_fn(x / 255)),
    )

    test_sup_model.test(
        test_train_sampler, test_test_sampler, inner_lr=3e-3,
    )

    train_pbar = tqdm(range(args.epochs), ncols=0)
    epoch_pbar = tqdm(total=len(train_sampler), ncols=0)

    loss_ema = aux_ema = None
    for epoch in train_pbar:
        epoch_pbar.reset()
        for i, (x, y) in enumerate(train_sampler):
            rng, rng_step = split(rng)
            loss, aux = trainer.train_step(epoch, rng_step, x, y)

            if loss_ema is None:
                loss_ema = loss
                aux_ema = aux
            else:
                (loss_ema, aux_ema) = jax.tree_multimap(
                    lambda ema, x: ema * 0.9 + x * 0.1,
                    (loss_ema, aux_ema),
                    (loss, aux),
                )

            if (i % 50) == 0:
                epoch_pbar.set_postfix(
                    loss=loss_ema.item(), acc=aux_ema["acc"].item(),
                )

            epoch_pbar.update()

        val_loss, val_acc = evaluate_supervised_accuracy(
            lambda x: trainer.jit_call_validate(normalize_fn(x / 255))[0], val_sampler,
        )
        print("\n")
        print("Sup validation stats:")
        print(val_loss, val_acc)

        print("CL validation stats:")

        test_sup_model.slow_params = trainer.params
        test_sup_model.slow_state = trainer.state
        for lr in [
            # 1e-2,
            # 3e-3,
            1e-3,
            3e-4,
            1e-4,
            5e-5,
            ]:
            (
                (test_train_loss, test_train_acc),
                (test_test_loss, test_test_acc),
            ) = test_sup_model.test(test_train_sampler, test_test_sampler, inner_lr=lr,)
            print(f"LR={lr}")
            print(f"Train loss: {test_train_loss}, Train acc: {test_train_acc}")
            print(f"Test loss: {test_test_loss}, Test acc: {test_test_acc}")
        print("\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
