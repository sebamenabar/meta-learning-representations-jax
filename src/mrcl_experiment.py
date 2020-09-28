import os.path as osp

# from typing import NamedTuple
from collections import namedtuple
from absl import logging

import numpy as onp

import jax
from jax.random import split
from jax import random, numpy as jnp

import optax as ox
import chex

from lib import (
    delayed_cosine_decay_schedule,
    fsl_inner_loop,
    outer_loop,
    batched_outer_loop,
    meta_step,
    mean_xe_and_acc_dict,
    reset_all,
    reset_by_idxs,
)
from models import prepare_model, make_params
from data import prepare_data, preprocess_images, augment
from data.sampling import fsl_sample, fsl_build


def bcast_local_devices(value):
    """Broadcasts an object to all local devices."""
    devices = jax.local_devices()

    def _replicate(x):
        """Replicate an object on each device."""
        x = jnp.array(x)
        return jax.api.device_put_sharded(len(devices) * [x], devices)

    return jax.tree_util.tree_map(_replicate, value)


def get_first(xs):
    """Gets values from the first device."""
    return jax.tree_map(lambda x: x[0], xs)


def replicate_array(x, num_devices):
    return jnp.broadcast_to(x, (num_devices,) + x.shape)


def get_host_rng(rng):
    return split(rng, jax.host_count())[jax.host_id()]


MetaLearnerState = namedtuple(
    "MetaLearnerState",
    ["slow_params", "fast_params", "slow_state", "fast_state", "inner_lr", "opt_state"],
)


def reshape_inputs(inputs):
    rets = []
    leading_dim = jax.local_device_count()
    for x in inputs:
        per_device_batch_size, ragged = divmod(x.shape[0], leading_dim)
        assert ragged == 0
        rets.append(x.reshape(leading_dim, per_device_batch_size, *x.shape[1:]))
    return tuple(rets)


class MetaLearner:
    def __init__(
        self,
        random_seed,
        dataset_name,
        data_root,
        model_cfg,
        train_cfg,
        # data_cfg,
        # sub_batch_size=None,
    ):
        self._dataset = dataset_name
        self._data_root = data_root
        # self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self._random_seed = random_seed

        self._encoder, self._classifier = prepare_model(dataset_name, **model_cfg)

        self._prefetch = train_cfg.prefetch  # Batch prefetching
        self._train_input = None  # Infinite iterator that generates batches
        self._learner_state = None

        # Sub batch size is used for gradient accumulation
        if train_cfg.sub_batch_size is None:
            self._sub_batch_size = train_cfg.batch_size
            self._apply_every = 1
        else:
            self._sub_batch_size = train_cfg.sub_batch_size
            self._apply_every, ragged = divmod(
                train_cfg.batch_size, train_cfg.sub_batch_size
            )
            assert ragged == 0

        num_devices = jax.device_count()
        global_batch_size = self._sub_batch_size
        self.per_device_batch_size, ragged = divmod(global_batch_size, num_devices)
        assert ragged == 0

        self._normalize_fn = None

        # with chex.fake_pmap_and_jit():
        self.reset_classifier = self._reset_classifier()

        self._scheduler = self._make_scheduler()
        self._optimizer = ox.chain(
            ox.clip(10),
            ox.scale(1 / self._apply_every),
            ox.apply_every(self._apply_every),
            #  ox.additive_weight_decay(self.train_cfg.weight_decay),
            ox.scale_by_adam(),
            ox.scale_by_schedule(self._scheduler),
        )

        self.update_pmap = jax.pmap(jax.partial(self._update_fn), axis_name="i")

    # def normalize_fn(self, *args, **kwargs):
    #     return self._normalize_fn(*args, **kwargs)

    def _reset_classifier(self):
        if self.train_cfg.reset_head == "none":
            print("No classifier reset")
            return None

        if "zero" in self.train_cfg.reset_head:
            print("Reset classifier to zero")
            initializer = lambda dtype: lambda rng, shape: jax.nn.initializers.zeros(
                rng,
                shape,
                dtype=dtype,
            )
        elif "glorot" in self.train_cfg.reset_head:
            print("Reset classifier to glorot")
            initializer = jax.nn.initializers.glorot_uniform
        elif "kaiming" in self.train_cfg.reset_head:
            print("Reset classifier to kaiming")
            initializer = jax.nn.initializers.he_normal

        if "all" in self.train_cfg.reset_head:
            print("Reset all classifier")
            return jax.partial(reset_all, initializer)
        elif "cls" in self.train_cfg.reset_head:
            print("Reset cls classifier")
            return jax.partial(reset_by_idxs, initializer)

    @staticmethod
    def inner_opt_update_fn(lr, updates, state, params):
        inner_opt = ox.sgd(lr)
        return inner_opt.update(updates, state, params)

    # @staticmethod
    def _update_fn(
        self,
        learner_state,
        global_step,
        rng,
        inputs,
        spt_classes,
        # normalize_fn,
        # augment,
        # augment_fn,
        # num_inner_steps,
        # slow_apply,
        # fast_apply,
        # opt_update_fn,
        # reset_fast_params_fn,
        # learn_inner_lr,
        # optimizer,
    ):
        x_spt, y_spt, x_qry, y_qry = inputs

        rng, rng_step, rng_aug = split(rng, 3)

        x_spt, x_qry = preprocess_images(
            rng_aug,
            x_spt,
            x_qry,
            self._normalize_fn,
            # normalize_fn,
            augment=self.train_cfg.augment,
            # augment=augment,
            # augment_fn=augment_fn,
            augment_fn=augment,
        )

        _inner_loop = jax.partial(
            fsl_inner_loop,
            is_training=True,
            num_steps=self.train_cfg.num_inner_steps,
            # num_steps=num_inner_steps,
            slow_apply=self._encoder.apply,
            # slow_apply=slow_apply,
            fast_apply=self._classifier.apply,
            # fast_apply=fast_apply,
            loss_fn=mean_xe_and_acc_dict,
            opt_update_fn=self.inner_opt_update_fn,
            # opt_update_fn=opt_update_fn,
        )
        _outer_loop = jax.partial(
            outer_loop,
            is_training=True,
            inner_loop=_inner_loop,
            slow_apply=self._encoder.apply,
            # slow_apply=slow_apply,
            fast_apply=self._classifier.apply,
            # fast_apply=fast_apply,
            loss_fn=mean_xe_and_acc_dict,
            reset_fast_params_fn=self.reset_classifier,
            # reset_fast_params_fn=reset_fast_params_fn,
        )
        _batched_outer_loop = jax.partial(batched_outer_loop, outer_loop=_outer_loop)

        inner_opt_state = ox.sgd(0).init(learner_state.fast_params)

        # if learn_inner_lr:
        if self.train_cfg.learn_inner_lr:
            positions = (0, 1, 2)
        else:
            positions = (0, 1)
        (outer_loss, (slow_state, fast_state, info)), grads = jax.value_and_grad(
            _batched_outer_loop, positions, has_aux=True
        )(
            learner_state.slow_params,
            learner_state.fast_params,
            learner_state.inner_lr,
            learner_state.slow_state,
            learner_state.fast_state,
            inner_opt_state,
            split(rng, x_spt.shape[0]),
            x_spt,
            y_spt,
            x_qry,
            y_qry,
            spt_classes,
        )

        grads = jax.tree_map(lambda v: jax.lax.pmean(v, axis_name="i"), grads)
        # if learn_inner_lr:
        if self.train_cfg.learn_inner_lr:
            updates, opt_state = self._optimizer.update(
                grads,
                learner_state.opt_state,
                (
                    learner_state.slow_params,
                    learner_state.fast_params,
                    learner_state.inner_lr,
                ),
            )
            slow_params, fast_params, inner_lr = ox.apply_updates(
                (
                    learner_state.slow_params,
                    learner_state.fast_params,
                    learner_state.inner_lr,
                ),
                updates,
            )
        else:
            updates, opt_state = self._optimizer.update(
                grads,
                learner_state.opt_state,
                (
                    learner_state.slow_params,
                    learner_state.fast_params,
                ),
            )
        slow_params, fast_params = ox.apply_updates(
            (
                learner_state.slow_params,
                learner_state.fast_params,
            ),
            updates,
        )
        inner_lr = learner_state.inner_lr

        out = (
            MetaLearnerState(
                slow_params=slow_params,
                fast_params=fast_params,
                slow_state=slow_state,
                fast_state=fast_state,
                inner_lr=inner_lr,
                opt_state=opt_state,
            ),
            info,
        )

        return out

    def step(self, *, global_step, rng):
        if self._train_input is None or self._learner_state is None or rng is None:
            rng = self._initialize_train()

        rng, rng_step = split(rng, 2)

        host_id = jax.host_id()
        local_device_count = jax.local_device_count()

        step_rng_device = jax.random.split(rng_step, num=jax.device_count())
        step_rng_device = step_rng_device[
            host_id * local_device_count : (host_id + 1) * local_device_count
        ]
        step_device = onp.broadcast_to(global_step, [local_device_count])

        inputs = next(self._train_input)
        spt_classes = onp.unique(inputs[1], axis=1)
        inputs = reshape_inputs(inputs)
        (spt_classes,) = reshape_inputs([spt_classes])

        self._learner_state, scalars = self.update_pmap(
            self._learner_state,
            step_device,
            step_rng_device,
            inputs,
            spt_classes,
            # normalize_fn=self._normalize_fn,
        )

        return rng, scalars

    def _initialize_train(self):
        rng = jax.random.PRNGKey(self._random_seed)
        rng_init, rng_data, rng_train = split(rng, 3)

        if self._train_input is None:
            _train_input = self._build_train_input(rng_data)
            if self._prefetch > 0:
                from acme.jax import utils as acme_utils

                self._train_input = acme_utils.prefetch(_train_input)
            else:
                self._train_input = _train_input
        if self._learner_state is None:
            print("Initializing parameters rather than restoring from checkpoint.")

            x_spt, *_ = reshape_inputs(next(self._train_input))
            init_fn = jax.pmap(self._make_initial_state, axis_name="i")

            rng_init = replicate_array(rng_init, jax.local_device_count())

            self._learner_state = init_fn(rng_init, x_spt[:, 0])

        # update_fn_kwargs = dict(
        #     normalize_fn=self._normalize_fn,
        #     augment=self.train_cfg.augment,
        #     augment_fn=augment,
        #     num_inner_steps=self.train_cfg.num_inner_steps,
        #     slow_apply=self._encoder.apply,
        #     fast_apply=self._classifier.apply,
        #     opt_update_fn=self.inner_opt_update_fn,
        #     reset_fast_params_fn=self.reset_classifier,
        #     learn_inner_lr=self.train_cfg.learn_inner_lr,
        #     optimizer=self._optimizer,
        # )
        # self.update_pmap = jax.pmap(
        #     jax.partial(self._update_fn, **update_fn_kwargs), axis_name="i"
        # )

        return get_host_rng(rng_train)

    def _build_train_input(self, rng):
        rng = get_host_rng(rng)
        logging.info(f"Host {jax.host_id()} random key: {rng}")

        if self._dataset == "miniimagenet":
            if self.train_cfg.method == "maml":
                logging.info("Preparing MAML MiniImageNet")
                dataset = MetaMiniImageNet(
                    rng,
                    "train",
                    self._data_root,
                    self.per_device_batch_size * jax.local_device_count(),
                    self.train_cfg.way,
                    self.train_cfg.shot,
                    self.train_cfg.qry_shot,
                    #  self.train_cfg.method == "maml",
                )
                self._normalize_fn = dataset._normalize

        return dataset

    def _make_initial_state(self, rng, dummy_input):
        slow_params, fast_params, slow_state, fast_state = make_params(
            rng,
            self._dataset,
            self._encoder.init,
            self._encoder.apply,
            self._classifier.init,
            self._normalize_fn(dummy_input / 255),
        )
        slow_state = jax.tree_map(
            jax.partial(replicate_array, num_devices=self.per_device_batch_size),
            slow_state,
        )

        fast_state = jax.tree_map(
            jax.partial(replicate_array, num_devices=self.per_device_batch_size),
            fast_state,
        )

        inner_lr = jnp.ones([]) * self.train_cfg.inner_lr
        if self.train_cfg.learn_inner_lr:
            opt_state = self._optimizer.init((slow_params, fast_params, inner_lr))
        else:
            opt_state = self._optimizer.init((slow_params, fast_params))

        return MetaLearnerState(
            slow_params,
            fast_params,
            slow_state,
            fast_state,
            inner_lr,
            opt_state,
        )

    def _make_scheduler(self):
        if self.train_cfg.scheduler == "cosine":
            return delayed_cosine_decay_schedule(
                -self.train_cfg.outer_lr,
                self.train_cfg.cosine_transition_begin * self._apply_every,
                self.train_cfg.cosine_decay_steps * self._apply_every,
                self.train_cfg.cosine_alpha,
            )
        elif self.train_cfg.scheduler == "step":
            return ox.piecewise_constant_schedule(
                -self.train_cfg.outer_lr,
                {
                    e: self.train_cfg.piecewise_constant_alpha
                    for e in self.train_cfg.piecewise_constant_schedule
                },
            )
        elif self.train_cfg.scheduler == "none":
            return ox.piecewise_constant_schedule(-self.train_cfg.outer_lr)

    # def _optimizer(self):
    #     return ox.chain(
    #         ox.clip(10),
    #         ox.scale(1 / self._apply_every),
    #         ox.apply_every(self._apply_every),
    #         #  ox.additive_weight_decay(self.train_cfg.weight_decay),
    #         ox.scale_by_adam(),
    #         ox.scale_by_schedule(self._make_scheduler()),
    #     )

    def get_first_state(self):
        _state = jax.tree_map(get_first, self._learner_state)
        return MetaLearnerState(
            _state.slow_params,
            _state.fast_params,
            jax.tree_map(get_first, _state.slow_state),
            jax.tree_map(get_first, _state.fast_state),
            _state.inner_lr,
            _state.opt_state,
        )

    def predict(self, inputs):
        inputs = self._normalize_fn(inputs)
        single_state = self.get_first_state()
        return self._classifier.apply(
            single_state.fast_params,
            single_state.fast_state,
            None,
            *self._encoder.apply(
                single_state.slow_params,
                single_state.slow_state,
                None,
                inputs,
                False,
            )[0],
            False,
        )[0]


class MetaMiniImageNet:
    def __init__(
        self,
        rng,
        split,
        data_root,
        batch_size,
        way,
        shot,
        qry_shot,
        shuffled_labels=True,
    ):
        self._rng = rng
        self._batch_size = batch_size
        self._way = way
        self._shot = shot
        self._qry_shot = qry_shot
        self._shuffled_labels = shuffled_labels

        if split == "train":
            self._fp = osp.join(
                data_root,
                "miniImageNet_category_split_train_phase_train_ordered.pickle",
            )
        self._images, self._labels, self._normalize = prepare_data(
            "miniimagenet", self._fp
        )

        if split == "val":
            self._labels = self._labels - 64

        self.fsl_sample = jax.partial(
            fsl_sample,
            images=self._images,
            labels=self._labels,
            num_tasks=batch_size,
            way=way,
            spt_shot=shot,
            qry_shot=qry_shot,
            disjoint=False,
            shuffled_labels=shuffled_labels,
        )
        self.fsl_build = jax.partial(
            fsl_build, batch_size=batch_size, way=way, shot=shot, qry_shot=qry_shot
        )

    def __next__(self):
        self._rng, rng = split(self._rng)
        return self.fsl_build(*self.fsl_sample(rng))