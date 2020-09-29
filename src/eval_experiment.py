import numpy as onp

import jax
import jax.numpy as jnp
from jax.random import split

import optax as ox

from data import augment
from lib import (
    flatten,
    fsl_inner_loop,
    outer_loop,
    batched_outer_loop,
    mean_xe_and_acc_dict,
)
from test_utils import lr_fit_eval


class LRTester:
    def __init__(
        self,
        slow_apply,
        num_tasks,
        batch_size,
        dataset,
        n_aug_samples,
        normalize_fn,
        keep_orig_aug=True,
    ):
        self.slow_apply = slow_apply
        self.num_tasks = num_tasks
        self.batch_size = batch_size
        self.num_tasks
        self.dataset = dataset
        self.n_aug_samples = n_aug_samples
        self.normalize_fn = normalize_fn
        self.keep_orig_aug = keep_orig_aug

        self.encode_batch = jax.jit(
            jax.partial(
                self._encode_batch,
                n_aug_samples,
                normalize_fn,
                slow_apply,
                keep_orig_aug,
            )
        )

    def eval(self, slow_params, slow_state):
        rng, rng_data = split(jax.random.PRNGKey(0), 2)
        # self.dataset.rng = rng_data
        preds = []
        targets = []
        for i in range(self.num_tasks // self.batch_size):
            rng, rng_step = split(rng)
            x_spt, y_spt, x_qry, y_qry = next(self.dataset)
            spt_features, y_spt, qry_features, y_qry = self.encode_batch(
                rng_step,
                slow_params,
                slow_state,
                x_spt,
                y_spt,
                x_qry,
                y_qry,
            )

            spt_features = onp.array(spt_features)
            qry_features = onp.array(qry_features)

            y_spt = onp.array(y_spt)
            y_qry = onp.array(y_qry)

            for i in range(x_spt.shape[0]):
                preds.append(lr_fit_eval(spt_features[i], y_spt[i], qry_features[i]))
            targets.append(y_qry)

        preds = onp.stack(preds)
        targets = onp.concatenate(targets)
        taskwise_acc = (preds == targets).astype(onp.float).mean(1)
        return taskwise_acc.mean(), taskwise_acc.std()

    @staticmethod
    def _encode_batch(
        n_aug_samples,
        normalize_fn,
        slow_apply,
        keep_orig_aug,
        rng,
        slow_params,
        slow_state,
        x_spt,
        y_spt,
        x_qry,
        y_qry,
    ):
        x_spt, x_qry = x_spt / 255, x_qry / 255
        # if self.n_aug_samples:
        print(f"Evaluating LR {n_aug_samples} augmented samples keep orig {keep_orig_aug}")
        if n_aug_samples:
            # print("Augmenting testing samples")
            # aug_x_spt = x_spt.repeat(self.n_aug_samples, 1)
            aug_x_spt = x_spt.repeat(n_aug_samples, 1)
            # aug_y_spt = y_spt.repeat(self.n_aug_samples, 1)
            aug_y_spt = y_spt.repeat(n_aug_samples, 1)
            rng, rng_aug = split(rng)
            aug_x_spt = augment(rng_aug, flatten(aug_x_spt, (0, 1))).reshape(
                *aug_x_spt.shape
            )

            if keep_orig_aug:
                x_spt = jnp.concatenate((x_spt, aug_x_spt), axis=1)
                y_spt = jnp.concatenate((y_spt, aug_y_spt), axis=1)
            else:
                x_spt = aug_x_spt
                y_spt = aug_y_spt

        x_spt = normalize_fn(x_spt)
        x_qry = normalize_fn(x_qry)

        spt_features = jax.vmap(
            jax.partial(slow_apply, slow_params, slow_state, None, is_training=False)
        )(x_spt)[0][0]
        qry_features = jax.vmap(
            jax.partial(slow_apply, slow_params, slow_state, None, is_training=False)
        )(x_qry)[0][0]

        return spt_features, y_spt, qry_features, y_qry


class MAMLTester:
    def __init__(
        self,
        slow_apply,
        fast_apply,
        num_tasks,
        batch_size,
        dataset,
        num_inner_steps,
        n_aug_samples,
        normalize_fn,
        keep_orig_aug=True,
    ):
        self.slow_apply = slow_apply
        self.fast_apply = fast_apply
        self.num_tasks = num_tasks
        self.batch_size = batch_size
        self.num_tasks
        self.dataset = dataset
        self.num_inner_steps = num_inner_steps
        self.n_aug_samples = n_aug_samples
        self.normalize_fn = normalize_fn
        self.keep_orig_aug = keep_orig_aug

        self.batch_adapt_jit = jax.jit(
            jax.partial(
                self._batch_adapt,
                self.n_aug_samples,
                self.normalize_fn,
                self.num_inner_steps,
                self.slow_apply,
                self.fast_apply,
                self.keep_orig_aug,
            ),
            #  static_argnums=(0,),
        )

    def eval(self, slow_params, fast_params, slow_state, fast_state, inner_lr):
        results = []
        rng, rng_data = split(jax.random.PRNGKey(0), 2)
        # self.dataset.rng = rng_data
        for i in range(self.num_tasks // self.batch_size):
            rng, rng_step = split(rng)
            x_spt, y_spt, x_qry, y_qry = next(self.dataset)
            results.append(
                self.batch_adapt_jit(
                    rng_step,
                    slow_params,
                    fast_params,
                    slow_state,
                    fast_state,
                    inner_lr,
                    x_spt,
                    y_spt,
                    x_qry,
                    y_qry,
                )
            )

        results = jax.tree_multimap(
            lambda x, *xs: jnp.concatenate(xs), results[0], *results
        )
        taskwise_acc = results["outer"]["final"]["aux"][0]["acc"]
        return taskwise_acc.mean().item(), taskwise_acc.std().item()

    @staticmethod  # JIT recompiled every time with self
    def _batch_adapt(
        # self,
        n_aug_samples,
        normalize_fn,
        num_inner_steps,
        slow_apply,
        fast_apply,
        keep_orig_aug,
        rng,
        slow_params,
        fast_params,
        slow_state,
        fast_state,
        inner_lr,
        x_spt,
        y_spt,
        x_qry,
        y_qry,
    ):


        x_spt, x_qry = x_spt / 255, x_qry / 255
        # if self.n_aug_samples:
        print(f"Evaluating MAML {n_aug_samples} augmented samples keep orig {keep_orig_aug}")
        if n_aug_samples:
            # print("Augmenting testing samples")
            # aug_x_spt = x_spt.repeat(self.n_aug_samples, 1)
            aug_x_spt = x_spt.repeat(n_aug_samples, 1)
            # aug_y_spt = y_spt.repeat(self.n_aug_samples, 1)
            aug_y_spt = y_spt.repeat(n_aug_samples, 1)
            rng, rng_aug = split(rng)
            aug_x_spt = augment(rng_aug, flatten(aug_x_spt, (0, 1))).reshape(
                *aug_x_spt.shape
            )

            if keep_orig_aug:
                x_spt = jnp.concatenate((x_spt, aug_x_spt), axis=1)
                y_spt = jnp.concatenate((y_spt, aug_y_spt), axis=1)
            else:
                x_spt = aug_x_spt
                y_spt = aug_y_spt
        

        # x_spt = self.normalize_fn(x_spt)
        # x_qry = self.normalize_fn(x_qry)
        x_spt = normalize_fn(x_spt)
        x_qry = normalize_fn(x_qry)

        inner_opt = ox.sgd(inner_lr)

        def inner_opt_update_fn(lr, updates, state, params):
            return inner_opt.update(updates, state, params)

        _inner_loop = jax.partial(
            fsl_inner_loop,
            is_training=False,
            num_steps=num_inner_steps,
            slow_apply=slow_apply,
            fast_apply=fast_apply,
            # num_steps=self.num_inner_steps,
            # slow_apply=self.slow_apply,
            # fast_apply=self.fast_apply,
            loss_fn=mean_xe_and_acc_dict,
            opt_update_fn=inner_opt_update_fn,
        )
        _outer_loop = jax.partial(
            outer_loop,
            is_training=False,
            inner_loop=_inner_loop,
            # slow_apply=self.slow_apply,
            # fast_apply=self.fast_apply,
            slow_apply=slow_apply,
            fast_apply=fast_apply,
            loss_fn=mean_xe_and_acc_dict,
            # reset_fast_params_fn=self.reset_classifier,
        )
        _batched_outer_loop = jax.partial(batched_outer_loop, outer_loop=_outer_loop)

        inner_opt_state = inner_opt.init(fast_params)
        (_, (_, _, info)) = _batched_outer_loop(
            slow_params,
            fast_params,
            inner_lr,
            slow_state,
            fast_state,
            inner_opt_state,
            split(rng, x_spt.shape[0]),
            x_spt,
            y_spt,
            x_qry,
            y_qry,
            None,
            # spt_classes,
        )

        return info
