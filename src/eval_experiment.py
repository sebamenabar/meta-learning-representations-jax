import numpy as onp

import jax
import jax.numpy as jnp
from jax.random import split

from tensorflow_probability.substrates import jax as tfp

import optax as ox

import time
from data import augment, BatchSampler
from lib import (
    flatten,
    fsl_inner_loop,
    outer_loop,
    batched_outer_loop,
    mean_xe_and_acc_dict,
    replicate_array,
)
from test_utils import lr_fit_eval

from tqdm.autonotebook import tqdm

from mrcl_experiment import reshape_inputs

def send_to_devices(tree):
    return jax.pmap(lambda x: x)(tree_replicate(tree, jax.device_count()))

# @jax.jit
def normalize(x):
    norm = jax.numpy.linalg.norm(x, axis=-1, keepdims=True)
    return x / norm


class LRTester:
    def __init__(
        self,
        slow_apply,
        num_tasks,
        #  batch_size,
        dataset,
        n_aug_samples,
        normalize_fn,
        keep_orig_aug=True,
        final_normalize=True,
    ):
        self.slow_apply = slow_apply
        self.num_tasks = num_tasks
        self.batch_size = dataset._batch_size
        self.num_tasks
        self.dataset = dataset
        self.n_aug_samples = n_aug_samples
        self.normalize_fn = normalize_fn
        self.keep_orig_aug = keep_orig_aug
        self.final_normalize = final_normalize

        self.encode_batch = jax.jit(
            # self.encode_batch = (
            jax.partial(
                self._encode_batch,
                n_aug_samples,
                normalize_fn,
                slow_apply,
                keep_orig_aug,
                final_normalize,
            )
        )

    def eval(self, slow_params, slow_state, num_tasks=None):
        if num_tasks is None:
            num_tasks = self.num_tasks
        rng, rng_data = split(jax.random.PRNGKey(0), 2)
        # self.dataset.rng = rng_data
        preds = []
        targets = []
        for i in tqdm(range(num_tasks // self.batch_size)):
            rng, rng_step = split(rng)
            x_spt, y_spt, x_qry, y_qry = next(self.dataset)

            now = time.time()
            spt_features, y_spt, qry_features, y_qry = self.encode_batch(
                rng_step,
                slow_params,
                slow_state,
                x_spt,
                y_spt,
                x_qry,
                y_qry,
            )

            # print("forward time:", time.time() - now)
            now = time.time()
            spt_features = onp.array(spt_features)
            qry_features = onp.array(qry_features)

            y_spt = onp.array(y_spt)
            y_qry = onp.array(y_qry)
            # print("move to cpu time:", time.time() - now)

            for i in range(x_spt.shape[0]):
                now = time.time()
                preds.append(lr_fit_eval(spt_features[i], y_spt[i], qry_features[i]))
                # print("fit lr time:", time.time() - now)
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
        final_normalize,
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
        print(
            f"Evaluating LR {n_aug_samples} augmented samples keep orig {keep_orig_aug}"
        )
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

        if final_normalize:
            spt_features = normalize(spt_features)
            qry_features = normalize(qry_features)

        return spt_features, y_spt, qry_features, y_qry


class MAMLTester:
    def __init__(
        self,
        slow_apply,
        fast_apply,
        num_tasks,
        # batch_size,
        dataset,
        num_inner_steps,
        n_aug_samples,
        normalize_fn,
        keep_orig_aug=True,
        reset_fn=None,
    ):
        self.slow_apply = slow_apply
        self.fast_apply = fast_apply
        self.num_tasks = num_tasks
        self.batch_size = dataset._batch_size
        self.num_tasks
        self.dataset = dataset
        self.num_inner_steps = num_inner_steps
        self.n_aug_samples = n_aug_samples
        self.normalize_fn = normalize_fn
        self.keep_orig_aug = keep_orig_aug
        self.reset_fn = reset_fn

        # self.batch_adapt_jit = jax.jit(
        self.batch_adapt_jit = jax.pmap(
            jax.partial(
                self._batch_adapt,
                self.n_aug_samples,
                self.normalize_fn,
                self.num_inner_steps,
                self.slow_apply,
                self.fast_apply,
                self.keep_orig_aug,
                self.reset_fn,
            ),
            #  static_argnums=(0,),
        )

    def eval(
        self, slow_params, fast_params, slow_state, fast_state, inner_lr, num_tasks=None
    ):
        if num_tasks is None:
            num_tasks = self.num_tasks

        per_device_batch_size = self.batch_size // jax.device_count()
        assert (self.batch_size % jax.device_count()) == 0, f"Val batch size: {self.batch_size}, num devices {jax.device_count()}"
        # slow_state = jax.tree_map(
        #     jax.partial(replicate_array, num_devices=per_device_batch_size), slow_state
        # )
        # fast_state = jax.tree_map(
        #     jax.partial(replicate_array, num_devices=per_device_batch_size), fast_state
        # )
        slow_state = tree_replicate(slow_state, per_device_batch_size)
        fast_state = tree_replicate(fast_state, per_device_batch_size)

        (slow_params, fast_params, slow_state, fast_state, inner_lr) = send_to_devices(
            (slow_params,
            fast_params,
            slow_state,
            fast_state,
            inner_lr,)
        )

        # slow_params = send_to_devices(slow_params)
        # fast_params = send_to_devices(fast_params)
        # slow_state = send_to_devices(slow_state)
        # fast_state = send_to_devices(fast_state)
        # inner_lr = send_to_devices(inner_lr)

        results = []
        rng, rng_data = split(jax.random.PRNGKey(0), 2)
        # self.dataset.rng = rng_data
        for i in tqdm(range(num_tasks // self.batch_size)):
            rng, rng_step = split(rng)
            inputs = next(self.dataset)
            spt_classes = onp.unique(inputs[1], axis=1)
            inputs = reshape_inputs(inputs)
            (spt_classes,) = reshape_inputs([spt_classes])
            results.append(
                self.batch_adapt_jit(
                    split(rng_step, jax.device_count()),
                    slow_params,
                    fast_params,
                    slow_state,
                    fast_state,
                    inner_lr,
                    *inputs,
                    spt_classes,
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
        reset_fn,
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
        spt_classes=None,
    ):
        x_spt, x_qry = x_spt / 255, x_qry / 255
        # if self.n_aug_samples:
        print(
            f"Evaluating MAML {n_aug_samples} augmented samples keep orig {keep_orig_aug}"
        )
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

        # spt_classes = None
        # spt_classes = onp.unique(y_spt, axis=1)

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
            reset_fast_params_fn=reset_fn,
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
            spt_classes,
            # spt_classes,
        )

        return info


def tree_replicate(tree, num):
    return jax.tree_map(lambda x: replicate_array(x, num), tree)


def xe_loss(logits, targets):
    return -jnp.take_along_axis(jax.nn.log_softmax(logits), targets[..., None], axis=-1)


def regression_batchwise_loss(W, num_classes, c, features, targets):
    batch_size = features.shape[0]
    num_features = features.shape[-1]
    b = W[:, :num_classes].reshape(batch_size, 1, num_classes)
    # b = W[:num_classes * batch_size].reshape(batch_size, 1, num_classes)
    w = W[:, num_classes:].reshape(batch_size, num_features, num_classes)
    # w = W[num_classes * batch_size:].reshape(batch_size, num_features, num_classes)
    logits = jnp.matmul(features, w) + b
    loss = c * xe_loss(logits, targets).sum((1, 2)) + (1 / 2) * (w ** 2).sum((1, 2))
    return loss.sum(), loss


def regression_value_and_grad_fn(num_classes, c, features, targets, W):
    grad, loss = jax.grad(regression_batchwise_loss, has_aux=True)(
        W, num_classes, c, features, targets
    )
    return loss, grad


def lr_fit_jax(features, y, num_classes):
    batch_size = features.shape[0]
    num_features = features.shape[-1]
    W_size = (batch_size, (num_features + 1) * num_classes)
    W = jnp.zeros(W_size, dtype=features.dtype)
    c = 1.0


    minimized = tfp.optimizer.lbfgs_minimize(
        value_and_gradients_function=jax.partial(
            regression_value_and_grad_fn, num_classes, c, features, y
        ),
        initial_position=W,
        previous_optimizer_results=None,
        num_correction_pairs=10,
        # tolerance=1e-08,
        tolerance=1e-4,
        x_tolerance=0,
        f_relative_tolerance=0,
        initial_inverse_hessian_estimate=None,
        max_iterations=1000,
        parallel_iterations=1,
        stopping_condition=None,
        max_line_search_iterations=50,
        name=None,
    )

    b = minimized.position[:, :num_classes].reshape(batch_size, 1, num_classes)
    w = minimized.position[:, num_classes:].reshape(
        batch_size, num_features, num_classes
    )
    return w, b


class GPUMultinomialRegression:
    def __init__(
        self,
        num_classes,
        slow_apply,
        num_tasks,
        #  batch_size,
        dataset,
        n_aug_samples,
        normalize_fn,
        keep_orig_aug=True,
        final_normalize=True,
    ):
        self.dataset = dataset
        self.num_classes = num_classes

        self.slow_apply = slow_apply
        self.num_tasks = num_tasks
        self.batch_size = dataset._batch_size
        self.num_tasks
        self.dataset = dataset
        self.n_aug_samples = n_aug_samples
        self.normalize_fn = normalize_fn
        self.keep_orig_aug = keep_orig_aug
        self.final_normalize = final_normalize

        self.predict_batch = jax.pmap(
            jax.partial(
                self._predict_batch,
                num_classes,
                n_aug_samples,
                normalize_fn,
                slow_apply,
                keep_orig_aug,
                final_normalize,
            )
        )

    def eval(self, slow_params, slow_state, num_tasks=None):
        if num_tasks is None:
            num_tasks = self.num_tasks
        rng, rng_data = split(jax.random.PRNGKey(0), 2)
        # self.dataset.rng = rng_data
        accs = []
        # targets = []
        # slow_params = jax.pmap(lambda x: x)(tree_replicate(slow_params, jax.device_count()))
        # slow_state = jax.pmap(lambda x: x)(tree_replicate(slow_state, jax.device_count()))
        slow_params, slow_state = send_to_devices((slow_params, slow_state))
        predict_batch = lambda rng, *args: self.predict_batch(
            rng,
            slow_params,
            slow_state,
            *args,
        )
        for i in tqdm(range(num_tasks // self.batch_size)):
            rng, rng_step = split(rng)
            x_spt, y_spt, x_qry, y_qry = reshape_inputs(next(self.dataset))

            now = time.time()
            taskwise_acc = predict_batch(
                split(rng_step, jax.device_count()),
                # slow_params,
                # slow_state,
                x_spt,
                y_spt,
                x_qry,
                y_qry,
            )
            accs.append(taskwise_acc)

            # print("forward time:", time.time() - now)
            now = time.time()
        accs = jnp.concatenate(accs)

        return accs.mean(), accs.std()

    @staticmethod
    def _predict_batch(
        num_classes,
        n_aug_samples,
        normalize_fn,
        slow_apply,
        keep_orig_aug,
        final_normalize,
        rng,
        slow_params,
        slow_state,
        x_spt,
        y_spt,
        x_qry,
        y_qry,
    ):
        # print(x_spt.shape)
        x_spt, x_qry = x_spt / 255, x_qry / 255
        # if self.n_aug_samples:
        print(
            f"Evaluating LR {n_aug_samples} augmented samples keep orig {keep_orig_aug}"
        )
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

        if final_normalize:
            spt_features = normalize(spt_features)
            qry_features = normalize(qry_features)

        w, b = lr_fit_jax(spt_features, y_spt, num_classes)
        qry_logits = jnp.matmul(qry_features, w) + b
        taskwise_acc = (
            (jax.nn.softmax(qry_logits).argmax(-1) == y_qry).astype(jnp.float32).mean(1)
        )

        return taskwise_acc


class ParallelSupervisedStandardTester:
    def __init__(
        self, rng, dataset, batch_size, slow_apply, fast_apply, normalize_fn=None,
    ):
        self.rng = rng
        self.x = dataset._images
        self.y = dataset._labels
        #self.batch_size = batch_size
        self.batch_size = 128
        self.normalize_fn = normalize_fn
        # self.eval_batch = jax.pmap(jax.partial(
        #     self._eval_batch, normalize_fn, slow_apply, fast_apply,
        #     ), axis_name="i")
        self.eval_batch = jax.jit(jax.partial(
            self._eval_batch, normalize_fn, slow_apply, fast_apply,
            ))

    def eval(self, slow_params, fast_params, slow_state, fast_state):
        rng = jax.random.PRNGKey(0)
        sampler = BatchSampler(rng, self.x, self.y, self.batch_size, shuffle=True, keep_last=True)
    
        # (slow_params, fast_params, slow_state, fast_state) = send_to_devices(
        #     (slow_params,
        #     fast_params,
        #     slow_state,
        #     fast_state,)
        # )

        total_corrects = 0
        total_samples = 0

        for inputs in sampler:
            num_samples = inputs[0].shape[0]
            # print(inputs[0].shape)

            # try:
            #     inputs = reshape_inputs(inputs, jax.device_count())
            
            batch_corrects = self.eval_batch(slow_params, fast_params, slow_state, fast_state, *inputs)
            total_corrects += batch_corrects.sum()
            total_samples += num_samples

        return total_corrects / total_samples
        

    @staticmethod
    def _eval_batch(normalize_fn, slow_apply, fast_apply, slow_params, fast_params, slow_state, fast_state, x, y):
        x = x / 255
        x = normalize_fn(x)

        slow_outputs = slow_apply(slow_params, slow_state, None, x, False)[0][0]
        pred = fast_apply(fast_params, fast_state, None, slow_outputs, False)[0]
        return (pred.argmax(-1) == y).sum()