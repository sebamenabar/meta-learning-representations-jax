import jax
from jax import value_and_grad, numpy as jnp
from jax.random import split

import optax as ox

from lib import evaluate_supervised_accuracy


class MetaLearningWrapper:
    def __init__(
        self,
        slow_apply,
        fast_apply,
        slow_params,
        fast_params,
        slow_state,
        fast_state,
        inner_lr=None,
        opt_state=None,
        training=True,
        loss_fn=None,
        optimizer=None,
        scheduler=None,
        init_inner_opt_state_fn=None,
        inner_opt_update_fn=None,
        num_inner_steps_train=5,
        num_inner_steps_test=10,
        preprocess_spt_fn=None,
        preprocess_qry_fn=None,
        reset_fast_params_fn=None,
        test_init_inner_opt_state_fn=None,
        test_inner_opt_update_fn=None,
        preprocess_test_fn=None,
    ):
        self.slow_apply = slow_apply
        self.fast_apply = fast_apply
        self.slow_params = slow_params
        self.fast_params = fast_params
        self.slow_state = slow_state
        self.fast_state = fast_state
        self.opt_state = opt_state

        self.inner_lr = inner_lr
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.inner_opt_update_fn = inner_opt_update_fn
        self.init_inner_opt_state_fn = init_inner_opt_state_fn
        self.num_inner_steps_train = num_inner_steps_train
        self.num_inner_steps_test = num_inner_steps_test
        self.preprocess_spt_fn = preprocess_spt_fn
        self.preprocess_qry_fn = preprocess_qry_fn
        self.reset_fast_params_fn = reset_fast_params_fn
        self.test_init_inner_opt_state = test_init_inner_opt_state_fn
        self.test_inner_opt_update_fn = test_inner_opt_update_fn
        self.preprocess_test_fn = preprocess_test_fn

        self.__training = training
        self.__step_fn = None
        self.__test_inner_loop = None

    def inner_loop(self, *args, **kwargs):
        raise NotImplementedError

    def training(self, val=None):
        if val is not None:
            self.__training = val
            return self
        return self.__training

    def init_opt_state(self, optimizer=None):
        if optimizer is None:
            optimizer = self.optimizer
        self.opt_state = optimizer.init((self.slow_params, self.fast_params))
        return self

    def set_step_fn(self, fn):
        self.__step_fn = fn
        return self

    def __call__(
        self,
        inputs,
        rng=None,
        training=None,
        slow_params=None,
        fast_params=None,
        slow_state=None,
        fast_state=None,
    ):
        if slow_params is None:
            slow_params = self.slow_params
        if fast_params is None:
            fast_params = self.fast_params
        if slow_state is None:
            slow_state = self.slow_state
        if fast_state is None:
            fast_state = self.fast_state
        if training is None:
            training = self.training()

        if rng is not None:
            rng_slow, rng_fast = split(rng)
        else:
            rng_slow = rng_fast = None
        slow_outputs, slow_state = self.slow_apply(
            slow_params, slow_state, rng_slow, inputs, training,
        )
        outputs, fast_state = self.fast_apply(
            fast_params, fast_state, rng_fast, *slow_outputs, training,
        )

        return outputs, (slow_state, fast_state)

    @jax.partial(jax.jit, static_argnums=0)
    def _jit_call_validate(
        self, slow_params, fast_params, slow_state, fast_state, inputs,
    ):
        return self(
            inputs,
            slow_params=slow_params,
            fast_params=fast_params,
            slow_state=slow_state,
            fast_state=fast_state,
            training=False,
        )

    def jit_call_validate(
        self,
        inputs,
        slow_params=None,
        fast_params=None,
        slow_state=None,
        fast_state=None,
    ):
        if slow_params is None:
            slow_params = self.slow_params
        if fast_params is None:
            fast_params = self.fast_params
        if slow_state is None:
            slow_state = self.slow_state
        if fast_state is None:
            fast_state = self.fast_state

        return self._jit_call_validate(
            slow_params, fast_params, slow_state, fast_state, inputs,
        )

    def test(
        self,
        spt_loader,
        qry_loader,
        opt_state=None,
        inner_opt_update_fn=None,
        test_init_inner_opt_state=None,
        inner_lr=None,
        preprocess_test_fn=None,
        fast_params=None,
        fast_state=None,
    ):
        if inner_opt_update_fn is None:
            inner_opt_update_fn = self.test_inner_opt_update_fn
        if test_init_inner_opt_state is None:
            test_init_inner_opt_state = self.test_init_inner_opt_state
        if inner_lr is None:
            inner_lr = self.inner_lr
        if preprocess_test_fn is None:
            preprocess_test_fn = self.preprocess_test_fn
        if fast_params is None:
            fast_params = self.fast_params
        if fast_state is None:
            fast_state = self.fast_state

        if opt_state is None:
            opt_state = test_init_inner_opt_state(fast_params)

        if self.__test_inner_loop is None:
            self.__test_inner_loop = jax.jit(
                lambda x, y, opt_state, inner_lr, slow_params, fast_params, slow_state, fast_state: self.inner_updates(
                    self.slow_apply(slow_params, slow_state, None, x, False)[0],
                    y,
                    inner_lr=inner_lr,
                    opt_state=opt_state,
                    # slow_params=slow_params,
                    fast_params=fast_params,
                    # slow_state=slow_state,
                    fast_state=fast_state,
                    training=False,
                    inner_opt_update_fn=inner_opt_update_fn,
                    fast_apply=self.fast_apply,
                    rng=None,
                    loss_fn=self.loss_fn,
                )
            )

        # fast_params, fast_state = self.fast_params, self.fast_state
        for x, y in spt_loader:
            fast_params, fast_state, opt_state, *_ = self.__test_inner_loop(
                preprocess_test_fn(x),
                y,
                opt_state,
                inner_lr,
                self.slow_params,
                fast_params,
                self.slow_state,
                fast_state,
            )

        train_loss, train_acc = evaluate_supervised_accuracy(
            lambda x: self.jit_call_validate(
                preprocess_test_fn(x), fast_params=fast_params, fast_state=fast_state,
            )[0],
            spt_loader,
        )
        test_loss, test_acc = evaluate_supervised_accuracy(
            lambda x: self.jit_call_validate(
                preprocess_test_fn(x), fast_params=fast_params, fast_state=fast_state,
            )[0],
            qry_loader,
        )

        return (train_loss, train_acc), (test_loss, test_acc)

    def fast_apply_and_loss(
        self,
        x,
        y,
        rng=None,
        training=None,
        fast_params=None,
        fast_state=None,
        loss_fn=None,
        fast_apply=None,
    ):
        if fast_params is None:
            fast_params = self.fast_params
        if fast_state is None:
            fast_state = self.fast_state
        if training is None:
            training = self.training()
        if fast_apply is None:
            fast_apply = self.fast_apply
        if loss_fn is None:
            loss_fn = self.loss_fn

        outputs, state = fast_apply(fast_params, fast_state, rng, *x, training)
        loss, aux = loss_fn(outputs, y)
        return loss, (state, aux)

    def single_inner_update(
        self,
        x,
        y,
        rng=None,
        inner_lr=None,
        training=None,
        fast_params=None,
        fast_state=None,
        opt_state=None,
        fast_apply=None,
        loss_fn=None,
        init_inner_opt_state_fn=None,
        inner_opt_update_fn=None,
    ):
        if training is None:
            training = self.training()
        if fast_params is None:
            fast_params = self.fast_params
        if inner_lr is None:
            inner_lr = self.inner_lr
        if fast_state is None:
            fast_state = self.fast_state
        if fast_apply is None:
            fast_apply = self.fast_apply
        if loss_fn is None:
            loss_fn = self.loss_fn
        if inner_opt_update_fn is None:
            inner_opt_update_fn = self.inner_opt_update_fn
        if init_inner_opt_state_fn is None:
            init_inner_opt_state_fn = self.init_inner_opt_state_fn

        if opt_state is None:
            opt_state = init_inner_opt_state_fn(fast_params)

        (loss, (new_fast_state, auxs)), grads = value_and_grad(
            self.fast_apply_and_loss, argnums=4, has_aux=True,
        )(x, y, rng, training, fast_params, fast_state, loss_fn, fast_apply)

        updates, opt_state = inner_opt_update_fn(
            inner_lr, grads, opt_state, fast_params,
        )
        fast_params = ox.apply_updates(fast_params, updates)

        return fast_params, new_fast_state, opt_state, loss, auxs

    def outer_loop(
        self,
        x_spt,
        y_spt,
        x_qry,
        y_qry,
        spt_classes=jnp.empty((0,), jnp.int32),
        rng=None,
        slow_params=None,
        fast_params=None,
        slow_state=None,
        fast_state=None,
        inner_lr=None,
        inner_opt_state=None,
        training=None,
        slow_apply=None,
        fast_apply=None,
        loss_fn=None,
        num_inner_steps=None,
        init_inner_opt_state_fn=None,
        inner_opt_update_fn=None,
        reset_fast_params_fn=None,
    ):
        if training is None:
            training = self.training()
        if slow_apply is None:
            slow_apply = self.slow_apply
        if fast_apply is None:
            fast_apply = self.fast_apply
        if loss_fn is None:
            loss_fn = self.loss_fn
        if num_inner_steps is None:
            if training:
                num_inner_steps = self.num_inner_steps_train
            else:
                num_inner_steps = self.num_inner_steps_test
        if inner_lr is None:
            inner_lr = self.inner_lr
        if slow_params is None:
            slow_params = self.slow_params
        if fast_params is None:
            fast_params = self.fast_params
        if slow_state is None:
            slow_state = self.slow_state
        if fast_state is None:
            fast_state = self.fast_state
        if init_inner_opt_state_fn is None:
            init_inner_opt_state_fn = self.init_inner_opt_state_fn
        if inner_opt_update_fn is None:
            inner_opt_update_fn = self.inner_opt_update_fn
        if reset_fast_params_fn is None:
            reset_fast_params_fn = self.reset_fast_params_fn

        if inner_opt_state is None:
            inner_opt_state = init_inner_opt_state_fn(fast_params)

        if rng is not None:
            rng_reset_params, rng_outer_slow, rng_outer_fast, rng_inner = split(rng, 4)
        else:
            rng_reset_params = rng_outer_slow = rng_outer_fast = rng_inner = None

        if reset_fast_params_fn is not None:
            fast_params = reset_fast_params_fn(
                rng_reset_params, spt_classes, fast_params,
            )

        slow_outputs, outer_slow_state = slow_apply(
            slow_params, slow_state, rng_outer_slow, x_qry, training,
        )
        (
            inner_fast_params,
            inner_slow_state,
            fast_state,
            inner_opt_state,
            # inner_loss,
            inner_auxs,
        ) = self.inner_loop(
            x_spt,
            y_spt,
            rng_inner,
            slow_params,
            fast_params,
            slow_state,
            fast_state,
            inner_lr,
            inner_opt_state,
            training,
            slow_apply,
            fast_apply,
            loss_fn,
            num_inner_steps,
            init_inner_opt_state_fn,
            inner_opt_update_fn,
        )

        initial_loss, (_, initial_aux) = self.fast_apply_and_loss(
            slow_outputs,
            y_qry,
            rng_outer_fast,
            training,
            fast_params,
            fast_state,
            loss_fn,
            fast_apply,
        )
        final_loss, (_, final_aux) = self.fast_apply_and_loss(
            slow_outputs,
            y_qry,
            rng_outer_fast,
            training,
            inner_fast_params,
            fast_state,
            loss_fn,
            fast_apply,
        )

        return (
            final_loss,
            (
                slow_state,
                fast_state,
                {
                    "inner": inner_auxs,
                    "outer": {
                        "initial": {**initial_aux, "loss": initial_loss},
                        "final": {**final_aux, "loss": final_loss},
                    },
                },
            ),
        )

    def single_step(
        self,
        step_num,
        bx_spt,
        by_spt,
        bx_qry,
        by_qry,
        # opt_state,
        opt_state=None,
        rng=None,
        spt_classes=None,
        slow_params=None,
        fast_params=None,
        slow_state=None,
        fast_state=None,
        inner_lr=None,
        inner_opt_state=None,
        training=None,
        slow_apply=None,
        fast_apply=None,
        loss_fn=None,
        num_inner_steps=None,
        init_inner_opt_state_fn=None,
        inner_opt_update_fn=None,
        reset_fast_params_fn=None,
        optimizer=None,
        scheduler=None,
    ):
        if training is None:
            training = self.training()
        if slow_apply is None:
            slow_apply = self.slow_apply
        if fast_apply is None:
            fast_apply = self.fast_apply
        if loss_fn is None:
            loss_fn = self.loss_fn
        if num_inner_steps is None:
            if training:
                num_inner_steps = self.num_inner_steps_train
            else:
                num_inner_steps = self.num_inner_steps_test
        if inner_lr is None:
            inner_lr = self.inner_lr
        if slow_params is None:
            slow_params = self.slow_params
        if fast_params is None:
            fast_params = self.fast_params
        if slow_state is None:
            slow_state = self.slow_state
        if fast_state is None:
            fast_state = self.fast_state
        if init_inner_opt_state_fn is None:
            init_inner_opt_state_fn = self.init_inner_opt_state_fn
        if inner_opt_update_fn is None:
            inner_opt_update_fn = self.inner_opt_update_fn
        if reset_fast_params_fn is None:
            reset_fast_params_fn = self.reset_fast_params_fn
        if optimizer is None:
            optimizer = self.optimizer
        if opt_state is None:
            opt_state = self.opt_state
            if opt_state is None:
                opt_state = optimizer.init((slow_params, fast_params))
        if scheduler is None:
            scheduler = self.scheduler

        if inner_opt_state is None:
            inner_opt_state = init_inner_opt_state_fn(fast_params)

        bsz = bx_spt.shape[0]
        if rng is not None:
            rng_step, rng_pre_spt, rng_pre_qry = split(rng, 3)
            bx_spt = jax.vmap(self.preprocess_spt_fn)(split(rng_pre_spt, bsz), bx_spt)
            bx_qry = jax.vmap(self.preprocess_qry_fn)(split(rng_pre_qry, bsz), bx_qry)
            rng_step = split(rng_step, bsz)
        else:
            rng_step = [None] * bsz
            bx_spt = jax.vmap(jax.partial(self.preprocess_spt_fn, None))(bx_spt)
            bx_qry = jax.vmap(jax.partial(self.preprocess_qry_fn, None))(bx_qry)
        if spt_classes is None:
            spt_classes = jnp.empty((bsz, 0), jnp.int32)

        def __batched_outer_loop(
            slow_params, fast_params,
        ):
            bloss, aux = jax.vmap(
                jax.partial(
                    self.outer_loop,
                    slow_params=slow_params,
                    fast_params=fast_params,
                    slow_state=slow_state,
                    fast_state=fast_state,
                    inner_lr=inner_lr,
                    inner_opt_state=inner_opt_state,
                    training=training,
                    slow_apply=slow_apply,
                    fast_apply=fast_apply,
                    loss_fn=loss_fn,
                    num_inner_steps=num_inner_steps,
                    init_inner_opt_state_fn=init_inner_opt_state_fn,
                    inner_opt_update_fn=inner_opt_update_fn,
                    reset_fast_params_fn=reset_fast_params_fn,
                )
            )(bx_spt, by_spt, bx_qry, by_qry, spt_classes, rng_step)
            return bloss.mean(), aux

        (outer_loss, (slow_state, fast_state, aux)), grads = value_and_grad(
            __batched_outer_loop, has_aux=True, argnums=(0, 1),
        )(slow_params, fast_params)

        # For multi-device
        # grads = jax.tree_map(lambda v: jax.lax.pmean(v, axis_name="i"), grads)

        updates, opt_state = optimizer.update(
            grads, opt_state, (slow_params, fast_params),
        )
        updates = scheduler(step_num, updates)
        (slow_params, fast_params) = ox.apply_updates(
            (slow_params, fast_params), updates
        )

        return (
            slow_params,
            fast_params,
            slow_state,
            fast_state,
            opt_state,
            # inner_lr,
            outer_loss,
            aux,
        )

    def train_step(
        self,
        step_num,
        rng,
        x_spt,
        y_spt,
        x_qry,
        y_qry,
        spt_classes,
        inner_lr=None,
        training=True,
    ):
        if inner_lr is None:
            inner_lr = self.inner_lr

        (
            self.slow_params,
            self.fast_params,
            self.slow_state,
            self.fast_state,
            self.opt_state,
            loss,
            aux,
        ) = self.__step_fn(
            step_num,
            x_spt,
            y_spt,
            x_qry,
            y_qry,
            opt_state=self.opt_state,
            rng=rng,
            spt_classes=spt_classes,
            slow_params=self.slow_params,
            fast_params=self.fast_params,
            slow_state=self.slow_state,
            fast_state=self.fast_state,
            inner_lr=inner_lr,
            # training=training,
        )

        return loss, aux

    def inner_loop(
        self,
        x,
        y,
        rng=None,
        slow_params=None,
        fast_params=None,
        slow_state=None,
        fast_state=None,
        inner_lr=None,
        opt_state=None,
        training=None,
        slow_apply=None,
        fast_apply=None,
        loss_fn=None,
        num_inner_steps=None,
        init_inner_opt_state_fn=None,
        inner_opt_update_fn=None,
    ):
        if training is None:
            training = self.training()
        if slow_apply is None:
            slow_apply = self.slow_apply
        if fast_apply is None:
            fast_apply = self.fast_apply
        if loss_fn is None:
            loss_fn = self.loss_fn
        if slow_params is None:
            slow_params = self.slow_params
        if fast_params is None:
            fast_params = self.fast_params
        if slow_state is None:
            slow_state = self.slow_state
        if fast_state is None:
            fast_state = self.fast_state
        if inner_lr is None:
            inner_lr = self.inner_lr
        if init_inner_opt_state_fn is None:
            init_inner_opt_state_fn = self.init_inner_opt_state_fn
        if inner_opt_update_fn is None:
            inner_opt_update_fn = self.inner_opt_update_fn
        if num_inner_steps is None:
            if training:
                num_inner_steps = self.num_inner_steps_train
            else:
                num_inner_steps = self.num_inner_steps_test

        if opt_state is None:
            opt_state = init_inner_opt_state_fn(fast_params)

        if rng is not None:
            rng_slow, rng_updates, rng_last = split(rng, 3)
        else:
            rng_slow = rng_updates = rng_last = None
        slow_outputs, slow_state = slow_apply(
            slow_params, slow_state, rng_slow, x, training,
        )

        fast_params, new_fast_state, opt_state, losses, auxs = self.inner_updates(
            slow_outputs,
            y,
            rng_updates,
            training,
            fast_params,
            fast_state,
            opt_state,
            fast_apply,
            loss_fn,
            inner_lr,
            inner_opt_update_fn,
            init_inner_opt_state_fn,
            num_inner_steps,
        )
        loss, (new_fast_state, aux) = self.fast_apply_and_loss(
            slow_outputs,
            y,
            rng_last,
            training,
            fast_params,
            fast_state,
            loss_fn,
            fast_apply,
        )
        # losses = jnp.append(losses, jnp.expand_dims(loss, 0), axis=0)
        # auxs = jax.tree_multimap(lambda x, xs: jnp.append(x, jnp.expand_dims(xs, 0), axis=0), auxs, aux)

        return fast_params, slow_state, fast_state, opt_state, (loss, aux, losses, auxs)

    def inner_updates(
        self,
        slow_outputs,
        y,
        rng,
        training,
        fast_params,
        fast_state,
        opt_state,
        fast_apply,
        loss_fn,
        inner_lr=None,
        inner_opt_update_fn=None,
        init_inner_opt_state_fn=None,
        num_inner_steps=None,
    ):
        raise NotImplementedError


class FSLWrapper(MetaLearningWrapper):
    def inner_updates(
        self,
        slow_outputs,
        y,
        rng,
        training,
        fast_params,
        fast_state,
        opt_state,
        fast_apply,
        loss_fn,
        inner_lr=None,
        inner_opt_update_fn=None,
        init_inner_opt_state_fn=None,
        num_inner_steps=None,
    ):
        # bsz = y.shape[0]
        if rng is None:
            rngs = [None] * num_inner_steps
        else:
            rngs = split(rng, num_inner_steps)
        losses = []
        auxs = []
        for i in range(num_inner_steps):
            (
                fast_params,
                new_fast_state,
                opt_state,
                loss,
                aux,
            ) = self.single_inner_update(
                slow_outputs,
                y,
                rngs[i],
                inner_lr,
                training,
                fast_params,
                fast_state,
                opt_state,
                fast_apply,
                loss_fn,
                inner_opt_update_fn,
            )

            losses.append(loss)
            auxs.append(aux)

        losses = jnp.stack(losses)
        auxs = jax.tree_multimap(lambda x, *xs: jnp.stack(xs), auxs[0], *auxs)

        return fast_params, fast_state, opt_state, losses, auxs


class CLWrapper(MetaLearningWrapper):
    def inner_updates(
        self,
        slow_outputs,
        y,
        rng,
        training,
        fast_params,
        fast_state,
        opt_state,
        fast_apply,
        loss_fn,
        inner_lr=None,
        inner_opt_update_fn=None,
        init_inner_opt_state_fn=None,
        num_inner_steps=None,
    ):

        bsz = y.shape[0]
        if rng is None:
            rngs = [None] * bsz
        else:
            rngs = split(rng, bsz)

        def scan_fun(
            carry,
            xs,
            # rng,
            # single_inner_update,
            # inner_lr,
            # training,
            # fast_apply,
            # loss_fn,
            # inner_opt_update_fn,
        ):
            (_params, _state, _opt_state) = carry
            (*_x, _y, _rng) = xs

            if rng is None:
                # Ugly patch
                _rng = None

            _params, _state, _opt_state, loss, aux = self.single_inner_update(
                _x,
                _y,
                _rng,
                inner_lr,
                training,
                _params,
                _state,
                _opt_state,
                fast_apply,
                loss_fn,
                init_inner_opt_state_fn,
                inner_opt_update_fn,
            )
            return (_params, _state, _opt_state), (loss, aux)

        slow_outputs = [jnp.expand_dims(_x, 1) for _x in slow_outputs]
        y = jnp.expand_dims(y, 1)
        carry, (losses, auxs) = jax.lax.scan(
            scan_fun, (fast_params, fast_state, opt_state), (*slow_outputs, y, rngs),
        )

        return (*carry, losses, auxs)
