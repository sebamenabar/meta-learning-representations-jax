import jax
import jax.numpy as jnp
from jax.random import split
from haiku.data_structures import merge
import optax as ox

from utils import use_self_as_default, expand, first_leaf_shape
from utils.losses import mean_xe_and_acc_dict


def make_simple_opt_update(opt):
    def f(lr, updates, state, params):
        return opt(lr).update(updates, state, params)

    return f


sgd_opt_update = make_simple_opt_update(ox.sgd)
adam_opt_update = make_simple_opt_update(ox.adam)


def make_simple_init_opt(opt):
    return opt(0).init


sgd_init_opt = make_simple_init_opt(ox.sgd)
adam_init_opt = make_simple_init_opt(ox.adam)


class ContinualLearningWrapperBase:
    def __init__(
        self,
        apply,
        params,
        state,
        slow_phase,
        fast_phase,
        get_slow_params,
        get_fast_params,
        get_slow_state,
        get_fast_state,
        lr,
        training=True,
        loss_fn=mean_xe_and_acc_dict,
        init_opt_state=sgd_init_opt,
        opt_update=sgd_opt_update,
    ):
        self.params = params
        self.state = state
        self.apply = apply
        self.training = training
        self.lr = lr
        self.__slow_apply = jax.partial(apply, phase=slow_phase)
        self.__fast_apply = jax.partial(apply, phase=fast_phase)
        self.get_sp = get_slow_params
        self.get_fp = get_fast_params
        self.get_ss = get_slow_state
        self.get_fs = get_fast_state
        self.loss_fn = loss_fn
        self.init_opt_state = init_opt_state
        self.opt_update = opt_update

    @use_self_as_default("params", "state", "training")
    def fast_apply(
        self,
        x,
        rng=None,
        params=None,
        state=None,
        training=None,
        fast_params=None,
        fast_state=None,
    ):
        if fast_params is not None:
            params = merge(params, fast_params)
        if fast_state is not None:
            state = merge(state, fast_state)
        return self.__fast_apply(params, state, rng, x, training=training)

    @use_self_as_default("params", "state", "training")
    def slow_apply(self, x, rng=None, params=None, state=None, training=None):
        return self.__slow_apply(params, state, rng, x, training=training)

    def bmap_slow_apply(self, x, *args, **kwargs):
        s1, s2 = first_leaf_shape(x)[:2] # Get shape of first leaf of x
        x = jax.tree_map(lambda t: t.reshape(s1 * s2, *t.shape[2:]), x)
        slow_outputs, slow_state = self.slow_apply(x, *args, **kwargs)
        slow_outputs = jax.tree_map(
            lambda t: t.reshape(s1, s2, *t.shape[1:]), slow_outputs
        )
        return slow_outputs, slow_state

    @use_self_as_default("loss_fn")
    def fast_apply_and_loss(
        self,
        x,
        y,
        rng=None,
        params=None,
        state=None,
        training=None,
        fast_params=None,
        fast_state=None,
        loss_fn=None,
    ):
        outputs, new_state = self.fast_apply(
            x,
            rng=rng,
            params=params,
            state=state,
            training=training,
            fast_params=fast_params,
            fast_state=fast_state,
        )
        loss, loss_aux = loss_fn(outputs, y)
        return loss, (new_state, loss_aux, outputs)

    def __single_fast_step(
        self,
        x,
        y,
        rng,
        params,
        state,
        fast_params,
        fast_state,
        lr,
        training,
        loss_fn,
        init_opt_state,
        opt_update,
        opt_state,
    ):
        if fast_params is None:
            fast_params = self.get_fp(params)
        if opt_state is None:
            opt_state = init_opt_state(fast_params)

        (loss, (new_state, loss_aux, outputs)), grads = jax.value_and_grad(
            lambda _fast_params: self.fast_apply_and_loss(
                x,
                y,
                params=params,
                state=state,
                rng=rng,
                fast_params=_fast_params,
                fast_state=fast_state,
                loss_fn=loss_fn,
                training=training,
            ),
            has_aux=True,
        )(fast_params)
        updates, opt_state = opt_update(
            lr,
            grads,
            opt_state,
            fast_params,
        )
        fast_params = ox.apply_updates(fast_params, updates)
        return fast_params, new_state, opt_state, (loss, loss_aux, outputs)

    @use_self_as_default("params", "lr", "init_opt_state", "opt_update")
    def single_fast_step(
        self,
        x,
        y,
        rng=None,
        params=None,
        state=None,
        fast_params=None,
        fast_state=None,
        lr=None,
        training=None,
        loss_fn=None,
        init_opt_state=None,
        opt_update=None,
        opt_state=None,
    ):
        kwargs = {}
        if rng is not None:
            kwargs["rng"] = split(rng, first_leaf_shape(y)[0])
        if fast_params is not None:
            kwargs["fast_params"] = fast_params
        if fast_state is not None:
            kwargs["fast_state"] = fast_state

        def fast_apply_and_loss_helper(x, y, kwargs={}):
            return self.__single_fast_step(
                x,
                y,
                rng=kwargs.get("rng", None),
                params=params,
                state=state,
                fast_params=kwargs.get("fast_params", None),
                fast_state=kwargs.get("fast_state", None),
                lr=lr,
                training=training,
                loss_fn=loss_fn,
                init_opt_state=init_opt_state,
                opt_update=opt_update,
                opt_state=opt_state,
            )

        return jax.vmap(fast_apply_and_loss_helper)(x, y, kwargs)

    def single_step(
        self,
        x,
        y,
        rng=None,
        params=None,
        state=None,
        fast_params=None,
        fast_state=None,
        lr=None,
        training=None,
        loss_fn=None,
        init_opt_state=None,
        opt_update=None,
        opt_state=None,
    ):
        slow_outputs, slow_state = self.bmap_slow_apply(
            x,
            rng=rng,
            params=params,
            state=state,
            training=training,
        )

        (
            fast_params,
            fast_state,
            opt_state,
            (loss, loss_aux, outputs),
        ) = self.single_fast_step(
            slow_outputs,
            y,
            rng=rng,
            params=params,
            state=slow_state,
            fast_params=fast_params,
            fast_state=fast_state,
            lr=lr,
            training=training,
            loss_fn=loss_fn,
            init_opt_state=init_opt_state,
            opt_update=opt_update,
            opt_state=opt_state,
        )

        return (fast_params, slow_state, fast_state, opt_state), (
            loss,
            loss_aux,
            outputs,
        )


class _BaseContinualLearner:
    def __init__(
        self,
        apply,
        model_class,
        params=None,
        state=None,
        lr=None,
        training=True,
        train_init_opt=sgd_init_opt,
        test_init_opt=None,
        train_opt_update=sgd_opt_update,
        test_opt_update=None,
        loss_fn=mean_xe_and_acc_dict,
        bmap=True,  # Use hk.BatchApply instead of jax.vmap
    ):
        if test_init_opt is None:
            test_init_opt = train_init_opt
        if test_opt_update is None:
            test_opt_update = train_opt_update

        # super().__init__(*args, **kwargs)
        self.apply = apply
        self.model_class = model_class
        self.params = params
        self.state = state
        self.lr = lr
        self.training = training
        self.loss_fn = loss_fn
        self.train_init_opt = train_init_opt
        self.test_init_opt = test_init_opt
        self.train_opt_update = train_opt_update
        self.test_opt_update = test_opt_update
        self.bmap = bmap

        self.train_slow_phase = self.model_class.train_slow_phase
        self.train_fast_phase = self.model_class.train_fast_phase
        self.test_slow_phase = self.model_class.test_slow_phase
        self.test_fast_phase = self.model_class.test_fast_phase

        self.get_train_sp = self.model_class.get_train_slow_params
        self.get_train_fp = self.model_class.get_train_fast_params
        self.get_test_sp = self.model_class.get_test_slow_params
        self.get_test_fp = self.model_class.get_test_slow_params

        self.get_train_ss = self.model_class.get_train_slow_state
        self.get_train_fs = self.model_class.get_train_fast_state
        self.get_test_ss = self.model_class.get_test_slow_state
        self.get_test_fs = self.model_class.get_test_fast_state

    # @use_self_as_default("training")
    def init_opt_state(self, training):
        if training:
            return self.train_init_opt
        return self.test_init_opt

    # @use_self_as_default("training")
    def opt_update(self, training):
        if training:
            return self.train_opt_update
        return self.test_opt_update

    # @use_self_as_default("training")
    def get_phases(self, training=None):
        if training:
            return self.train_slow_phase, self.train_fast_phase
        return self.test_slow_phase, self.test_fast_phase

    # @use_self_as_default("params", "training")
    def get_sp(self, params=None, training=None):
        if training:
            return self.get_train_sp(params)
        return self.get_test_sp(params)

    # @use_self_as_default("params", "training")
    def get_fp(self, params=None, training=None):
        if training:
            return self.get_train_fp(params)
        return self.get_test_fp(params)

    # @use_self_as_default("state", "training")
    def get_ss(self, state=None, training=None):
        if training:
            return self.get_train_ss(state)
        return self.get_test_ss(state)

    # @use_self_as_default("state", "training")
    def get_fs(self, state=None, training=None):
        if training:
            return self.get_train_fs(state)
        return self.get_test_fs(state)

    # @use_self_as_default("training"):
    def slow_phase(self, training):
        if training:
            return self.train_slow_phase
        return self.test_slow_phase

    # @use_self_as_default("training"):
    def fast_phase(self, training):
        if training:
            return self.train_fast_phase
        return self.test_fast_phase

    @use_self_as_default(
        "params", "state", "training", "loss_fn", fast_phase=["training"]
    )
    def fast_apply_and_loss(
        self,
        x,
        y,
        params=None,
        state=None,
        rng=None,
        fast_params=None,
        fast_state=None,
        training=None,
        loss_fn=None,
        fast_phase=None,
    ):
        if fast_params is not None:
            params = merge(params, fast_params)
        if fast_state is not None:
            state = merge(state, fast_state)
        outputs, new_state = self.apply(
            params,
            state,
            rng,
            x,
            phase=fast_phase,
            training=training,
        )
        loss, loss_aux = loss_fn(outputs, y)
        return loss, (self.get_fs(new_state), loss_aux, outputs)

    @use_self_as_default(
        "params", "state", "training", slow_phase=["training"], fast_phase=["training"]
    )
    def apply_and_loss(
        self,
        x,
        y,
        params=None,
        state=None,
        rng=None,
        training=None,
        loss_fn=None,
        slow_phase=None,
        fast_phase=None,
    ):
        if rng is None:
            rng_slow = rng_fast = None
        else:
            rng_slow, rng_fast = split(rng)

        outputs, slow_state = self.apply(
            params,
            state,
            rng_slow,
            x,
            phase=slow_phase,
            training=training,
        )
        loss, (fast_state, *others) = self.fast_apply_and_loss(
            outputs,
            y,
            params,
            state,
            rng_fast,
            training,
            loss_fn,
            fast_phase,
        )

        return loss, (merge(slow_state, fast_state), *others)

    @use_self_as_default(
        "training",
        "lr",
        init_opt_state=["training"],
        opt_update=["training"],
        fast_phase=["training"],
    )
    def single_fast_step(
        self,
        x,
        y,
        params=None,
        state=None,
        rng=None,
        fast_params=None,
        fast_state=None,
        lr=None,
        training=None,
        loss_fn=None,
        init_opt_state=None,
        opt_update=None,
        opt_state=None,
        fast_phase=None,
    ):
        if fast_params is None:
            fast_params = self.get_fp(params, training)
        if opt_state is None:
            opt_state = init_opt_state(fast_params)

        (loss, (fast_state, loss_aux, outputs)), grads = jax.value_and_grad(
            self.fast_apply_and_loss,
            argnums=5,
            has_aux=True,
        )(
            x,
            y,
            params,
            state,
            rng,
            fast_params,
            fast_state,
            training,
            loss_fn,
            fast_phase,
        )
        updates, opt_state = opt_update(
            lr,
            grads,
            opt_state,
            fast_params,
        )
        fast_params = ox.apply_updates(fast_params, updates)
        return fast_params, fast_state, opt_state, (loss, loss_aux, outputs)

    @use_self_as_default(
        "params",
        "state",
        "training",
        slow_phase=["training"],
    )
    def slow_apply(
        self,
        x,
        params=None,
        state=None,
        rng=None,
        training=None,
        slow_phase=None,
    ):
        return self.apply(
            params,
            state,
            rng,
            x,
            phase=slow_phase,
            training=training,
        )

    def bmap_slow_apply(
        self,
        x,
        *args,
        **kwargs,
    ):
        s1, s2 = x.shape[:2]  # Batch size and number of samples per batch
        x = jax.tree_map(lambda _x: _x.reshape(s1 * s2, *_x.shape[2:]), x)
        slow_outputs, slow_state = self.slow_apply(
            x,
            *args,
            **kwargs,
        )
        slow_outputs = jax.tree_map(
            lambda _x: _x.reshape(s1, s2, *_x.shape[1:]), slow_outputs
        )
        return slow_outputs, slow_state

    def vmap_slow_apply(
        self,
        x,
        params,
        state,
        rng,
        *args,
        **kwargs,
    ):
        if rng is None:
            if state is None:
                return jax.vmap(
                    lambda _x: self.slow_apply(_x, params, state, rng, *args, **kwargs)
                )(x)
            else:
                return jax.vmap(
                    lambda _x, _state: self.slow_apply(
                        _x, params, _state, rng, *args, **kwargs
                    )
                )(x, state)
        else:
            bsz = x.shape[0]
            rng = split(rng, bsz)
            if state is None:
                return jax.vmap(
                    lambda _x, _rng: self.slow_apply(
                        _x, params, state, _rng, *args, **kwargs
                    )
                )(x, rng)
            else:
                return jax.vmap(
                    lambda _x, _state, _rng: self.slow_apply(
                        _x, params, _state, _rng, *args, **kwargs
                    )
                )(x, state, rng)

    @use_self_as_default(
        "params",
        "state",
        "training",
        init_opt_state=["training"],
    )
    def bmap_inner_loop(
        self,
        x_spt,
        y_spt,
        params=None,
        state=None,
        rng=None,
        fast_params=None,
        fast_state=None,
        training=None,
        opt_state=None,
        lr=None,
        loss_fn=None,
        slow_phase=None,
        fast_phase=None,
        init_opt_state=None,
        opt_update=None,
        counter=None,
    ):
        if rng is None:
            rng_slow = rng_fast = None
        else:
            rng_slow, rng_fast = split(rng)

        slow_outputs, slow_state = self.bmap_slow_apply(
            x_spt, params, state, rng_slow, training, slow_phase
        )

        if fast_state is None:
            fast_state = expand(self.get_fs(slow_state, training), len(x_spt))

        return self._inner_loop_fast(
            expand(slow_outputs, 1, 2),
            expand(y_spt, 1, 2),
            params,
            slow_state,
            rng_fast,
            fast_params,
            fast_state,
            training,
            opt_state,
            lr,
            loss_fn,
            fast_phase,
            init_opt_state,
            opt_update,
            counter,
        )

    def inner_loop(self, *args, **kwargs):
        if self.bmap:
            return self.bmap_inner_loop(*args, **kwargs)

    def _inner_loop_fast(
        self,
        x,
        y,
        params,
        state,
        rng,
        fast_params,
        fast_state,
        training,
        opt_state,
        lr,
        loss_fn,
        fast_phase,
        init_opt_state,
        opt_update,
        counter,
    ):
        def scan_fun(carry, xs):
            _fast_params = carry.get("fast_params", None)
            _fast_state = carry.get("fast_state", None)
            _opt_state = carry.get("opt_state", None)
            if len(xs) == 2:
                _x, _y = xs
                _rng = None
            else:
                _x, _y, _rng = xs
            (
                fast_params,
                fast_state,
                opt_state,
                (loss, loss_aux, outputs),
            ) = self.single_fast_step(
                _x,
                _y,
                params,
                state,
                _rng,
                _fast_params,
                _fast_state,
                lr,
                training,
                loss_fn,
                init_opt_state,
                opt_update,
                _opt_state,
                fast_phase,
            )

            out = {
                "fast_params": fast_params,
                "fast_state": fast_state,
                "opt_state": opt_state,
                "counter": carry["counter"] + 1,
            }

            return out, (loss, loss_aux, outputs)

        if rng is not None:
            rng = (split(rng, len(x)),)
        else:
            rng = tuple()

        xss = (
            x,
            y,
            *rng,
        )

        if counter is None:
            counter = jnp.zeros([])
        if fast_params is None:
            fast_params = self.get_fp(params, training)
        if opt_state is None:
            opt_state = init_opt_state(fast_params)

        carries = expand(
            {
                "counter": counter,
                "fast_params": fast_params,
                "opt_state": opt_state,
            },
            y.shape[0],
        )

        carries["fast_state"] = fast_state

        last_carry, (loss, loss_aux, outputs) = jax.vmap(
            lambda carry, xs: jax.lax.scan(scan_fun, carry, xs)
        )(carries, xss)

        return last_carry, (loss, loss_aux, outputs)