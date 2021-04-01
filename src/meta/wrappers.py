import jax
import jax.numpy as jnp
from jax.random import split
from haiku.data_structures import merge
import optax as ox

from .base import MetaBase
from utils import (
    use_self_as_default,
    expand,
    first_leaf_shape,
    split_rng_or_none,
)


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


class MetaLearnerBase(MetaBase):
    def __init__(
        self,
        *args,
        init_opt_state=sgd_init_opt,
        opt_update=sgd_opt_update,
        **kwargs,
        # self,
        # apply,
        # params,
        # state,
        # slow_phase,
        # fast_phase,
        # get_slow_params,
        # get_fast_params,
        # get_slow_state,
        # get_fast_state,
        # lr,
        # training=True,
        # loss_fn=mean_xe_and_acc_dict,
    ):
        super().__init__(*args, **kwargs)
        # self.params = params
        # self.state = state
        # self.apply = apply
        # self.training = training
        # self.lr = lr
        # self.__slow_apply = jax.partial(apply, phase=slow_phase)
        # self.__fast_apply = jax.partial(apply, phase=fast_phase)
        # self.get_slow_params = get_slow_params
        # self.get_fast_params = get_fast_params
        # self.get_slow_state = get_slow_state
        # self.get_fast_state = get_fast_state
        # self.loss_fn = loss_fn
        self.init_opt_state = init_opt_state
        self.opt_update = opt_update

    # @use_self_as_default("params", "state", "training")
    # def fast_apply(
    #     self,
    #     x,
    #     rng=None,
    #     params=None,
    #     state=None,
    #     training=None,
    #     fast_params=None,
    #     fast_state=None,
    # ):
    #     if fast_params is not None:
    #         params = merge(params, fast_params)
    #     if fast_state is not None:
    #         state = merge(state, fast_state)
    #     return self.__fast_apply(params, state, rng, x, training=training)

    # @use_self_as_default("params", "state", "training")
    # def _slow_apply(self, x, rng=None, params=None, state=None, training=None):
    #     return self.__slow_apply(params, state, rng, x, training=training)

    # @use_self_as_default("loss_fn")
    # def fast_apply_and_loss(
    #     self,
    #     x,
    #     y,
    #     rng=None,
    #     params=None,
    #     state=None,
    #     training=None,
    #     fast_params=None,
    #     fast_state=None,
    #     loss_fn=None,
    # ):
    #     outputs, new_state = self.fast_apply(
    #         x,
    #         rng=rng,
    #         params=params,
    #         state=state,
    #         training=training,
    #         fast_params=fast_params,
    #         fast_state=fast_state,
    #     )
    #     loss, loss_aux = loss_fn(outputs, y)
    #     return loss, (new_state, loss_aux, outputs)

    @use_self_as_default("params", "lr", "init_opt_state", "opt_update")
    def _single_fast_step(
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
        if fast_params is None:
            fast_params = self.get_fp(params)
            # fast_params = expand(self.get_fp(params), first_leaf_shape(x)[0])
        if opt_state is None:
            opt_state = init_opt_state(fast_params)

        (loss, (new_state, loss_aux, outputs)), grads = jax.value_and_grad(
            lambda _fast_params: self._fast_apply_and_loss(
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
            return self._single_fast_step(
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
        slow_outputs, slow_state = self.slow_apply(
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


class MetaLearnerBaseB(MetaLearnerBase):
    # @property
    # def get_sp(self):
    #     return self.get_slow_params

    # @property
    # def get_fp(self):
    #     return self.get_fast_params

    # @property
    # def get_ss(self):
    #     return self.get_slow_state

    # @property
    # def get_fs(self):
    #     return self.get_fast_state

    def slow_apply(self, x, *args, **kwargs):
        s1, s2 = first_leaf_shape(x)[:2]  # Get shape of first leaf of x
        x = jax.tree_map(lambda t: t.reshape(s1 * s2, *t.shape[2:]), x)
        slow_outputs, slow_state = self._slow_apply(x, *args, **kwargs)
        slow_outputs = jax.tree_map(
            lambda t: t.reshape(s1, s2, *t.shape[1:]), slow_outputs
        )
        return slow_outputs, slow_state


class MetaLearnerBaseV(MetaLearnerBase):
    def slow_apply(self, x, *args, **kwargs):
        # TODO implement slow apply with vmap instead of flatten and unflatten
        pass


class ContinualLearnerB(MetaLearnerBaseB):
    @use_self_as_default("init_opt_state")
    def inner_loop(
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
        init_opt_state=None,
        opt_update=None,
        counter=None,
    ):
        bsz, traj_length = first_leaf_shape(x_spt)[:2]
        rng_slow, rng_fast = split_rng_or_none(rng)
        slow_outputs, slow_state = self.slow_apply(
            x_spt,
            rng=rng_slow,
            params=params,
            state=state,
            training=training,
        )

        if fast_params is None:
            fast_params = expand(self.get_fp(params or self.params), bsz)
        if fast_state is None:
            fast_state = expand(
                self.get_fs(merge(state or self.state, slow_state)), bsz
            )
        if opt_state is None:
            opt_state = init_opt_state(fast_params)

        def scan_fun(carry, xs):
            nonlocal rng_fast
            (fast_params, fast_state, opt_state) = carry
            if rng_fast is None:
                _rng = None
                x, y = xs
            else:
                x, y, _rng = xs
            (
                fast_params,
                new_state,
                opt_state,
                (loss, loss_aux, outputs),
            ) = self.single_fast_step(
                x,
                y,
                rng=_rng,
                params=params,
                state=merge(state or self.state, slow_state),
                fast_params=fast_params,
                fast_state=fast_state,
                lr=lr,
                training=training,
                loss_fn=loss_fn,
                opt_update=opt_update,
                init_opt_state=init_opt_state,
                opt_state=opt_state,
            )
            out_carry = (fast_params, self.get_fs(new_state), opt_state)

            return out_carry, (loss, loss_aux, outputs)

        # Tranpose inputs to perform scan over trajectory
        scan_inputs = [slow_outputs, y_spt]
        scan_inputs = jax.tree_map(
            lambda x: jnp.transpose(x, (1, 0, *jnp.arange(2, len(x.shape)))),
            scan_inputs,
        )
        scan_inputs = expand(scan_inputs, 1, 2)
        if rng_fast is not None:
            scan_inputs.append(split(rng, traj_length))

        (fast_params, fast_state, opt_state), aux = jax.lax.scan(
            scan_fun, (fast_params, fast_state, opt_state), scan_inputs
        )
        aux = jax.tree_map(
            lambda x: jnp.transpose(x, (1, 0, *jnp.arange(2, len(x.shape)))),
            aux,
        )
        (loss, loss_aux, outputs) = aux
        outputs = jnp.squeeze(outputs, 2)

        return dict(
            fast_params=fast_params,
            slow_state=slow_state,
            fast_state=fast_state,
            opt_state=opt_state,
            loss=loss,
            loss_aux=loss_aux,
            slow_outputs=slow_outputs,
            outputs=outputs,
        )