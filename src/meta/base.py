import jax
from jax.random import split
from haiku.data_structures import merge

from utils import use_self_as_default, first_leaf_shape
from utils.losses import mean_xe_and_acc_dict


class MetaBase:
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
        lr=None,
        training=True,
        loss_fn=mean_xe_and_acc_dict,
    ):
        self.__apply = apply
        self.params = params
        self.state = state
        self.slow_phase = slow_phase
        self.fast_phase = fast_phase
        self.lr = lr
        self.__slow_apply = jax.partial(apply, phase=slow_phase)
        self.__fast_apply = jax.partial(apply, phase=fast_phase)
        self.get_slow_params = get_slow_params
        self.get_fast_params = get_fast_params
        self.get_slow_state = get_slow_state
        self.get_fast_state = get_fast_state
        self.loss_fn = loss_fn
        self.training = training

    @property
    def get_sp(self):
        return self.get_slow_params

    @property
    def get_fp(self):
        return self.get_fast_params

    @property
    def get_ss(self):
        return self.get_slow_state

    @property
    def get_fs(self):
        return self.get_fast_state

    @use_self_as_default("params", "state", "training")
    def _fast_apply(
        self,
        x,
        y=None,
        rng=None,
        params=None,
        state=None,
        training=None,
        fast_params=None,
        fast_state=None,
        loss_fn=None,
    ):
        if fast_params is not None:
            params = merge(params, fast_params)
        if fast_state is not None:
            state = merge(state, fast_state)
        outputs, new_state = self.__fast_apply(params, state, rng, x, training=training)
        if loss_fn is not None:
            loss, loss_aux = loss_fn(outputs, y)
            return loss, (new_state, loss_aux, outputs)
        else:
            return outputs, new_state

    # @use_self_as_default("loss_fn")
    def _fast_apply_and_loss(
        self,
        *args,
        loss_fn=None,
        **kwargs,
    ):
        # Decorator does not work with this arguments style
        if loss_fn is None:
            loss_fn = self.loss_fn
        assert loss_fn is not None, "Unspecified loss_fn"
        return self._fast_apply(
            *args,
            loss_fn=loss_fn,
            **kwargs,
        )

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
        alt=False,
    ):
        def f(_x=x, _y=y, _rng=None, _fast_params=None, _fast_state=None):
            return self._fast_apply_and_loss(
                _x,
                _y,
                rng=_rng,
                params=params,
                state=state,
                training=training,
                loss_fn=loss_fn,
                fast_params=_fast_params,
                fast_state=_fast_state,
            )

        kwargs = {}
        if rng is not None:
            if not alt:
                kwargs["_rng"] = split(rng, first_leaf_shape(x)[0])
            else:
                kwargs["_rng"] = split(rng, first_leaf_shape(fast_params)[0])
        if fast_params is not None:
            kwargs["_fast_params"] = fast_params
        if fast_state is not None:
            kwargs["_fast_state"] = fast_state
        if not alt:
            kwargs["_x"] = x
            kwargs["_y"] = y

        loss, (new_state, loss_aux, outputs) = jax.vmap(f)(**kwargs)
        return loss, (new_state, loss_aux, outputs)

    @use_self_as_default("params", "state", "training")
    def _slow_apply(self, x, rng=None, params=None, state=None, training=None):
        return self.__slow_apply(params, state, rng, x, training=training)

    def slow_apply(self, *args, **kwargs):
        raise NotImplementedError

    @use_self_as_default("loss_fn")
    def old_fast_apply_and_loss(
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

    def apply(
        self,
        x,
        rng=None,
        params=None,
        state=None,
        training=None,
        fast_params=None,
        fast_state=None,
    ):
        slow_outputs, slow_state = self.slow_apply(
            x,
            rng,
            params,
            state,
            training,
        )
        return self.fast_apply(
            slow_outputs,
            rng,
            params,
            slow_state,
            training,
            fast_params,
            fast_state,
        )

    def apply_and_loss(
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
        slow_outputs, slow_state = self.slow_apply(
            x,
            rng,
            params,
            state,
            training,
        )

        loss, (new_state, loss_aux, outputs) = self.fast_apply_and_loss(
            slow_outputs,
            y,
            rng=rng,
            params=params,
            state=slow_state,
            training=training,
            fast_params=fast_params,
            fast_state=fast_state,
            loss_fn=loss_fn,
        )

        return dict(
            loss=loss,
            fast_state=new_state,
            loss_aux=loss_aux,
            outputs=outputs,
            slow_outputs=slow_outputs,
            slow_state=slow_state,
        )
