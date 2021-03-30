import jax
from haiku.data_structures import merge

from utils import use_self_as_default
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
    def _slow_apply(self, x, rng=None, params=None, state=None, training=None):
        return self.__slow_apply(params, state, rng, x, training=training)

    def slow_apply(self, *args, **kwargs):
        raise NotImplementedError

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

    def apply(
        self,
        x,
        rng,
        params,
        state,
        training,
        fast_params,
        fast_state,
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
        rng,
        params,
        state,
        training,
        fast_params,
        fast_state,
        loss_fn,
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
            rng,
            params,
            slow_state,
            training,
            fast_params,
            fast_state,
            loss_fn,
        )

        return dict(
            loss=loss,
            fast_state=new_state,
            loss_aux=loss_aux,
            outputs=outputs,
            slow_outputs=slow_outputs,
            slow_state=slow_state,
        )
