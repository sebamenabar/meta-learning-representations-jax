import jax
from jax import numpy as jnp
from jax.random import split
import optax as ox
from haiku.data_structures import merge
from utils import tree_shape, use_self_as_default, tree_flatten_array, split_rng_or_none
from .wrappers import MetaLearnerBaseB

class MetaTrainerB(MetaLearnerBaseB):
    def __init__(
        self,
        learner,
        *args,
        opt_state=None,
        inner_lr=None,
        augmentation="none",
        augmentation_fn=None,
        alt=True,
        train_lr=False,
        optimizer=None,
        scheduler=None,
        reset_fast_params=None,
        reset_before_outer_loop=True,
        cross_replica_axis=None,
        include_spt=False,
        **kwargs,
        # inner_lr, training, loss_fn, init_opt_state, opt_update
    ):
        super().__init__(*args, **kwargs)
        self.learner = learner
        self.alt = alt
        self.augmentation = augmentation
        self.augmentation_fn = augmentation_fn
        self.train_lr = train_lr
        self.inner_lr = jnp.ones([]) * inner_lr
        self.reset_fast_params = reset_fast_params
        self.reset_before_outer_loop = reset_before_outer_loop
        self.cross_replica_axis = cross_replica_axis
        self.include_spt = include_spt
        self.opt_state = opt_state
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.fast_state = None

        update = jax.partial(
            lambda rng, step_num, x_spt, y_spt, x_qry, y_qry, x_qry_cl, y_qry_cl, spt_classes, params, state, opt_state, fast_state=None: self._update(
                rng,
                step_num,
                x_spt,
                y_spt,
                x_qry,
                y_qry,
                x_qry_cl,
                y_qry_cl,
                spt_classes,
                params,
                state,
                opt_state=opt_state,
                fast_state=fast_state,
            )
        )
        
        if self.cross_replica_axis is not None:
            self.update = jax.pmap(update, axis_name="i")
        else:
            self.update = jax.jit(update)

        # if self.train_lr:
        #     self.params = (self.params, self.inner_lr)
        # else:
        #     self.params = self.params,

    def initialize_opt_state(self):
        if self.train_lr:
            params = (self.params, self.inner_lr)
        else:
            params = (self.params,)
        if self.cross_replica_axis is not None:
            opt_state = jax.pmap(self.optimizer.init)(params)
        else:
            opt_state = self.optimizer.init(params)
        self.opt_state = opt_state
        return self

    def replicate_state(self):
        # self.params = jax.device_put_replicated(self.params, jax.local_devices())
        # self.state = jax.device_put_replicated(self.state, jax.local_devices())
        if self.train_lr:
            self.inner_lr = jax.device_put_replicated(
                self.inner_lr, jax.local_devices()
            )
        # self.opt_state = jax.device_put_replicated(self.opt_state, jax.local_devices())
        return self

    def apply_augmentation(self, rng, _input):
        if self.augmentation_fn is not None:
            return self.augmentation_fn(rng, _input)
        return _input

    def augments(self, rng, x_spt, x_qry, x_qry_cl):
        rng = split(rng, 3)
        if self.augmentation in ["spt", "all"]:
            x_spt = self.apply_augmentation(rng[0], x_spt)
        if self.augmentation in ["qry", "all"]:
            x_qry = self.apply_augmentation(rng[0], x_qry)
            x_qry_cl = self.apply_augmentation(rng[0], x_qry_cl)

        return x_spt, x_qry, x_qry_cl

    def step(
        self,
        rng,
        step_num,
        x_spt,
        y_spt,
        x_qry,
        y_qry,
        x_qry_cl,
        y_qry_cl,
        spt_classes=None,
    ):
        # spt_classes = onp.unique(y_spt, axis=-1)

        if self.cross_replica_axis is not None:
            rng = split(rng, jax.local_device_count())
            step_num = jax.device_put_replicated(step_num, jax.local_devices())
            if spt_classes is None:
                spt_classes = [None] * jax.local_device_count()

        if self.train_lr:
            params = (self.params, self.inner_lr)
        else:
            params = self.params

        if self.fast_state is not None:
            args = (self.fast_state,)
        else:
            args = tuple()

        loss, params, opt_state, out = self.update(
            rng,
            step_num,
            x_spt,
            y_spt,
            x_qry,
            y_qry,
            x_qry_cl,
            y_qry_cl,
            spt_classes,
            params,
            self.state,
            self.opt_state,
            *args,
        )
        if self.train_lr:
            self.params = params[0]
            self.inner_lr = params[1]
        else:
            self.params = params
        self.state = merge(
            out["initial_outer_out"]["slow_state"],
            # out["inner_out"]["fast_state"],
        )
        self.fast_state = out["inner_out"]["fast_state"]
        self.opt_state = opt_state

        return loss, out, self.inner_lr

    @use_self_as_default(
        "training",
        # "init_opt_state",
        "reset_fast_params",
        "reset_before_outer_loop",
        "include_spt",
        "scheduler",
        "inner_lr",
        "loss_fn",
        # "opt_update",
        "optimizer",
    )
    def _update(
        self,
        rng,
        step_num,
        x_spt,
        y_spt,
        x_qry,
        y_qry,
        x_qry_cl,
        y_qry_cl,
        spt_classes,
        params,
        state,
        training=None,
        loss_fn=None,
        opt_state=None,
        inner_lr=None,
        # init_opt_state=None,
        # opt_update=None,
        optimizer=None,
        alt=None,
        reset_fast_params=None,
        reset_before_outer_loop=None,
        include_spt=None,
        scheduler=None,
        inner_opt_state=None,
        inner_init_opt_state=None,
        inner_opt_update=None,
        fast_state=None,
    ):
        
        rng_data, rng_reset, rng_pre, rng = split(rng, 4)

        # inputs come as uint8 for speedier transfer
        x_spt, x_qry, x_qry_cl = self.augments(
            rng_data, x_spt / 255, x_qry / 255, x_qry_cl / 255
        )

        if self.train_lr:
            _params = params
        else:
            _params = (params,)

        if opt_state is None:
            # opt_state = self.init_opt_state(params)
            opt_state = optimizer.init(params)

        if reset_before_outer_loop and reset_fast_params:
            params = (reset_fast_params(rng_reset, params[0], spt_classes), *params[1:])

        # _, pre_slow_state = self.slow_apply(
        #     x_qry_cl,
        #     rng_pre,
        #     params,
        #     state,
        #     training,
        # )

        if include_spt:
            x_qry = jnp.concatenate((x_spt, x_qry, x_qry_cl))
            y_qry = jnp.concatenate((y_spt, y_qry, y_qry_cl))
        else:
            x_qry = jnp.concatenate((x_qry, x_qry_cl))
            y_qry = jnp.concatenate((y_qry, y_qry_cl))

        def helper(_params, _lr=inner_lr):
            loss, out = self.outer_loss(
                x_spt,
                y_spt,
                x_qry,
                y_qry,
                spt_classes=spt_classes,
                rng=rng,
                params=_params,
                # state=pre_slow_state,
                state=state,
                training=training,
                loss_fn=loss_fn,
                opt_state=inner_opt_state,
                lr=_lr,
                init_opt_state=inner_init_opt_state,
                opt_update=inner_opt_update,
                alt=alt,
                reset_fast_params=reset_fast_params,
                reset_before_outer_loop=reset_before_outer_loop,
                fast_state=fast_state,
            )
            if self.cross_replica_axis is not None:
                loss = jax.lax.pmean(loss, self.cross_replica_axis)
            else:
                loss = jnp.mean(loss)
            return jnp.mean(loss), out

        (loss, out), grads = jax.value_and_grad(
            helper, has_aux=True, argnums=tuple(range(len(_params)))
        )(*_params)
        if self.cross_replica_axis is not None:
            grads = jax.lax.pmean(grads, self.cross_replica_axis)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        if self.scheduler is not None:
            updates = self.scheduler(updates)
        params = ox.apply_updates(params, updates)

        return loss, params, opt_state, out

    @use_self_as_default(
        "alt",
        "params",
        "state",
        "training",
        # "lr",
        "loss_fn",
        # "init_opt_state",
        # "opt_update",
        "reset_fast_params",
        "reset_before_outer_loop",
    )
    def outer_loss(
        self,
        x_spt,
        y_spt,
        x_qry,
        y_qry,
        spt_classes=None,
        rng=None,
        params=None,
        state=None,
        training=None,
        loss_fn=None,
        opt_state=None,
        lr=None,
        init_opt_state=None,
        opt_update=None,
        alt=None,
        reset_fast_params=None,
        reset_before_outer_loop=None,
        fast_params=None,
        fast_state=None,
    ):
        rng_inner, rng_outer_slow, rng_outer_fast, rng_reset = split_rng_or_none(rng, 4)

        if (lr is not None) and (not self.train_lr):
            lr = jax.lax.stop_gradient(lr)

        if (spt_classes is not None) and reset_before_outer_loop and reset_fast_params:
            params = reset_fast_params(rng_reset, params, spt_classes)

        slow_outputs, slow_state = self.learner.slow_apply(
            x_qry,
            rng_outer_slow,
            params,
            state,
            training,
        )

        loss, (new_state, loss_aux, outputs) = self.learner.fast_apply_and_loss(
            slow_outputs,
            y_qry,
            rng=rng_outer_fast,
            params=params,
            state=slow_state,
            training=training,
            loss_fn=loss_fn,
            # alt=alt,
        )
        initial_outer_out = dict(
            loss=loss,
            slow_state=slow_state,
            fast_state=new_state,
            loss_aux=loss_aux,
            outputs=outputs,
        )

        inner_out = self.learner.inner_loop(
            x_spt,
            y_spt,
            params,
            slow_state,
            rng_inner,
            training=training,
            opt_state=opt_state,
            lr=lr,
            loss_fn=loss_fn,
            init_opt_state=init_opt_state,
            opt_update=opt_update,
            fast_params=fast_params,
            fast_state=fast_state,
        )

        if alt:
            slow_outputs, y_qry = tree_flatten_array((slow_outputs, y_qry))

        loss, (new_state, loss_aux, outputs) = self.learner.fast_apply_and_loss(
            slow_outputs,
            y_qry,
            rng=split(rng_outer_fast)[0],
            params=params,
            state=slow_state,
            training=training,
            fast_params=inner_out["fast_params"],
            fast_state=inner_out["fast_state"],
            loss_fn=loss_fn,
            alt=alt,
        )
        outer_out = dict(
            loss=loss, fast_state=new_state, loss_aux=loss_aux, outputs=outputs
        )

        return (outer_out["loss"]), dict(
            inner_out=inner_out,
            outer_out=outer_out,
            initial_outer_out=initial_outer_out,
        )