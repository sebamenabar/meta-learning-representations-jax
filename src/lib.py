import numpy as onp
import jax
from jax.tree_util import Partial as partial
from jax import jit, numpy as jnp, value_and_grad, vmap

from jax.experimental import optix


def flatten(array, dims=None):
    shape = array.shape
    if dims is None:
        return array.reshape(-1)
    elif isinstance(dims, tuple):
        assert (0 <= dims[0] < len(shape)) and (0 <= dims[1] < len(shape))
        final_shape = (
            *shape[: dims[0]],
            onp.prod(shape[dims[0] : dims[1] + 1]),
            *shape[dims[1] + 1 :],
        )
        return array.reshape(final_shape)
    else:
        assert 0 <= dims < len(shape)
        final_shape = (onp.prod(shape[: dims + 1]), *shape[dims + 1 :])
        return array.reshape(final_shape)


def setup_device(gpus=0, default_platform="cpu"):
    if gpus != 0:
        gpu = jax.devices("gpu")[0]
    else:
        gpu = None
        default_platform = "cpu"
    cpu = jax.devices("cpu")[0]
    jax.config.update("jax_platform_name", default_platform)

    return cpu, gpu


@jit
def xe_loss(logits, targets):
    return -jnp.take_along_axis(jax.nn.log_softmax(logits), targets[..., None], axis=-1)


@jit
def mean_xe_loss(logits, targets):
    return xe_loss(logits, targets).mean()


@jit
def xe_and_acc(logits, targets):
    acc = (logits.argmax(1) == targets).astype(jnp.float32)
    return xe_loss(logits, targets), acc


@jit
def mean_xe_and_acc(logits, targets):
    loss, acc = xe_and_acc(logits, targets)
    return loss.mean(), acc.mean()


@jit
def mean_xe_and_acc_dict(logits, targets):
    loss, acc = mean_xe_and_acc(logits, targets)
    return loss, {"acc": acc}


def make_fsl_inner_loop(
    slow_apply, fast_apply_and_loss_fn, opt_update_fn, num_steps, update_state=False
):
    def inner_loop(
        rng,
        slow_params,
        fast_params,
        slow_state,
        fast_state,
        is_training,
        opt_state,
        x_spt,
        y_spt,
        num_steps=num_steps,
        update_state=update_state,
    ):
        slow_outputs, slow_state = slow_apply(
            rng, slow_params, slow_state, is_training, x_spt,
        )
        for i in range(num_steps):
            (loss, (new_fast_state, *aux)), grads = value_and_grad(
                fast_apply_and_loss_fn, 1, has_aux=True
            )(rng, fast_params, fast_state, is_training, *slow_outputs, y_spt)
            if update_state:
                fast_state = new_fast_state
            if i == 0:
                initial_loss = loss
                initial_aux = aux
            updates, opt_state = opt_update_fn(grads, opt_state, fast_params)
            fast_params = optix.apply_updates(fast_params, updates)

        final_loss, (final_fast_state, *final_aux) = fast_apply_and_loss_fn(
            rng, fast_params, fast_state, False, *slow_outputs, y_spt
        )

        return (
            fast_params,
            slow_state,
            fast_state,
            {
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "initial_aux": initial_aux,
                "final_aux": final_aux,
            },
        )

    return inner_loop


def make_outer_loop(
    slow_apply, fast_apply_and_loss_fn, inner_loop, num_steps, update_state=False
):
    def outer_loop(
        rng,
        slow_params,
        fast_params,
        slow_state,
        fast_state,
        is_training,
        inner_opt_state,
        x_spt,
        y_spt,
        x_qry,
        y_qry,
        num_steps=num_steps,
        update_state=update_state,
    ):
        slow_outputs, initial_slow_state = slow_apply(
            rng, slow_params, slow_state, is_training, x_qry,
        )
        initial_loss, (initial_fast_state, *initial_aux) = fast_apply_and_loss_fn(
            rng, fast_params, fast_state, is_training, *slow_outputs, y_qry,
        )
        fast_params, slow_state, fast_state, inner_info = inner_loop(
            rng,
            slow_params,
            fast_params,
            slow_state,
            fast_state,
            is_training,
            inner_opt_state,
            x_spt,
            y_spt,
            num_steps,
            update_state,
        )
        final_loss, (final_fast_state, *final_aux) = fast_apply_and_loss_fn(
            rng, fast_params, fast_state, is_training, *slow_outputs, y_qry,
        )

        return (
            final_loss,
            (
                slow_state,
                fast_state,
                {
                    "inner": inner_info,
                    "outer": {
                        "initial_loss": initial_loss,
                        "final_loss": final_loss,
                        "initial_aux": initial_aux,
                        "final_aux": final_aux,
                    },
                },
            ),
        )

    return outer_loop


def make_batched_outer_loop(outer_loop):
    def helper_fn(rng, x_spt, y_spt, x_qry, y_qry, **kwargs):
        return outer_loop(
            rng=rng, x_spt=x_spt, y_spt=y_spt, x_qry=x_qry, y_qry=y_qry, **kwargs
        )

    def batched_outer_loop(
        rng,  # Assume rng is already split
        slow_params,
        fast_params,
        slow_state,
        fast_state,
        is_training,
        inner_opt_state,
        bx_spt,
        by_spt,
        bx_qry,
        by_qry,
        num_steps,
    ):

        losses, (slow_states, fast_states, infos) = vmap(
            partial(
                helper_fn,
                slow_params=slow_params,
                fast_params=fast_params,
                slow_state=slow_state,
                fast_state=fast_state,
                is_training=is_training,
                inner_opt_state=inner_opt_state,
                num_steps=num_steps,
            )
        )(rng, bx_spt, by_spt, bx_qry, by_qry)

        return losses.mean(), (slow_states, fast_states, infos)

    return batched_outer_loop


def make_fsl_inner_outer_loop(
    slow_apply,
    fast_apply_and_loss_fn,
    inner_opt_update_fn,
    num_steps,
    update_state=False,
):
    inner_loop = make_fsl_inner_loop(
        slow_apply,
        fast_apply_and_loss_fn,
        inner_opt_update_fn,
        num_steps=num_steps,
        update_state=False,
    )
    outer_loop = make_outer_loop(
        slow_apply,
        fast_apply_and_loss_fn,
        inner_loop,
        num_steps=num_steps,
        update_state=update_state,
    )
    return inner_loop, outer_loop

