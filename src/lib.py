import jax
from jax.tree_util import Partial as partial
from jax import jit, numpy as jnp, value_and_grad, vmap

from jax.experimental import optix


def setup_device(gpus=0):
    if gpus != 0:
        gpu = jax.devices("gpu")[0]
    else:
        gpu = None
    cpu = jax.devices("cpu")[0]
    default_platform = "cpu"
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


def make_fsl_inner_loop(
    apply_and_loss_fn, opt_update_fn, num_steps, update_state=False
):
    # def apply_and_loss_fn(
    #     rng, slow_params, fast_params, state, is_training, inputs, targets
    # ):
    #     outputs, state, *rest1 = apply_fn(
    #         rng, slow_params, fast_params, state, is_training, inputs,
    #     )
    #     loss, *rest2 = loss_fn(outputs, targets)
    #     return loss, (state, *rest1, *rest2)

    def inner_loop(
        rng,
        slow_params,
        fast_params,
        state,
        opt_state,
        is_training,
        x_spt,
        y_spt,
        num_steps=num_steps,
        update_state=update_state,
    ):
        for i in range(num_steps):
            (loss, (new_state, *aux)), grads = value_and_grad(
                apply_and_loss_fn, 2, has_aux=True
            )(rng, slow_params, fast_params, state, is_training, x_spt, y_spt,)
            if update_state:
                state = new_state
            if i == 0:
                initial_loss = loss
                initial_aux = aux
            updates, opt_state = opt_update_fn(grads, opt_state, fast_params)
            fast_params = optix.apply_updates(fast_params, updates)

        (final_loss, (final_state, *final_aux)), grads = value_and_grad(
            apply_and_loss_fn, 1, has_aux=True
        )(rng, slow_params, fast_params, state, False, x_spt, y_spt,)

        return (
            fast_params,
            state,
            {
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "initial_aux": initial_aux,
                "final_aux": final_aux,
            },
        )

    return inner_loop


def make_outer_loop(apply_and_loss_fn, inner_loop, num_steps, update_state=False):
    def outer_loop(
        rng,
        slow_params,
        fast_params,
        state,
        inner_opt_state,
        is_training,
        x_spt,
        y_spt,
        x_qry,
        y_qry,
        num_steps=num_steps,
        update_state=update_state,
    ):
        initial_loss, (initial_state, *initial_aux) = apply_and_loss_fn(
            rng, slow_params, fast_params, state, False, x_qry, y_qry,
        )
        fast_params, state, inner_info = inner_loop(
            rng,
            slow_params,
            fast_params,
            state,
            inner_opt_state,
            is_training,
            x_spt,
            y_spt,
            num_steps,
            update_state,
        )
        final_loss, (final_state, *final_aux) = apply_and_loss_fn(
            rng, slow_params, fast_params, state, False, x_qry, y_qry,
        )

        return (
            final_loss,
            (
                state,
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
        state,
        inner_opt_state,
        is_training,
        bx_spt,
        by_spt,
        bx_qry,
        by_qry,
    ):

        losses, (states, infos) = vmap(
            partial(
                helper_fn,
                slow_params=slow_params,
                fast_params=fast_params,
                state=state,
                inner_opt_state=inner_opt_state,
                is_training=is_training,
            )
        )(rng, bx_spt, by_spt, bx_qry, by_qry)

        return losses.mean(), (states, infos)

    return batched_outer_loop


def make_fsl_inner_outer_loop(
    apply_and_loss_fn, inner_opt_update_fn, num_steps, update_state=False
):
    inner_loop = make_fsl_inner_loop(
        apply_and_loss_fn, inner_opt_update_fn, num_steps=num_steps, update_state=False,
    )
    outer_loop = make_outer_loop(
        apply_and_loss_fn, inner_loop, num_steps=num_steps, update_state=update_state
    )
    return inner_loop, outer_loop

