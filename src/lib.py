import functools
from easydict import EasyDict as edict

import numpy as onp

import jax
from jax.random import split
from jax.tree_util import Partial as partial
from jax import jit, numpy as jnp, value_and_grad, vmap, tree_multimap, ops

import haiku as hk
import optax as ox


def parse_and_build_cfg(args):
    cfg = edict()
    for argname, argval in vars(args).items():
        rsetattr(cfg, argname, argval)
    return cfg


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


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


# @jit
def xe_loss(logits, targets):
    return -jnp.take_along_axis(jax.nn.log_softmax(logits), targets[..., None], axis=-1)


# @jit
def mean_xe_loss(logits, targets):
    return xe_loss(logits, targets).mean()


# @jit
def xe_and_acc(logits, targets):
    acc = (logits.argmax(1) == targets).astype(jnp.float32)
    return xe_loss(logits, targets), acc


# @jit
def mean_xe_and_acc(logits, targets):
    loss, acc = xe_and_acc(logits, targets)
    return loss.mean(), acc.mean()


# @jit
def mean_xe_and_acc_dict(logits, targets):
    loss, acc = mean_xe_and_acc(logits, targets)
    return loss, {"acc": acc}


def apply_and_loss_fn(
    slow_params,
    fast_params,
    slow_state,
    fast_state,
    rng,
    inputs,
    targets,
    is_training,
    slow_apply,
    fast_apply,
    loss_fn,
):
    slow_rng, fast_rng = split(rng)

    slow_outputs, slow_state = slow_apply(
        slow_params,
        slow_state,
        slow_rng,
        inputs,
        is_training,
    )
    fast_outputs, fast_state = fast_apply(
        fast_params, fast_state, fast_rng, *slow_outputs, is_training
    )
    loss, *aux = loss_fn(fast_outputs, targets)

    return loss, ((slow_state, fast_state), aux)


def fast_apply_and_loss_fn(
    fast_params,
    fast_state,
    rng,
    slow_outputs,
    is_training,
    targets,
    fast_apply,
    loss_fn,
):
    outputs, state = fast_apply(
        fast_params, fast_state, rng, *slow_outputs, is_training
    )
    loss, aux = loss_fn(outputs, targets)
    return loss, (state, aux)


def batched_outer_loop(
    slow_params,
    fast_params,
    inner_lr,
    replicated_slow_state,
    replicated_fast_state,
    inner_opt_state,
    brng,
    bx_spt,
    by_spt,
    bx_qry,
    by_qry,
    bspt_classes,
    outer_loop,
):
    def helper(slow_state, fast_state, *args):
        return outer_loop(
            slow_params, fast_params, inner_lr, slow_state, fast_state, inner_opt_state, *args
        )

    losses, aux = vmap(helper)(
        replicated_slow_state,
        replicated_fast_state,
        brng,
        bx_spt,
        by_spt,
        bx_qry,
        by_qry,
        bspt_classes,
    )
    return losses.mean(), aux


def outer_loop(
    slow_params,
    fast_params,
    inner_lr,
    slow_state,
    fast_state,
    inner_opt_state,
    rng,
    x_spt,
    y_spt,
    x_qry,
    y_qry,
    spt_classes,
    is_training,
    inner_loop,  # instantiated inner_loop
    slow_apply,
    fast_apply,
    loss_fn,
    reset_fast_params_fn=None,
    track_slow_state="none",
):
    if reset_fast_params_fn:
        rng, rng_reset = split(rng)
        tree_flat, tree_struct = jax.tree_flatten(fast_params)
        rng_tree = jax.tree_unflatten(tree_struct, split(rng_reset, len(tree_flat)))
        fast_params = jax.tree_multimap(
            partial(reset_fast_params_fn, spt_classes), rng_tree, fast_params
        )

    _fast_apply_and_loss_fn = partial(
        fast_apply_and_loss_fn, fast_apply=fast_apply, loss_fn=loss_fn
    )
    rng_outer_slow, rng_outer_fast, rng_inner = split(rng, 3)
    # To have proper statitics on the first step we need to compute
    # them before the first inner loop
    slow_outputs, outer_slow_state = slow_apply(
        slow_params,
        slow_state,
        rng_outer_slow,
        x_qry,
        is_training,
    )
    if "outer" in track_slow_state:
        print("Using outer statistics")
        slow_state = outer_slow_state

    (
        inner_fast_params,
        inner_slow_state,
        fast_state,
        inner_opt_state,
        inner_auxs,
    ) = inner_loop(
        slow_params,
        fast_params,
        inner_lr,
        slow_state,
        fast_state,
        inner_opt_state,
        rng_inner,
        x_spt,
        y_spt,
    )
    if "inner" in track_slow_state:
        print("Using inner statistics")
        slow_state = inner_slow_state
    initial_loss, (_, *initial_aux) = _fast_apply_and_loss_fn(
        fast_params,
        fast_state,
        rng_outer_fast,
        slow_outputs,
        is_training,
        y_qry,
    )
    final_loss, (final_fast_state, *final_aux) = _fast_apply_and_loss_fn(
        inner_fast_params,
        fast_state,
        rng_outer_fast,
        slow_outputs,
        is_training,
        y_qry,
    )
    return (
        final_loss,
        (
            slow_state,
            fast_state,
            {
                "inner": inner_auxs,
                "outer": {
                    "initial": {"aux": initial_aux, "loss": initial_loss},
                    "final": {"aux": final_aux, "loss": final_loss},
                },
            },
        ),
    )


def fsl_inner_loop(
    slow_params,
    fast_params,
    inner_lr,
    slow_state,
    fast_state,
    opt_state,
    rng,
    inputs,
    targets,
    is_training,
    num_steps,
    slow_apply,
    fast_apply,
    loss_fn,
    opt_update_fn,
    update_state=False,
    return_history=True,
):
    _fast_apply_and_loss_fn = partial(
        fast_apply_and_loss_fn, fast_apply=fast_apply, loss_fn=loss_fn
    )
    rng_slow, *rngs = split(rng, num_steps + 2)
    slow_outputs, slow_state = slow_apply(
        slow_params,
        slow_state,
        rng_slow,
        inputs,
        is_training,
    )

    losses = []
    auxs = []

    for i in range(num_steps):
        (loss, (new_fast_state, *aux)), grads = value_and_grad(
            _fast_apply_and_loss_fn, has_aux=True
        )(fast_params, fast_state, rngs[i], slow_outputs, is_training, targets)
        if update_state:
            fast_state = new_fast_state
        if i == 0:
            initial_loss = loss
            initial_aux = aux

        if return_history:
            losses.append(loss)
            auxs.append(aux)

        updates, opt_state = opt_update_fn(inner_lr, grads, opt_state, fast_params)
        fast_params = ox.apply_updates(fast_params, updates)

    final_loss, (final_fast_state, *final_aux) = _fast_apply_and_loss_fn(
        fast_params,
        fast_state,
        rngs[i + 1],
        slow_outputs,
        is_training,
        targets,
    )
    if return_history:
        losses.append(final_loss)
        auxs.append(final_aux)
        info = {
            "losses": jnp.stack(losses),
            "auxs": tree_multimap(lambda x, *xs: jnp.stack(xs), auxs[0], *auxs),
        }
    else:
        info = (
            {
                "losses": {"initial": initial_loss, "final": final_loss},
                "auxs": {"initial": initial_aux, "final": final_aux},
            },
        )

    return (
        fast_params,
        slow_state,
        fast_state,
        opt_state,
        info,
    )


def cl_inner_loop(
    slow_params,
    fast_params,
    inner_lr,
    slow_state,
    fast_state,
    opt_state,
    rng,
    inputs,
    targets,
    is_training,
    slow_apply,
    fast_apply,
    loss_fn,
    opt_update_fn,
    num_steps=None,
    update_state=False,
    return_history=True,
):
    _fast_apply_and_loss_fn = partial(
        fast_apply_and_loss_fn, fast_apply=fast_apply, loss_fn=loss_fn
    )
    rng_slow, rng_fast, *rngs = split(rng, inputs.shape[0] + 2)
    slow_outputs, slow_state = slow_apply(
        slow_params,
        slow_state,
        rng_slow,
        inputs,
        is_training,
    )

    initial_loss, (_, *initial_aux) = _fast_apply_and_loss_fn(
        fast_params,
        fast_state,
        rng_fast,
        slow_outputs,
        is_training,
        targets,
    )

    losses = []
    auxs = []

    for i in range(inputs.shape[0]):
        (loss, (new_fast_state, *aux)), grads = value_and_grad(
            _fast_apply_and_loss_fn, has_aux=True
        )(
            fast_params,
            fast_state,
            rngs[i],
            [so[[i]] for so in slow_outputs],
            is_training,
            targets[[i]],
        )
        if update_state:
            fast_state = new_fast_state
        if return_history:
            losses.append(loss)
            auxs.append(aux)

        updates, opt_state = opt_update_fn(inner_lr, grads, opt_state, fast_params)
        fast_params = ox.apply_updates(fast_params, updates)

    final_loss, (final_fast_state, *final_aux) = _fast_apply_and_loss_fn(
        fast_params,
        fast_state,
        rng_fast,
        slow_outputs,
        is_training,
        targets,
    )
    if return_history:
        # losses.append(final_loss)
        # auxs.append(final_aux)
        info = {
            "losses": jnp.stack(losses),
            "auxs": tree_multimap(lambda x, *xs: jnp.stack(xs), auxs[0], *auxs),
            "initial": {"loss": initial_loss, "aux": initial_aux},
            "final": {"loss": final_loss, "aux": final_aux},
        }
    else:
        info = (
            {
                "losses": {"initial": initial_loss, "final": final_loss},
                "auxs": {"initial": initial_aux, "final": final_aux},
            },
        )

    return (
        fast_params,
        slow_state,
        fast_state,
        opt_state,
        info,
    )


def meta_step(
    rng,
    step_num,
    outer_opt_state,
    slow_params,
    fast_params,
    inner_lr,
    slow_state,
    fast_state,
    x_spt,
    y_spt,
    x_qry,
    y_qry,
    spt_classes,
    inner_opt_init,
    outer_opt_update,
    batched_outer_loop_ins,
    reset_fast_params_fn=None,
    preprocess_images_fn=None,
    train_inner_lr=False,
):
    rng, rng_preprocess = split(rng)
    if preprocess_images_fn:
        x_spt, x_qry = preprocess_images_fn(rng_preprocess, x_spt, x_qry)
    if reset_fast_params_fn:
        rng, rng_reset = split(rng)
        tree_flat, tree_struct = jax.tree_flatten(fast_params)
        rng_tree = jax.tree_unflatten(tree_struct, split(rng_reset, len(tree_flat)))
        fast_params = jax.tree_multimap(
            partial(reset_fast_params_fn, spt_classes), rng_tree, fast_params
        )
    inner_opt_state = inner_opt_init(fast_params)

    if train_inner_lr:
        positions = (0, 1, 2)
    else:
        positions = (0, 1)
    (outer_loss, (slow_state, fast_state, info)), grads = value_and_grad(
        batched_outer_loop_ins, positions, has_aux=True
    )(
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
    )
    grads = jax.tree_map(lambda v: jax.lax.pmean(v, axis_name="i"), grads)
    if train_inner_lr:
        updates, outer_opt_state = outer_opt_update(
            grads, outer_opt_state, (slow_params, fast_params, inner_lr)
        )
        slow_params, fast_params, inner_lr = ox.apply_updates((slow_params, fast_params, inner_lr), updates)
    else:
        updates, outer_opt_state = outer_opt_update(
            grads, outer_opt_state, (slow_params, fast_params)
        )
        slow_params, fast_params = ox.apply_updates((slow_params, fast_params), updates)

    return outer_opt_state, slow_params, fast_params, inner_lr, slow_state, fast_state, info


def reset_by_idxs(w_make_fn, idxs, rng, array):
    print("Resetting head by indexes")
    return jax.ops.index_update(
        array,
        jax.ops.index[:, idxs],
        w_make_fn(dtype=array.dtype)(rng, (array.shape[0], idxs.shape[0])),
    )


def reset_all(w_make_fn, idxs, rng, array):
    print("Resetting all head")
    return w_make_fn(dtype=array.dtype)(rng, array.shape)


def delayed_cosine_decay_schedule(
    init_value: float,
    transition_begin: int,
    decay_steps: int,
    alpha: float = 0.0,
):
    """Returns a function which implements cosine learning rate decay.

    For more details see:
      https://openreview.net/forum?id=Skq89Scxx&noteId=Skq89Scxx

    Args:
      init_value: An initial value `init_v`.
      decay_steps: Positive integer - the number of steps for which to apply
        the decay for.
      alpha: Float. The minimum value of the multiplier used to adjust the
        learning rate.

    Returns:
      schedule: A function that maps step counts to values.
    """
    if not decay_steps > 0:
        raise ValueError("The cosine_decay_schedule requires positive decay_steps!")

    def schedule(count):
        count = jnp.minimum(count - transition_begin, decay_steps)
        cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * count / decay_steps))
        decayed = (1 - alpha) * cosine_decay + alpha
        return jnp.where(count <= 0, init_value, init_value * decayed)

    return schedule


# return jnp.where(count <= 0, init_value, init_value * jnp.power(decay_rate, p))