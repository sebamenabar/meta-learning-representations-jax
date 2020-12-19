import numpy as onp
import jax
from jax import numpy as jnp, partial, value_and_grad, tree_multimap
from jax.random import split

import optax as ox

from losses import xe_and_acc


zero_initializer = lambda dtype: lambda rng, shape: jax.nn.initializers.zeros(
    rng, shape, dtype=dtype,
)


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


def evaluate_supervised_accuracy(model, data_loader):
    num_total_corrects = 0
    num_total = 0
    total_loss = 0.0
    for x, y in data_loader:
        logits = model(x)
        loss, num_corrects = xe_and_acc(logits, y)
        num_total_corrects = num_total_corrects + num_corrects.sum()
        num_total = num_total + y.shape[0]
        total_loss = total_loss + loss.sum()

    return total_loss / num_total, num_total_corrects / num_total
