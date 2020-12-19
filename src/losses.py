import jax
from jax import numpy as jnp


def xe_loss(logits, targets):
    return -jnp.take_along_axis(jax.nn.log_softmax(logits), targets[..., None], axis=-1)


def mean_xe_loss(logits, targets):
    return xe_loss(logits, targets).mean()


def xe_and_acc(logits, targets):
    acc = (logits.argmax(1) == targets).astype(jnp.float32)
    return xe_loss(logits, targets), acc


def mean_xe_and_acc(logits, targets):
    loss, acc = xe_and_acc(logits, targets)
    return loss.mean(), acc.mean()


def mean_xe_and_acc_dict(logits, targets):
    loss, acc = mean_xe_and_acc(logits, targets)
    return loss, {"acc": acc}
