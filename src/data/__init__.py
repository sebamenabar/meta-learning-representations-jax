from jax.tree_util import Partial as partial

from .sampling import *
from .miniimagenet import (
    prepare_data as prepare_mi_data,
    mean as mi_mean,
    std as mi_std,
)
from .omniglot import (
    prepare_data as prepare_omn_data,
    # mean as omn_mean,
    # std as omn_std,
)


statistics = {
    "miniimagenet": (mi_mean, mi_std),
    "omniglot": (None, None),
}


def preprocess_data(x, mean, std):
    return (x / 255) - mean / std


def prepare_data(dataset, data_dir, cpu, device, prefetch_data_gpu=False):
    mean, std = statistics[dataset]
    if dataset == "miniimagenet":
        train_images, train_labels, _ = prepare_mi_data(data_dir, "train")
        val_images, val_labels, _ = prepare_mi_data(data_dir, "val")
        preprocess_fn = partial(prepare_data, mean=mean, std=std)
        mean = jax.device_put(mean, device)
        std = jax.device_put(std, device)
    elif dataset == "omniglot":
        train_images, train_labels = prepare_omn_data(data_dir, "train")
        val_images, val_labels = prepare_omn_data(data_dir, "val")
        preprocess_fn = lambda x: x

    val_images = jax.device_put(val_images, cpu)
    val_labels = jax.device_put(val_labels, cpu)
    if prefetch_data_gpu:
        train_images = jax.device_put(train_images, device)
        train_labels = jax.device_put(train_labels, device)
        if dataset == "omniglot":
            val_images = jax.device_put(val_images, device)
            val_labels = jax.device_put(val_labels, device)
    else:
        train_images = jax.device_put(train_images, cpu)
        train_labels = jax.device_put(train_labels, cpu)

    return train_images, train_labels, val_images, val_labels, preprocess_fn
