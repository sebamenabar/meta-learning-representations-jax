from jax.tree_util import Partial as partial

from .sampling import *
from .miniimagenet import (
    prepare_data as prepare_mi_data,
    mean as mi_mean,
    std as mi_std,
)
from .cifar100 import (
    prepare_data as prepare_cifar100_data,
    mean as cifar100_mean,
    std as cifar100_std,
)
from .omniglot import (
    prepare_data as prepare_omn_data,
    # mean as omn_mean,
    # std as omn_std,
)


statistics = {
    "miniimagenet": (mi_mean, mi_std),
    "omniglot": (None, None),
    "cifar100": (cifar100_mean, cifar100_std),
}


def preprocess_data(x, mean, std):
    return (x / 255) - mean / std


def prepare_data(dataset, data_dir, cpu, device, prefetch_data_gpu=False, normalized=False):
    mean, std = statistics[dataset]
    if dataset == "miniimagenet":
        train_images, train_labels, _ = prepare_mi_data(data_dir, "train")
        val_images, val_labels, _ = prepare_mi_data(data_dir, "val")
        mean = jax.device_put(mean, device)
        std = jax.device_put(std, device)
        preprocess_fn = partial(preprocess_data, mean=mean, std=std)
    if dataset == "cifar100":
        train_images, train_labels = prepare_cifar100_data(data_dir, "train", normalized)
        val_images, val_labels = prepare_cifar100_data(data_dir, "val", normalized)
        mean = jax.device_put(mean, device)
        std = jax.device_put(std, device)
        if normalized:
            preprocess_fn = lambda x: x
        else:
            preprocess_fn = partial(preprocess_data, mean=mean, std=std)
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
