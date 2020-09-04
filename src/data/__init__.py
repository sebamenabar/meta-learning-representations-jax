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


def normalize_fn(x, mean, std):
    return (x - mean) / std


def prepare_data(dataset, data_fp, device):
    mean, std = statistics[dataset]
    if dataset == "miniimagenet":
        images, labels, _ = prepare_mi_data(data_fp, "train")
        # val_images, val_labels, _ = prepare_mi_data(data_dir, "val")
        mean = jax.device_put(mean, device)
        std = jax.device_put(std, device)
        _normalize_fn = partial(normalize_fn, mean=mean, std=std)

    images = jnp.array(images)
    labels = jnp.array(labels)

    return images, labels, _normalize_fn


# def prepare_data(dataset, data_dir, device, prefetch_data_gpu=False, normalized=False):
#     mean, std = statistics[dataset]
#     if dataset == "miniimagenet":
#         if data_dir is None:
#             # TEMP
#             data_dir = "/mnt/ialabnas/homes/samenabar/data/FSL/mini-imagenet"
#         train_images, train_labels, _ = prepare_mi_data(data_dir, "train")
#         val_images, val_labels, _ = prepare_mi_data(data_dir, "val")
#         mean = jax.device_put(mean, device)
#         std = jax.device_put(std, device)
#         preprocess_fn = partial(preprocess_data, mean=mean, std=std)
#     if dataset == "cifar100":
#         train_images, train_labels = prepare_cifar100_data(data_dir, "train", normalized)
#         val_images, val_labels = prepare_cifar100_data(data_dir, "val", normalized)
#         mean = jax.device_put(mean, device)
#         std = jax.device_put(std, device)
#         if normalized:
#             preprocess_fn = lambda x: x
#         else:
#             preprocess_fn = partial(preprocess_data, mean=mean, std=std)
#     elif dataset == "omniglot":
#         train_images, train_labels = prepare_omn_data(data_dir, "train")
#         val_images, val_labels = prepare_omn_data(data_dir, "val")
#         preprocess_fn = lambda x: x

#     val_images = jnp.array(val_images)
#     val_labels = jnp.array(val_labels)
#     if prefetch_data_gpu:
#         train_images = jax.device_put(train_images, device)
#         train_labels = jax.device_put(train_labels, device)
#         if dataset == "omniglot":
#             val_images = jax.device_put(val_images, device)
#             val_labels = jax.device_put(val_labels, device)
#     else:
#         train_images = jnp.array(train_images)
#         train_labels = jnp.array(train_labels)

#     return train_images, train_labels, val_images, val_labels, preprocess_fn
