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
import data.augmentations as augmentations

statistics = {
    "miniimagenet": (mi_mean, mi_std),
    "omniglot": (None, None),
    "cifar100": (cifar100_mean, cifar100_std),
}


def _normalize_fn(x, mean, std):
    return (x - mean) / std


def prepare_data(dataset, data_fp):
    mean, std = statistics[dataset]
    if dataset == "miniimagenet":
        images, labels, _ = prepare_mi_data(data_fp)
        normalize_fn = partial(_normalize_fn, mean=mean, std=std)

    return images, labels, normalize_fn, (mean, std)


def augment(rng, imgs, color_jitter_prob=1.0):
    rng_crop, rng_color, rng_flip = split(rng, 3)
    imgs = augmentations.random_crop(
        imgs, rng_crop, 84, ((8, 8), (8, 8), (0, 0))
    )  # Padding 8 pixels on each spatial dim, none on channel dim
    imgs = augmentations.color_transform(
        imgs,
        rng_color,
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.0,
        color_jitter_prob=color_jitter_prob,
        to_grayscale_prob=0.0,
    )
    imgs = augmentations.random_flip(imgs, rng_flip)
    return imgs