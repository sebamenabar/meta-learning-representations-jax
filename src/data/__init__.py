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


def normalize_fn(x, mean, std):
    return (x - mean) / std


def prepare_data(dataset, data_fp, device=None):
    mean, std = statistics[dataset]
    if dataset == "miniimagenet":
        images, labels, _ = prepare_mi_data(data_fp, "train")
        # val_images, val_labels, _ = prepare_mi_data(data_dir, "val")
        # mean = jax.device_put(mean, device)
        # std = jax.device_put(std, device)
        _normalize_fn = partial(normalize_fn, mean=mean, std=std)

    images = onp.array(images)
    labels = onp.array(labels)

    return images, labels, _normalize_fn


def augment(rng, imgs, color_jitter_prob=1.0):
    rng_crop, rng_color, rng_flip = split(rng, 3)
    imgs = augmentations.random_crop(imgs, rng_crop, 84, ((8, 8), (8, 8), (0, 0)))
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
    # imgs = normalize_fn(imgs)
    return imgs


def preprocess_images(rng, x_spt, x_qry, normalize_fn, augment="none", augment_fn=None):
    
    print("start of preprocess images")
    
    x_spt = x_spt / 255
    x_qry = x_qry / 255
    if augment == "all":
        print("\nAugmenting support and query")
        rng_spt, rng_qry = split(rng)
        x_spt = augment_fn(rng_spt, flatten(x_spt, (0, 1))).reshape(*x_spt.shape)
        x_qry = augment_fn(rng_qry, flatten(x_qry, (0, 1))).reshape(*x_qry.shape)
    elif augment == "spt":
        print("\nAugmenting support only")
        x_spt = augment_fn(rng, flatten(x_spt, (0, 1))).reshape(*x_spt.shape)
    elif augment == "qry":
        print("\nAugmenting query only")
        x_qry = augment_fn(rng, flatten(x_qry, (0, 1))).reshape(*x_qry.shape)
    elif augment == "none":
        print("\nNo augmentation")
    else:
        raise NameError(f"Unkwown augmentation {augment}")

    x_spt = normalize_fn(x_spt)
    x_qry = normalize_fn(x_qry)

    print("end of preprocess images")

    return x_spt, x_qry