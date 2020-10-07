from jax.tree_util import Partial as partial
import pickle
import os.path as osp
import numpy as onp
from lib import flatten


# mean = onp.array([0.4707837, 0.4494574, 0.4026407])
# std = onp.array([0.28429058, 0.27527657, 0.29029518])

mean = onp.array([120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0])
std = onp.array([70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0])


def normalize_fn(x, mean, std):
    return (x - mean) / std


def prepare_data(data_fp, split="train"):
    #  fname = f"mini-imagenet-cache-{split}.pkl"
    with open(data_fp, "rb") as f:
        data = pickle.load(f)

    images = data["data"]
    labels = data["labels"]

    return images, labels, data["catname2label"]


class MiniImageNetDataset:
    def __init__(self, split, data_root):
        if split == "train":
            self._fp = osp.join(
                data_root,
                "miniImageNet_category_split_train_phase_train_ordered.pickle",
            )
            self._images, self._labels, _ = prepare_data(self._fp)
            self._images = flatten(self._images, (0, 1))
            self._labels = flatten(self._labels, (0, 1))
        elif split == "train_val":
            self._fp = osp.join(
                data_root,
                "miniImageNet_category_split_train_phase_val_ordered.pickle",
            )
            self._images, self._labels, _ = prepare_data(self._fp)
        self._normalize = partial(normalize_fn, mean=mean, std=std)