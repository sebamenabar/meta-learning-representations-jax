import pickle
import os.path as osp
import numpy as onp


def prepare_data(data_dir, split="train"):
    fname = f"mini-imagenet-cache-{split}.pkl"
    with open(osp.join(data_dir, fname), "rb") as f:
        data = pickle.load(f)

    images = data["image_data"]
    num_classes = len(data["class_dict"])
    num_samples = images.shape[0] // num_classes
    labels = onp.tile(onp.arange(num_classes).reshape(num_classes, 1), num_samples)
    images = images.reshape(num_classes, num_samples, *images.shape[1:])

    return images, labels, data["class_dict"]
