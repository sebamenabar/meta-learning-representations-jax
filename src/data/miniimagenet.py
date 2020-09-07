import pickle
import os.path as osp
import numpy as onp


# mean = onp.array([0.4707837, 0.4494574, 0.4026407])
# std = onp.array([0.28429058, 0.27527657, 0.29029518])

# Statistics used in A Good Embedding is All you Need?
mean = onp.array([120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0])
std = onp.array([70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0])


def prepare_data(data_fp):
    with open(data_fp, "rb") as f:
        data = pickle.load(f)

    images = data["data"]
    labels = data["labels"]

    return images, labels, data["catname2label"]
