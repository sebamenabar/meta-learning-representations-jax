import pickle
import os.path as osp
import numpy as onp


mean = onp.array([0.50689849, 0.48627395, 0.44093623])
std = onp.array([0.26766447, 0.25680128, 0.27642309])


def prepare_data(data_dir, split="train", normalized=False):
    if normalized:
        fname = f"cifar-100-cache-{split}-normalized.pkl"
    else:
        fname = f"cifar-100-cache-{split}.pkl"
    with open(osp.join(data_dir, fname), "rb") as f:
        data = pickle.load(f)

    images = data["image_date"]  # typo in file
    labels = data["labels"]
    return images, labels
