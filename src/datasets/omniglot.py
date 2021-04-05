import numpy as onp
from utils.data import ImageDataset

OMNIGLOT_FILES = {
    "train": "val_train_train",
    "train+val": "train",
    "val": "val_test_train",
    "val_train": "val_test_train",
    "val_test": "val_test_test",
    "test": "test_train",
    "test_train": "test_train",
    "test_test": "test_test",
}


def get_omniglot_dataset(
    split,
    image_size,
    train=True,
    all=False,
):
    split = OMNIGLOT_FILES[split]
    if train:
        split = f"background_{split}"
    if all:
        split = f"{split}_all"
    split = f"{split}_{image_size}.npz"
    fp = f"/home/samenabar/storage/code/continual_learning/data/from_torch/omniglot/{split}"
    data = onp.load(fp)

    return ImageDataset(
        data["images"].transpose(0, 2, 3, 1),
        data["targets"],
        data["mean_0_1"],
        data["std_0_1"],
    )
