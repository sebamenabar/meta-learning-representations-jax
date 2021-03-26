from utils.data import ImageDataset

TIM_FILES = {
    "train": "train",
    "val_train": "val",
    "test_train": "test",
}


def get_tiered_imagenet_dataset(
    split,
    image_size=None,  # compatibility,
    train=None,
    all=None,
):
    split = TIM_FILES["split"]
    return ImageDataset(
        f"/home/samenabar/storage/code/continual_learning/data/tiered-imagenet/{split}.npz",
        transpose=False,
    )