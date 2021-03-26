from datasets.omniglot import get_omniglot_dataset
from datasets.tiered_imagenet import get_tiered_imagenet_dataset


def get_dataset(
    name,
    split,
    all=None,
    train=None,
    image_size=None,
):
    if name == "omniglot":
        assert (all is not None) and (train is not None) and (image_size is not None), (
            "get_dataset's arguments 'all', "
            "'train' and 'image_size' can't "
            "be None for omniglot dataset, values "
            f"where ({all}, {train}, {image_size})"
        )

        return get_omniglot_dataset(
            split,
            image_size,
            train,
            all,
        )
    elif name == "tiered-imagenet":
        return get_tiered_imagenet_dataset(split)
    else:
        raise NameError(f"Unknown dataset {name}")