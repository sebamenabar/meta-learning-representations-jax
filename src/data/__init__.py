from jax.random import split
import data.augmentations as augmentations


def augment(rng, imgs, color_jitter_prob=1.0, out_size=84):
    rng_crop, rng_color, rng_flip = split(rng, 3)
    # print("crop")
    imgs = augmentations.random_crop(imgs, rng_crop, out_size, ((8, 8), (8, 8), (0, 0)))
    # print("color")
    # imgs = augmentations.color_transform(
    #     imgs,
    #     rng_color,
    #     brightness=0.4,
    #     contrast=0.4,
    #     saturation=0.4,
    #     hue=0.0,
    #     color_jitter_prob=color_jitter_prob,
    #     to_grayscale_prob=0.0,
    # )
    # print("flip")
    # imgs = augmentations.random_flip(imgs, rng_flip)
    # imgs = normalize_fn(imgs)
    return imgs


class TensorDataset:
    def __init__(self, *tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple((t[index] for t in self.tensors))

    def __len__(self):
        return len(self.tensors[0])
