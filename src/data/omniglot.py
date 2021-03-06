import pickle
import os.path as osp


# from PIL import Image
# import numpy as onp

# import tensorflow_datasets as tfds

# mean = onp.array((0.9220594763755798))
# std = onp.array((0.2477845698595047))


# def prepare_data(split="train"):
#     (data,) = tfds.as_numpy(
#         tfds.load(name="omniglot", shuffle_files=False, batch_size=-1, split=[split],)
#     )

#     convert = lambda x: Image.fromarray(x).convert("L").resize((28, 28), Image.LANCZOS)
#     images = onp.stack([convert(img) for img in data["image"]]).astype(onp.float32)

#     labels = data["label"]
#     order = onp.argsort(labels)
#     uniques, counts = onp.unique(labels, return_counts=True)
#     assert (counts[0] == counts).all()

#     images = images[order]
#     images = images.reshape(len(uniques), counts[0], *images.shape[1:], 1)
#     labels = labels[order].reshape(len(uniques), counts[0], *labels.shape[1:])

#     return images, labels, None


def prepare_data(data_dir, split="train"):
    fname = f"omniglot-cache-{split}.pkl"
    with open(osp.join(data_dir, fname), "rb") as f:
        data = pickle.load(f)

    images = data["image_data"]
    labels = data["labels"]
    return images, labels
