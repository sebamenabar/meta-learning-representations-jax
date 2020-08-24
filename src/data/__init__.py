from .sampling import *
from .miniimagenet import prepare_data as prepare_mi_data, mean as mi_mean, std as mi_std


statistics = {
    "miniimagenet": (mi_mean, mi_std),
}


def prepare_data(dataset, data_dir, cpu, gpu, prefetch_data_gpu=False):
    if dataset == "miniimagenet":
        train_images, train_labels, _ = prepare_mi_data(data_dir, "train")
        val_images, val_labels, _ = prepare_mi_data(data_dir, "val")

    if prefetch_data_gpu:
        train_images = jax.device_put(train_images, gpu)
        train_labels = jax.device_put(train_labels, gpu)
    else:
        train_images = jax.device_put(train_images, cpu)
        train_labels = jax.device_put(train_labels, cpu)

    val_images = jax.device_put(val_images, cpu)
    val_labels = jax.device_put(val_labels, cpu)

    mean, std = statistics[dataset]
    mean = jax.device_put(mean, gpu)
    std = jax.device_put(std, gpu)

    return train_images, train_labels, val_images, val_labels, mean, std
