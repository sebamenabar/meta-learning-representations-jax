import numpy as onp
import jax
from jax.random import split
from jax import numpy as jnp, random


def shuffle_along_axis(rng, a, axis):
    idx = random.uniform(rng, a.shape).argsort(axis=axis)
    return jnp.take_along_axis(a, idx, axis=axis)


def fsl_sample_tasks(rng, images, labels, num_tasks, way, shot, disjoint=True):
    rng_classes, rng_idxs = split(rng)
    if not disjoint:
        # For each task take way classes
        sampled_classes = shuffle_along_axis(
            rng_classes, jnp.arange(images.shape[0])[None, :].repeat(num_tasks, 0), 1
        )[:, :way]
        # Flatten everything for indexing
        sampled_classes = sampled_classes.reshape(num_tasks * way, 1)
        # Repeat each class idx shot times
        sampled_classes = sampled_classes.repeat(shot, 1)
    else:
        # Sample from all the classes without repetition between tasks
        sampled_classes = random.choice(
            rng_classes,
            jnp.arange(images.shape[0]),
            (num_tasks * way, 1),
            replace=False,
        )
        # Repeat each class idx shot times
        sampled_classes = sampled_classes.repeat(shot, 1)

    # For each task for each class sample shot indexes
    sampled_idxs = shuffle_along_axis(
        rng_idxs, jnp.arange(images.shape[1])[None].repeat(num_tasks * way, 0), 1
    )[:, :shot]

    sampled_images = images[sampled_classes, sampled_idxs]
    sampled_labels = labels[sampled_classes, sampled_idxs]

    image_shape = images.shape[2:]
    sampled_images = sampled_images.reshape(num_tasks, way, shot, *image_shape)
    sampled_labels = sampled_labels.reshape(num_tasks, way, shot)

    return sampled_images, sampled_labels


def fsl_sample_transfer_and_build(
    rng,
    preprocess_fn,
    images,
    labels,
    num_tasks,
    way,
    spt_shot,
    qry_shot,
    device=None,
    disjoint=True,
):
    sampled_images, sampled_labels = fsl_sample_tasks(
        rng,
        images,
        labels,
        num_tasks=num_tasks,
        way=way,
        shot=spt_shot + qry_shot,
        disjoint=disjoint,
    )
    shuffled_labels = (
        jnp.repeat(jnp.arange(way), spt_shot + qry_shot)
        .reshape(way, spt_shot + qry_shot)[None, :]
        .repeat(num_tasks, 0)
    )

    # Transfer ints but operate on gpu
    sampled_images = jax.device_put(sampled_images, device)
    shuffled_labels = jax.device_put(shuffled_labels, device)

    images_shape = sampled_images.shape[3:]
    # sampled_images = ((sampled_images / 255) - mean) / std
    sampled_images = preprocess_fn(sampled_images)

    # Transfer floats but operate on cpu
    # sampled_images = jax.device_put(sampled_images, gpu)
    # shuffled_labels = jax.device_put(shuffled_labels, gpu)

    x_spt, x_qry = jnp.split(sampled_images, (spt_shot,), 2)
    x_spt = x_spt.reshape(num_tasks, way * spt_shot, *images_shape)
    x_qry = x_qry.reshape(num_tasks, way * qry_shot, *images_shape)
    y_spt, y_qry = jnp.split(shuffled_labels, (spt_shot,), 2)
    y_spt = y_spt.reshape(num_tasks, way * spt_shot)
    y_qry = y_qry.reshape(num_tasks, way * qry_shot)

    return x_spt, y_spt, x_qry, y_qry


def batch_sampler(rng, X, y, batch_size):
    order = random.permutation(rng, jnp.arange(len(X)))
    current = 0
    while current + batch_size < len(X):
        idxs = order[current : current + batch_size]
        current += batch_size
        yield X[idxs], y[idxs]


class BatchSampler:
    def __init__(self, rng, X, y, batch_size, shuffle=True, keep_last=False):
        self.rng = rng
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.__len = (len(X) // batch_size) + (
            ((len(X) % batch_size) > 0) and keep_last
        )
        self.shuffle = shuffle
        self.keep_last = keep_last

    def __len__(self):
        return self.__len

    def __iter__(self):
        self.order = jnp.arange(len(self.X))
        if self.shuffle:
            rng, rng_shuffle = split(self.rng)
            self.rng = rng
            self.order = random.permutation(rng_shuffle, self.order)
        self.n = 0
        return self

    def __next__(self):
        if self.n + self.batch_size < len(self.X):
            idxs = self.order[self.n : self.n + self.batch_size]
            self.n += self.batch_size
            return self.X[idxs], self.y[idxs]
        elif self.n < len(self.X) and self.keep_last:
            idxs = self.order[self.n :]
            self.n += self.batch_size
            return self.X[idxs], self.y[idxs]
        else:
            raise StopIteration
