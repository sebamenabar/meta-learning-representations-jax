import numpy as onp
import jax
from jax.random import split
from jax import numpy as jnp, random
from lib import flatten


def shuffle_along_axis(rng, a, axis):
    idx = random.uniform(rng, a.shape).argsort(axis=axis)
    return jnp.take_along_axis(a, idx, axis=axis)


class FSLSampler:
    def __init__(self, X, y, shuffle_labels=False):
        self.X = X
        self.y = y
        self.data_shape = X.shape[2:]
        self.num_classes = X.shape[0]
        self.samples_per_class = X.shape[1]
        self.shuffle_labels = shuffle_labels

    def sample(self, rng, batch_size, way, shot):
        rng_classes, rng_idxs = split(rng)
        sampled_classes = shuffle_along_axis(
            rng_classes, onp.arange(self.num_classes)[None, :].repeat(batch_size, 0), 1
        )[:, :way]
        # Flatten everything for indexing
        sampled_classes = sampled_classes.reshape(batch_size * way, 1)
        # Repeat each class idx shot times
        sampled_classes = sampled_classes.repeat(shot, 1)

        # For each task for each class sample shot indexes
        sampled_idxs = shuffle_along_axis(
            rng_idxs,
            onp.arange(self.samples_per_class)[None].repeat(batch_size * way, 0),
            1,
        )[:, :shot]

        sampled_images = self.X[sampled_classes, sampled_idxs]
        sampled_images = sampled_images.reshape(batch_size, way, shot, *self.data_shape)

        if self.shuffle_labels:
            sampled_labels = (
                onp.repeat(onp.arange(way), shot)
                .reshape(way, shot)[None, :]
                .repeat(batch_size, 0)
            )
        else:
            sampled_labels = self.y[sampled_classes, sampled_idxs]
            sampled_labels = sampled_labels.reshape(batch_size, way, shot)

        return sampled_images, sampled_labels


def sample_tasks(rng, images, labels, num_tasks, way, shot, disjoint=True):
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


def fsl_sample(
    rng,
    images,
    labels,
    num_tasks,
    way,
    spt_shot,
    qry_shot,
    disjoint=True,
    shuffled_labels=True,
):
    sampled_images, sampled_labels = sample_tasks(
        rng,
        images,
        labels,
        num_tasks=num_tasks,
        way=way,
        shot=spt_shot + qry_shot,
        disjoint=disjoint,
    )
    if shuffled_labels:
        labels = (
            jnp.repeat(jnp.arange(way), spt_shot + qry_shot)
            .reshape(way, spt_shot + qry_shot)[None, :]
            .repeat(num_tasks, 0)
        )
    else:
        labels = sampled_labels
    return sampled_images, labels


def fsl_build(
    images, labels, num_devices, batch_size, way, shot, qry_shot,
):
    image_shape = images.shape[-3:]
    x_spt, x_qry = onp.split(images, (shot,), 2)
    x_spt = x_spt.reshape(num_devices, batch_size, way * shot, *image_shape)
    x_qry = x_qry.reshape(num_devices, batch_size, way * qry_shot, *image_shape)
    y_spt, y_qry = onp.split(labels, (shot,), 2)
    y_spt = y_spt.reshape(num_devices, batch_size, way * shot)
    y_qry = y_qry.reshape(num_devices, batch_size, way * qry_shot)
    return x_spt, y_spt, x_qry, y_qry


def fsl_sample_transfer_build(
    rng,
    images,
    labels,
    batch_size,
    way,
    shot,
    qry_shot,
    preprocess_fn,
    device=None,
    disjoint=False,
    shuffled_labels=True,
    stratified=None,  # For compatibility
):
    x, y = fsl_sample(
        rng, images, labels, batch_size, way, shot, qry_shot, disjoint, shuffled_labels,
    )
    x = jax.device_put(x, device)
    x = x / 255
    x = preprocess_fn(x)
    y = jax.device_put(y, device)

    image_shape = x.shape[-3:]
    x_spt, x_qry = jnp.split(x, (shot,), 2)
    x_spt = x_spt.reshape(batch_size, way * shot, *image_shape)
    x_qry = x_qry.reshape(batch_size, way * qry_shot, *image_shape)
    y_spt, y_qry = jnp.split(y, (shot,), 2)
    y_spt = y_spt.reshape(batch_size, way * shot)
    y_qry = y_qry.reshape(batch_size, way * qry_shot)
    return x_spt, y_spt, x_qry, y_qry


def continual_learning_sample(
    rng, images, labels, num_tasks, way, shot, qry_shot, stratified=True, disjoint=True
):
    rng_spt, rng_qry = split(rng)
    x_spt, y_spt = sample_tasks(rng, images, labels, num_tasks, way, shot, disjoint)

    if stratified:
        x_qry, y_qry = sample_tasks(
            rng,
            images,
            labels,
            num_tasks,
            min(qry_shot, images.shape[0]),
            max(1, qry_shot // images.shape[0]),
            disjoint,
        )
        x_qry = flatten(x_qry, (1, 2))
        y_qry = flatten(y_qry, (1, 2))
    else:
        idxs = shuffle_along_axis(
            rng_qry,
            jnp.arange(images.shape[0] * images.shape[1])[None, :].repeat(num_tasks, 0),
            1,
        )[:, :qry_shot]

        x_qry = images[idxs // images.shape[1], idxs % images.shape[1]]
        y_qry = labels[idxs // images.shape[1], idxs % images.shape[1]]

    return x_spt, y_spt, x_qry, y_qry


def continual_learning_sample_build_transfer(
    rng,
    images,
    labels,
    batch_size,
    way,
    shot,
    qry_shot,
    preprocess_fn,
    device,
    stratified=True,
    disjoint=False,
    shuffled_labels=None,  # Compatibility
):
    x_spt, y_spt, x_qry, y_qry = continual_learning_sample(
        rng, images, labels, batch_size, way, shot, qry_shot, stratified, disjoint,
    )

    image_shape = x_spt.shape[-3:]
    x_spt = x_spt.reshape(batch_size, way * shot, *image_shape)
    y_spt = y_spt.reshape(batch_size, way * shot)

    x_spt = preprocess_fn(jax.device_put(x_spt, device))
    y_spt = jax.device_put(y_spt, device)
    x_qry = preprocess_fn(jax.device_put(x_qry, device))
    y_qry = jax.device_put(y_qry, device)
    x_qry = jnp.concatenate((x_spt, x_qry), 1)
    y_qry = jnp.concatenate((y_spt, y_qry), 1)

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
        if self.n + self.batch_size <= len(self.X):
            idxs = self.order[self.n : self.n + self.batch_size]
            self.n += self.batch_size
            return self.X[idxs], self.y[idxs]
        elif self.n < len(self.X) and self.keep_last:
            idxs = self.order[self.n :]
            self.n += self.batch_size
            return self.X[idxs], self.y[idxs]
        else:
            raise StopIteration
