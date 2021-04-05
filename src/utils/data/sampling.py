import numpy as onp
from numpy import arange
from jax import random
from jax.random import split, permutation
from .datasets import Subset


class BatchSampler:
    def __init__(
        self,
        rng,
        dataset,
        batch_size,
        shuffle=True,
        keep_last=False,
        collate_fn=lambda x: x,
        dataset_is_array=False,
    ):
        self.rng = rng
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.keep_last = keep_last
        self.collate_fn = collate_fn
        self.dataset_is_array = dataset_is_array

        self.__len = (len(dataset) // batch_size) + (
            ((len(dataset) % batch_size) > 0) and keep_last
        )

    def __len__(self):
        return self.__len

    def __iter__(self):
        order = arange(len(self.dataset))
        if self.shuffle:
            self.rng, rng_shuffle = split(self.rng)
            order = permutation(rng_shuffle, order)
        n = 0

        def __iter():
            nonlocal n
            while n < len(order):
                indexes = None
                if n + self.batch_size <= len(order):
                    indexes = order[n : n + self.batch_size]
                elif self.keep_last:
                    indexes = order[n:]

                if indexes is None:
                    raise Exception("indexes should never be None")

                if self.dataset_is_array:
                    yield self.dataset[
                        list(indexes)
                    ]  # For some reason using nparray fails
                else:
                    yield self.collate_fn([self.dataset[o] for o in order[indexes]])

                n += self.batch_size
                if (n > len(order)) or (
                    ((len(order) - n) < self.batch_size) and (not self.keep_last)
                ):
                    break

        return __iter()


def sample_trajectory(
    rng,
    dataset,
    targets,
    traj_length,
    num_train_samples=15,
    sort=False,
    shuffle=True,
):
    rng, rng_classes = split(rng)
    selected_classes = random.choice(
        rng_classes, onp.unique(targets), (traj_length,), replace=False
    )

    if sort:
        selected_classes = sorted(selected_classes)

    train_indexes = []
    test_indexes = []
    for _class in selected_classes:
        _indexes = onp.nonzero(_class == targets)[0]
        if shuffle:
            rng, rng_shuffle = split(rng)

            _indexes = random.permutation(rng_shuffle, _indexes)

        train_indexes.append(_indexes[:num_train_samples])
        test_indexes.append(_indexes[num_train_samples:])

    train_dataset = Subset(
        dataset,
        onp.concatenate(train_indexes),
    )
    test_dataset = Subset(dataset, onp.concatenate(test_indexes))

    return train_dataset, test_dataset


def make_test_iterators(
    rng,
    test_dataset,
    targets,
    n,
    traj_length,
    sort,
    shuffle,
    batch_size=256,
    dataset_is_array=True,
):
    test_train_iterators, test_test_iterators = [], []
    for _ in range(n):
        rng, rng_classes = split(rng)

        test_train_dataset, test_test_dataset = sample_trajectory(
            rng_classes,
            test_dataset,
            targets,
            traj_length,
            sort=sort,
            shuffle=shuffle,
        )

        test_train_iterator = BatchSampler(
            random.PRNGKey(0),  # Since we do not shuffle the random key does not matter
            test_train_dataset,
            batch_size,
            shuffle=False,
            keep_last=True,
            dataset_is_array=dataset_is_array,
        )

        test_test_iterator = BatchSampler(
            random.PRNGKey(0),
            test_test_dataset,
            batch_size,
            shuffle=False,
            keep_last=True,
            dataset_is_array=dataset_is_array,
        )

        test_train_iterators.append(test_train_iterator)
        test_test_iterators.append(test_test_iterator)

    return test_train_iterators, test_test_iterators