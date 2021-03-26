from numpy import arange
from jax.random import split, permutation


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