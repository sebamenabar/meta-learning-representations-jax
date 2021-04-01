import numpy as onp
from jax.random import split, choice
from utils import use_self_as_default, shuffle_along_axis


class Subset:
    def __init__(self, dataset, indexes):
        self.dataset = dataset
        self.indexes = indexes
        assert onp.max(indexes) <= len(dataset)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        return self.dataset[self.indexes[index]]


class SimpleDataset:
    def __init__(self, inputs, targets, *args, **kwargs):
        self.inputs = inputs
        self.targets = targets
        self.args = args
        self.kwargs = kwargs

        self.__len = len(self.inputs)

    def __len__(self):
        return self.__len

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def take(self, index):
        # Useful if your data consists of ndarrays
        return self.inputs.take(index, axis=0), self.targets.take(index, axis=0)

    @property
    def isarray(self):
        return (type(self.inputs) is onp.ndarray) and (
            type(self.targets) is onp.ndarray
        )

    def get_classes_subset(self, classes, sort_by_class=False):
        if sort_by_class:
            classes = sorted(classes)

        if self.isarray:
            return self.__get_classes_subset_ndarray(classes)
        else:
            return self.__get_classes_subset(classes)

    def __get_classes_subset(self, classes):
        inputs = []
        targets = []

        for _class in classes:
            indexes = [i for (i, t) in enumerate(self.targets) if t == _class]
            assert len(
                indexes
            ), f"Class {_class} not in targets (all classes: {classes})"
            inputs += [self.inputs[i] for i in indexes]
            targets += [self.targets[i] for i in indexes]

        return type(self)(
            inputs,
            targets,
            *self.args,
            **self.kwargs,
        )

    def __get_classes_subset_ndarray(self, classes):
        inputs = []
        targets = []

        for _class in classes:
            indexes = self.targets == _class
            assert (
                indexes.any()
            ), f"Class {_class} not in targets (all classes: {classes})"
            inputs.append(self.inputs[indexes])
            targets.append(self.targets[indexes])

        return type(self)(
            onp.concatenate(inputs),
            onp.concatenate(targets),
            *self.args,
            **self.kwargs,
        )


class ImageDataset(SimpleDataset):
    def __init__(self, inputs, targets, mean, std):
        super().__init__(inputs, targets, mean, std)
        self.mean = mean
        self.std = std

    def normalize(self, input):
        return (input - self.mean) / self.std


class MetaDataset:
    def __init__(
        self,
        dataset,
        batch_size,
        way,
        shot,
        qry_shot,
        cl_qry_shot,
        disjoint=False,
        disjoint_cl_qry=False,
        targets_key="targets",
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.way = way
        self.shot = shot
        self.qry_shot = qry_shot
        self.cl_qry_shot = cl_qry_shot
        self.disjoint = disjoint
        self.disjoint_cl_qry = disjoint_cl_qry
        self.targets = self.make_targets_mapping(getattr(dataset, targets_key))
        self.num_samples_per_class = {t: len(idxs) for t, idxs in self.targets.items()}

    def get_sampler(self, rng):
        def yielder():
            nonlocal rng
            while True:
                rng, _rng = split(rng)
                yield self.sample(_rng)

        return yielder()

    @staticmethod
    def make_targets_mapping(targets):
        uniques = set(targets)
        return {
            j: onp.array([i for (i, _t) in enumerate(targets) if _t == t])
            for j, t in enumerate(uniques)
        }

    @staticmethod
    def sample_indexes_per_class(*args, **kwargs):
        return sample_indexes_per_class(*args, **kwargs)

    def get_spt_qry_indexes(self, sampled_classes, indexes):
        dataset_indexes = []
        for i, (_classes, _indexes) in enumerate(zip(sampled_classes, indexes)):
            dataset_indexes.append([])
            for _class, _class_indexes in zip(_classes, _indexes):
                dataset_indexes[i].append(self.targets[_class][_class_indexes])
        return dataset_indexes
        # return jnp.array(dataset_indexes)

    def get_spt_qry_datapoints(self, indexes):
        inputs = []
        targets = []
        for _indexes in indexes:
            _inputs = []
            _targets = []
            for _class_indexes in _indexes:
                for _index in _class_indexes:
                    _input, _target = self.dataset[_index]
                    _inputs.append(_input)
                    _targets.append(_target)
            inputs.append(_inputs)
            targets.append(_targets)

        return inputs, targets

    def get_cl_qry_datapoints(self, indexes):
        inputs = []
        targets = []
        for _indexes in indexes:
            _inputs = []
            _targets = []
            for _index in _indexes:
                _input, _target = self.dataset[_index]
                _inputs.append(_input)
                _targets.append(_target)
            inputs.append(_inputs)
            targets.append(_targets)
        return inputs, targets

    @use_self_as_default(
        "batch_size",
        "way",
        "shot",
        "qry_shot",
        "cl_qry_shot",
        "disjoint",
        "disjoint_cl_qry",
    )
    def sample(
        self,
        rng,
        batch_size=None,
        way=None,
        shot=None,
        qry_shot=None,
        cl_qry_shot=None,
        disjoint=None,
        disjoint_cl_qry=None,
    ):
        rng, rng_classes, rng_samples, rng_cl = split(rng, 4)
        sampled_classes = sample_classes_indexes(
            rng_classes, len(self.targets), batch_size, way, disjoint
        )
        spt_indexes, qry_indexes = self.sample_indexes_per_class(
            rng_samples, self.num_samples_per_class, sampled_classes, shot, qry_shot
        )

        cl_qry_indexes = sample_cl_qry(
            rng_cl, len(self.dataset), batch_size, cl_qry_shot, disjoint_cl_qry
        )

        spt_dataset_indexes = self.get_spt_qry_indexes(sampled_classes, spt_indexes)
        qry_dataset_indexes = self.get_spt_qry_indexes(sampled_classes, qry_indexes)

        spt_inputs, spt_targets = self.get_spt_qry_datapoints(spt_dataset_indexes)
        qry_inputs, qry_targets = self.get_spt_qry_datapoints(qry_dataset_indexes)
        cl_qry_inputs, cl_qry_targets = self.get_cl_qry_datapoints(cl_qry_indexes)

        return (
            spt_inputs,
            spt_targets,
            qry_inputs,
            qry_targets,
            cl_qry_inputs,
            cl_qry_targets,
        )


def sample_cl_qry(rng, dataset_size, batch_size, shot, disjoint):
    return sample_classes_indexes(rng, dataset_size, batch_size, shot, disjoint)


def sample_classes_indexes(rng, num_classes, batch_size, way, disjoint):
    if disjoint:
        sampled_classes = choice(
            rng,
            onp.arange(num_classes),
            (batch_size * way, 1),
            replace=False,
        )
    else:
        sampled_classes = shuffle_along_axis(
            rng, onp.arange(num_classes)[None, :].repeat(batch_size, 0), 1
        )[:, :way]

    sampled_classes = sampled_classes.reshape(batch_size, way)

    return sampled_classes


def sample_indexes_per_class(
    rng,
    num_samples_per_class,
    sampled_classes,
    shot,
    qry_shot,
):
    spt_indexes = [[] for _ in range(len(sampled_classes))]
    qry_indexes = [[] for _ in range(len(sampled_classes))]
    # Iterate over batch dimension
    for i, indexes in enumerate(sampled_classes):
        for class_index in indexes:
            rng, _rng = split(rng)
            sampled_indexes_per_class = list(
                choice(
                    _rng,
                    onp.arange((num_samples_per_class[class_index])),
                    (shot + qry_shot,),
                    replace=False,
                )
            )
            spt_indexes[i].append(sampled_indexes_per_class[:shot])
            qry_indexes[i].append(sampled_indexes_per_class[shot:])

    return onp.array(spt_indexes), onp.array(qry_indexes)


class MetaDatasetArray(MetaDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_samples_per_class = onp.array([len(l) for l in self.targets.values()])
        assert (
            num_samples_per_class[0] == num_samples_per_class
        ).all(), "All classes need to have the same number of samples to use an array dataset"
        self.num_samples_per_class = num_samples_per_class[0]
        self.targets = onp.stack(list(self.targets.values()), 0)

    @staticmethod
    def sample_indexes_per_class(
        rng, num_samples_per_class, sampled_classes, shot, qry_shot
    ):
        sampled_indexes = shuffle_along_axis(
            rng,
            onp.arange(num_samples_per_class)[None].repeat(
                sampled_classes.shape[0] * sampled_classes.shape[1], 0
            ),
            1,
        )[:, : shot + qry_shot].reshape(
            sampled_classes.shape[0], sampled_classes.shape[1], shot + qry_shot
        )

        return sampled_indexes[:, :, :shot], sampled_indexes[:, :, shot:]

    def get_spt_qry_indexes(self, sampled_classes, indexes):
        return self.targets[
            onp.expand_dims(sampled_classes, -1).repeat(indexes.shape[2], -1), indexes
        ]

    def get_spt_qry_datapoints(self, indexes):
        return self.dataset.take(indexes)

    def get_cl_qry_datapoints(self, indexes):
        return self.dataset.take(indexes)