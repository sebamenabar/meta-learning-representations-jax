from multiprocessing import Pool
from sklearn.linear_model import LogisticRegression

import numpy as onp

import jax
from jax.random import split
from jax import vmap, tree_multimap, numpy as jnp, partial

from lib import xe_and_acc, flatten

# from data import augment as augment_fn
from data.sampling import BatchSampler, fsl_build


def test_fsl_maml(
    rng,
    slow_params,
    fast_params,
    inner_lr,
    slow_state,
    fast_state,
    num_batches,
    inner_opt_init,
    sample_fn,
    batched_outer_loop,
    normalize_fn,
    build_fn,
    augment_fn=None,
    device=None,
):
    results = []
    for _ in range(num_batches):
        rng, rng_augment, rng_sample, rng_apply = split(rng, 4)
        x, y = sample_fn(rng_sample)
        x = jax.device_put(x, device)
        y = jax.device_put(y, device)
        x = x / 255
        x_spt, y_spt, x_qry, y_qry = build_fn(x, y)
        if augment_fn:
            x_spt = augment_fn(rng_augment, flatten(x_spt, (0, 1))).reshape(
                *x_spt.shape
            )
        x_spt = normalize_fn(x_spt)
        x_qry = normalize_fn(x_qry)

        loss, (_, _, info) = batched_outer_loop(
            slow_params,
            fast_params,
            inner_lr,
            slow_state,
            fast_state,
            inner_opt_init(fast_params),
            split(rng_apply, x_spt.shape[0]),
            x_spt,
            y_spt,
            x_qry,
            y_qry,
        )
        results.append((loss, info))
    results = tree_multimap(lambda x, *xs: jnp.stack(xs), results[0], *results)
    return results


# @jax.jit
def normalize(x):
    norm = jax.numpy.linalg.norm(x, axis=-1, keepdims=True)
    return x / norm


def test_fsl_embeddings(
    rng,
    apply_fn,
    sample_fn,
    num_batches,
    classifier="LR",
    is_norm=True,
    device=None,
    pool=0,
    n_jobs=None,
):

    if pool > 0:
        pool = Pool(pool)
    else:
        pool = None

    if classifier == "LR":
        fit_fn = lr_fit_eval

    preds = []
    targets = []
    for i in range(num_batches):
        rng, rng_sample = split(rng)
        x_spt, y_spt, x_qry, y_qry = sample_fn(rng)  # Batched tasks
        batch_size = x_spt.shape[0]
        # x_spt = jax.device_put(x_spt, device)
        # x_qry = jax.device_put(x_qry, device)
        spt_features = vmap(apply_fn)(x_spt)
        qry_features = vmap(apply_fn)(x_qry)

        spt_features = spt_features.reshape(batch_size, x_spt.shape[1], -1)
        qry_features = qry_features.reshape(batch_size, x_qry.shape[1], -1)

        if is_norm:
            spt_features = normalize(spt_features)
            qry_features = normalize(qry_features)

        # spt_features = jax.device_get(spt_features)
        # qry_features = jax.device_get(qry_features)
        # y_spt = jax.device_get(y_spt)
        # y_qry = jax.device_get(y_qry)
        spt_features = onp.array(spt_features)
        qry_features = onp.array(qry_features)
        y_spt = onp.array(y_spt)
        y_qry = onp.array(y_qry)

        targets.append(y_qry)

        if pool is not None:
            preds.append(
                pool.starmap_async(fit_fn, zip(spt_features, y_spt, qry_features))
            )
        else:
            for i in range(batch_size):
                preds.append(
                    lr_fit_eval(spt_features[i], y_spt[i], qry_features[i], n_jobs=n_jobs)
                )

    if pool is not None:
        preds = onp.concatenate([onp.stack(p.get(5)) for p in preds])
    else:
        preds = onp.stack(preds)
    targets = onp.concatenate(targets)
    return preds, targets


def lr_fit_eval(X, y, X_test, n_jobs=None, predict_train=False):
    clf = LogisticRegression(
        penalty="l2",
        random_state=0,
        C=10,
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        multi_class="multinomial",
        n_jobs=n_jobs,
    )
    clf.fit(X, y)
    query_y_pred = clf.predict(X_test)
    if predict_train:
        spt_y_pred = clf.predict(X)
        return spt_y_pred, query_y_pred

    return query_y_pred


def forward_loader(pred_fn, loader, device, is_norm=False, normalize_fn=None):
    all_preds = []
    all_targets = []
    for X, y in loader:
        X = jax.device_put(X, device)
        X = X / 255
        if normalize_fn:
            X = normalize_fn(X)
        preds = pred_fn(X)
        if is_norm:
            preds = normalize(preds.reshape(preds.shape[0], -1))
        all_preds.append(onp.array(preds))
        all_targets.append(onp.array(y))
    return onp.concatenate(all_preds), onp.concatenate(all_targets)


def test_sup_lr(emb_fn, spt_loader, qry_loader, device, preprocess_fn=None, n_jobs=2):
    spt_preds, spt_targets = forward_loader(
        emb_fn, spt_loader, device, True, preprocess_fn
    )
    qry_preds, qry_targets = forward_loader(
        emb_fn, qry_loader, device, True, preprocess_fn
    )

    cosine_distance = spt_preds @ qry_preds.transpose()
    max_idx = onp.argmax(cosine_distance, axis=0)
    return spt_targets[max_idx], qry_targets


def test_sup_cosine(pred_fn, spt_loader, qry_loader, device, preprocess_fn=None):
    spt_preds, spt_targets = forward_loader(
        pred_fn, spt_loader, device, True, preprocess_fn
    )
    qry_preds, qry_targets = forward_loader(
        pred_fn, qry_loader, device, True, preprocess_fn
    )

    cosine_distance = spt_preds @ qry_preds.transpose()
    max_idx = onp.argmax(cosine_distance, axis=0)
    return spt_targets[max_idx], qry_targets


class SupervisedCosineTester:
    def __init__(
        self,
        rng,
        spt_images,
        spt_labels,
        qry_images,
        qry_labels,
        batch_size,
        # forward_fn,
        preprocess_fn=None,
        device=None,
    ):
        self.device = device
        # self.forward_fn = forward_fn
        self.preprocess_fn = preprocess_fn
        rng_spt, rng_qry = split(rng)
        self.spt_loader = BatchSampler(
            rng_spt, spt_images, spt_labels, batch_size, True, True
        )
        self.qry_loader = BatchSampler(
            rng_spt, qry_images, qry_labels, batch_size, True, True
        )

    def __call__(self, forward_fn):
        # forward_fn = partial(forward_fn, slow_params, slow_state, rng,)
        preds, targets = test_sup_cosine(
            forward_fn,
            self.spt_loader,
            self.qry_loader,
            self.device,
            self.preprocess_fn,
        )
        acc = (preds == targets).astype(onp.float).mean()
        return acc


class FSLLRTester:
    def __init__(
        self,
        images,
        labels,
        batch_size,
        total_num_tasks,
        way,
        shot,
        qry_shot,
        preprocess_fn=None,
        device=None,
        pool=0,
    ):
        self.total_num_tasks = total_num_tasks
        self.batch_size = batch_size
        self.pool = pool
        self.sample_fn = partial(
            fsl_sample_transfer_build,
            images=images,
            labels=labels,
            batch_size=batch_size,
            way=way,
            shot=shot,
            qry_shot=qry_shot,
            device=device,
            disjoint=False,
            preprocess_fn=preprocess_fn,
            shuffled_labels=True,
        )

    def __call__(
        self, forward_fn, rng,
    ):
        preds, targets = test_fsl_embeddings(
            rng,
            forward_fn,
            self.sample_fn,
            self.total_num_tasks // self.batch_size,
            pool=self.pool,
        )
        acc = (preds == targets).astype(onp.float).mean()
        return acc


class SupervisedStandardTester:
    def __init__(
        self, rng, test_images, test_labels, batch_size, normalize_fn=None, device=None,
    ):
        self.test_images = test_images
        self.test_images = test_labels
        self.loader = BatchSampler(
            rng, test_images, test_labels, batch_size, True, True,
        )
        self.normalize_fn = normalize_fn
        self.device = device

    def __call__(
        self, forward_fn,
    ):
        preds, targets = forward_loader(
            forward_fn, self.loader, self.device, normalize_fn=self.normalize_fn,
        )
        loss, acc = xe_and_acc(preds, targets)

        return loss.mean(), acc.mean()
