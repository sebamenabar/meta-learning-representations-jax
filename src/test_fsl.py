from multiprocessing import Pool
from sklearn.linear_model import LogisticRegression

import numpy as onp

import jax
from jax import vmap
from jax.random import split


def meta_test(
    rng,
    apply_fn,
    sample_fn,
    num_batches,
    classifier="LR",
    is_norm=True,
    device=None,
    pool=0,
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
        x_spt = jax.device_put(x_spt, device)
        x_qry = jax.device_put(x_qry, device)
        spt_features = vmap(apply_fn)(x_spt)
        qry_features = vmap(apply_fn)(x_qry)

        spt_features = spt_features.reshape(batch_size, x_spt.shape[1], -1)
        qry_features = qry_features.reshape(batch_size, x_qry.shape[1], -1)

        if is_norm:
            spt_features = jax.nn.normalize(spt_features)
            qry_features = jax.nn.normalize(qry_features)

        spt_features = jax.device_get(spt_features)
        qry_features = jax.device_get(qry_features)
        y_spt = jax.device_get(y_spt)
        y_qry = jax.device_get(y_qry)

        targets.append(y_qry)

        if pool is not None:
            preds.append(pool.starmap_async(fit_fn, zip(spt_features, y_spt, qry_features)))
        else:
            for i in range(batch_size):
                preds.append(lr_fit_eval(spt_features[i], y_spt[i], qry_features[i], n_jobs=8))
    
    if pool is not None:
        preds = onp.concatenate([onp.stack(p.get(5)) for p in preds])
    else:
        preds = onp.stack(preds)
    targets = onp.concatenate(targets)
    return preds, targets


def lr_fit_eval(X, y, X_test, n_jobs=1):
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
    return query_y_pred
