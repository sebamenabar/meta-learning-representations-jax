from multiprocessing import Pool
from sklearn.linear_model import LogisticRegression

import numpy as onp

import jax
from jax.random import split
from jax import vmap, tree_multimap, numpy as jnp


def test_fsl_maml(
    rng,
    slow_params,
    fast_params,
    slow_state,
    fast_state,
    num_batches,
    inner_opt_init,
    sample_fn,
    batched_outer_loop,
):
    results = []
    for _ in range(num_batches):
        rng, rng_sample, rng_apply = split(rng, 3)
        x_spt_test, y_spt_test, x_qry_test, y_qry_test = sample_fn(rng_sample)

        loss, (_, _, info) = batched_outer_loop(
            slow_params,
            fast_params,
            slow_state,
            fast_state,
            inner_opt_init(fast_params),
            split(rng_apply, x_spt_test.shape[0]),
            x_spt_test,
            y_spt_test,
            x_qry_test,
            y_qry_test,
        )
        results.append((loss, info))
    results = tree_multimap(lambda x, *xs: jnp.stack(xs), results[0], *results)
    return results


@jax.jit
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
                    lr_fit_eval(spt_features[i], y_spt[i], qry_features[i], n_jobs=4)
                )

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
