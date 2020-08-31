import numpy as onp

import jax
from test_utils import normalize, lr_fit_eval


def forward_loader(pred_fn, loader, device, is_norm=False, preprocess_fn=None):
    all_preds = []
    all_targets = []
    for X, y in loader:
        if preprocess_fn:
            X = preprocess_fn(X)
        X = jax.device_put(X, device)
        preds = pred_fn(X)
        if is_norm:
            preds = normalize(preds.reshape(preds.shape[0], -1))
        all_preds.append(onp.array(preds))
        all_targets.append(onp.array(y))
    return onp.concatenate(all_preds), onp.concatenate(all_targets)


def test_sup(pred_fn, data_loader, device):
    all_preds, all_targets = forward_loader(pred_fn, data_loader, device)
    # all_preds = onp.concatenate(all_preds)
    # all_targets = onp.concatenate(all_targets)

    return all_preds, all_targets


def test_sup_lr(pred_fn, spt_loader, qry_loader, is_norm, device, n_jobs=4):
    spt_preds, spt_targets = forward_loader(pred_fn, spt_loader, device, is_norm)
    qry_preds, qry_targets = forward_loader(pred_fn, qry_loader, device, is_norm)

    qry_y_preds = lr_fit_eval(
        spt_preds.reshape(spt_preds.shape[0], -1),
        spt_targets,
        qry_preds.reshape(qry_preds.shape[0], -1),
        n_jobs=n_jobs,
    )
    return qry_y_preds, qry_targets


def test_sup_cosine(pred_fn, spt_loader, qry_loader, device, preprocess_fn=None):
    spt_preds, spt_targets = forward_loader(pred_fn, spt_loader, device, True, preprocess_fn)
    qry_preds, qry_targets = forward_loader(pred_fn, qry_loader, device, True, preprocess_fn)

    cosine_distance = spt_preds @ qry_preds.transpose()
    max_idx = onp.argmax(cosine_distance, axis=0)
    return spt_targets[max_idx], qry_targets
