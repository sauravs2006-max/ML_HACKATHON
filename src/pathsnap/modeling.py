import os, json, time
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from .features import FEATURES
from .evaluate import candidate_metrics, sequence_metrics, pick
from . import io

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
ART = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts")
SEED = 42


def load_tables():
    d = os.path.join(OUT, "data")
    return pd.read_parquet(f"{d}/train.parquet"), pd.read_parquet(f"{d}/student_test.parquet")


def network_refs(split, trips):
    arcs, routes = {}, {}
    for t in trips:
        _, _, a, r = io.load_raw(split, t)
        arcs[t], routes[t] = a, r
    return arcs, routes


def proba(model, X):
    return model.predict_proba(X)[:, 1]


def cross_validate(model, tr, folds=5):
    gkf = GroupKFold(n_splits=folds)
    X, y, g = tr[FEATURES].values, tr.y.values, tr.trip.values
    res, oof = [], np.zeros(len(tr))
    for f, (a, b) in enumerate(gkf.split(X, y, g)):
        m = clone(model).fit(X[a], y[a])
        oof[b] = proba(m, X[b])
        res.append(candidate_metrics(tr.iloc[b], oof[b]))
    r = pd.DataFrame(res)
    return r, oof


def run(name, family, config, model, tr, te, folds=5, save=True):
    t0 = time.time()
    cv, _ = cross_validate(model, tr, folds)
    cv_time = time.time() - t0
    t0 = time.time()
    m = clone(model).fit(tr[FEATURES].values, tr.y.values)
    train_time = time.time() - t0
    t0 = time.time()
    p = proba(m, te[FEATURES].values)
    infer = time.time() - t0
    test = candidate_metrics(te, p)
    test.update(sequence_metrics(te, pick(te, p), *network_refs("student_test", te.trip.unique())))
    tr_m = candidate_metrics(tr, proba(m, tr[FEATURES].values))
    res = dict(
        model=name, family=family, config=config,
        cv_macro_f1=cv.macro_f1.mean(), cv_macro_f1_std=cv.macro_f1.std(), cv_top1=cv.top1_acc.mean(),
        cv_folds=cv.macro_f1.round(4).tolist(),
        train_macro_f1=tr_m["macro_f1"], train_top1=tr_m["top1_acc"],
        **{f"test_{k}": v for k, v in test.items()},
        train_time_s=train_time, cv_time_s=cv_time,
        infer_ms_per_point=1000 * infer / te.groupby(["trip", "point"]).ngroups,
    )
    if save:
        os.makedirs(f"{OUT}/models", exist_ok=True)
        os.makedirs(f"{ART}/models", exist_ok=True)
        json.dump(res, open(f"{OUT}/models/{name}.json", "w"), indent=2)
        joblib.dump(m, f"{ART}/models/{name}.joblib")
    return res, m, p
