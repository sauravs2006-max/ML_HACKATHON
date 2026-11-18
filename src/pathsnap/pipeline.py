import os, json, time
import numpy as np
import joblib
from . import io, labels, hmm
from .features import candidate_table, FEATURES
from .evaluate import continuity, route_f1

ART = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts")


def load_artifacts(art=ART):
    cfg = json.load(open(os.path.join(art, "config.json")))
    return joblib.load(os.path.join(art, cfg["model_file"])), cfg


def candidate_probs(trip, model):
    df, (C, D, T) = candidate_table(trip)
    P = np.zeros(C.shape)
    P[df.point.values, df["rank"].values] = model.predict_proba(df[FEATURES].values)[:, 1]
    return C, D, T, P


def emissions(P, D, mode="classifier", sigma=7.0):
    if mode == "distance":
        return -0.5 * (D / sigma) ** 2
    Pn = P / np.maximum(P.sum(1, keepdims=True), 1e-12)
    return np.log(np.clip(Pn, 1e-6, 1))


def predict(trip, model, hmm_params=None, mode="classifier"):
    t0 = time.time()
    C, D, T, P = candidate_probs(trip, model)
    clf = np.where((C >= 0).any(1), C[np.arange(len(C)), np.argmax(np.where(C >= 0, P, -1), 1)], -1)
    dec, breaks = hmm.viterbi(trip, C, T, emissions(P, D, mode), hmm_params)
    return dict(classifier=clf, hmm=dec, breaks=breaks, C=C, seconds=time.time() - t0)


def score(trip, pred_seq, lab):
    ok = pred_seq >= 0
    seq = pred_seq[ok]
    return dict(
        top1_acc=float(np.mean(pred_seq == lab)),
        route_continuity=continuity(trip["arcs"], seq),
        route_f1=route_f1(seq, trip["route"]),
    )


def evaluate(trip, out):
    lab, _ = labels.align_route(trip)
    return {k: score(trip, out[k], lab) for k in ("classifier", "hmm")} | dict(
        candidate_miss_rate=float(np.mean(~(out["C"] == lab[:, None]).any(1))))
