import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, accuracy_score


def pick(df, score):
    """Per GPS point keep the candidate with highest score -> 0/1 selection vector aligned with df."""
    s = pd.Series(score, index=df.index)
    best = s.groupby([df.trip, df.point]).idxmax()
    sel = np.zeros(len(df), int)
    sel[df.index.get_indexer(best.values)] = 1
    return sel


def continuity(arcs, seq):
    seq = np.asarray(seq)
    ch = seq[1:] != seq[:-1]
    if not ch.any():
        return 1.0
    a, b = seq[:-1][ch], seq[1:][ch]
    return float(np.mean(arcs[a, 1] == arcs[b, 0]))


def route_f1(pred_arcs, ref_arcs):
    p, r = set(pred_arcs), set(ref_arcs)
    tp = len(p & r)
    return 0.0 if tp == 0 else 2 * tp / (len(p) + len(r))


def candidate_metrics(df, prob, sel=None):
    sel = pick(df, prob) if sel is None else sel
    y = df.y.values
    pts = df.assign(hit=sel * y).groupby(["trip", "point"]).hit.max()
    out = dict(
        top1_acc=float(pts.mean()),
        macro_f1=float(f1_score(y, sel, average="macro")),
        precision=float(precision_score(y, sel)),
        recall=float(recall_score(y, sel)),
        cand_accuracy=float(accuracy_score(y, sel)),
    )
    if prob is not None:
        out["log_loss"] = float(log_loss(y, np.clip(prob, 1e-6, 1 - 1e-6)))
    return out


def sequence_metrics(df, sel, arcs_by_trip, routes_by_trip):
    cont, rf = [], []
    for tid, g in df[sel == 1].sort_values(["trip", "point"]).groupby("trip"):
        seq = g.arc.values
        cont.append(continuity(arcs_by_trip[tid], seq))
        rf.append(route_f1(seq, routes_by_trip[tid]))
    return dict(route_continuity=float(np.mean(cont)), route_f1=float(np.mean(rf)))
