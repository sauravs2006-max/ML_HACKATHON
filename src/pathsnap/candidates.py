import numpy as np
from scipy.spatial import cKDTree
from .geo import seg_dist

K, RADIUS, STEP = 8, 100.0, 15.0


def arc_index(trip, step=STEP):
    n = np.maximum(np.ceil(trip["alen"] / step).astype(int), 1) + 1
    aid = np.repeat(np.arange(len(n)), n)
    f = np.concatenate([np.linspace(0, 1, k) for k in n])
    pts = trip["A"][aid] + f[:, None] * (trip["B"] - trip["A"])[aid]
    return cKDTree(pts), aid


def generate(trip, k=K, radius=RADIUS, pool=96):
    """Return (N,k) arc ids (-1 = empty), distances and projection fractions."""
    tree, aid = trip.get("_tree") or arc_index(trip)
    trip["_tree"] = (tree, aid)
    q = min(pool, len(aid))
    _, nn = tree.query(trip["pxy"], k=q)
    ids = aid[nn.reshape(len(trip["pxy"]), -1)]
    d, t = seg_dist(trip["pxy"][:, None, :], trip["A"][ids], trip["B"][ids])
    n = len(ids)
    C, Dk, Tk = np.full((n, k), -1), np.full((n, k), np.inf), np.zeros((n, k))
    for i in range(n):
        u, first = np.unique(ids[i], return_index=True)
        o = np.argsort(d[i, first])[:k]
        o = o[d[i, first][o] <= radius]
        m = len(o)
        C[i, :m], Dk[i, :m], Tk[i, :m] = u[o], d[i, first][o], t[i, first][o]
    return C, Dk, Tk


def miss_rate(C, labels):
    return float(np.mean(~(C == labels[:, None]).any(1)))
