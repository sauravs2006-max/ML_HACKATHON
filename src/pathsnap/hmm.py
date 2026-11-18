import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

DEFAULTS = dict(beta=5.0, emis_weight=1.0, limit=2000.0, gap_break=120.0)


class RoadGraph:
    def __init__(self, trip, cache=400):
        a = trip["arcs"]
        n = len(trip["nodes"])
        self.G = csr_matrix((trip["alen"] + 1e-3, (a[:, 0], a[:, 1])), shape=(n, n))
        self.cache, self.size = {}, cache

    def dist_from(self, node, limit):
        key = (node, limit)
        if key not in self.cache:
            if len(self.cache) > self.size:
                self.cache.clear()
            self.cache[key] = dijkstra(self.G, indices=node, limit=limit)
        return self.cache[key]


def transition(trip, graph, ca, ta, cb, tb, gc, dt, p):
    """Log transition matrix (len(ca) x len(cb)) using |route distance - great-circle distance| / beta."""
    arcs, L = trip["arcs"], trip["alen"]
    limit = max(p["limit"], 3 * gc)
    route = np.full((len(ca), len(cb)), np.inf)
    for i, a in enumerate(ca):
        if a < 0:
            continue
        sp = graph.dist_from(arcs[a, 1], limit)
        ok = cb >= 0
        route[i, ok] = (1 - ta[i]) * L[a] + sp[arcs[cb[ok], 0]] + tb[ok] * L[cb[ok]]
        same = cb == a
        route[i, same] = np.abs(tb[same] - ta[i]) * L[a]
    beta = p["beta"] * max(1.0, np.sqrt(dt))
    return -np.abs(route - gc) / beta


def viterbi(trip, C, T, emis, p=None):
    p = {**DEFAULTS, **(p or {})}
    graph = RoadGraph(trip)
    valid = np.nonzero((C >= 0).any(1))[0]
    out = np.full(len(C), -1)
    if len(valid) == 0:
        return out, 0
    pxy, t = trip["pxy"], trip["t"]
    E = np.where(C >= 0, p["emis_weight"] * emis, -np.inf)
    n, k = len(valid), C.shape[1]
    score = np.empty((n, k))
    back = np.full((n, k), -1)
    score[0] = E[valid[0]]
    breaks = 0
    for s in range(1, n):
        i, j = valid[s - 1], valid[s]
        dt = t[j] - t[i]
        gc = float(np.hypot(*(pxy[j] - pxy[i])))
        if dt <= p["gap_break"]:
            M = score[s - 1][:, None] + transition(trip, graph, C[i], T[i], C[j], T[j], gc, dt, p)
            back[s] = M.argmax(0)
            best = M.max(0)
        else:
            best = np.full(k, -np.inf)
        if not np.isfinite(best).any():
            breaks += 1
            back[s] = -1
            best = np.zeros(k)
        score[s] = best + E[j]
    j = int(np.argmax(score[-1]))
    for s in range(n - 1, -1, -1):
        out[valid[s]] = C[valid[s], j]
        j = back[s, j] if back[s, j] >= 0 else (int(np.argmax(score[s - 1])) if s > 0 else 0)
    return out, breaks
