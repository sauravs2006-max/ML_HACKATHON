import numpy as np
from .geo import seg_dist


def align_route(trip):
    """Monotonic DP: assign each GPS point to a route arc so that route order is never reversed
    and the total point-to-arc distance is minimal."""
    r = trip["route"]
    D, _ = seg_dist(trip["pxy"][:, None, :], trip["A"][r][None], trip["B"][r][None])
    n, m = D.shape
    idx = np.arange(m)
    back = np.empty((n, m), np.int32)
    cost = D[0].copy()
    back[0] = idx
    for i in range(1, n):
        pm = np.minimum.accumulate(cost)
        arg = np.maximum.accumulate(np.where(cost == pm, idx, 0))
        cost = D[i] + pm
        back[i] = arg
    j = int(np.argmin(cost))
    pos = np.empty(n, np.int64)
    for i in range(n - 1, -1, -1):
        pos[i] = j
        j = back[i, j]
    return r[pos], D[np.arange(n), pos]
