import numpy as np
import pandas as pd
from .geo import seg_dist, headings, speeds, angdiff
from .candidates import generate

FEATURES = [
    "dist", "rank", "dist_gap", "dist_ratio", "head_diff", "move", "speed", "dt",
    "frac", "alen", "n_cand", "n_cand30", "ctx_prev", "ctx_next", "ctx_mean",
    "out_deg", "in_deg", "has_rev",
]


def candidate_table(trip, labels=None, step=1, ctx=3):
    C, D, T = generate(trip)
    pxy, n = trip["pxy"], len(trip["pxy"])
    head, move = headings(pxy)
    spd, dt = speeds(pxy, trip["t"])
    arcs = trip["arcs"]
    out_deg = np.bincount(arcs[:, 0], minlength=len(trip["nodes"]))
    in_deg = np.bincount(arcs[:, 1], minlength=len(trip["nodes"]))
    pairs = set(map(tuple, arcs))
    has_rev = np.array([(b, a) in pairs for a, b in arcs])

    pi = np.arange(0, n, step)
    Cs, Ds, Ts = C[pi], D[pi], T[pi]
    valid = Cs >= 0
    rows, cols = np.nonzero(valid)
    p, c = pi[rows], Cs[rows, cols]
    dmin = Ds[:, 0][rows]
    ip, inx = np.clip(p - ctx, 0, n - 1), np.clip(p + ctx, 0, n - 1)
    dp, _ = seg_dist(pxy[ip], trip["A"][c], trip["B"][c])
    dn, _ = seg_dist(pxy[inx], trip["A"][c], trip["B"][c])
    dist = Ds[rows, cols]
    df = pd.DataFrame(dict(
        trip=trip["id"], point=p, arc=c,
        dist=dist, rank=cols, dist_gap=dist - dmin, dist_ratio=dist / (dmin + 1.0),
        head_diff=angdiff(head[p], trip["abear"][c]), move=move[p], speed=spd[p], dt=dt[p],
        frac=Ts[rows, cols], alen=trip["alen"][c],
        n_cand=valid.sum(1)[rows], n_cand30=(Ds <= 30).sum(1)[rows],
        ctx_prev=dp, ctx_next=dn, ctx_mean=(dp + dn + dist) / 3,
        out_deg=out_deg[arcs[c, 1]], in_deg=in_deg[arcs[c, 0]], has_rev=has_rev[c].astype(int),
    ))
    if labels is not None:
        df["y"] = (c == labels[p]).astype(int)
    return df, (C, D, T)
