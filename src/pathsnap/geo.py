import numpy as np


def seg_dist(p, A, B):
    """p:(...,2), A/B:(...,2) broadcastable -> (distance, fraction along segment)"""
    ab = B - A
    l2 = (ab ** 2).sum(-1)
    t = np.clip(((p - A) * ab).sum(-1) / np.where(l2 > 0, l2, 1), 0, 1)
    proj = A + t[..., None] * ab
    return np.hypot(*(p - proj).T).T, t


def headings(pxy, w=2):
    n = len(pxy)
    i0, i1 = np.clip(np.arange(n) - w, 0, n - 1), np.clip(np.arange(n) + w, 0, n - 1)
    d = pxy[i1] - pxy[i0]
    return np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360, np.hypot(d[:, 0], d[:, 1])


def speeds(pxy, t):
    d = np.r_[0, np.hypot(*np.diff(pxy, axis=0).T)]
    dt = np.r_[1, np.diff(t)]
    return d / np.maximum(dt, 1e-6), dt


def angdiff(a, b):
    d = np.abs(a - b) % 360
    return np.minimum(d, 360 - d)
