import os
import numpy as np

DATA_DIR = os.environ.get("PATHSNAP_DATA", os.path.join(os.path.dirname(__file__), "..", "..", "data"))
EARTH_R = 6371008.8


def split_dir(split):
    return os.path.join(DATA_DIR, split)


def list_trips(split):
    return sorted(d for d in os.listdir(split_dir(split)) if not d.startswith("."))


def load_raw(split, tid):
    p = os.path.join(split_dir(split), tid, tid)
    track = np.loadtxt(p + ".track", ndmin=2)
    nodes = np.loadtxt(p + ".nodes", ndmin=2)
    arcs = np.loadtxt(p + ".arcs", dtype=np.int64, ndmin=2)
    route = np.loadtxt(p + ".route", dtype=np.int64, ndmin=1) if os.path.exists(p + ".route") else None
    return track, nodes, arcs, route


def clean_track(track):
    t = track[np.isfinite(track).all(1)]
    t = t[(np.abs(t[:, 0]) <= 180) & (np.abs(t[:, 1]) <= 90)]
    t = t[np.argsort(t[:, 2], kind="stable")]
    _, keep = np.unique(t[:, 2], return_index=True)
    return t[np.sort(keep)]


def project(lon, lat, lon0, lat0):
    x = np.radians(lon - lon0) * EARTH_R * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_R
    return np.c_[x, y]


def load_trip(split, tid):
    track, nodes, arcs, route = load_raw(split, tid)
    track = clean_track(track)
    lon0, lat0 = track[:, 0].mean(), track[:, 1].mean()
    pxy = project(track[:, 0], track[:, 1], lon0, lat0)
    nxy = project(nodes[:, 0], nodes[:, 1], lon0, lat0)
    a, b = nxy[arcs[:, 0]], nxy[arcs[:, 1]]
    d = b - a
    return dict(
        id=tid, split=split, track=track, t=track[:, 2], pxy=pxy,
        nodes=nodes, nxy=nxy, arcs=arcs, route=route,
        A=a, B=b, alen=np.hypot(d[:, 0], d[:, 1]),
        abear=np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360,
    )
