import numpy as np
import folium

LEGEND = """<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:8px;
border:1px solid #888;font-size:13px">{}</div>"""


def arc_lines(trip, arc_ids):
    ids = np.asarray(arc_ids)
    keep = np.r_[True, ids[1:] != ids[:-1]]
    ids = ids[keep]
    n, a = trip["nodes"], trip["arcs"]
    return [[(n[a[i, 0], 1], n[a[i, 0], 0]), (n[a[i, 1], 1], n[a[i, 1], 0])] for i in ids]


def path_coords(trip, arc_ids):
    ids = [i for k, i in enumerate(arc_ids) if k == 0 or i != arc_ids[k - 1]]
    n, a = trip["nodes"], trip["arcs"]
    pts = []
    for i in ids:
        for v in (a[i, 0], a[i, 1]):
            p = (float(n[v, 1]), float(n[v, 0]))
            if not pts or pts[-1] != p:
                pts.append(p)
    return pts


def nearby_arcs(trip, margin=150.0, max_arcs=4000):
    lo, hi = trip["pxy"].min(0) - margin, trip["pxy"].max(0) + margin
    m = ((trip["A"] >= lo) & (trip["A"] <= hi)).all(1) | ((trip["B"] >= lo) & (trip["B"] <= hi)).all(1)
    return np.nonzero(m)[0][:max_arcs]


def trip_map(trip, pred=None, ref=None, network=True, gps_step=1):
    tr = trip["track"]
    m = folium.Map(location=[tr[:, 1].mean(), tr[:, 0].mean()], tiles="OpenStreetMap", control_scale=True)
    items = []
    if network:
        g = folium.FeatureGroup("Road network", show=True)
        for seg in arc_lines(trip, nearby_arcs(trip)):
            folium.PolyLine(seg, color="#9e9e9e", weight=1, opacity=0.6).add_to(g)
        g.add_to(m)
        items.append(("#9e9e9e", "Road network"))
    g = folium.FeatureGroup("GPS points")
    for lon, lat, t in tr[::gps_step]:
        folium.CircleMarker((lat, lon), radius=1.5, color="#1565c0", fill=True, opacity=0.4, tooltip=f"t={t:.0f}").add_to(g)
    g.add_to(m)
    items.append(("#1565c0", "GPS points"))
    if ref is not None:
        folium.PolyLine(path_coords(trip, list(ref)), color="#2e7d32", weight=6, opacity=0.7, tooltip="Reference").add_to(
            folium.FeatureGroup("Reference route").add_to(m))
        items.append(("#2e7d32", "Reference route"))
    if pred is not None:
        folium.PolyLine(path_coords(trip, list(pred)), color="#d32f2f", weight=3, opacity=0.9, tooltip="Predicted").add_to(
            folium.FeatureGroup("Predicted route").add_to(m))
        items.append(("#d32f2f", "Predicted route"))
    folium.Marker((tr[0, 1], tr[0, 0]), tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker((tr[-1, 1], tr[-1, 0]), tooltip="End", icon=folium.Icon(color="red")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(LEGEND.format("<br>".join(
        f'<span style="color:{c}">&#9632;</span> {t}' for c, t in items))))
    m.fit_bounds([[tr[:, 1].min(), tr[:, 0].min()], [tr[:, 1].max(), tr[:, 0].max()]])
    return m
