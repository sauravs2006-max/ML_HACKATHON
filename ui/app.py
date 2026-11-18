import os
import requests
import folium
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap · Map-matching evaluator")

api = st.sidebar.text_input("API URL", os.environ.get("PATHSNAP_API", "http://localhost:8000"))

try:
    info = requests.get(f"{api}/trips", timeout=10).json()
except Exception as e:
    st.error(f"Cannot reach API at {api}: {e}")
    st.stop()

st.sidebar.success(f"Connected · split: {info['split']}")
trip = st.selectbox("Evaluation trip", info["trips"])

if st.button("Reconstruct route", type="primary"):
    with st.spinner("Running pipeline on the API..."):
        r = requests.post(f"{api}/predict", json={"trip_id": trip}, timeout=600)
    if r.status_code != 200:
        st.error(r.text)
        st.stop()
    st.session_state["res"] = r.json()

res = st.session_state.get("res")
if res:
    m = res["metrics"]
    c = st.columns(5)
    c[0].metric("GPS points", res["n_points"])
    if m:
        c[1].metric("Top-1 accuracy (HMM)", f"{m['hmm']['top1_acc']:.3f}",
                    f"{m['hmm']['top1_acc'] - m['classifier']['top1_acc']:+.3f} vs classifier")
        c[2].metric("Route continuity", f"{m['hmm']['route_continuity']:.3f}")
        c[3].metric("Route F1", f"{m['hmm']['route_f1']:.3f}")
    c[4].metric("API time (s)", res["total_seconds"])
    if m:
        st.dataframe(pd.DataFrame({k: m[k] for k in ("classifier", "hmm")}).T.round(4))

    gps = res["gps"]
    fmap = folium.Map(tiles="OpenStreetMap")
    g = folium.FeatureGroup("GPS points").add_to(fmap)
    for lat, lon in gps:
        folium.CircleMarker((lat, lon), radius=1.5, color="#1565c0", opacity=0.4, fill=True).add_to(g)
    folium.PolyLine(res["predicted_path"], color="#d32f2f", weight=4, tooltip="Predicted route").add_to(
        folium.FeatureGroup("Predicted route").add_to(fmap))
    folium.Marker(gps[0], tooltip="Start", icon=folium.Icon(color="green")).add_to(fmap)
    folium.Marker(gps[-1], tooltip="End", icon=folium.Icon(color="red")).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[min(p[0] for p in gps), min(p[1] for p in gps)], [max(p[0] for p in gps), max(p[1] for p in gps)]])
    st.caption(f"Trip {res['trip_id']} · blue = GPS observations, red = predicted route")
    html = fmap.get_root().render()
    st.iframe(html, height=620) if hasattr(st, "iframe") else components.html(html, height=620)
