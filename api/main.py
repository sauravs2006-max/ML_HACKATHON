import os, sys, time, logging
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap import io, pipeline, viz

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pathsnap")

SPLIT = os.environ.get("PATHSNAP_EVAL_SPLIT", "evaluator_test")
if not os.path.isdir(io.split_dir(SPLIT)):
    SPLIT = "student_test"
MODEL, CFG = pipeline.load_artifacts()
log.info("model=%s split=%s hmm=%s", CFG["model_file"], SPLIT, CFG["hmm"])

app = FastAPI(title="PathSnap API")


class Req(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "split": SPLIT, "model": CFG["model_file"]}


@app.get("/trips")
def trips():
    return {"split": SPLIT, "trips": io.list_trips(SPLIT)}


@app.post("/predict")
def predict(req: Req):
    if req.trip_id not in io.list_trips(SPLIT):
        raise HTTPException(404, f"unknown trip {req.trip_id}")
    log.info("request trip=%s", req.trip_id)
    t0 = time.time()
    trip = io.load_trip(SPLIT, req.trip_id)
    out = pipeline.predict(trip, MODEL, CFG["hmm"])
    pred = out["hmm"][out["hmm"] >= 0]
    metrics = pipeline.evaluate(trip, out) if trip["route"] is not None else None
    step = max(1, len(trip["track"]) // 3000)
    res = dict(
        trip_id=req.trip_id,
        n_points=int(len(trip["track"])),
        gps=trip["track"][::step, [1, 0]].round(6).tolist(),
        predicted_path=viz.path_coords(trip, list(pred)),
        n_predicted_arcs=int(len(np.unique(pred))),
        metrics=metrics,
        inference_seconds=round(out["seconds"], 3),
        total_seconds=round(time.time() - t0, 3),
    )
    log.info("done trip=%s points=%d hmm_top1=%s continuity=%s time=%.2fs", req.trip_id, res["n_points"],
             metrics and round(metrics["hmm"]["top1_acc"], 4), metrics and round(metrics["hmm"]["route_continuity"], 4),
             res["total_seconds"])
    return res
