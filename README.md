# PathSnap · Map-matching & route reconstruction

Noisy GPS points → candidate road segments → classifier probabilities → HMM/Viterbi → road route.

## Setup (once)
```bash
pip install -r requirements.txt
pip install -e .
```
Data: copy `train/` and `student_test/` into `data/` (or extract `data.zip` in the repo folder):
```
data/train/00000001/00000001.track ...
data/student_test/00000000/...
```
`data/` is git-ignored. Another location can be set with the `PATHSNAP_DATA` environment variable.

## Run order
| # | Notebook | Owner |
|---|---|---|
| 01 | data inventory, missing values, timestamp checks | Person 1 |
| 02 | EDA plots, road-network connectivity | Person 2 |
| 03 | per-point labels from `.route`, candidate miss rate | Person 3 |
| 04 | candidate features, first Folium map | Person 4 |
| 05 | build train / student-test candidate tables | Person 1 |
| 06 | models 1–3: logistic regression, Gaussian NB, KNN | Person 1 |
| 07 | models 4–7: random forest, HistGradientBoosting, LightGBM, MLP | Person 2 |
| 08 | model comparison and selection | Person 2 |
| 09 | Bayesian optimisation (Optuna) of LightGBM | Person 3 |
| 10 | HMM/Viterbi decoding, writes `artifacts/config.json` | Person 3 |
| 11 | Folium maps of predicted vs reference routes | Person 4 |

Notebooks 01–04 are independent. 05 must run before 06–10.

## Layout
| Folder | Content |
|---|---|
| `src/pathsnap/` | `io` loading/cleaning · `geo` geometry · `labels` route→point alignment · `candidates` · `features` · `evaluate` metrics · `modeling` CV harness · `hmm` Viterbi · `pipeline` inference · `viz` Folium |
| `notebooks/` | 01 → 11 |
| `outputs/` | CSV results, figures, Folium maps |
| `artifacts/` | `config.json` + final model |
| `api/` | FastAPI backend (Laptop 1) |
| `ui/` | Streamlit UI (Laptop 2) |

## Method
* Coordinates projected to local metres around each trip's centroid.
* Labels: each GPS point aligned to a reference-route arc with a monotonic DP. Trips 00000033 and 00000054 are far from their
  reference route and excluded from training.
* Candidates: 8 nearest arcs within 100 m (miss rate ≈ 1.2 % train, 0.4 % student-test).
* Classifier: "is this candidate the reference arc", GroupKFold over trips.
* HMM: states = candidates, emission = normalised classifier probability, transition = exp(−|road dist − straight dist| / β).

## Results (student-test, 10 trips)
| Method | Top-1 accuracy | Route continuity |
|---|---|---|
| Nearest arc | 0.575 | 0.47 |
| LightGBM | 0.900 | 0.75 |
| LightGBM + HMM | 0.952 | 0.91 |

## Deployment
Laptop 1:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Laptop 2:
```bash
set PATHSNAP_API=http://<laptop1-ip>:8000      # Windows (macOS/Linux: export ...)
streamlit run ui/app.py
```
Endpoints: `GET /health`, `GET /trips`, `POST /predict {"trip_id": "00000091"}`. The API serves `data/evaluator_test`
(falls back to `student_test`) and never returns the reference route.
