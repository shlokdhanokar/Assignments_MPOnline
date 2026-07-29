# End-to-End ML Deployment on Render

A complete machine-learning service: train a model, wrap it in a Flask web app with both an HTML interface and a JSON API, and deploy it to [Render](https://render.com) as a public web service.

This is the project where the model stops being a notebook artefact and becomes something a person or another program can actually call.

## Dataset

- **Source:** UCI Machine Learning Repository — Adult Census Income
- **Link:** https://archive.ics.uci.edu/dataset/2/adult
- **Size:** 32,561 records, 14 attributes
- **Licence:** public domain

Downloaded automatically during the build. Never committed.

## Libraries Used
flask · gunicorn · scikit-learn · pandas · numpy · joblib

## Architecture

```
  Browser ─── GET  /            ──▶  Flask ──▶  HTML form (templates/index.html)
          ─── POST /            ──▶  Flask ──▶  form + prediction
  Client  ─── POST /api/predict ──▶  Flask ──▶  JSON response
  Render  ─── GET  /health      ──▶  Flask ──▶  liveness probe

  Build time:  train_model.py ──▶ model.joblib + model_metadata.json
  Runtime:     app.py loads the pipeline ONCE at import
```

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Renders the input form |
| `/` | POST | Renders the form with a prediction |
| `/api/predict` | POST | JSON in → JSON out |
| `/health` | GET | Health check used by Render |

### Design decisions worth noting

**The model is trained at build time, not committed.** `render.yaml` runs `python train_model.py` as part of `buildCommand`. Binary `.joblib` files in git bloat the repository, go stale silently, and break whenever scikit-learn changes its pickle format. Training during the build means the artefact always matches the installed library versions.

**The pipeline is loaded once at import, not per request.** Deserialising a model on every call would dominate response latency.

**Ten input fields, not fourteen.** The full Adult dataset has 14 attributes, but a public web form with 14 fields is unusable. These ten carry most of the signal (see project 01's permutation-importance results). `fnlwgt` is excluded on principle — it is a census *sampling weight*, not a personal attribute.

**Preprocessing lives inside the pickled pipeline.** Scaling and encoding travel with the model, so the web app cannot drift out of sync with the transformations the model was trained on — the single most common cause of "works in the notebook, wrong in production."

## Model

Gradient boosting (`HistGradientBoostingClassifier`, 200 iterations) inside a `Pipeline` with median/most-frequent imputation, `StandardScaler` and one-hot encoding.

| Metric | Value |
|---|---|
| Holdout accuracy | **0.8773** |
| Holdout ROC-AUC | **0.9298** |
| Training rows | 26,048 |
| Serialised size | 439 KB |

## Testing

`test_app.py` runs 14 tests against the Flask test client — **all passing**:

```
Health endpoint
  PASS  returns 200 / reports ok / exposes model metrics
HTML form
  PASS  form renders / contains the age field / form POST renders a prediction
JSON API
  PASS  high earner -> 200 / high earner -> >50K / low earner -> <=50K
  PASS  high earner scores above low earner
Input validation
  PASS  missing fields -> 400 / error names the missing fields
  PASS  non-numeric age -> 400 / non-JSON body -> 400

14 passed, 0 failed
```

The tests assert *behaviour*, not just status codes: a plausible high earner must score above a plausible low earner, which catches a model wired up backwards — something a 200-response check would happily miss.

Verified against a live server as well:

```console
$ curl http://127.0.0.1:5000/health
{"accuracy":0.8773,"model_loaded":true,"roc_auc":0.9298,"status":"ok"}

$ curl -X POST http://127.0.0.1:5000/api/predict -H "Content-Type: application/json" \
    -d '{"age":45,"education-num":16,"hours-per-week":60,"capital-gain":15000,
         "capital-loss":0,"workclass":"Private","marital-status":"Married-civ-spouse",
         "occupation":"Exec-managerial","relationship":"Husband","sex":"Male"}'
{"confidence":0.9997,"prediction":">50K","probability_above_50k":0.9997}
```

## Run Locally

```bash
pip install -r requirements.txt
python train_model.py     # writes model.joblib
python app.py             # http://127.0.0.1:5000
python test_app.py        # 14 tests
```

> `gunicorn` is Unix-only and will not start on Windows. Locally use `python app.py`; Render runs Linux, where the gunicorn start command applies.

## Deploy to Render

1. Push this repository to GitHub (public or connected private).
2. On [render.com](https://render.com) → **New** → **Web Service** → connect the repo.
3. Render reads `render.yaml` automatically. If configuring by hand, set **Root Directory** to `08-end-to-end-render-deployment` and:
   - **Build command:** `pip install -r requirements.txt && python train_model.py`
   - **Start command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
   - **Health check path:** `/health`
4. Deploy. The first build takes a few minutes (installing scikit-learn and training).

**Live URL:** _add your Render URL here after deploying_ — e.g. `https://income-predictor-xxxx.onrender.com`

> Render's free tier sleeps after ~15 minutes of inactivity, so the first request after idling takes 30–60 seconds to wake the service. This is expected, not a bug.

`$PORT` matters: Render assigns the port at runtime and a service hardcoded to 5000 will fail its health check and be marked unhealthy. `app.py` reads `os.environ.get("PORT", 5000)`.

## Files

| File | Purpose |
|---|---|
| `train_model.py` | Downloads data, trains the pipeline, writes `model.joblib` |
| `app.py` | Flask app — HTML routes, JSON API, health check |
| `templates/index.html` | Responsive form UI, light and dark theme |
| `test_app.py` | 14 endpoint and validation tests |
| `render.yaml` | Render infrastructure-as-code |
| `Procfile` | Process definition (Heroku-compatible) |
| `requirements.txt` | Pinned dependencies |

## Conclusion

This project closes the loop from dataset to callable service. The model itself is ordinary gradient boosting at 87.7 % accuracy; what matters here is everything around it — preprocessing sealed inside the serialised pipeline so training and serving cannot diverge, the artefact rebuilt at deploy time rather than committed, configuration read from the environment instead of hardcoded, a health endpoint for the platform to probe, input validation returning 400 rather than a 500 stack trace, and tests that assert the predictions are directionally sane. Those are the parts that decide whether a model survives contact with real traffic, and none of them are visible in a notebook.
