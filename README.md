# Credence AI – Credit Card Default Prediction System

Credence AI is a backend system that predicts whether a credit card customer will **default on next month's payment**, using their historical financial and repayment information. It serves predictions and portfolio-level risk analytics through a **FastAPI** REST API. **No database is required**: predictions are kept in a temporary in-memory history while the service runs.

**Domain:** Financial Technology / Credit Risk Prediction
**Stakeholder:** Credit Portfolio Strategist (the only user role)

---

## 1. Problem statement

Credit card defaults create financial risk for lending institutions. If customers who are likely to miss next month's payment can be spotted early, a **Credit Portfolio Strategist** can:

* assess the default risk of individual customers,
* score an entire portfolio and see how much credit exposure is at risk,
* find risky **customer segments** (for example by repayment history, education or age group),
* make portfolio-level credit decisions (limit reviews, provisioning, collection priorities).

Credence AI is the machine learning backend that supports these decisions. It is **backend-only**: it has no frontend, dashboard or login. FastAPI's Swagger UI at `/docs` is used to explore and test the API.

---

## 2. Technology stack

| Layer | Technology |
|---|---|
| Machine learning | Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib |
| Backend | FastAPI, Uvicorn, Pydantic |
| Storage | None (no database). Temporary in-memory prediction history, cleared on restart |
| Testing | Pytest, FastAPI TestClient, HTTPX |
| Development & deployment | VS Code, Jupyter Notebook (analysis), Git & GitHub, Render (API hosting), Postman/Newman (API testing) |

---

## 3. Project structure

```
Credence AI/
├── app/                        # FastAPI backend
│   ├── main.py                 # app, lifespan (loads model, creates in-memory store), /, /health, /model/info
│   ├── config.py               # settings from environment / .env
│   ├── schemas.py              # Pydantic request/response models + input validation
│   ├── store.py                # temporary in-memory prediction history + portfolio aggregations
│   ├── predictor.py            # loads the joblib pipeline and makes predictions
│   ├── dependencies.py         # shared FastAPI dependencies
│   └── routers/
│       ├── predictions.py      # /predictions endpoints
│       └── portfolio.py        # /portfolio analytics endpoints
├── ml/                         # machine learning package
│   ├── config.py               # features, target, paths, split settings, risk bands
│   ├── data.py                 # robust CSV loading + 80:20 stratified split
│   ├── eda.py                  # exploratory data analysis -> figures + JSON summary
│   ├── pipeline.py             # preprocessing ColumnTransformer + 4 model definitions
│   ├── evaluate.py             # metrics, threshold selection, evaluation plots
│   ├── train.py                # tuning, comparison, model selection, saving the model
│   ├── imputation.py           # missing-value fill values (training split) + robustness evaluation
│   └── plotting.py             # shared figure styling
├── notebooks/
│   └── credence_ai_credit_default.ipynb   # self-contained analysis notebook (EDA → models → conclusions)
├── data/raw/default of credit card clients.csv
├── models/
│   ├── credence_model.joblib   # trained pipeline (preprocessing + Random Forest)
│   ├── model_metadata.json     # model name, threshold, metrics, features
│   └── imputation_values.json  # training-split medians/modes used for missing inputs
├── reports/
│   ├── figures/                # 14 EDA and evaluation figures
│   └── metrics/                # eda_summary.json, model_comparison.{csv,json}, classification_reports.txt,
│                               # missing_value_robustness.json
├── tests/                      # pytest suite (48 tests, no database needed)
├── postman/                    # Postman collection + environments + Newman reports
├── requirements.txt
├── render.yaml                 # Render deployment blueprint
├── .env.example                # optional settings (no credentials needed)
└── pytest.ini
```

---

## 4. Dataset

**Default of Credit Card Clients**: 30,000 credit card customers in Taiwan (April–September 2005), from the UCI Machine Learning Repository.

* **Target:** `default.payment.next.month`, where 0 = no default and 1 = default.
* **Input features (23):**

| Group | Features | Meaning |
|---|---|---|
| Credit limit | `LIMIT_BAL` | Credit limit in NT dollars (includes family/supplementary credit) |
| Demographics | `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` | 1=male, 2=female · 1=graduate school, 2=university, 3=high school, 4=others · 1=married, 2=single, 3=others · years |
| Repayment status | `PAY_0`, `PAY_2`…`PAY_6` | Sep → Apr 2005. −2 = no consumption, −1 = paid in full, 0 = revolving credit, 1–8 = months of payment delay |
| Bill amounts | `BILL_AMT1`…`BILL_AMT6` | Bill statement amount, Sep → Apr (negative = credit balance) |
| Payment amounts | `PAY_AMT1`…`PAY_AMT6` | Amount paid, Sep → Apr |

**Quirks of the actual file, and how they are handled** (`ml/data.py`):

* The first line is a placeholder header (`,X1,X2,…,Y`). The real column names are on line 2. The loader checks for this and skips the first line only when it is actually present.
* The target column is spelled `default payment next month` and is renamed to `default.payment.next.month`.
* The `ID` column is an identifier, not a feature, so it is dropped.

---

## 5. Exploratory data analysis

Run `python -m ml.eda` to generate `reports/figures/eda_*.png` and `reports/metrics/eda_summary.json`.

| Check | Result | Decision |
|---|---|---|
| Shape | 30,000 rows × 23 features + target | — |
| Data types | all integer-coded | — |
| Missing values | **0** | no imputation needed |
| Duplicates | 35 rows identical once `ID` is removed; all IDs are unique | **kept**: these are distinct customers (mostly inactive accounts) |
| Target balance | **77.9% no default / 22.1% default** | stratified split, class weighting, F1/recall-focused evaluation |
| Undocumented codes | `EDUCATION` 0/5/6 (345 rows), `MARRIAGE` 0 (54 rows) | **kept**, grouped with "others" by the encoder (see below) |
| Outliers | many IQR outliers in all monetary features | **kept**: they are real high-limit customers. A power transform reduces their influence |
| Negative bills | 3,932 negative `BILL_AMT` values | valid credit balances, handled by Yeo-Johnson |
| Correlation | `BILL_AMT1..6` inter-correlated (r ≈ 0.8–0.95). `PAY_0` has the strongest link to default (r = +0.32) | all 23 features kept. RF is unaffected by collinearity and LR is L2-regularised |

**Key insights:**

* Customers whose **latest payment (`PAY_0`) is 2+ months late default 50–78% of the time**, against 13–17% for customers with no delay.
* 52% of eventual defaulters had a payment delay in September, compared with 15% of non-defaulters.
* Defaulters have lower credit limits and make smaller payments. Demographic differences are small.

---

## 6. Preprocessing (leak-free, identical at training and inference)

All preprocessing is a scikit-learn `ColumnTransformer` placed **inside the same `Pipeline` as the classifier** (`ml/pipeline.py`):

| Feature type | Columns | Transformation | Justification |
|---|---|---|---|
| Nominal categorical | SEX, EDUCATION, MARRIAGE | `OneHotEncoder(min_frequency=0.02, handle_unknown="infrequent_if_exist")` | The codes have no numeric order. Categories seen in fewer than 2% of *training* rows (EDUCATION 0/4/5/6, MARRIAGE 0/3) are merged into one "infrequent" (= *others*) bucket, and codes never seen in training are routed there too |
| Ordinal | PAY_0, PAY_2…PAY_6 | `StandardScaler` | The values are ordered (a larger value means a longer delay), so they stay numeric |
| Monetary | LIMIT_BAL, BILL_AMT1–6, PAY_AMT1–6 | `PowerTransformer(method="yeo-johnson")` | Fixes heavy right skew, handles negative values, and reduces the pull of extreme values without deleting rows |
| Continuous | AGE | `StandardScaler` | Puts AGE on a comparable scale for LR and KNN |

**How leakage is prevented:**

1. The 80:20 split happens first: `test_size=0.2, random_state=42, stratify=y`.
2. Transformers are fitted on the training data only, and re-fitted inside every CV fold.
3. The test set is used only for the final evaluation.
4. The saved `.joblib` artifact contains the fitted preprocessing, so the API applies **exactly** the same transformations. The pipeline uses only built-in scikit-learn classes, so a model trained in the notebook can also be loaded by the API.

---

## 7. Model training and comparison

Run `python -m ml.train` (about 1 minute on an M2). Each of the four required algorithms is tuned with **5-fold stratified cross-validation on the training set**, scored by **F1 for the default class**:

| Model | Imbalance handling | Grid |
|---|---|---|
| Logistic Regression | `class_weight="balanced"` | C ∈ {0.01, 0.1, 1, 10} |
| Decision Tree | `class_weight="balanced"` | max_depth ∈ {4, 6, 8, 10}, min_samples_leaf ∈ {20, 50, 100} |
| Random Forest (300 trees) | `class_weight="balanced_subsample"` | max_depth ∈ {8, 12}, min_samples_leaf ∈ {5, 10, 20} |
| K-Nearest Neighbors | none available | k ∈ {15, 31, 51}, weights ∈ {uniform, distance} |

KNN was **not skipped**: with 24k training rows it trains in a few seconds. XGBoost is not used.

### Results on the held-out test set (6,000 customers, threshold 0.5)

| Model | CV F1 (1) | Accuracy | Precision (1) | Recall (1) | **F1 (1)** | ROC-AUC | False positives | False negatives |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.507 | 0.714 | 0.404 | **0.625** | 0.491 | 0.741 | 1,221 | **498** |
| Decision Tree | 0.529 | 0.766 | 0.476 | 0.564 | 0.516 | 0.748 | 825 | 578 |
| **Random Forest** ✅ | **0.546** | 0.794 | 0.532 | 0.560 | **0.546** | **0.775** | 654 | 584 |
| K-Nearest Neighbors | 0.441 | **0.809** | **0.621** | 0.345 | 0.444 | 0.742 | **280** | 869 |

Full classification reports: `reports/metrics/classification_reports.txt`. Figures: `model_confusion_matrices.png`, `model_roc_curves.png`, `model_metric_comparison.png`, `best_model_feature_importance.png`, `best_model_threshold_tradeoff.png`.

### Why the Random Forest was selected, and why accuracy was not used

The model is selected by **cross-validated F1 for the default class**, which is computed on training folds only, so the test set plays no part in the choice. ROC-AUC is used to break near-ties. Random Forest has the best CV F1 **and** the best test F1 and ROC-AUC.

**Accuracy is misleading here.** A model that always predicts "no default" would score 77.9%. KNN has the *highest* accuracy (80.9%) yet misses **65% of actual defaulters** (869 false negatives).

### Understanding the errors (for a Credit Portfolio Strategist)

| Term | Definition | Business meaning |
|---|---|---|
| **False negative** | A defaulter predicted as "no default" | **Unexpected credit loss**: no limit review and no early collection, so portfolio risk is under-stated. Usually the most expensive error. |
| **False positive** | A good customer flagged as a defaulter | **Opportunity cost**: needless limit cuts or interventions and an unhappy customer, so risk is over-stated. |
| **Precision (1)** | TP / (TP + FP) | Of the customers we flag, the share who really default. Measures how efficient interventions are. |
| **Recall (1)** | TP / (TP + FN) | Of all real defaulters, the share we catch. Measures how much loss we prevent. |
| **F1 (1)** | Harmonic mean of precision and recall | A single score that is high only when both are good. Suited to the minority class. |
| **ROC-AUC** | Probability that a random defaulter is ranked above a random non-defaulter | Ranking quality, independent of the threshold. Useful for prioritising reviews. |

### Decision threshold

The decision threshold (**0.5184**) is chosen to maximise F1 on **out-of-fold training predictions**. Because class weighting already balances the classes, it ends up very close to 0.5, and test F1 is practically unchanged (0.542 vs 0.546). The threshold is a **business lever**: lowering it catches more defaulters (higher recall) at the cost of more false alarms (see `best_model_threshold_tradeoff.png`).

### Final model (`models/credence_model.joblib`)

Random Forest, `max_depth=12`, `min_samples_leaf=5`, 300 trees.
Test results at the tuned threshold: accuracy 0.799, precision (1) 0.545, recall (1) 0.540, F1 (1) 0.542, ROC-AUC 0.775.
The most important features are `PAY_0`, `PAY_2`, `PAY_3`, `LIMIT_BAL` and `PAY_AMT1`.

### Limitations

* **Probabilities are risk scores, not calibrated probabilities.** Class weighting pushes scores upwards. On 200 test customers the mean predicted probability was 0.40, while the actual default rate was 0.25. Use the scores to **rank** customers and for the LOW/MEDIUM/HIGH bands, not as literal default frequencies.
* The data covers one Taiwanese bank in 2005, so it does not transfer automatically to other markets or periods.
* An F1 of about 0.55 shows that repayment history alone cannot separate all defaulters. Combine the scores with expert judgement.
* SEX, EDUCATION, MARRIAGE and AGE are model inputs in this project. In a real deployment, using protected attributes in credit decisions needs a fairness and regulatory review.

---

## 8. Prediction history (no database)

Credence AI uses **no database** (no MySQL, SQLite or MongoDB). Every prediction is kept in a **temporary in-memory history** (`app/store.py`) for as long as the API process runs:

* `GET /predictions`, `GET /predictions/{id}`, `DELETE /predictions/{id}` and the `/portfolio` endpoints work on this history.
* **Nothing is persistent.** The history is empty after every restart, redeploy or free-plan sleep on Render.
* At most `MAX_STORED_PREDICTIONS` predictions are kept (default 10,000). The oldest are evicted first, so memory stays bounded.
* Each prediction keeps the **23 values the model actually used** (after imputation), `imputed_features`, the probability, prediction, risk band, threshold and model version, plus a `batch_id` when it came from a batch.
* Access is thread-safe. The service runs as a single uvicorn process (`render.yaml`) so that all requests share one history.

---

## 9. API reference

Start the server with `uvicorn app.main:app --reload` and open **http://127.0.0.1:8000/docs**.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Service information |
| GET | `/health` | Model loaded? Imputation available? Storage type and number of predictions in memory |
| GET | `/model/info` | Selected model, threshold, features, CV and test metrics |
| POST | `/predictions` | Score **one customer** (missing features imputed); added to the in-memory history |
| POST | `/predictions/batch` | Score **a portfolio** (up to 1,000 customers) and get a summary; added to the history |
| GET | `/predictions` | List predictions in the in-memory history, **riskiest first**. Filters: `risk_band`, `predicted_default`, `batch_id`, `customer_ref`, `min_probability`, plus `limit` and `offset` |
| GET | `/predictions/{id}` | One prediction from the history, with its inputs |
| DELETE | `/predictions/{id}` | Remove a prediction from the history |
| GET | `/portfolio/summary` | Portfolio KPIs: predicted default rate, mean risk score, **share of credit limit at risk**, risk-band breakdown, over the in-memory history (optional `batch_id`) |
| GET | `/portfolio/segments?dimension=…` | Predicted risk by `PAY_0`, `EDUCATION`, `MARRIAGE`, `SEX` or `AGE_GROUP`, riskiest segment first |

**Risk bands** (on the default probability): LOW < 0.30 ≤ MEDIUM < 0.60 ≤ HIGH. `predicted_default` uses the model's decision threshold (0.5184), so a MEDIUM customer may still be flagged.

**Example: score one customer**

```bash
curl -X POST http://127.0.0.1:8000/predictions \
  -H "Content-Type: application/json" \
  -d '{"customer_ref":"CUST-000123","LIMIT_BAL":50000,"SEX":2,"EDUCATION":2,"MARRIAGE":1,"AGE":37,
       "PAY_0":2,"PAY_2":2,"PAY_3":2,"PAY_4":0,"PAY_5":0,"PAY_6":0,
       "BILL_AMT1":46990,"BILL_AMT2":48233,"BILL_AMT3":49291,"BILL_AMT4":28314,"BILL_AMT5":28959,"BILL_AMT6":29547,
       "PAY_AMT1":0,"PAY_AMT2":0,"PAY_AMT3":1000,"PAY_AMT4":1100,"PAY_AMT5":1069,"PAY_AMT6":1000}'
```

```json
{"prediction_id": 1, "stored": true, "customer_ref": "CUST-000123", "default_probability": 0.749109,
 "predicted_default": true, "prediction_label": "Default", "risk_band": "HIGH",
 "decision_threshold": 0.5184, "model_name": "Random Forest", "model_version": "1.0.0",
 "imputed_features": [], "created_at": "2026-11-20T15:50:13"}
```

### Missing values

Real portfolio records can be incomplete. **Any feature may be omitted or sent as `null`, up to 11 of the 23.**

* **How values are filled:** numeric and ordinal features (`LIMIT_BAL`, `AGE`, `PAY_*`, `BILL_AMT*`, `PAY_AMT*`) get the **median**, and `SEX`, `EDUCATION`, `MARRIAGE` get the **most frequent value**. These values are computed from the **training split only** (`ml/imputation.py` → `models/imputation_values.json`), so there is no leakage.
* **The trained model was not retrained or changed** (`credence_model.joblib` is byte-identical). The training data had no missing values, so an imputer inside the pipeline would have done nothing during training. Filling values *before* the unchanged pipeline therefore keeps training and inference consistent.
* **Complete records give exactly the same predictions as before** (this is tested).
* The response lists `imputed_features`, and the history records them, so the strategist can see which predictions rest partly on imputed data.
* **More than 11 missing → HTTP 422.** A prediction based mostly on medians would be misleading.

Example: `BILL_AMT3` omitted → `"imputed_features": ["BILL_AMT3"]`.

**Measured effect on the held-out test set** (`reports/metrics/missing_value_robustness.json`; values removed at random, then imputed):

| Cells missing | 0% | 5% | 10% | 20% | 30% |
|---|---|---|---|---|---|
| ROC-AUC | 0.7747 | 0.7685 | 0.7668 | 0.7623 | 0.7558 |
| F1 (default) | 0.5424 | 0.5265 | 0.5158 | 0.4839 | 0.4455 |

Ranking quality (ROC-AUC) degrades gracefully. Losing **`PAY_0`** (the most recent repayment status) costs the most: it drops AUC by 0.031 on its own, so it is the most important field to collect.

**Input validation:** values that are present must be in range. For example, `SEX=3`, `AGE<18`, `PAY_0<-2` or negative payments are rejected with HTTP 422, as are unknown fields, including the target column.

---

## 10. Setup guide for teammates

No database or credentials are needed. You only need Python.

### Prerequisites

* **Python 3.12 or newer.** The project was developed and tested on Python 3.14. Check with `python3 --version` (on Windows, `python --version`).
* Git, if you clone rather than use the ZIP.

### Step 1: Get the project

```bash
# Option A - clone from GitHub
git clone <repository-url> "Credence AI"
cd "Credence AI"

# Option B - from the ZIP: extract Credence-AI.zip, then
cd "Credence AI"
```

### Step 2: Create and activate a virtual environment

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Once it is active, your prompt starts with `(.venv)`. Activate it again in every new terminal.

### Step 3: Install requirements

```bash
pip install -r requirements.txt
```

### Step 4 (optional): Settings

The defaults work as they are. To change the in-memory history size or the batch limit, run `cp .env.example .env` (on Windows, `copy .env.example .env`) and edit `.env`.

### Step 5: Start FastAPI

```bash
uvicorn app.main:app --reload
```

### Step 6: Open Swagger

Go to **http://127.0.0.1:8000/docs**. To check that everything works:

1. **GET /health** → `"status":"ok"`, `"model_loaded":true`, `"storage":"in-memory (temporary, cleared on restart)"`.
2. **POST /predictions** → *Try it out* → *Execute* (an example customer is pre-filled) → `201` with a prediction and a `prediction_id`.
3. **GET /predictions** → the prediction you just made. It disappears when the server restarts.

### Optional

```bash
pytest -v            # run the 48 tests (no database needed)
python -m ml.eda     # regenerate EDA figures
python -m ml.train   # retrain the model (~1 min); a trained model is already included
```

### Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` | The virtual environment isn't activated (step 2), or requirements aren't installed (step 3) |
| Error while loading the model | Use the exact `scikit-learn` version from `requirements.txt` |
| `GET /predictions` is empty | Expected after a restart: the history is in memory only |
| `Address already in use` | Another server is running on port 8000. Stop it, or use `--port 8001` |

---

## 11. Testing

```bash
pytest -v
```

There are **48 tests**, and **none of them needs a database**:

* **ML tests:** dataset schema, the 80:20 stratified and deterministic split, the preprocessor being fitted on training data only, grouping of rare and unseen category codes, risk-band boundaries, the model artifact, probability validity, and the imputation values coming from the training split only.
* **API tests** (TestClient/HTTPX): every endpoint, input validation, in-memory history read-back, deletion, batch scoring, the portfolio summary and segment calculations, and 404s.
* **Missing-value tests:** omitted and `null` values, the 11-feature limit, imputed values recorded in the history, and complete records being unchanged.
* **Database-free tests:**
  * no database library (SQLAlchemy, PyMySQL, sqlite3, pymongo…) is imported by the app;
  * the history is empty after a restart;
  * oldest-first eviction at the capacity limit;
  * unique ids under concurrent requests;
  * deleted predictions leave the portfolio figures.

### Postman / Newman

`postman/` contains a **Postman collection** of 19 requests with automated assertions, plus two environments (`local`: `http://127.0.0.1:8000`, `render`: the public URL).

* **Postman app:** *Import* the three JSON files, select an environment, then *Run collection*.
* **Command line** (Node.js required; Newman is fetched on demand and is not a project dependency):

```bash
npx -p newman -p newman-reporter-htmlextra newman run postman/Credence-AI.postman_collection.json \
  -e postman/Credence-AI-local.postman_environment.json -r cli,htmlextra \
  --reporter-htmlextra-export postman/reports/newman-local.html
```

Run the folders in order: later requests use the prediction and batch ids that earlier ones create in the in-memory history.

Recorded runs (HTML and JSON reports in `postman/reports/`):

| Target | Requests | Assertions | Failed |
|---|---|---|---|
| Local server, no database (`newman-local.html`) | 19 | 51 | 0 |
| Live Render deployment | *re-run after the database-free version is deployed* | | |

---

## 12. Jupyter notebook

`notebooks/credence_ai_credit_default.ipynb` is self-contained and already executed, so all outputs and charts are saved in it. It covers EDA, the preprocessing rationale, training all four models, evaluation, the false-positive/false-negative analysis, threshold selection and conclusions.

To re-run it, open it in VS Code (select the `.venv` interpreter) and choose **Run All**. It reads `../data/raw/…` and writes its outputs to `notebooks/outputs/`, which is git-ignored. It reproduces the same metrics as `ml/train.py`.

---

## 13. Deployment on Render

**Live API:** https://credence-ai-api.onrender.com (Swagger: https://credence-ai-api.onrender.com/docs).

The service needs **no database** and no secrets.

1. Push the repository to GitHub. `models/credence_model.joblib` (about 11 MB) and `models/imputation_values.json` are committed, so Render does not need to retrain.
2. Sign in at https://render.com with GitHub, then choose **New → Blueprint** and select the repository. `render.yaml` configures:
   * plan: free
   * build: `pip install -r requirements.txt`
   * start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT` (a single process, so all requests share one in-memory history)
   * health check: `/health`
   * `PYTHON_VERSION=3.14.3`, `MAX_STORED_PREDICTIONS=10000`
3. Click **Apply** and wait for the build (a few minutes). Because the repository is connected, every later `git push` redeploys automatically.
4. Verify at `/health` (expect `"storage":"in-memory (temporary, cleared on restart)"`), then open `/docs`, or run the Postman collection with the `render` environment.

**Free plan notes:**

* The service sleeps after about 15 minutes without traffic, so the first request after that takes roughly 30–60 seconds.
* **The in-memory history is cleared whenever the service sleeps, restarts or redeploys.**
* Peak memory measured locally was about 280 MB, within the 512 MB limit.

The `scikit-learn` version in `requirements.txt` is pinned to the version used for training (1.9.1). This matters because joblib models must be loaded with the same scikit-learn version.

---

## 14. Reproducibility

* `random_state=42` is used for the split, the CV folds and all models.
* All dependency versions are pinned in `requirements.txt`.
* `models/model_metadata.json` records the model, hyper-parameters, threshold, metrics, scikit-learn version and training timestamp.
