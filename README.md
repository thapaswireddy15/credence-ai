# Credence AI – Credit Card Default Prediction System

Credence AI is a backend system that predicts whether a credit card customer will **default on next month's payment**, using their historical financial and repayment information. It stores every prediction in **MySQL** and provides portfolio-level risk analytics through a **FastAPI** REST API.

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
| Database | **MySQL 8**, SQLAlchemy ORM, PyMySQL |
| Testing | Pytest, FastAPI TestClient, HTTPX |
| Development & deployment | VS Code, Git & GitHub, Kaggle Notebooks (experimentation), Render (API hosting) |

---

## 3. Project structure

```
Credence AI/
├── app/                        # FastAPI backend
│   ├── main.py                 # app, lifespan (loads model + creates tables), /, /health, /model/info
│   ├── config.py               # settings from environment / .env
│   ├── database.py             # SQLAlchemy engine, session, MySQL database creation
│   ├── models.py               # ORM tables: predictions, prediction_batches
│   ├── schemas.py              # Pydantic request/response models + input validation
│   ├── crud.py                 # database queries and portfolio aggregations
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
│   └── credence_ai_credit_default.ipynb   # self-contained, Kaggle-compatible notebook
├── data/raw/default of credit card clients.csv
├── models/
│   ├── credence_model.joblib   # trained pipeline (preprocessing + Random Forest)
│   ├── model_metadata.json     # model name, threshold, metrics, features
│   └── imputation_values.json  # training-split medians/modes used for missing inputs
├── reports/
│   ├── figures/                # 14 EDA and evaluation figures
│   └── metrics/                # eda_summary.json, model_comparison.{csv,json}, classification_reports.txt,
│                               # missing_value_robustness.json
├── scripts/init_db.py          # creates the MySQL database + tables, exports sql/schema.sql
├── sql/schema.sql              # MySQL DDL, generated from the ORM models
├── tests/                      # pytest suite (49 tests)
├── postman/                    # Postman collection + environments + Newman reports
├── requirements.txt
├── render.yaml                 # Render deployment blueprint
├── .env.example                # copy to .env and fill in
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
4. The saved `.joblib` artifact contains the fitted preprocessing, so the API applies **exactly** the same transformations. The pipeline uses only built-in scikit-learn classes, so a model trained in the Kaggle notebook can also be loaded by the API.

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

## 8. Database design (MySQL)

Tables are defined with the SQLAlchemy ORM (`app/models.py`) and created automatically when the API starts. The DDL is in `sql/schema.sql`.

**`prediction_batches`**: one row per portfolio scoring run.
Columns: `id` PK, `name`, `total_customers`, `predicted_defaults`, `average_default_probability`, `created_at`.

**`predictions`**: one row per customer prediction, storing the **exact 23 inputs** together with the output for auditability.
Columns: `id` PK, `batch_id` FK → `prediction_batches.id` (nullable, `ON DELETE SET NULL`), `customer_ref`, the 23 features, `default_probability` (DOUBLE), `predicted_default`, `risk_band` ENUM('LOW','MEDIUM','HIGH'), `decision_threshold`, `model_name`, `model_version`, `imputed_features`, `created_at`.
For imputed inputs, the feature columns hold the **values the model actually used**, and `imputed_features` lists which ones were filled in (NULL = complete record). Databases created before this column existed are upgraded automatically on startup. The upgrade only adds the column and never drops data.
Indexes on `batch_id`, `customer_ref`, `risk_band` and `created_at`. Engine: InnoDB, charset utf8mb4.

A batch and all of its predictions are written in **one transaction**, so a failure leaves no partial batch behind.

**Database modes (`DATABASE_ENABLED`):**

| Mode | When | Behaviour |
|---|---|---|
| `true` (default) | Local development, teammates, any deployment with MySQL | Every prediction is stored. History and portfolio endpoints work |
| `false` (predict-only) | Public Render deployment | No MySQL needed. `/predictions` and `/predictions/batch` return results with `stored: false` and `prediction_id: null`. History, delete and portfolio endpoints return **503** with an explanation |

---

## 9. API reference

Start the server with `uvicorn app.main:app --reload` and open **http://127.0.0.1:8000/docs**.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Service information |
| GET | `/health` | Model loaded? Imputation available? Database mode and connectivity |
| GET | `/model/info` | Selected model, threshold, features, CV and test metrics |
| POST | `/predictions` | Score **one customer** (missing features imputed) and store the result |
| POST | `/predictions/batch` | Score **a portfolio** (up to 1,000 customers) and get a summary |
| GET | `/predictions` | List stored predictions, **riskiest first**. Filters: `risk_band`, `predicted_default`, `batch_id`, `customer_ref`, `min_probability`, plus `limit` and `offset` |
| GET | `/predictions/{id}` | One stored prediction with its inputs |
| DELETE | `/predictions/{id}` | Delete a stored prediction |
| GET | `/portfolio/summary` | Portfolio KPIs: predicted default rate, mean risk score, **share of credit limit at risk**, risk-band breakdown (optional `batch_id`) |
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
* The response lists `imputed_features`, and stored rows record them, so the strategist can see which predictions rest partly on imputed data.
* **More than 11 missing → HTTP 422.** A prediction based mostly on medians would be misleading.

Example: `BILL_AMT3` omitted → `"imputed_features": ["BILL_AMT3"]`.

**Measured effect on the held-out test set** (`reports/metrics/missing_value_robustness.json`; values removed at random, then imputed):

| Cells missing | 0% | 5% | 10% | 20% | 30% |
|---|---|---|---|---|---|
| ROC-AUC | 0.7747 | 0.7685 | 0.7668 | 0.7623 | 0.7558 |
| F1 (default) | 0.5424 | 0.5265 | 0.5158 | 0.4839 | 0.4455 |

Ranking quality (ROC-AUC) degrades gracefully. Losing **`PAY_0`** (the most recent repayment status) costs the most: it drops AUC by 0.031 on its own, so it is the most important field to collect.

**Input validation:** values that are present must be in range. For example, `SEX=3`, `AGE<18`, `PAY_0<-2` or negative payments are rejected with HTTP 422, as are unknown fields, including the target column. If MySQL is enabled but unreachable, the API returns HTTP 503 instead of crashing.

---

## 10. Setup guide for teammates

> **Every teammate uses their own MySQL server and credentials.** The repository and ZIP never contain a `.env` file or any password. You create your own `.env` in step 4, and it stays on your machine (git ignores it).

### Prerequisites

* **Python 3.12 or newer.** The project was developed and tested on Python 3.14. Check with `python3 --version` (on Windows, `python --version`).
* **MySQL 8 Server**, installed and running, plus a MySQL user and password that you know, e.g. `root`. Download it from https://dev.mysql.com/downloads/mysql/ or use `brew install mysql` on macOS.
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

### Step 4: Configure `.env` with your own MySQL credentials

```bash
cp .env.example .env           # Windows: copy .env.example .env
```

Open **`.env`** (not `.env.example`) and set your own values:

```
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root                      # your MySQL user
MYSQL_PASSWORD=<your own password>   # replace the placeholder
MYSQL_DATABASE=credence_ai
```

Never put a real password in `.env.example`, because that file is committed.

### Step 5: Create the MySQL database and tables

```bash
python -m scripts.init_db
```

This connects with your `.env` credentials and creates the `credence_ai` database (if it doesn't exist) with the `predictions` and `prediction_batches` tables. It should end with `ready with tables: prediction_batches, predictions`.

*Alternative (manual):* run `sql/schema.sql` in MySQL Workbench or with `mysql -u root -p < sql/schema.sql`. The API also creates missing tables automatically when it starts.

### Step 6: Start FastAPI

```bash
uvicorn app.main:app --reload
```

### Step 7: Open Swagger

Go to **http://127.0.0.1:8000/docs**. To check that everything works:

1. **GET /health** → `"status":"ok"`, `"model_loaded":true`, `"database_mode":"enabled"`, `"database_connected":true`.
2. **POST /predictions** → *Try it out* → *Execute* (an example customer is pre-filled) → `201` with a prediction.

### Optional

```bash
pytest -v            # run the 49 tests (uses a separate credence_ai_test database)
python -m ml.eda     # regenerate EDA figures
python -m ml.train   # retrain the model (~1 min); a trained model is already included
```

### Troubleshooting

| Symptom | Fix |
|---|---|
| `Access denied for user 'root'@'localhost'` | Wrong user or password in `.env`, or you edited `.env.example` instead of `.env` |
| `Can't connect to MySQL server` | The MySQL server isn't running. Start it (macOS: System Settings → MySQL, or `brew services start mysql`) |
| `/health` shows `"database_connected": false` | Same causes as above. Re-run `python -m scripts.init_db` to see the exact error |
| `ModuleNotFoundError` | The virtual environment isn't activated (step 2), or requirements aren't installed (step 3) |
| Error while loading the model | Use the exact `scikit-learn` version from `requirements.txt` |

---

## 11. Testing

```bash
pytest -v
```

There are **49 tests**:

* **ML tests:** dataset schema, the 80:20 stratified and deterministic split, the preprocessor being fitted on training data only, grouping of rare and unseen category codes, risk-band boundaries, the model artifact, probability validity, and the imputation values coming from the training split only.
* **API tests** (TestClient/HTTPX): every endpoint, input validation, MySQL persistence and read-back, deletion, batch scoring, and the portfolio summary and segment calculations.
* **Missing-value tests:** omitted and `null` values, the 11-feature limit, stored imputed values, and complete records being unchanged.
* **Predict-only mode tests**, and a test of the additive schema migration.

Database tests use a **separate MySQL database, `credence_ai_test`** (created automatically, and its rows are cleared around each test), so your real data is never touched. Override it with `TEST_DATABASE_URL` if needed. If MySQL cannot be reached, those tests are **skipped with an explanation**. They never fall back to SQLite.

---

### Postman / Newman

`postman/` contains a **Postman collection** of 19 requests with automated assertions, plus two environments (`local`: `http://127.0.0.1:8000`, `render`: the public URL).

* **Postman app:** *Import* the three JSON files, select an environment, then *Run collection*.
* **Command line** (Node.js required; Newman is fetched on demand and is not a project dependency):

```bash
npx -p newman -p newman-reporter-htmlextra newman run postman/Credence-AI.postman_collection.json \
  -e postman/Credence-AI-local.postman_environment.json -r cli,htmlextra \
  --reporter-htmlextra-export postman/reports/newman-local.html
```

`GET /health` records the server's `database_mode`, and the later tests adapt to it. Against a MySQL-backed server they check storage, history and deletion. Against the predict-only deployment they check `stored: false` and the 503 responses.

Recorded runs (HTML and JSON reports in `postman/reports/`):

| Target | Requests | Assertions | Failed |
|---|---|---|---|
| Local, MySQL mode (`credence_ai_test` database) | 19 | 51 | 0 |
| Local, predict-only mode (`DATABASE_ENABLED=false`) | 19 | 45 | 0 |
| **Live Render deployment** (https://credence-ai-api.onrender.com, predict-only) | 19 | 45 | 0 |

---

## 12. Kaggle notebook

`notebooks/credence_ai_credit_default.ipynb` is self-contained. It covers EDA, the preprocessing rationale, training all four models, evaluation, the FP/FN analysis, threshold selection and conclusions.

* **On Kaggle:** create a notebook, add the *Default of Credit Card Clients* dataset as input, and upload this notebook. It finds the CSV under `/kaggle/input/` automatically and writes outputs to `/kaggle/working/`.
* **Locally:** run it from the `notebooks/` folder. It reads `../data/raw/…` and writes to `notebooks/outputs/` (git-ignored).

The notebook reproduces the same metrics as `ml/train.py`. Kaggle is used **only for experimentation**, never to host the API.

---

## 13. Deployment on Render

**Live API:** https://credence-ai-api.onrender.com (Swagger: https://credence-ai-api.onrender.com/docs). It passed the full Postman collection, 45/45 assertions (`postman/reports/newman-render.html`).

The public deployment runs in **predict-only mode** (`DATABASE_ENABLED=false`, set in `render.yaml`), so it needs **no database**. Your local MySQL is never connected to the internet.

1. Push the repository to GitHub. `models/credence_model.joblib` (about 11 MB) and `models/imputation_values.json` are committed, so Render does not need to retrain.
2. Sign in at https://render.com with GitHub, then choose **New → Blueprint** and select the repository. `render.yaml` configures:
   * plan: free
   * build: `pip install -r requirements.txt`
   * start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   * health check: `/health`
   * `DATABASE_ENABLED=false` and `PYTHON_VERSION=3.14.3`
3. Click **Apply** and wait for the build (a few minutes). The service URL is shown on its page, e.g. `https://credence-ai-api.onrender.com`.
4. Verify it at `/health` (expect `"database_mode":"disabled"`), then open `/docs`, or run the Postman collection with the `render` environment (set `baseUrl` to your URL).

**Free plan notes:** the service sleeps after about 15 minutes without traffic, so the first request after that takes roughly 30–60 seconds. Peak memory measured locally was about 280 MB, within the 512 MB limit.

**Optional, enabling storage in the cloud:** create a hosted MySQL 8 database (e.g. Aiven, Railway or AWS RDS), then in Render set `DATABASE_ENABLED=true` and the secret `DATABASE_URL=mysql+pymysql://user:password@host:3306/credence_ai?charset=utf8mb4` (URL-encode special characters in the password).

The `scikit-learn` version in `requirements.txt` is pinned to the version used for training (1.9.1). This matters because joblib models must be loaded with the same scikit-learn version.

---

## 14. Reproducibility

* `random_state=42` is used for the split, the CV folds and all models.
* All dependency versions are pinned in `requirements.txt`.
* `models/model_metadata.json` records the model, hyper-parameters, threshold, metrics, scikit-learn version and training timestamp.
