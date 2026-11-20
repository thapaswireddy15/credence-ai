"""Preprocessing and model pipeline definitions.

The preprocessing is built ONLY from scikit-learn transformers and lives inside
the same ``Pipeline`` object as the classifier. Consequences:

* transformations are fit on the training data only (``Pipeline.fit``), and
  during cross-validation they are re-fit inside every fold -> no data leakage;
* the saved ``.joblib`` artifact contains the preprocessing, so the API applies
  exactly the same transformations at inference time;
* no custom Python classes are pickled, so a model trained in the Kaggle
  notebook is loadable by the API as well.
"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ml.config import (
    CATEGORICAL_FEATURES,
    CATEGORY_MIN_FREQUENCY,
    CONTINUOUS_FEATURES,
    MONETARY_FEATURES,
    RANDOM_STATE,
    REPAYMENT_STATUS_FEATURES,
)


def build_preprocessor() -> ColumnTransformer:
    """Feature-type-specific preprocessing.

    * Categorical codes (SEX, EDUCATION, MARRIAGE): one-hot encoding. Codes are
      nominal, so treating them as numbers would invent a false order. Rare and
      undocumented codes are merged into an "infrequent" bucket; codes never
      seen in training are also routed there instead of crashing the API.
    * Repayment status (PAY_0..PAY_6): ordinal -> kept numeric and standardised.
    * Monetary amounts (LIMIT_BAL, BILL_AMT*, PAY_AMT*): Yeo-Johnson power
      transform + standardisation. It handles the strong right skew AND the
      negative bill amounts (credit balances), and shrinks the influence of
      extreme values without deleting any customer.
    * AGE: standardised.
    """
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=CATEGORY_MIN_FREQUENCY,
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
            ("repayment_status", StandardScaler(), REPAYMENT_STATUS_FEATURES),
            ("monetary", PowerTransformer(method="yeo-johnson", standardize=True), MONETARY_FEATURES),
            ("continuous", StandardScaler(), CONTINUOUS_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_pipeline(classifier) -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor()), ("classifier", classifier)])


def model_candidates() -> dict:
    """The four required algorithms with small, justified hyper-parameter grids.

    ``class_weight="balanced"`` counteracts the 78/22 class imbalance so the
    models do not simply learn to predict "no default". KNN has no class-weight
    option; it is tuned as-is and its lower recall is discussed in the report.
    """
    return {
        "Logistic Regression": {
            "estimator": LogisticRegression(
                class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
            ),
            "param_grid": {"classifier__C": [0.01, 0.1, 1.0, 10.0]},
        },
        "Decision Tree": {
            "estimator": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            "param_grid": {
                "classifier__max_depth": [4, 6, 8, 10],
                "classifier__min_samples_leaf": [20, 50, 100],
            },
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                n_jobs=1,  # GridSearchCV already parallelises across folds
                random_state=RANDOM_STATE,
            ),
            "param_grid": {
                "classifier__max_depth": [8, 12],
                "classifier__min_samples_leaf": [5, 10, 20],
            },
        },
        "K-Nearest Neighbors": {
            "estimator": KNeighborsClassifier(),
            "param_grid": {
                "classifier__n_neighbors": [15, 31, 51],
                "classifier__weights": ["uniform", "distance"],
            },
        },
    }
