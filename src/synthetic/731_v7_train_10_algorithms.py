"""
731 V7 - Ten-Algorithm Recommendation Benchmark
================================================

Purpose
-------
Compare ten machine-learning approaches for ranking technically valid
731 configuration candidates.

Pointwise candidate-scoring models:
    1. Logistic Regression
    2. Decision Tree
    3. k-NN
    4. SVM
    5. Random Forest
    6. Extra Trees
    7. Gradient Boosting
    8. HistGradientBoosting
    9. AdaBoost

Dedicated ranking model:
    10. Pairwise Logistic Learning-to-Rank

Important methodological rule
-----------------------------
ML does NOT determine technical validity.

The deterministic 731 configuration / constraint layer determines the
candidate pool. ML only ranks candidates within that valid pool.

The script preserves:
    - RFQ-level TRAIN / VALIDATION / TEST splits
    - leakage-safe feature governance
    - three feature regimes
    - one positive synthetic engineer choice per RFQ
    - deterministic candidate-ID tie-breaking
    - common ranking metrics

Outputs
-------
data/processed/ml/v7_10_algorithms/
"""

from __future__ import annotations

from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


warnings.filterwarnings("ignore")

RANDOM_STATE = 731

ROOT = Path(__file__).resolve().parents[2]

DATASET = ROOT / "data/processed/ml/731_v4_1_ml_candidate_dataset.csv"
MANIFEST = ROOT / "data/processed/ml/731_v4_2_1_model_feature_manifest.csv"

OUTPUT_DIR = ROOT / "data/processed/ml/v7_10_algorithms"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GROUP = "rfq_id"
CONFIG_ID = "canonical_configuration_id"
TARGET = "ml_target"
SPLIT = "ml_split"

REGIMES = [
    "REALISTIC_BASELINE",
    "PROVENANCE_AWARE",
    "MECHANISTIC_UPPER_BOUND",
]


# ---------------------------------------------------------------------
# GOVERNANCE
# ---------------------------------------------------------------------

FORBIDDEN = {
    TARGET,
    "ground_truth_configuration_id",
    "ground_truth_canonical_configuration_id",
    "is_ground_truth_configuration",
    "engineer_rank",
    "engineer_score",
    "engineer_utility",
    "synthetic_engineer_rank",
    "synthetic_engineer_score",
    "synthetic_engineer_utility",
    "preference_rank",
    "preference_score",
    "preference_utility",
    "tie_break_jitter",
    "ground_truth_evidence_level",
    "ground_truth_context",
    "ground_truth_record_type",
    "ground_truth_generation_method",
}

EVALUATION_ONLY = {
    GROUP,
    "customer_id",
    CONFIG_ID,
    "synthetic_record_id",
    "source_canonical_configuration_id",
    SPLIT,
    "candidate_count",
    "pairwise_comparison_count",
    "target_package",
}


# ---------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------

POINTWISE_MODELS = {
    "LOGISTIC_REGRESSION": LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),

    "DECISION_TREE": DecisionTreeClassifier(
        class_weight="balanced",
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=7,
        weights="distance",
        n_jobs=-1,
    ),

    "SVM": SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),

    "RANDOM_FOREST": RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),

    "EXTRA_TREES": ExtraTreesClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),

    "GRADIENT_BOOSTING": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE,
    ),

    "HIST_GRADIENT_BOOSTING": HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    ),

    "ADABOOST": AdaBoostClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
    ),
}


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def normalize_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)

    mapping = {
        "TRUE": 1,
        "FALSE": 0,
        "YES": 1,
        "NO": 0,
        "1": 1,
        "0": 0,
    }

    normalized = (
        series.astype(str)
        .str.strip()
        .str.upper()
        .map(mapping)
    )

    if normalized.isna().any():
        bad = sorted(series[normalized.isna()].astype(str).unique())
        raise ValueError(f"Could not normalize target values: {bad[:20]}")

    return normalized.astype(int)


def normalize_split(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def is_numeric_like(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True

    converted = pd.to_numeric(series, errors="coerce")
    non_null = series.notna().sum()

    if non_null == 0:
        return False

    return converted.notna().sum() / non_null >= 0.95


def build_preprocessor(X: pd.DataFrame):
    numeric = [
        c for c in X.columns
        if is_numeric_like(X[c])
    ]

    categorical = [
        c for c in X.columns
        if c not in numeric
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ])

    transformers = []

    if numeric:
        transformers.append(
            ("numeric", numeric_pipeline, numeric)
        )

    if categorical:
        transformers.append(
            ("categorical", categorical_pipeline, categorical)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return preprocessor, numeric, categorical


# ---------------------------------------------------------------------
# RANKING EVALUATION
# ---------------------------------------------------------------------

def ranking_evaluation(frame: pd.DataFrame):
    required = {GROUP, CONFIG_ID, TARGET, "model_score"}

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"Ranking frame missing columns: {sorted(missing)}"
        )

    positive_counts = frame.groupby(GROUP)[TARGET].sum()

    if not (positive_counts == 1).all():
        raise ValueError(
            "Ranking evaluation requires exactly one positive per RFQ."
        )

    ranked_rows = []
    rfq_rows = []

    for rfq_id, group in frame.groupby(GROUP, sort=False):

        ranked = (
            group.sort_values(
                ["model_score", CONFIG_ID],
                ascending=[False, True],
                kind="mergesort",
            )
            .reset_index(drop=True)
            .copy()
        )

        ranked["predicted_rank"] = np.arange(
            1,
            len(ranked) + 1,
        )

        true_row = ranked.loc[ranked[TARGET] == 1]

        if len(true_row) != 1:
            raise ValueError(
                f"RFQ {rfq_id} does not have exactly one positive."
            )

        true_rank = int(true_row["predicted_rank"].iloc[0])

        ranked_rows.append(ranked)

        rfq_rows.append({
            GROUP: rfq_id,
            "candidate_count": len(ranked),
            "true_rank": true_rank,
            "top1": float(true_rank <= 1),
            "top3": float(true_rank <= 3),
            "top5": float(true_rank <= 5),
            "top10": float(true_rank <= 10),
            "mrr": 1.0 / true_rank,
            "ndcg3": (
                1.0 / np.log2(true_rank + 1)
                if true_rank <= 3 else 0.0
            ),
            "ndcg5": (
                1.0 / np.log2(true_rank + 1)
                if true_rank <= 5 else 0.0
            ),
            "ndcg10": (
                1.0 / np.log2(true_rank + 1)
                if true_rank <= 10 else 0.0
            ),
        })

    detail = pd.concat(
        ranked_rows,
        ignore_index=True,
    )

    per_rfq = pd.DataFrame(rfq_rows)

    summary = {
        "rfqs": len(per_rfq),
        "top1": per_rfq["top1"].mean(),
        "top3": per_rfq["top3"].mean(),
        "top5": per_rfq["top5"].mean(),
        "top10": per_rfq["top10"].mean(),
        "mrr": per_rfq["mrr"].mean(),
        "ndcg3": per_rfq["ndcg3"].mean(),
        "ndcg5": per_rfq["ndcg5"].mean(),
        "ndcg10": per_rfq["ndcg10"].mean(),
        "mean_true_rank": per_rfq["true_rank"].mean(),
        "median_true_rank": per_rfq["true_rank"].median(),
    }

    return detail, per_rfq, summary


# ---------------------------------------------------------------------
# BOOTSTRAP CONFIDENCE INTERVALS
# ---------------------------------------------------------------------

def bootstrap_ranking_metrics(
    per_rfq: pd.DataFrame,
    n_bootstrap: int = 2000,
    seed: int = RANDOM_STATE,
):
    rng = np.random.default_rng(seed)

    metric_columns = [
        "top1",
        "top3",
        "top5",
        "top10",
        "mrr",
        "ndcg3",
        "ndcg5",
        "ndcg10",
        "true_rank",
    ]

    n = len(per_rfq)

    if n == 0:
        return pd.DataFrame()

    bootstrap_values = {
        metric: []
        for metric in metric_columns
    }

    for _ in range(n_bootstrap):

        idx = rng.integers(
            low=0,
            high=n,
            size=n,
        )

        sample = per_rfq.iloc[idx]

        for metric in metric_columns:
            bootstrap_values[metric].append(
                sample[metric].mean()
            )

    rows = []

    for metric, values in bootstrap_values.items():

        arr = np.asarray(values)

        display_metric = (
            "mean_true_rank"
            if metric == "true_rank"
            else metric
        )

        point_estimate = (
            per_rfq[metric].mean()
        )

        rows.append({
            "metric": display_metric,
            "estimate": point_estimate,
            "ci_lower_95": np.quantile(arr, 0.025),
            "ci_upper_95": np.quantile(arr, 0.975),
            "bootstrap_replicates": n_bootstrap,
        })

    # Median rank needs separate bootstrap handling.
    median_values = []

    for _ in range(n_bootstrap):

        idx = rng.integers(
            low=0,
            high=n,
            size=n,
        )

        sample = per_rfq.iloc[idx]

        median_values.append(
            sample["true_rank"].median()
        )

    rows.append({
        "metric": "median_true_rank",
        "estimate": per_rfq["true_rank"].median(),
        "ci_lower_95": np.quantile(median_values, 0.025),
        "ci_upper_95": np.quantile(median_values, 0.975),
        "bootstrap_replicates": n_bootstrap,
    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# CLASSIFICATION METRICS
# ---------------------------------------------------------------------

def classification_metrics(
    y_true,
    probability,
    regime,
    model_name,
    split_name,
    rows,
    rfqs,
):
    prediction = (probability >= 0.5).astype(int)

    return {
        "regime": regime,
        "model": model_name,
        "split": split_name,
        "rows": rows,
        "rfqs": rfqs,
        "accuracy": accuracy_score(y_true, prediction),
        "precision": precision_score(
            y_true,
            prediction,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            prediction,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            prediction,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probability,
        ),
        "log_loss": log_loss(
            y_true,
            probability,
            labels=[0, 1],
        ),
    }


# ---------------------------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------------------------

def extract_feature_importance(
    pipeline,
    regime,
    model_name,
):
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]

    try:
        names = preprocessor.get_feature_names_out()
    except Exception:
        return pd.DataFrame()

    if hasattr(estimator, "coef_"):
        values = np.asarray(estimator.coef_)

        if values.ndim == 2:
            values = values[0]

        importance_type = "coefficient"

    elif hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        importance_type = "feature_importance"

    else:
        return pd.DataFrame()

    n = min(
        len(names),
        len(values),
    )

    result = pd.DataFrame({
        "regime": regime,
        "model": model_name,
        "transformed_feature": names[:n],
        "importance": values[:n],
        "absolute_importance": np.abs(values[:n]),
        "importance_type": importance_type,
    })

    return result.sort_values(
        "absolute_importance",
        ascending=False,
    )


# ---------------------------------------------------------------------
# PAIRWISE LOGISTIC LTR
# ---------------------------------------------------------------------

def make_pairwise_training_data(
    candidate_df: pd.DataFrame,
    features: list[str],
):
    """
    Construct preference pairs from the candidate dataset.

    For every RFQ:
        positive candidate > every negative candidate

    Both directions are added:
        positive, negative -> label 1
        negative, positive -> label 0

    Pairwise features are constructed AFTER ordinary preprocessing.
    Therefore categorical one-hot values and numeric standardized values
    can all be represented by directional vector differences.
    """

    X = candidate_df[features].copy()
    y = candidate_df[TARGET].to_numpy()

    preprocessor, _, _ = build_preprocessor(X)

    transformed = preprocessor.fit_transform(X)

    pair_X = []
    pair_y = []

    groups = candidate_df.groupby(GROUP, sort=False).indices

    for _, indices in groups.items():

        indices = np.asarray(indices)

        local_y = y[indices]

        positive = indices[local_y == 1]
        negative = indices[local_y == 0]

        if len(positive) != 1:
            raise ValueError(
                "Pairwise training requires exactly one "
                "positive candidate per RFQ."
            )

        p = positive[0]

        for n in negative:

            difference = transformed[p] - transformed[n]

            pair_X.append(difference)
            pair_y.append(1)

            pair_X.append(-difference)
            pair_y.append(0)

    return (
        preprocessor,
        np.asarray(pair_X),
        np.asarray(pair_y),
    )


def pairwise_candidate_scores(
    model,
    preprocessor,
    candidate_df: pd.DataFrame,
    features: list[str],
):
    """
    Compare every candidate against all other candidates within its RFQ.

    Candidate score =
        mean P(candidate preferred over opponent)
    """

    transformed = preprocessor.transform(
        candidate_df[features].copy()
    )

    scores = np.zeros(len(candidate_df), dtype=float)

    groups = candidate_df.groupby(
        GROUP,
        sort=False,
    ).indices

    for _, indices in groups.items():

        indices = np.asarray(indices)

        if len(indices) == 1:
            scores[indices[0]] = 1.0
            continue

        for candidate_idx in indices:

            opponent_indices = indices[
                indices != candidate_idx
            ]

            differences = (
                transformed[candidate_idx]
                - transformed[opponent_indices]
            )

            probabilities = model.predict_proba(
                differences
            )[:, 1]

            scores[candidate_idx] = probabilities.mean()

    return scores


# ---------------------------------------------------------------------
# FEATURE REGIMES
# ---------------------------------------------------------------------

def load_regime_features(
    df: pd.DataFrame,
    manifest: pd.DataFrame,
):
    baseline = manifest.loc[
        manifest["final_status"] == "REALISTIC_BASELINE",
        "feature",
    ].tolist()

    provenance = manifest.loc[
        manifest["final_status"] == "PROVENANCE_AWARE",
        "feature",
    ].tolist()

    mechanistic = manifest.loc[
        manifest["final_status"].isin([
            "REALISTIC_BASELINE",
            "PROVENANCE_AWARE",
            "MECHANISTIC_ONLY",
        ]),
        "feature",
    ].tolist()

    # Preserve V4.3 governance correction.
    baseline = [
        c for c in baseline
        if c != "unmentioned_features"
    ]

    def clean_features(features):
        return [
            c for c in features
            if c in df.columns
            and c not in FORBIDDEN
            and c not in EVALUATION_ONLY
        ]

    baseline = clean_features(baseline)
    provenance = clean_features(provenance)
    mechanistic = clean_features(mechanistic)

    regime_features = {
        "REALISTIC_BASELINE": baseline,

        "PROVENANCE_AWARE": sorted(
            set(baseline + provenance)
        ),

        "MECHANISTIC_UPPER_BOUND": sorted(
            set(mechanistic)
        ),
    }

    return regime_features


# ---------------------------------------------------------------------
# SAVE RANKING RESULT
# ---------------------------------------------------------------------

def append_ranking_result(
    ranking_summaries,
    confidence_intervals,
    per_rfq_results,
    prediction_results,
    regime,
    model_name,
    split_name,
    source_frame,
    scores,
):
    ranking_frame = source_frame[
        [GROUP, CONFIG_ID, TARGET]
    ].copy()

    ranking_frame["model_score"] = scores

    detail, per_rfq, summary = ranking_evaluation(
        ranking_frame
    )

    summary.update({
        "regime": regime,
        "model": model_name,
        "split": split_name,
    })

    ranking_summaries.append(summary)

    per_rfq["regime"] = regime
    per_rfq["model"] = model_name
    per_rfq["split"] = split_name

    per_rfq_results.append(per_rfq)

    detail["regime"] = regime
    detail["model"] = model_name
    detail["split"] = split_name

    prediction_results.append(detail)

    if split_name == "TEST":

        ci = bootstrap_ranking_metrics(
            per_rfq,
            n_bootstrap=2000,
            seed=RANDOM_STATE,
        )

        ci["regime"] = regime
        ci["model"] = model_name
        ci["split"] = split_name

        confidence_intervals.append(ci)

    return summary


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print("=" * 80)
    print("731 V7 - TEN-ALGORITHM RECOMMENDATION BENCHMARK")
    print("=" * 80)

    # -------------------------------------------------------------
    # 1. INPUTS
    # -------------------------------------------------------------

    print("\n[1] Checking inputs")

    require_file(DATASET)
    require_file(MANIFEST)

    print(f"Candidate dataset: {DATASET}")
    print(f"Feature manifest:  {MANIFEST}")
    print(f"Output directory:  {OUTPUT_DIR}")

    # -------------------------------------------------------------
    # 2. LOAD
    # -------------------------------------------------------------

    print("\n[2] Loading data")

    df = pd.read_csv(
        DATASET,
        low_memory=False,
    )

    manifest = pd.read_csv(
        MANIFEST,
        low_memory=False,
    )

    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print(f"RFQs:    {df[GROUP].nunique():,}")

    # -------------------------------------------------------------
    # 3. SCHEMA
    # -------------------------------------------------------------

    print("\n[3] Validating schema")

    required = {
        GROUP,
        CONFIG_ID,
        TARGET,
        SPLIT,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing candidate columns: {sorted(missing)}"
        )

    manifest_required = {
        "feature",
        "final_status",
    }

    missing_manifest = (
        manifest_required - set(manifest.columns)
    )

    if missing_manifest:
        raise ValueError(
            "Feature manifest missing columns: "
            f"{sorted(missing_manifest)}"
        )

    df[TARGET] = normalize_target(df[TARGET])
    df[SPLIT] = normalize_split(df[SPLIT])

    # -------------------------------------------------------------
    # 4. SPLIT INTEGRITY
    # -------------------------------------------------------------

    print("\n[4] Checking RFQ-level split integrity")

    expected_splits = {
        "TRAIN",
        "VALIDATION",
        "TEST",
    }

    actual_splits = set(
        df[SPLIT].dropna().unique()
    )

    if not expected_splits.issubset(actual_splits):
        raise ValueError(
            f"Expected {expected_splits}; found {actual_splits}"
        )

    cross_split = (
        df.groupby(GROUP)[SPLIT]
        .nunique()
    )

    leaking_rfqs = int(
        (cross_split > 1).sum()
    )

    print(f"RFQs crossing splits: {leaking_rfqs}")

    if leaking_rfqs:
        raise ValueError(
            "RFQ leakage detected across splits."
        )

    for split_name in [
        "TRAIN",
        "VALIDATION",
        "TEST",
    ]:
        count = df.loc[
            df[SPLIT] == split_name,
            GROUP,
        ].nunique()

        print(
            f"{split_name:10s}: "
            f"{count:,} RFQs"
        )

    # -------------------------------------------------------------
    # 5. TARGET INTEGRITY
    # -------------------------------------------------------------

    print("\n[5] Checking target integrity")

    positives = (
        df.groupby(GROUP)[TARGET]
        .sum()
    )

    exactly_one = int(
        (positives == 1).sum()
    )

    zero = int(
        (positives == 0).sum()
    )

    multiple = int(
        (positives > 1).sum()
    )

    print(f"Exactly one positive: {exactly_one:,}")
    print(f"Zero positives:       {zero:,}")
    print(f"Multiple positives:   {multiple:,}")

    if not (positives == 1).all():
        raise ValueError(
            "Expected exactly one positive candidate per RFQ."
        )

    # -------------------------------------------------------------
    # 6. FEATURE REGIMES
    # -------------------------------------------------------------

    print("\n[6] Building feature regimes")

    regime_features = load_regime_features(
        df,
        manifest,
    )

    for regime, features in regime_features.items():

        if not features:
            raise ValueError(
                f"{regime} contains zero usable features."
            )

        forbidden_found = (
            set(features)
            & (FORBIDDEN | EVALUATION_ONLY)
        )

        if forbidden_found:
            raise ValueError(
                f"{regime} contains forbidden fields: "
                f"{sorted(forbidden_found)}"
            )

        print(
            f"{regime:28s}: "
            f"{len(features):3d} features"
        )

    # -------------------------------------------------------------
    # 7. SPLITS
    # -------------------------------------------------------------

    train = df[
        df[SPLIT] == "TRAIN"
    ].copy()

    validation = df[
        df[SPLIT] == "VALIDATION"
    ].copy()

    test = df[
        df[SPLIT] == "TEST"
    ].copy()

    # -------------------------------------------------------------
    # RESULT CONTAINERS
    # -------------------------------------------------------------

    classification_results = []
    ranking_summaries = []
    confidence_intervals = []
    per_rfq_results = []
    prediction_results = []
    importance_results = []
    runtime_results = []

    # -------------------------------------------------------------
    # 8. POINTWISE MODELS
    # -------------------------------------------------------------

    print("\n[7] Training nine pointwise models")

    for regime, features in regime_features.items():

        print("\n" + "=" * 80)
        print(f"REGIME: {regime}")
        print("=" * 80)

        X_train = train[features].copy()
        y_train = train[TARGET].copy()

        X_val = validation[features].copy()
        y_val = validation[TARGET].copy()

        X_test = test[features].copy()
        y_test = test[TARGET].copy()

        preprocessor_template, numeric, categorical = (
            build_preprocessor(X_train)
        )

        print(
            f"Numeric={len(numeric)} | "
            f"Categorical={len(categorical)}"
        )

        for model_name, estimator_template in POINTWISE_MODELS.items():

            print(
                f"\nTraining {model_name} "
                f"[{regime}]"
            )

            start = time.perf_counter()

            pipeline = Pipeline([
                (
                    "preprocessor",
                    clone(preprocessor_template),
                ),
                (
                    "model",
                    clone(estimator_template),
                ),
            ])

            pipeline.fit(
                X_train,
                y_train,
            )

            fit_seconds = (
                time.perf_counter() - start
            )

            # Validation
            val_probability = (
                pipeline.predict_proba(X_val)[:, 1]
            )

            classification_results.append(
                classification_metrics(
                    y_true=y_val,
                    probability=val_probability,
                    regime=regime,
                    model_name=model_name,
                    split_name="VALIDATION",
                    rows=len(validation),
                    rfqs=validation[GROUP].nunique(),
                )
            )

            val_summary = append_ranking_result(
                ranking_summaries,
                confidence_intervals,
                per_rfq_results,
                prediction_results,
                regime,
                model_name,
                "VALIDATION",
                validation,
                val_probability,
            )

            # Test
            test_probability = (
                pipeline.predict_proba(X_test)[:, 1]
            )

            classification_results.append(
                classification_metrics(
                    y_true=y_test,
                    probability=test_probability,
                    regime=regime,
                    model_name=model_name,
                    split_name="TEST",
                    rows=len(test),
                    rfqs=test[GROUP].nunique(),
                )
            )

            test_summary = append_ranking_result(
                ranking_summaries,
                confidence_intervals,
                per_rfq_results,
                prediction_results,
                regime,
                model_name,
                "TEST",
                test,
                test_probability,
            )

            importance = extract_feature_importance(
                pipeline,
                regime,
                model_name,
            )

            if not importance.empty:
                importance_results.append(
                    importance
                )

            runtime_results.append({
                "regime": regime,
                "model": model_name,
                "training_seconds": fit_seconds,
            })

            print(
                f"Validation Top1="
                f"{val_summary['top1']:.4f} | "
                f"Test Top1="
                f"{test_summary['top1']:.4f} | "
                f"Test MRR="
                f"{test_summary['mrr']:.4f}"
            )

    # -------------------------------------------------------------
    # 9. PAIRWISE LOGISTIC LTR
    # -------------------------------------------------------------

    print("\n" + "=" * 80)
    print("[8] Training Pairwise Logistic Learning-to-Rank")
    print("=" * 80)

    for regime, features in regime_features.items():

        model_name = "PAIRWISE_LOGISTIC_LTR"

        print(
            f"\nTraining {model_name} "
            f"[{regime}]"
        )

        start = time.perf_counter()

        (
            pair_preprocessor,
            pair_X_train,
            pair_y_train,
        ) = make_pairwise_training_data(
            train,
            features,
        )

        print(
            f"Pairwise training rows: "
            f"{len(pair_y_train):,}"
        )

        pair_model = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

        pair_model.fit(
            pair_X_train,
            pair_y_train,
        )

        fit_seconds = (
            time.perf_counter() - start
        )

        val_scores = pairwise_candidate_scores(
            pair_model,
            pair_preprocessor,
            validation,
            features,
        )

        val_summary = append_ranking_result(
            ranking_summaries,
            confidence_intervals,
            per_rfq_results,
            prediction_results,
            regime,
            model_name,
            "VALIDATION",
            validation,
            val_scores,
        )

        test_scores = pairwise_candidate_scores(
            pair_model,
            pair_preprocessor,
            test,
            features,
        )

        test_summary = append_ranking_result(
            ranking_summaries,
            confidence_intervals,
            per_rfq_results,
            prediction_results,
            regime,
            model_name,
            "TEST",
            test,
            test_scores,
        )

        runtime_results.append({
            "regime": regime,
            "model": model_name,
            "training_seconds": fit_seconds,
        })

        # Pairwise coefficients
        try:
            transformed_names = (
                pair_preprocessor
                .get_feature_names_out()
            )

            coefficients = (
                pair_model.coef_[0]
            )

            n = min(
                len(transformed_names),
                len(coefficients),
            )

            pair_importance = pd.DataFrame({
                "regime": regime,
                "model": model_name,
                "transformed_feature": transformed_names[:n],
                "importance": coefficients[:n],
                "absolute_importance": np.abs(
                    coefficients[:n]
                ),
                "importance_type": "pairwise_coefficient",
            }).sort_values(
                "absolute_importance",
                ascending=False,
            )

            importance_results.append(
                pair_importance
            )

        except Exception:
            pass

        print(
            f"Validation Top1="
            f"{val_summary['top1']:.4f} | "
            f"Test Top1="
            f"{test_summary['top1']:.4f} | "
            f"Test MRR="
            f"{test_summary['mrr']:.4f}"
        )

    # -------------------------------------------------------------
    # 10. SAVE OUTPUTS
    # -------------------------------------------------------------

    print("\n[9] Saving outputs")

    classification_df = pd.DataFrame(
        classification_results
    )

    ranking_df = pd.DataFrame(
        ranking_summaries
    )

    runtime_df = pd.DataFrame(
        runtime_results
    )

    classification_df.to_csv(
        OUTPUT_DIR / "731_v7_classification_metrics.csv",
        index=False,
    )

    ranking_df.to_csv(
        OUTPUT_DIR / "731_v7_ranking_metrics.csv",
        index=False,
    )

    runtime_df.to_csv(
        OUTPUT_DIR / "731_v7_runtime_summary.csv",
        index=False,
    )

    if confidence_intervals:
        pd.concat(
            confidence_intervals,
            ignore_index=True,
        ).to_csv(
            OUTPUT_DIR
            / "731_v7_bootstrap_confidence_intervals.csv",
            index=False,
        )

    if per_rfq_results:
        pd.concat(
            per_rfq_results,
            ignore_index=True,
        ).to_csv(
            OUTPUT_DIR
            / "731_v7_per_rfq_ranking_results.csv",
            index=False,
        )

    if prediction_results:
        pd.concat(
            prediction_results,
            ignore_index=True,
        ).to_csv(
            OUTPUT_DIR
            / "731_v7_candidate_predictions.csv",
            index=False,
        )

    if importance_results:
        pd.concat(
            importance_results,
            ignore_index=True,
        ).to_csv(
            OUTPUT_DIR
            / "731_v7_feature_importance.csv",
            index=False,
        )

    # -------------------------------------------------------------
    # 11. THESIS COMPARISON TABLE
    # -------------------------------------------------------------

    test_results = ranking_df[
        ranking_df["split"] == "TEST"
    ].copy()

    metric_order = [
        "regime",
        "model",
        "rfqs",
        "top1",
        "top3",
        "top5",
        "top10",
        "mrr",
        "ndcg3",
        "ndcg5",
        "ndcg10",
        "mean_true_rank",
        "median_true_rank",
    ]

    thesis_table = (
        test_results[metric_order]
        .sort_values(
            [
                "regime",
                "top1",
                "mrr",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    thesis_table.to_csv(
        OUTPUT_DIR
        / "731_v7_thesis_algorithm_comparison.csv",
        index=False,
    )

    # Overall model summary.
    overall = (
        test_results
        .groupby("model", as_index=False)
        .agg({
            "top1": "mean",
            "top3": "mean",
            "top5": "mean",
            "top10": "mean",
            "mrr": "mean",
            "ndcg3": "mean",
            "ndcg5": "mean",
            "ndcg10": "mean",
            "mean_true_rank": "mean",
        })
        .sort_values(
            ["top1", "mrr"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )

    overall.insert(
        0,
        "overall_rank",
        np.arange(1, len(overall) + 1),
    )

    overall.to_csv(
        OUTPUT_DIR
        / "731_v7_overall_algorithm_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # 12. EXPERIMENT METADATA
    # -------------------------------------------------------------

    metadata = {
        "experiment": "731 V7 ten-algorithm benchmark",
        "random_state": RANDOM_STATE,
        "pointwise_models": list(
            POINTWISE_MODELS.keys()
        ),
        "ranking_models": [
            "PAIRWISE_LOGISTIC_LTR"
        ],
        "regimes": REGIMES,
        "bootstrap_replicates": 2000,
        "candidate_dataset": str(DATASET),
        "feature_manifest": str(MANIFEST),
        "technical_validity_decided_by_ml": False,
        "ml_role": (
            "Rank candidates already admitted by the "
            "configuration/constraint workflow."
        ),
    }

    with open(
        OUTPUT_DIR / "731_v7_experiment_metadata.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
        )

    # -------------------------------------------------------------
    # 13. TERMINAL SUMMARY
    # -------------------------------------------------------------

    print("\n" + "=" * 80)
    print("V7 TEST RESULTS")
    print("=" * 80)

    display_columns = [
        "regime",
        "model",
        "top1",
        "top3",
        "top5",
        "mrr",
        "ndcg5",
        "mean_true_rank",
    ]

    print(
        thesis_table[display_columns]
        .to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("OVERALL ALGORITHM SUMMARY")
    print("=" * 80)

    print(
        overall[
            [
                "overall_rank",
                "model",
                "top1",
                "mrr",
                "ndcg5",
                "mean_true_rank",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("V7 COMPLETE")
    print("=" * 80)

    print(
        f"\nOutputs written to:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()