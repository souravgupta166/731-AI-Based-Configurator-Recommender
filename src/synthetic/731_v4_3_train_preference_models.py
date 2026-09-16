#!/usr/bin/env python3
"""
731 V4.3 — Synthetic Engineer Preference Model Training
=========================================================

Goal
----
Train and evaluate candidate-selection models for the synthetic engineer
preference task.

Unit of prediction
------------------
One row = one RFQ × candidate configuration.

Target
------
ml_target:
    1 = candidate selected by the synthetic engineer
    0 = candidate not selected

Primary task
------------
Within each RFQ, rank candidate configurations so the selected candidate
appears as high as possible.

Feature regimes
---------------
A) REALISTIC_BASELINE
   Information available before engineer choice:
   - package context
   - candidate technical configuration
   - pre-choice requirement/candidate matching features

B) PROVENANCE_AWARE
   Realistic baseline + synthetic provenance/process variables.

C) MECHANISTIC_UPPER_BOUND
   Realistic + provenance + variables used by the synthetic engineer
   preference-generation mechanism.

Important methodological rules
------------------------------
- Existing RFQ-level TRAIN/VALIDATION/TEST split is reused.
- Candidates from the same RFQ never cross splits.
- Ground-truth configuration IDs and synthetic engineer scores/ranks are
  excluded.
- canonical_configuration_id is used only for grouping/evaluation.
- Ranking metrics are primary; row-level classification metrics are
  secondary.
- No SciPy dependency.
- Models are deliberately transparent baselines:
    1. Logistic Regression
    2. Random Forest
    3. HistGradientBoosting

Outputs
-------
data/processed/ml/731_v4_3_model_metrics.csv
data/processed/ml/731_v4_3_ranking_metrics.csv
data/processed/ml/731_v4_3_predictions.csv
data/processed/ml/731_v4_3_feature_importance.csv
data/processed/ml/731_v4_3_model_summary.csv
data/processed/ml/731_v4_3_report.txt
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
ML_DIR = ROOT / "data/processed/ml"

DATASET = ML_DIR / "731_v4_1_ml_candidate_dataset.csv"
MANIFEST = ML_DIR / "731_v4_2_1_model_feature_manifest.csv"

OUTPUT_METRICS = ML_DIR / "731_v4_3_model_metrics.csv"
OUTPUT_RANKING = ML_DIR / "731_v4_3_ranking_metrics.csv"
OUTPUT_PREDICTIONS = ML_DIR / "731_v4_3_predictions.csv"
OUTPUT_IMPORTANCE = ML_DIR / "731_v4_3_feature_importance.csv"
OUTPUT_SUMMARY = ML_DIR / "731_v4_3_model_summary.csv"
OUTPUT_REPORT = ML_DIR / "731_v4_3_report.txt"


# ---------------------------------------------------------------------
# MODEL / GOVERNANCE SETTINGS
# ---------------------------------------------------------------------

TARGET = "ml_target"
GROUP = "rfq_id"
CONFIG_ID = "canonical_configuration_id"
SPLIT = "ml_split"

REGIMES = [
    "REALISTIC_BASELINE",
    "PROVENANCE_AWARE",
    "MECHANISTIC_UPPER_BOUND",
]

MODELS = {
    "LOGISTIC_REGRESSION": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=731,
    ),
    "RANDOM_FOREST": RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=731,
    ),
    "HIST_GRADIENT_BOOSTING": HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=731,
    ),
}


# ---------------------------------------------------------------------
# FORBIDDEN / NON-PREDICTIVE FIELDS
# ---------------------------------------------------------------------

FORBIDDEN = {
    "ml_target",
    "ground_truth_configuration_id",
    "is_ground_truth_configuration",
    "engineer_rank",
    "synthetic_engineer_score",
    "synthetic_engineer_utility",
    "tie_break_jitter",
    "synthetic_engineer_choice",
    "ground_truth_engineer_rank",
    "ground_truth_engineer_score",
    "ground_truth_selected_by_engineer",
    "rfq_ground_truth_record_type",
    "rfq_ground_truth_evidence_level",
    "rfq_ground_truth_nearest_real_distance",
    "rfq_ground_truth_configuration_id",
    "rfq_ground_truth_package",
    "ground_truth_record_type",
    "ground_truth_evidence_level",
    "ground_truth_nearest_real_distance",
}

EVALUATION_ONLY = {
    "rfq_id",
    "customer_id",
    "canonical_configuration_id",
    "synthetic_record_id",
    "source_canonical_configuration_id",
    "ml_split",
    "candidate_count",
    "pairwise_comparison_count",
}


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def normalize_split(series):
    return (
        series.astype("string")
        .str.strip()
        .str.upper()
    )


def normalize_target(series):
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().all() and set(numeric.unique()).issubset({0, 1}):
        return numeric.astype(int)

    normalized = series.astype(str).str.strip().str.lower()
    mapped = normalized.map({
        "true": 1,
        "false": 0,
        "yes": 1,
        "no": 0,
    })

    if mapped.isna().any():
        raise ValueError("Unable to normalize ml_target to 0/1.")

    return mapped.astype(int)


def is_numeric_like(series):
    if pd.api.types.is_numeric_dtype(series):
        return True

    converted = pd.to_numeric(series, errors="coerce")
    return converted.notna().mean() >= 0.95


def ndcg_at_k(relevance, k):
    """
    Binary relevance NDCG@K for one RFQ.

    With exactly one positive candidate:
      DCG = 1/log2(rank+1)
      IDCG = 1
    for K >= rank.
    """
    rel = list(relevance[:k])

    dcg = 0.0
    for i, value in enumerate(rel, start=1):
        if value > 0:
            dcg += 1.0 / np.log2(i + 1)

    ideal = 1.0 if np.sum(relevance) > 0 else 0.0

    return dcg / ideal if ideal > 0 else np.nan


def reciprocal_rank(relevance):
    for i, value in enumerate(relevance, start=1):
        if value > 0:
            return 1.0 / i
    return 0.0


def ranking_evaluation(frame, score_col="model_score"):
    """
    Evaluate candidate ranking within RFQ.

    Returns RFQ-level macro metrics.
    """
    rows = []

    for rfq_id, group in frame.groupby(GROUP, sort=False):
        ranked = group.sort_values(
            [score_col, CONFIG_ID],
            ascending=[False, True],
            kind="mergesort",
        )

        relevance = ranked[TARGET].astype(int).to_numpy()

        positive_positions = np.flatnonzero(relevance == 1)

        if len(positive_positions) == 0:
            continue

        true_rank = int(positive_positions[0] + 1)
        candidate_count = len(ranked)

        rows.append({
            GROUP: rfq_id,
            "candidate_count": candidate_count,
            "true_rank": true_rank,
            "hit_at_1": int(true_rank <= 1),
            "hit_at_3": int(true_rank <= 3),
            "hit_at_5": int(true_rank <= 5),
            "hit_at_10": int(true_rank <= 10),
            "reciprocal_rank": reciprocal_rank(relevance),
            "ndcg_at_3": ndcg_at_k(relevance, 3),
            "ndcg_at_5": ndcg_at_k(relevance, 5),
            "ndcg_at_10": ndcg_at_k(relevance, 10),
        })

    ranking = pd.DataFrame(rows)

    if ranking.empty:
        raise ValueError("No RFQs available for ranking evaluation.")

    summary = {
        "rfq_count": len(ranking),
        "mean_candidate_count": ranking["candidate_count"].mean(),
        "median_candidate_count": ranking["candidate_count"].median(),
        "min_candidate_count": ranking["candidate_count"].min(),
        "max_candidate_count": ranking["candidate_count"].max(),
        "top1_accuracy": ranking["hit_at_1"].mean(),
        "top3_accuracy": ranking["hit_at_3"].mean(),
        "top5_accuracy": ranking["hit_at_5"].mean(),
        "top10_accuracy": ranking["hit_at_10"].mean(),
        "mrr": ranking["reciprocal_rank"].mean(),
        "mean_true_rank": ranking["true_rank"].mean(),
        "median_true_rank": ranking["true_rank"].median(),
        "ndcg_at_3": ranking["ndcg_at_3"].mean(),
        "ndcg_at_5": ranking["ndcg_at_5"].mean(),
        "ndcg_at_10": ranking["ndcg_at_10"].mean(),
    }

    return ranking, summary


def build_preprocessor(X):
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

    transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return transformer, numeric, categorical


def transformed_feature_names(preprocessor):
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return []


def extract_importance(model_pipeline, regime, model_name):
    """
    Extract coefficient/importance information when available.
    """
    preprocessor = model_pipeline.named_steps["preprocessor"]
    estimator = model_pipeline.named_steps["model"]

    names = transformed_feature_names(preprocessor)

    if hasattr(estimator, "coef_"):
        values = estimator.coef_[0]
        importance_type = "coefficient"
    elif hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        importance_type = "feature_importance"
    else:
        return pd.DataFrame()

    n = min(len(names), len(values))

    return pd.DataFrame({
        "regime": regime,
        "model": model_name,
        "transformed_feature": names[:n],
        "importance": values[:n],
        "absolute_importance": np.abs(values[:n]),
        "importance_type": importance_type,
    }).sort_values(
        "absolute_importance",
        ascending=False
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print("=" * 80)
    print("731 V4.3 SYNTHETIC ENGINEER PREFERENCE MODEL TRAINING")
    print("=" * 80)

    # -------------------------------------------------------------
    # 1. CHECK INPUTS
    # -------------------------------------------------------------
    print("\n[1] Checking inputs")

    require_file(DATASET)
    require_file(MANIFEST)

    print(f"Dataset:  {DATASET}")
    print(f"Manifest: {MANIFEST}")

    # -------------------------------------------------------------
    # 2. LOAD
    # -------------------------------------------------------------
    print("\n[2] Loading data")

    df = pd.read_csv(DATASET, low_memory=False)
    manifest = pd.read_csv(MANIFEST, low_memory=False)

    print(f"Dataset rows:    {len(df):,}")
    print(f"Dataset columns: {len(df.columns):,}")
    print(f"Manifest rows:   {len(manifest):,}")

    # -------------------------------------------------------------
    # 3. VALIDATE
    # -------------------------------------------------------------
    print("\n[3] Validating schema")

    required = {
        GROUP,
        CONFIG_ID,
        TARGET,
        SPLIT,
    }

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    manifest_required = {
        "feature",
        "final_status",
    }

    missing_manifest = sorted(
        manifest_required - set(manifest.columns)
    )

    if missing_manifest:
        raise ValueError(
            f"Manifest missing required columns: {missing_manifest}"
        )

    df[TARGET] = normalize_target(df[TARGET])
    df[SPLIT] = normalize_split(df[SPLIT])

    expected_splits = {"TRAIN", "VALIDATION", "TEST"}
    actual_splits = set(df[SPLIT].dropna().unique())

    if not expected_splits.issubset(actual_splits):
        raise ValueError(
            f"Expected splits {expected_splits}; found {actual_splits}"
        )

    # -------------------------------------------------------------
    # 4. SPLIT INTEGRITY
    # -------------------------------------------------------------
    print("\n[4] RFQ-level split integrity")

    split_counts = (
        df.groupby(SPLIT)[GROUP]
        .nunique()
        .to_dict()
    )

    print(f"TRAIN RFQs:      {split_counts.get('TRAIN', 0):,}")
    print(f"VALIDATION RFQs: {split_counts.get('VALIDATION', 0):,}")
    print(f"TEST RFQs:       {split_counts.get('TEST', 0):,}")

    cross_split = (
        df.groupby(GROUP)[SPLIT]
        .nunique()
    )

    leaking_rfqs = int((cross_split > 1).sum())

    print(f"RFQs crossing splits: {leaking_rfqs:,}")

    if leaking_rfqs:
        raise ValueError("RFQ leakage detected across train/validation/test.")

    # -------------------------------------------------------------
    # 5. POSITIVE LABEL INTEGRITY
    # -------------------------------------------------------------
    print("\n[5] Target integrity")

    positives_per_rfq = df.groupby(GROUP)[TARGET].sum()

    print(
        "RFQs with exactly one positive:",
        int((positives_per_rfq == 1).sum())
    )
    print(
        "RFQs with zero positives:",
        int((positives_per_rfq == 0).sum())
    )
    print(
        "RFQs with multiple positives:",
        int((positives_per_rfq > 1).sum())
    )

    if not (positives_per_rfq == 1).all():
        raise ValueError(
            "Expected exactly one synthetic engineer choice per RFQ."
        )

    # -------------------------------------------------------------
    # 6. FEATURE MANIFEST
    # -------------------------------------------------------------
    print("\n[6] Loading feature regimes")

    baseline = manifest.loc[
        manifest["final_status"] == "REALISTIC_BASELINE",
        "feature"
    ].tolist()

    provenance = manifest.loc[
        manifest["final_status"] == "PROVENANCE_AWARE",
        "feature"
    ].tolist()

    mechanistic = manifest.loc[
        manifest["final_status"].isin([
            "REALISTIC_BASELINE",
            "PROVENANCE_AWARE",
            "MECHANISTIC_ONLY",
        ]),
        "feature"
    ].tolist()

    # Governance correction:
    # unmentioned_features is a pre-choice variable, but it is also directly
    # used in the synthetic engineer's label-generating utility. Therefore it
    # is excluded from the strict realistic baseline and retained only in the
    # mechanistic upper-bound regime.
    baseline = [
        c for c in baseline
        if c != "unmentioned_features"
    ]

    # Remove forbidden/evaluation fields defensively.
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

    for regime, features in regime_features.items():
        print(f"{regime}: {len(features):,} features")

    # -------------------------------------------------------------
    # 7. FEATURE VALIDATION
    # -------------------------------------------------------------
    print("\n[7] Validating feature regimes")

    # target_package is evaluation/context information from upstream files,
    # not a column in the V4.1 ML candidate dataset. Never require it here.
    if "target_package" not in df.columns:
        print("target_package: not present in ML dataset (expected; not required)")


    for regime, features in regime_features.items():

        if not features:
            raise ValueError(f"{regime} has zero features.")

        forbidden_found = (
            set(features) & (FORBIDDEN | EVALUATION_ONLY)
        )

        if forbidden_found:
            raise ValueError(
                f"{regime} contains forbidden/evaluation fields: "
                f"{sorted(forbidden_found)}"
            )

        print(f"{regime}: PASS")

    # -------------------------------------------------------------
    # 8. MODEL TRAINING
    # -------------------------------------------------------------
    print("\n[8] Training models")

    train = df[df[SPLIT] == "TRAIN"].copy()
    validation = df[df[SPLIT] == "VALIDATION"].copy()
    test = df[df[SPLIT] == "TEST"].copy()

    all_metrics = []
    all_ranking_metrics = []
    all_predictions = []
    all_importance = []
    model_summary = []

    for regime, features in regime_features.items():

        print("\n" + "-" * 80)
        print(f"REGIME: {regime}")
        print(f"Features: {len(features)}")
        print("-" * 80)

        X_train = train[features].copy()
        y_train = train[TARGET].copy()

        X_val = validation[features].copy()
        y_val = validation[TARGET].copy()

        X_test = test[features].copy()
        y_test = test[TARGET].copy()

        preprocessor_template, numeric, categorical = build_preprocessor(
            X_train
        )

        print(
            f"Numeric features: {len(numeric)} | "
            f"Categorical features: {len(categorical)}"
        )

        for model_name, estimator in MODELS.items():

            print(f"\nTraining: {model_name}")

            preprocessor = preprocessor_template

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", estimator),
            ])

            pipeline.fit(X_train, y_train)

            # -----------------------------------------------------
            # VALIDATION
            # -----------------------------------------------------
            val_probability = pipeline.predict_proba(X_val)[:, 1]

            val_prediction = (
                val_probability >= 0.5
            ).astype(int)

            val_auc = roc_auc_score(
                y_val,
                val_probability
            )

            val_logloss = log_loss(
                y_val,
                val_probability,
                labels=[0, 1],
            )

            validation_metrics = {
                "regime": regime,
                "model": model_name,
                "split": "VALIDATION",
                "rows": len(validation),
                "rfqs": validation[GROUP].nunique(),
                "accuracy": accuracy_score(y_val, val_prediction),
                "precision": precision_score(
                    y_val, val_prediction, zero_division=0
                ),
                "recall": recall_score(
                    y_val, val_prediction, zero_division=0
                ),
                "f1": f1_score(
                    y_val, val_prediction, zero_division=0
                ),
                "roc_auc": val_auc,
                "log_loss": val_logloss,
            }

            all_metrics.append(validation_metrics)

            # -----------------------------------------------------
            # TEST
            # -----------------------------------------------------
            test_probability = pipeline.predict_proba(X_test)[:, 1]

            test_prediction = (
                test_probability >= 0.5
            ).astype(int)

            test_auc = roc_auc_score(
                y_test,
                test_probability
            )

            test_logloss = log_loss(
                y_test,
                test_probability,
                labels=[0, 1],
            )

            test_metrics = {
                "regime": regime,
                "model": model_name,
                "split": "TEST",
                "rows": len(test),
                "rfqs": test[GROUP].nunique(),
                "accuracy": accuracy_score(y_test, test_prediction),
                "precision": precision_score(
                    y_test, test_prediction, zero_division=0
                ),
                "recall": recall_score(
                    y_test, test_prediction, zero_division=0
                ),
                "f1": f1_score(
                    y_test, test_prediction, zero_division=0
                ),
                "roc_auc": test_auc,
                "log_loss": test_logloss,
            }

            all_metrics.append(test_metrics)

            # -----------------------------------------------------
            # TEST RANKING
            # -----------------------------------------------------
            ranking_frame = test[
                [GROUP, CONFIG_ID, TARGET]
            ].copy()

            ranking_frame["model_score"] = test_probability

            ranking_detail, ranking_summary = ranking_evaluation(
                ranking_frame,
                score_col="model_score",
            )

            ranking_summary.update({
                "regime": regime,
                "model": model_name,
                "split": "TEST",
                "feature_count": len(features),
            })

            all_ranking_metrics.append(ranking_summary)

            # -----------------------------------------------------
            # PREDICTIONS
            # -----------------------------------------------------
            prediction_frame = test[
                [
                    GROUP,
                    CONFIG_ID,
                    TARGET,
                ]
            ].copy()

            prediction_frame["regime"] = regime
            prediction_frame["model"] = model_name
            prediction_frame["model_probability"] = test_probability
            prediction_frame["model_prediction"] = test_prediction

            # Add within-RFQ rank.
            prediction_frame["predicted_rank"] = (
                prediction_frame.groupby(GROUP)[
                    "model_probability"
                ]
                .rank(
                    ascending=False,
                    method="first"
                )
                .astype(int)
            )

            prediction_frame["is_top1"] = (
                prediction_frame["predicted_rank"] == 1
            ).astype(int)

            prediction_frame["is_top3"] = (
                prediction_frame["predicted_rank"] <= 3
            ).astype(int)

            prediction_frame["is_top5"] = (
                prediction_frame["predicted_rank"] <= 5
            ).astype(int)

            prediction_frame["is_top10"] = (
                prediction_frame["predicted_rank"] <= 10
            ).astype(int)

            all_predictions.append(prediction_frame)

            # -----------------------------------------------------
            # FEATURE IMPORTANCE
            # -----------------------------------------------------
            importance = extract_importance(
                pipeline,
                regime,
                model_name,
            )

            if not importance.empty:
                all_importance.append(importance)

            model_summary.append({
                "regime": regime,
                "model": model_name,
                "feature_count": len(features),
                "numeric_feature_count": len(numeric),
                "categorical_feature_count": len(categorical),
                "train_rows": len(train),
                "validation_rows": len(validation),
                "test_rows": len(test),
            })

            print(
                f"TEST Top-1: {ranking_summary['top1_accuracy']:.4f} | "
                f"Top-3: {ranking_summary['top3_accuracy']:.4f} | "
                f"Top-5: {ranking_summary['top5_accuracy']:.4f} | "
                f"Top-10: {ranking_summary['top10_accuracy']:.4f} | "
                f"MRR: {ranking_summary['mrr']:.4f} | "
                f"NDCG@5: {ranking_summary['ndcg_at_5']:.4f}"
            )

    # -------------------------------------------------------------
    # 9. SAVE
    # -------------------------------------------------------------
    print("\n[9] Saving outputs")

    metrics_df = pd.DataFrame(all_metrics)
    ranking_df = pd.DataFrame(all_ranking_metrics)
    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True
    )
    importance_df = (
        pd.concat(all_importance, ignore_index=True)
        if all_importance
        else pd.DataFrame()
    )
    summary_df = pd.DataFrame(model_summary)

    metrics_df.to_csv(
        OUTPUT_METRICS,
        index=False
    )

    ranking_df.to_csv(
        OUTPUT_RANKING,
        index=False
    )

    predictions_df.to_csv(
        OUTPUT_PREDICTIONS,
        index=False
    )

    importance_df.to_csv(
        OUTPUT_IMPORTANCE,
        index=False
    )

    summary_df.to_csv(
        OUTPUT_SUMMARY,
        index=False
    )

    # -------------------------------------------------------------
    # 10. BEST MODEL
    # -------------------------------------------------------------
    ranking_sorted = ranking_df.sort_values(
        ["top1_accuracy", "mrr", "ndcg_at_5"],
        ascending=False,
    )

    best = ranking_sorted.iloc[0]

    # -------------------------------------------------------------
    # 11. REPORT
    # -------------------------------------------------------------
    print("\n[10] Writing report")

    report = [
        "=" * 80,
        "731 V4.3 SYNTHETIC ENGINEER PREFERENCE MODEL REPORT",
        "=" * 80,
        "",
        "Task:",
        "Rank candidate configurations within each RFQ so that the",
        "synthetic engineer-selected candidate appears as high as possible.",
        "",
        "Dataset:",
        str(DATASET),
        "",
        f"Candidate rows: {len(df):,}",
        f"RFQs: {df[GROUP].nunique():,}",
        "",
        "RFQ-level split:",
        f"  TRAIN:      {train[GROUP].nunique():,} RFQs / {len(train):,} rows",
        f"  VALIDATION: {validation[GROUP].nunique():,} RFQs / {len(validation):,} rows",
        f"  TEST:       {test[GROUP].nunique():,} RFQs / {len(test):,} rows",
        "",
        "Feature regimes:",
    ]

    for regime, features in regime_features.items():
        report.append(
            f"  {regime}: {len(features):,} features"
        )

    report.extend([
        "",
        "Primary ranking metrics:",
        "  Top-1, Top-3, Top-5, Top-10, MRR, NDCG@3, NDCG@5, NDCG@10",
        "",
        "Model results:",
        "",
    ])

    for _, row in ranking_df.iterrows():
        report.append(
            f"{row['regime']} | {row['model']} | "
            f"Top1={row['top1_accuracy']:.4f} | "
            f"Top3={row['top3_accuracy']:.4f} | "
            f"Top5={row['top5_accuracy']:.4f} | "
            f"Top10={row['top10_accuracy']:.4f} | "
            f"MRR={row['mrr']:.4f} | "
            f"NDCG@5={row['ndcg_at_5']:.4f}"
        )

    report.extend([
        "",
        "Best test model by Top-1, then MRR, then NDCG@5:",
        f"  Regime: {best['regime']}",
        f"  Model: {best['model']}",
        f"  Top-1: {best['top1_accuracy']:.4f}",
        f"  Top-3: {best['top3_accuracy']:.4f}",
        f"  Top-5: {best['top5_accuracy']:.4f}",
        f"  Top-10: {best['top10_accuracy']:.4f}",
        f"  MRR: {best['mrr']:.4f}",
        f"  NDCG@5: {best['ndcg_at_5']:.4f}",
        "",
        "Interpretation:",
        "These models estimate the synthetic engineer preference mechanism.",
        "They do not represent historical human engineer decisions.",
        "The realistic baseline is the primary thesis result; provenance-aware",
        "and mechanistic models are sensitivity/upper-bound comparisons.",
        "",
        "Leakage controls:",
        "Ground-truth IDs, engineer ranks/scores/utilities, target labels,",
        "and hidden ground-truth context were excluded from predictive features.",
        "RFQ-level splitting prevents candidates from the same RFQ crossing",
        "train/validation/test partitions.",
        "",
    ])

    OUTPUT_REPORT.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    # -------------------------------------------------------------
    # 12. FINAL
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("V4.3 MODEL TRAINING COMPLETE")
    print("=" * 80)

    print(
        f"\nBest model: {best['regime']} / {best['model']}"
    )
    print(
        f"Top-1:  {best['top1_accuracy']:.4f}"
    )
    print(
        f"Top-3:  {best['top3_accuracy']:.4f}"
    )
    print(
        f"Top-5:  {best['top5_accuracy']:.4f}"
    )
    print(
        f"Top-10: {best['top10_accuracy']:.4f}"
    )
    print(
        f"MRR:    {best['mrr']:.4f}"
    )
    print(
        f"NDCG@5: {best['ndcg_at_5']:.4f}"
    )

    print("\nOutputs:")
    for path in [
        OUTPUT_METRICS,
        OUTPUT_RANKING,
        OUTPUT_PREDICTIONS,
        OUTPUT_IMPORTANCE,
        OUTPUT_SUMMARY,
        OUTPUT_REPORT,
    ]:
        print(f"  {path}")

    print("\nNext step:")
    print(
        "Compare realistic vs provenance vs mechanistic performance and "
        "inspect feature importance before proceeding to V4.4."
    )


if __name__ == "__main__":
    main()
