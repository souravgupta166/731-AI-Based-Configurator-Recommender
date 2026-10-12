#!/usr/bin/env python3
"""
731 V7 HGB INFERENCE ARTIFACT EXPORTER
======================================

Purpose
-------

Reproduce the V7 HistGradientBoosting training setup for the
REALISTIC_BASELINE regime and serialize the fitted sklearn Pipeline
for application inference.

This script does NOT modify the existing V7 benchmark outputs.

It creates a separate inference artifact using the same:

    - V4.1 candidate dataset
    - V4.2.1 feature manifest
    - REALISTIC_BASELINE feature governance
    - TRAIN split
    - preprocessing
    - HistGradientBoosting hyperparameters
    - random state

Architecture
------------

    V4.1 candidate dataset
             |
             v
    V4.2.1 feature manifest
             |
             v
    REALISTIC_BASELINE features
             |
             v
    TRAIN RFQs only
             |
             v
    V7 preprocessing
             |
             v
    HistGradientBoosting
             |
             v
    serialized inference artifact


IMPORTANT
---------

This is an inference-artifact exporter.

It does NOT:

- change V7 benchmark CSV files
- change V7 metrics
- change the V7 split
- invent new features
- include forbidden fields
- include evaluation-only fields
- use the TEST split for fitting
- generate Emerson model codes
- decide technical validity
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


# ============================================================================
# 1. PROJECT ROOT
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================================
# 2. INPUTS
# ============================================================================

DATASET = (
    ROOT
    / "data"
    / "processed"
    / "ml"
    / "731_v4_1_ml_candidate_dataset.csv"
)

MANIFEST = (
    ROOT
    / "data"
    / "processed"
    / "ml"
    / "731_v4_2_1_model_feature_manifest.csv"
)


# ============================================================================
# 3. OUTPUTS
# ============================================================================

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "ml"
    / "v7_10_algorithms"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_OUTPUT = (
    OUTPUT_DIR
    / "731_v7_HIST_GRADIENT_BOOSTING_REALISTIC_BASELINE.joblib"
)

METADATA_OUTPUT = (
    OUTPUT_DIR
    / "731_v7_HIST_GRADIENT_BOOSTING_REALISTIC_BASELINE_metadata.json"
)

FEATURES_OUTPUT = (
    OUTPUT_DIR
    / "731_v7_HIST_GRADIENT_BOOSTING_REALISTIC_BASELINE_features.json"
)


# ============================================================================
# 4. V7 CONSTANTS
# ============================================================================

RANDOM_STATE = 731

GROUP = "rfq_id"
CONFIG_ID = "canonical_configuration_id"
TARGET = "ml_target"
SPLIT = "ml_split"


# ============================================================================
# 5. V7 GOVERNANCE
# ============================================================================

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


# ============================================================================
# 6. MODEL
# ============================================================================

MODEL_NAME = (
    "HIST_GRADIENT_BOOSTING"
)

REGIME = (
    "REALISTIC_BASELINE"
)

ESTIMATOR = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=RANDOM_STATE,
)


# ============================================================================
# 7. HELPERS
# ============================================================================

def require_file(
    path: Path,
) -> None:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def normalize_target(
    series: pd.Series,
) -> pd.Series:

    if pd.api.types.is_numeric_dtype(
        series
    ):

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

        bad = sorted(
            series[
                normalized.isna()
            ]
            .astype(str)
            .unique()
        )

        raise ValueError(
            "Could not normalize target values: "
            f"{bad[:20]}"
        )

    return normalized.astype(int)


def normalize_split(
    series: pd.Series,
) -> pd.Series:

    return (
        series
        .astype(str)
        .str.strip()
        .str.upper()
    )


def is_numeric_like(
    series: pd.Series,
) -> bool:

    if pd.api.types.is_numeric_dtype(
        series
    ):

        return True

    converted = pd.to_numeric(
        series,
        errors="coerce",
    )

    non_null = series.notna().sum()

    if non_null == 0:
        return False

    return (
        converted.notna().sum()
        / non_null
        >= 0.95
    )


def build_preprocessor(
    X: pd.DataFrame,
):
    """
    Exact preprocessing logic used by V7.
    """

    numeric = [
        column
        for column in X.columns
        if is_numeric_like(
            X[column]
        )
    ]

    categorical = [
        column
        for column in X.columns
        if column not in numeric
    ]

    numeric_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    transformers = []

    if numeric:

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric,
            )
        )

    if categorical:

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return (
        preprocessor,
        numeric,
        categorical,
    )


# ============================================================================
# 8. FEATURE REGIME
# ============================================================================

def load_realistic_baseline_features(
    df: pd.DataFrame,
    manifest: pd.DataFrame,
) -> list[str]:
    """
    Reproduce V7 REALISTIC_BASELINE feature selection exactly.
    """

    if "feature" not in manifest.columns:

        raise ValueError(
            "Feature manifest does not contain "
            "'feature'."
        )

    if "final_status" not in manifest.columns:

        raise ValueError(
            "Feature manifest does not contain "
            "'final_status'."
        )

    baseline = manifest.loc[
        manifest["final_status"]
        == "REALISTIC_BASELINE",
        "feature",
    ].tolist()

    # V4.3 governance correction.
    baseline = [
        column
        for column in baseline
        if column
        != "unmentioned_features"
    ]

    features = [
        column
        for column in baseline
        if column in df.columns
        and column not in FORBIDDEN
        and column not in EVALUATION_ONLY
    ]

    if not features:

        raise ValueError(
            "REALISTIC_BASELINE contains "
            "zero usable features."
        )

    forbidden_found = (
        set(features)
        & (
            FORBIDDEN
            | EVALUATION_ONLY
        )
    )

    if forbidden_found:

        raise ValueError(
            "Forbidden/evaluation-only fields "
            "entered the feature set: "
            f"{sorted(forbidden_found)}"
        )

    return features


# ============================================================================
# 9. MAIN
# ============================================================================

def main() -> None:

    print()

    print(
        "=" * 80
    )

    print(
        "731 V7 HGB INFERENCE ARTIFACT EXPORTER"
    )

    print(
        "=" * 80
    )

    print()

    # ========================================================================
    # INPUT CHECK
    # ========================================================================

    print(
        "[1] Checking inputs"
    )

    require_file(
        DATASET
    )

    require_file(
        MANIFEST
    )

    print(
        f"  Dataset:  {DATASET}"
    )

    print(
        f"  Manifest: {MANIFEST}"
    )

    print()

    # ========================================================================
    # LOAD DATA
    # ========================================================================

    print(
        "[2] Loading V7 candidate dataset"
    )

    df = pd.read_csv(
        DATASET,
        low_memory=False,
    )

    manifest = pd.read_csv(
        MANIFEST,
        low_memory=False,
    )

    print(
        f"  Rows:    {len(df):,}"
    )

    print(
        f"  Columns: {len(df.columns):,}"
    )

    print(
        f"  RFQs:    {df[GROUP].nunique():,}"
    )

    print()

    # ========================================================================
    # SCHEMA
    # ========================================================================

    print(
        "[3] Validating schema"
    )

    required_columns = {
        GROUP,
        CONFIG_ID,
        TARGET,
        SPLIT,
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Candidate dataset is missing: "
            f"{sorted(missing)}"
        )

    manifest_required = {
        "feature",
        "final_status",
    }

    missing_manifest = (
        manifest_required
        - set(manifest.columns)
    )

    if missing_manifest:

        raise ValueError(
            "Feature manifest is missing: "
            f"{sorted(missing_manifest)}"
        )

    df[TARGET] = normalize_target(
        df[TARGET]
    )

    df[SPLIT] = normalize_split(
        df[SPLIT]
    )

    print(
        "  Schema: PASS"
    )

    print()

    # ========================================================================
    # SPLIT INTEGRITY
    # ========================================================================

    print(
        "[4] Checking RFQ split integrity"
    )

    cross_split = (
        df.groupby(GROUP)[SPLIT]
        .nunique()
    )

    leaking_rfqs = int(
        (
            cross_split
            > 1
        ).sum()
    )

    print(
        f"  RFQs crossing splits: "
        f"{leaking_rfqs}"
    )

    if leaking_rfqs:

        raise ValueError(
            "RFQ leakage detected."
        )

    train = df[
        df[SPLIT]
        == "TRAIN"
    ].copy()

    validation = df[
        df[SPLIT]
        == "VALIDATION"
    ].copy()

    test = df[
        df[SPLIT]
        == "TEST"
    ].copy()

    print(
        f"  TRAIN RFQs:      "
        f"{train[GROUP].nunique():,}"
    )

    print(
        f"  VALIDATION RFQs: "
        f"{validation[GROUP].nunique():,}"
    )

    print(
        f"  TEST RFQs:       "
        f"{test[GROUP].nunique():,}"
    )

    print()

    # ========================================================================
    # TARGET INTEGRITY
    # ========================================================================

    print(
        "[5] Checking target integrity"
    )

    positives = (
        df.groupby(GROUP)[TARGET]
        .sum()
    )

    if not (
        positives == 1
    ).all():

        raise ValueError(
            "V7 candidate dataset does not contain "
            "exactly one positive per RFQ."
        )

    print(
        "  Exactly one positive per RFQ: PASS"
    )

    print()

    # ========================================================================
    # FEATURE REGIME
    # ========================================================================

    print(
        "[6] Building REALISTIC_BASELINE feature set"
    )

    features = (
        load_realistic_baseline_features(
            df,
            manifest,
        )
    )

    print(
        f"  Features: {len(features)}"
    )

    for feature in features:

        print(
            f"    - {feature}"
        )

    print()

    # ========================================================================
    # TRAIN FEATURES
    # ========================================================================

    X_train = train[
        features
    ].copy()

    y_train = train[
        TARGET
    ].copy()

    X_validation = validation[
        features
    ].copy()

    y_validation = validation[
        TARGET
    ].copy()

    X_test = test[
        features
    ].copy()

    y_test = test[
        TARGET
    ].copy()

    # ========================================================================
    # PREPROCESSOR
    # ========================================================================

    print(
        "[7] Building V7 preprocessing pipeline"
    )

    (
        preprocessor,
        numeric_features,
        categorical_features,
    ) = build_preprocessor(
        X_train
    )

    print(
        f"  Numeric features:     "
        f"{len(numeric_features)}"
    )

    print(
        f"  Categorical features: "
        f"{len(categorical_features)}"
    )

    print()

    # ========================================================================
    # PIPELINE
    # ========================================================================

    pipeline = Pipeline(
        [
            (
                "preprocessor",
                clone(
                    preprocessor
                ),
            ),
            (
                "model",
                clone(
                    ESTIMATOR
                ),
            ),
        ]
    )

    # ========================================================================
    # FIT
    # ========================================================================

    print(
        "[8] Fitting HistGradientBoosting"
    )

    print(
        "  Regime: REALISTIC_BASELINE"
    )

    print(
        "  Training split only"
    )

    print(
        "  Estimator:"
    )

    print(
        "    HistGradientBoostingClassifier"
    )

    print(
        "    max_iter=300"
    )

    print(
        "    learning_rate=0.05"
    )

    print(
        "    max_leaf_nodes=31"
    )

    print(
        "    l2_regularization=1.0"
    )

    print(
        "    random_state=731"
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    print(
        "  Fit: PASS"
    )

    print()

    # ========================================================================
    # SANITY CHECK
    # ========================================================================

    print(
        "[9] Running inference sanity check"
    )

    validation_scores = (
        pipeline.predict_proba(
            X_validation
        )[:, 1]
    )

    test_scores = (
        pipeline.predict_proba(
            X_test
        )[:, 1]
    )

    if len(
        validation_scores
    ) != len(
        validation
    ):

        raise ValueError(
            "Validation score count mismatch."
        )

    if len(
        test_scores
    ) != len(
        test
    ):

        raise ValueError(
            "Test score count mismatch."
        )

    if not np.isfinite(
        validation_scores
    ).all():

        raise ValueError(
            "Validation scores contain "
            "non-finite values."
        )

    if not np.isfinite(
        test_scores
    ).all():

        raise ValueError(
            "Test scores contain "
            "non-finite values."
        )

    print(
        f"  Validation rows scored: "
        f"{len(validation_scores):,}"
    )

    print(
        f"  Test rows scored:       "
        f"{len(test_scores):,}"
    )

    print(
        "  Inference sanity check: PASS"
    )

    print()

    # ========================================================================
    # FEATURE NAMES
    # ========================================================================

    transformed_feature_names = []

    try:

        transformed_feature_names = (
            pipeline
            .named_steps[
                "preprocessor"
            ]
            .get_feature_names_out()
            .tolist()
        )

    except Exception:

        transformed_feature_names = []

    # ========================================================================
    # SAVE MODEL
    # ========================================================================

    print(
        "[10] Saving inference artifact"
    )

    joblib.dump(
        pipeline,
        MODEL_OUTPUT,
    )

    print(
        f"  Model:"
    )

    print(
        f"    {MODEL_OUTPUT}"
    )

    # ========================================================================
    # SAVE FEATURE METADATA
    # ========================================================================

    feature_metadata = {
        "experiment": (
            "731 V7 ten-algorithm benchmark"
        ),
        "artifact_type": (
            "serialized sklearn inference pipeline"
        ),
        "model": MODEL_NAME,
        "feature_regime": REGIME,
        "random_state": RANDOM_STATE,
        "candidate_dataset": str(
            DATASET
        ),
        "feature_manifest": str(
            MANIFEST
        ),
        "feature_count": len(
            features
        ),
        "features": features,
        "numeric_features": (
            numeric_features
        ),
        "categorical_features": (
            categorical_features
        ),
        "transformed_feature_count": (
            len(
                transformed_feature_names
            )
        ),
        "transformed_features": (
            transformed_feature_names
        ),
        "forbidden_fields": sorted(
            FORBIDDEN
        ),
        "evaluation_only_fields": sorted(
            EVALUATION_ONLY
        ),
        "technical_validity_decided_by_ml": (
            False
        ),
        "ml_role": (
            "Rank candidates already admitted by the "
            "configuration/constraint workflow."
        ),
        "model_code_generation": (
            "NOT_IMPLEMENTED"
        ),
    }

    with open(
        FEATURES_OUTPUT,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            feature_metadata,
            handle,
            indent=2,
        )

    # ========================================================================
    # SAVE ARTIFACT METADATA
    # ========================================================================

    metadata = {
        "experiment": (
            "731 V7 ten-algorithm benchmark"
        ),
        "artifact_type": (
            "inference model"
        ),
        "model": MODEL_NAME,
        "feature_regime": REGIME,
        "random_state": RANDOM_STATE,
        "estimator_parameters": {
            "max_iter": 300,
            "learning_rate": 0.05,
            "max_leaf_nodes": 31,
            "l2_regularization": 1.0,
            "random_state": RANDOM_STATE,
        },
        "training": {
            "dataset": str(DATASET),
            "manifest": str(MANIFEST),
            "train_rows": int(
                len(train)
            ),
            "train_rfqs": int(
                train[GROUP].nunique()
            ),
            "validation_rows": int(
                len(validation)
            ),
            "validation_rfqs": int(
                validation[GROUP].nunique()
            ),
            "test_rows": int(
                len(test)
            ),
            "test_rfqs": int(
                test[GROUP].nunique()
            ),
        },
        "governance": {
            "technical_validity_decided_by_ml": (
                False
            ),
            "ml_role": (
                "Rank candidates already admitted by the "
                "configuration/constraint workflow."
            ),
            "model_code_generation": (
                "NOT_IMPLEMENTED"
            ),
        },
        "files": {
            "model": str(
                MODEL_OUTPUT
            ),
            "feature_metadata": str(
                FEATURES_OUTPUT
            ),
        },
    }

    with open(
        METADATA_OUTPUT,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            metadata,
            handle,
            indent=2,
        )

    # ========================================================================
    # FINAL
    # ========================================================================

    print(
        f"  Feature metadata:"
    )

    print(
        f"    {FEATURES_OUTPUT}"
    )

    print(
        f"  Artifact metadata:"
    )

    print(
        f"    {METADATA_OUTPUT}"
    )

    print()

    print(
        "=" * 80
    )

    print(
        "V7 HGB INFERENCE ARTIFACT COMPLETE"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "MODEL"
    )

    print(
        f"  {MODEL_OUTPUT}"
    )

    print()

    print(
        "REGIME"
    )

    print(
        f"  {REGIME}"
    )

    print()

    print(
        "FEATURES"
    )

    print(
        f"  {len(features)} raw V7 features"
    )

    print(
        f"  {len(transformed_feature_names)} "
        "transformed features"
    )

    print()

    print(
        "IMPORTANT"
    )

    print(
        "  This artifact ranks candidates."
    )

    print(
        "  It does not determine technical validity."
    )

    print(
        "  It does not generate an Emerson model code."
    )

    print()


if __name__ == "__main__":
    main()