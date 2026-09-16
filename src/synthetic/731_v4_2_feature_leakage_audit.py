#!/usr/bin/env python3
"""
================================================================================
731 V4.2 FEATURE & LEAKAGE AUDIT
================================================================================

Purpose
-------
Audit the V4.1 ML-ready candidate dataset before model training.

The audit creates three modelling regimes:

1. REALISTIC_BASELINE
   Technical candidate attributes and requirement-derived features that are
   available before the synthetic engineer makes a choice.

2. PROVENANCE_AWARE
   Adds synthetic provenance/process metadata. This is a sensitivity analysis,
   not the primary thesis feature set.

3. MECHANISTIC_UPPER_BOUND
   Adds variables directly used by the synthetic engineer preference mechanism.
   This measures how well ML can recover the known synthetic preference process.

Forbidden/evaluation-only fields are never included in an approved modelling
regime.

Important:
-----------
The synthetic engineer labels are NOT historical engineer decisions.
The resulting models estimate the synthetic preference mechanism only.

Dependencies
------------
This script intentionally requires only NumPy and Pandas. Spearman correlation
is calculated as Pearson correlation over rank-transformed values, so SciPy is
NOT required.
"""

from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd


# =============================================================================
# 0. PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = PROJECT_ROOT / "data" / "processed" / "ml"

INPUT_DATASET = ML_DIR / "731_v4_1_ml_candidate_dataset.csv"

OUTPUT_AUDIT = ML_DIR / "731_v4_2_feature_leakage_audit.csv"
OUTPUT_SUMMARY = ML_DIR / "731_v4_2_feature_regime_summary.csv"
OUTPUT_PROFILE = ML_DIR / "731_v4_2_feature_profile.csv"
OUTPUT_CORRELATIONS = ML_DIR / "731_v4_2_feature_correlations.csv"
OUTPUT_TARGET_RELATIONSHIPS = ML_DIR / "731_v4_2_target_relationships.csv"
OUTPUT_MANIFEST = ML_DIR / "731_v4_2_model_feature_manifest.csv"
OUTPUT_METRICS = ML_DIR / "731_v4_2_audit_metrics.csv"
OUTPUT_REPORT = ML_DIR / "731_v4_2_audit_report.txt"


# =============================================================================
# 1. FEATURE GOVERNANCE
# =============================================================================

# Direct target / post-choice / ground-truth information.
FORBIDDEN = {
    "ml_target",
    "is_synthetic_engineer_choice",
    "engineer_rank",
    "synthetic_engineer_score",
    "synthetic_engineer_utility",
    "tie_break_jitter",
    "ground_truth_configuration_id",
    "is_ground_truth_configuration",
    "true_configuration_id",
    "recommended_configuration_id",
    "ground_truth_engineer_rank",
    "ground_truth_engineer_score",
    "ground_truth_selected_by_engineer",
}

# Identifiers, split/group fields, and evaluation-only fields.
EVALUATION_ONLY = {
    "rfq_id",
    "customer_id",
    "canonical_configuration_id",
    "ml_split",
    "target_package",
    "package_context",
    "config_package_context",
    "candidate_count",
    "pairwise_comparison_count",
}

# Direct components of the synthetic engineer utility.
MECHANISTIC = {
    "unmentioned_features",
    "source_frequency_score",
    "real_proximity_score",
    "low_change_score",
    "observed_configuration_score",
    "evidence_score_normalized",
}

# Synthetic generation/provenance information.
PROVENANCE = {
    "record_type",
    "generation_method",
    "evidence_level",
    "source_frequency",
    "nearest_real_distance",
    "num_changed_characteristics",
    "evidence_score",
    "preference_label_provenance",
    "preference_generation_method",
}

# Technical candidate configuration characteristics.
TECHNICAL = {
    "devCategory",
    "characteristic",
    "housing",
    "powerSupply",
    "numberOfChannels",
    "explosionApproval",
    "protectionArea",
    "certification",
    "dataInterface",
    "stromSchaltbar",
    "stromEingaenge",
    "temperaturEingaenge",
    "binaerDigitalOpenColl_MN",
    "binaerOpenColl_MP",
    "waveInjector",
    "dev_advMeterVerification",
    "dynamicGasMaster",
    "customUserFluid",
    "steamApplication",
}

# Features derived from RFQ requirements and candidate configuration.
REQUIREMENT_DERIVED = {
    "mandatory_total",
    "mandatory_satisfied",
    "mandatory_violations",
    "preferred_total",
    "preferred_satisfied",
    "preferred_differences",
    "mandatory_valid",
    "mandatory_preferred_valid",
}

REALISTIC = TECHNICAL | REQUIREMENT_DERIVED


# =============================================================================
# 2. HELPERS
# =============================================================================

def normalize_bool(value):
    """Normalize common boolean representations."""
    if pd.isna(value):
        return np.nan

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    text = str(value).strip().lower()

    if text in {"true", "1", "yes", "y", "t"}:
        return True

    if text in {"false", "0", "no", "n", "f"}:
        return False

    return np.nan


def looks_like_identifier(column_name):
    """Defensive check for identifier-like fields."""
    name = column_name.lower()

    patterns = [
        "rfq_id",
        "customer_id",
        "configuration_id",
        "synthetic_record_id",
        "source_canonical_configuration_id",
        "true_configuration_id",
        "recommended_configuration_id",
    ]

    return any(pattern in name for pattern in patterns)


def cramers_v(df, feature, target):
    """
    Calculate Cramer's V for categorical feature-target association.

    Returns NaN where the statistic is not defined.
    """
    x = df[feature].astype("string").fillna("<NA>")
    y = df[target].astype("string").fillna("<NA>")

    table = pd.crosstab(x, y)

    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan

    observed = table.to_numpy(dtype=float)
    n = observed.sum()

    if n <= 1:
        return np.nan

    row_totals = observed.sum(axis=1, keepdims=True)
    col_totals = observed.sum(axis=0, keepdims=True)
    expected = (row_totals @ col_totals) / n

    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum(
            (observed - expected) ** 2
            / np.where(expected == 0, np.nan, expected)
        )

    phi2 = chi2 / n
    r, k = observed.shape

    phi2_corrected = max(
        0.0,
        phi2 - ((k - 1) * (r - 1)) / (n - 1),
    )

    r_corrected = r - ((r - 1) ** 2) / (n - 1)
    k_corrected = k - ((k - 1) ** 2) / (n - 1)

    denominator = min(
        k_corrected - 1,
        r_corrected - 1,
    )

    if denominator <= 0:
        return np.nan

    return float(np.sqrt(phi2_corrected / denominator))


def classify_feature(column):
    """
    Assign governance classification to a dataset column.

    Priority:
    FORBIDDEN
    -> EVALUATION_ONLY
    -> SYNTHETIC_MECHANISM
    -> PROVENANCE_PROCESS
    -> REALISTIC_TECHNICAL
    -> REALISTIC_REQUIREMENT
    -> IDENTIFIER
    -> UNCLASSIFIED
    """
    if column in FORBIDDEN:
        return (
            "FORBIDDEN",
            "Direct target, post-choice or ground-truth information",
            "NO",
            "Never use as predictor",
        )

    if column in EVALUATION_ONLY:
        return (
            "EVALUATION_ONLY",
            "Identifier, grouping or evaluation-only context",
            "NO",
            "Retain for grouping/evaluation only",
        )

    if column in MECHANISTIC:
        return (
            "SYNTHETIC_MECHANISM",
            "Direct component of synthetic engineer preference mechanism",
            "NO",
            "Use only in mechanistic upper-bound experiment",
        )

    if column in PROVENANCE:
        return (
            "PROVENANCE_PROCESS",
            "Synthetic provenance/process metadata",
            "NO",
            "Use only in provenance-aware sensitivity experiment",
        )

    if column in TECHNICAL:
        return (
            "REALISTIC_TECHNICAL",
            "Candidate technical configuration characteristic",
            "YES",
            "Realistic baseline",
        )

    if column in REQUIREMENT_DERIVED:
        return (
            "REALISTIC_REQUIREMENT",
            "Requirement-derived candidate satisfaction feature",
            "YES",
            "Realistic baseline; candidate-pool filtering caveat",
        )

    if looks_like_identifier(column):
        return (
            "IDENTIFIER",
            "Identifier-like field",
            "NO",
            "Never use as predictive feature",
        )

    return (
        "UNCLASSIFIED",
        "Not explicitly approved by V4.2 feature governance",
        "NO",
        "Exclude until manually reviewed",
    )


def get_numeric_series(series):
    """Convert a series to numeric where possible."""
    return pd.to_numeric(series, errors="coerce")


def is_numeric_feature(series):
    """Determine whether a column is sufficiently numeric."""
    numeric = get_numeric_series(series)

    return (
        numeric.notna().mean() >= 0.95
        and numeric.nunique(dropna=True) > 1
    )


# =============================================================================
# 3. MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("731 V4.2 FEATURE & LEAKAGE AUDIT")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # [1] INPUT CHECK
    # -------------------------------------------------------------------------
    print("\n[1] Checking input file")
    print(f"V4.1 dataset: {INPUT_DATASET}")

    if not INPUT_DATASET.exists():
        raise FileNotFoundError(
            "\nV4.1 ML dataset not found:\n"
            f"{INPUT_DATASET}\n\n"
            "Run V4.1 first."
        )

    # -------------------------------------------------------------------------
    # [2] LOAD
    # -------------------------------------------------------------------------
    print("\n[2] Loading V4.1 dataset")

    df = pd.read_csv(
        INPUT_DATASET,
        low_memory=False,
    )

    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    # -------------------------------------------------------------------------
    # [3] REQUIRED SCHEMA
    # -------------------------------------------------------------------------
    print("\n[3] Validating required columns")

    required_columns = {
        "rfq_id",
        "canonical_configuration_id",
        "ml_target",
        "ml_split",
    }

    missing_required = sorted(
        required_columns - set(df.columns)
    )

    if missing_required:
        raise ValueError(
            "V4.1 dataset is missing required columns:\n"
            + "\n".join(f"  - {c}" for c in missing_required)
        )

    print("Required columns: PASS")

    # -------------------------------------------------------------------------
    # [4] TARGET
    # -------------------------------------------------------------------------
    print("\n[4] Validating target")

    df["ml_target"] = pd.to_numeric(
        df["ml_target"],
        errors="coerce",
    )

    if df["ml_target"].isna().any():
        raise ValueError("ml_target contains missing/non-numeric values.")

    unique_targets = set(df["ml_target"].unique())

    if not unique_targets.issubset({0, 1}):
        raise ValueError(
            f"ml_target contains unexpected values: {sorted(unique_targets)}"
        )

    positive_rows = int((df["ml_target"] == 1).sum())
    negative_rows = int((df["ml_target"] == 0).sum())

    print(f"Positive rows: {positive_rows:,}")
    print(f"Negative rows: {negative_rows:,}")
    print(f"Positive rate: {positive_rows / len(df):.4f}")

    # -------------------------------------------------------------------------
    # [5] BOOLEAN NORMALIZATION
    # -------------------------------------------------------------------------
    print("\n[5] Normalizing boolean columns")

    boolean_columns = [
        "mandatory_valid",
        "mandatory_preferred_valid",
        "is_synthetic_engineer_choice",
        "is_ground_truth_configuration",
    ]

    for column in boolean_columns:
        if column in df.columns:
            df[column] = df[column].map(normalize_bool)

    print("Boolean normalization: PASS")

    # -------------------------------------------------------------------------
    # [6] STRUCTURAL INTEGRITY
    # -------------------------------------------------------------------------
    print("\n[6] Structural integrity")

    rfq_count = int(df["rfq_id"].nunique())

    duplicate_pairs = int(
        df.groupby(
            ["rfq_id", "canonical_configuration_id"],
            dropna=False,
        )
        .size()
        .gt(1)
        .sum()
    )

    print(f"RFQs: {rfq_count:,}")
    print(f"Duplicate RFQ/candidate pairs: {duplicate_pairs:,}")

    if duplicate_pairs != 0:
        raise ValueError(
            "Duplicate RFQ/candidate pairs detected."
        )

    print("Structural integrity: PASS")

    # -------------------------------------------------------------------------
    # [7] FEATURE CLASSIFICATION
    # -------------------------------------------------------------------------
    print("\n[7] Classifying every column")

    classification_rows = []

    for column in df.columns:
        regime, rationale, baseline, usage = classify_feature(column)

        series = df[column]

        unique_non_null = int(series.nunique(dropna=True))
        missing_count = int(series.isna().sum())
        missing_rate = missing_count / len(df)

        numeric = get_numeric_series(series)
        numeric_rate = numeric.notna().mean()

        if (
            numeric_rate >= 0.95
            and numeric.nunique(dropna=True) > 1
        ):
            feature_kind = "numeric"
        elif unique_non_null <= 20:
            feature_kind = "categorical_low_cardinality"
        else:
            feature_kind = "categorical_high_cardinality"

        classification_rows.append(
            {
                "feature": column,
                "regime": regime,
                "rationale": rationale,
                "use_in_realistic_baseline": baseline,
                "recommended_usage": usage,
                "dtype": str(series.dtype),
                "feature_kind": feature_kind,
                "unique_non_null": unique_non_null,
                "missing_count": missing_count,
                "missing_rate": missing_rate,
                "constant_feature": unique_non_null <= 1,
            }
        )

    audit = pd.DataFrame(classification_rows)

    # -------------------------------------------------------------------------
    # [8] TARGET RELATIONSHIPS
    # -------------------------------------------------------------------------
    print("\n[8] Measuring feature-target relationships")

    relationship_rows = []

    target = df["ml_target"].astype(float)

    for column in df.columns:
        if column == "ml_target":
            continue

        regime = audit.loc[
            audit["feature"].eq(column),
            "regime",
        ].iloc[0]

        numeric = get_numeric_series(df[column])

        if (
            numeric.notna().mean() >= 0.95
            and numeric.nunique(dropna=True) > 1
        ):
            valid = numeric.notna() & target.notna()

            if valid.sum() > 2:
                pearson = numeric[valid].corr(
                    target[valid],
                    method="pearson",
                )
                spearman = numeric[valid].rank().corr(
                    target[valid].rank(),
                    method="pearson",
                )
            else:
                pearson = np.nan
                spearman = np.nan

            categorical_association = np.nan
            association_type = "pearson_spearman"

        else:
            pearson = np.nan
            spearman = np.nan

            categorical_association = cramers_v(
                df,
                column,
                "ml_target",
            )

            association_type = "cramers_v"

        relationship_rows.append(
            {
                "feature": column,
                "regime": regime,
                "pearson_with_target": pearson,
                "spearman_with_target": spearman,
                "categorical_association": categorical_association,
                "association_type": association_type,
            }
        )

    target_relationships = pd.DataFrame(
        relationship_rows
    )

    # -------------------------------------------------------------------------
    # [9] MERGE RELATIONSHIPS INTO AUDIT
    # -------------------------------------------------------------------------
    print("\n[9] Screening target associations")

    audit = audit.merge(
        target_relationships[
            [
                "feature",
                "pearson_with_target",
                "spearman_with_target",
                "categorical_association",
            ]
        ],
        on="feature",
        how="left",
    )

    audit["high_association_flag"] = (
        audit["pearson_with_target"].abs().ge(0.80)
        |
        audit["spearman_with_target"].abs().ge(0.80)
        |
        audit["categorical_association"].ge(0.80)
    )

    high_association = audit[
        audit["high_association_flag"]
    ].copy()

    print(
        f"High-association features: {len(high_association):,}"
    )

    if len(high_association) > 0:
        print("\nHigh-association features:")

        display_columns = [
            "feature",
            "regime",
            "pearson_with_target",
            "spearman_with_target",
            "categorical_association",
        ]

        print(
            high_association[
                display_columns
            ]
            .sort_values("feature")
            .to_string(index=False)
        )

    # -------------------------------------------------------------------------
    # [10] CONSTANT FEATURES
    # -------------------------------------------------------------------------
    print("\n[10] Checking constant features")

    constant_features = sorted(
        audit.loc[
            audit["constant_feature"],
            "feature",
        ].tolist()
    )

    print(
        f"Constant features: {len(constant_features):,}"
    )

    if constant_features:
        for feature in constant_features:
            print(f"  - {feature}")

    constant_set = set(constant_features)
    all_columns = set(df.columns)

    # -------------------------------------------------------------------------
    # [11] FEATURE REGIMES
    # -------------------------------------------------------------------------
    print("\n[11] Building feature regimes")

    realistic_features = sorted(
        (
            (TECHNICAL | REQUIREMENT_DERIVED)
            & all_columns
            - constant_set
            - FORBIDDEN
            - EVALUATION_ONLY
        )
    )

    provenance_features = sorted(
        (
            (TECHNICAL | REQUIREMENT_DERIVED | PROVENANCE)
            & all_columns
            - constant_set
            - FORBIDDEN
            - EVALUATION_ONLY
        )
    )

    mechanistic_features = sorted(
        (
            (
                TECHNICAL
                | REQUIREMENT_DERIVED
                | PROVENANCE
                | MECHANISTIC
            )
            & all_columns
            - constant_set
            - FORBIDDEN
            - EVALUATION_ONLY
        )
    )

    print(
        f"Realistic baseline features:      {len(realistic_features)}"
    )
    print(
        f"Provenance-aware features:        {len(provenance_features)}"
    )
    print(
        f"Mechanistic upper-bound features: {len(mechanistic_features)}"
    )

    # -------------------------------------------------------------------------
    # [12] MANIFEST
    # -------------------------------------------------------------------------
    print("\n[12] Creating model feature manifest")

    manifest_rows = []

    for column in sorted(all_columns - {"ml_target"}):
        if column in realistic_features:
            model_role = "REALISTIC_BASELINE"
        elif column in provenance_features:
            model_role = "PROVENANCE_AWARE"
        elif column in mechanistic_features:
            model_role = "MECHANISTIC_UPPER_BOUND"
        elif column in MECHANISTIC:
            model_role = "MECHANISTIC_ONLY"
        elif column in PROVENANCE:
            model_role = "PROVENANCE_ONLY"
        elif column in FORBIDDEN:
            model_role = "FORBIDDEN"
        elif column in EVALUATION_ONLY:
            model_role = "EVALUATION_ONLY"
        else:
            model_role = "EXCLUDED_UNCLASSIFIED"

        manifest_rows.append(
            {
                "feature": column,
                "model_role": model_role,
                "in_realistic_baseline": column in realistic_features,
                "in_provenance_aware": column in provenance_features,
                "in_mechanistic_upper_bound": (
                    column in mechanistic_features
                ),
            }
        )

    manifest = pd.DataFrame(manifest_rows)

    # -------------------------------------------------------------------------
    # [13] NUMERIC CORRELATION AUDIT
    # -------------------------------------------------------------------------
    print("\n[13] Numeric feature correlation audit")

    correlation_candidates = sorted(
        (
            set(realistic_features)
            | (PROVENANCE & all_columns)
            | (MECHANISTIC & all_columns)
        )
    )

    numeric_data = {}

    for column in correlation_candidates:
        numeric = get_numeric_series(df[column])

        if (
            numeric.notna().mean() >= 0.95
            and numeric.nunique(dropna=True) > 1
        ):
            numeric_data[column] = numeric

    correlation_rows = []

    if len(numeric_data) >= 2:
        ranked_numeric_data = pd.DataFrame(
            numeric_data
        ).rank(method="average")

        correlation_matrix = ranked_numeric_data.corr(
            method="pearson"
        )

        columns = list(correlation_matrix.columns)

        for i, feature_a in enumerate(columns):
            for feature_b in columns[i + 1:]:
                correlation = correlation_matrix.loc[
                    feature_a,
                    feature_b,
                ]

                if pd.isna(correlation):
                    continue

                correlation = float(correlation)

                correlation_rows.append(
                    {
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "spearman_correlation": correlation,
                        "abs_spearman": abs(correlation),
                        "high_correlation_flag": (
                            abs(correlation) >= 0.90
                        ),
                    }
                )

    correlations = pd.DataFrame(
        correlation_rows,
        columns=[
            "feature_a",
            "feature_b",
            "spearman_correlation",
            "abs_spearman",
            "high_correlation_flag",
        ],
    )

    if len(correlations) > 0:
        correlations = (
            correlations
            .sort_values(
                "abs_spearman",
                ascending=False,
            )
            .reset_index(drop=True)
        )

    high_correlation_pair_count = (
        int(
            correlations[
                "high_correlation_flag"
            ].sum()
        )
        if len(correlations) > 0
        else 0
    )

    print(
        f"Numeric features audited: {len(numeric_data):,}"
    )
    print(
        f"High-correlation pairs: {high_correlation_pair_count:,}"
    )

    if high_correlation_pair_count > 0:
        print("\nTop high-correlation pairs:")
        print(
            correlations[
                correlations["high_correlation_flag"]
            ]
            .head(20)
            .to_string(index=False)
        )

    # -------------------------------------------------------------------------
    # [14] FEATURE PROFILE
    # -------------------------------------------------------------------------
    print("\n[14] Creating feature profile")

    profile = (
        audit
        .sort_values(
            [
                "regime",
                "high_association_flag",
                "feature",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # [15] LEAKAGE VALIDATION
    # -------------------------------------------------------------------------
    print("\n[15] Final leakage validation")

    forbidden_in_realistic = sorted(
        set(realistic_features) & FORBIDDEN
    )

    forbidden_in_provenance = sorted(
        set(provenance_features) & FORBIDDEN
    )

    forbidden_in_mechanistic = sorted(
        set(mechanistic_features) & FORBIDDEN
    )

    evaluation_in_realistic = sorted(
        set(realistic_features) & EVALUATION_ONLY
    )

    unknown_features = sorted(
        set(
            audit.loc[
                audit["regime"].eq("UNCLASSIFIED"),
                "feature",
            ]
        )
    )

    print(
        f"Forbidden in realistic baseline: {len(forbidden_in_realistic)}"
    )
    print(
        f"Forbidden in provenance-aware:   {len(forbidden_in_provenance)}"
    )
    print(
        f"Forbidden in mechanistic:         {len(forbidden_in_mechanistic)}"
    )
    print(
        f"Evaluation fields in baseline:    {len(evaluation_in_realistic)}"
    )
    print(
        f"Unclassified fields:              {len(unknown_features)}"
    )

    if forbidden_in_realistic:
        raise RuntimeError(
            "Forbidden field entered realistic baseline:\n"
            + "\n".join(forbidden_in_realistic)
        )

    if forbidden_in_provenance:
        raise RuntimeError(
            "Forbidden field entered provenance-aware regime:\n"
            + "\n".join(forbidden_in_provenance)
        )

    if forbidden_in_mechanistic:
        raise RuntimeError(
            "Forbidden field entered mechanistic regime:\n"
            + "\n".join(forbidden_in_mechanistic)
        )

    if evaluation_in_realistic:
        raise RuntimeError(
            "Evaluation-only field entered realistic baseline:\n"
            + "\n".join(evaluation_in_realistic)
        )

    print("Leakage governance: PASS")

    # -------------------------------------------------------------------------
    # [16] ADDITIONAL TARGET LEAKAGE CHECKS
    # -------------------------------------------------------------------------
    print("\n[16] Additional target leakage checks")

    direct_label_columns_present = sorted(
        set(df.columns) & FORBIDDEN
    )

    target_copy_features = []

    for column in df.columns:
        if column == "ml_target":
            continue

        numeric = get_numeric_series(df[column])

        if (
            numeric.notna().mean() >= 0.99
            and numeric.nunique(dropna=True) == 2
        ):
            values = set(
                numeric.dropna().unique()
            )

            if values.issubset({0, 1}):
                agreement = (
                    numeric
                    .eq(df["ml_target"])
                    .mean()
                )

                if agreement >= 0.999:
                    target_copy_features.append(
                        {
                            "feature": column,
                            "target_copy_agreement": float(
                                agreement
                            ),
                        }
                    )

    print(
        f"Direct forbidden/evaluation fields present in dataset: "
        f"{len(direct_label_columns_present)}"
    )

    print(
        f"Potential exact target-copy features: "
        f"{len(target_copy_features)}"
    )

    if target_copy_features:
        print(
            pd.DataFrame(
                target_copy_features
            ).to_string(index=False)
        )

    # -------------------------------------------------------------------------
    # [17] METRICS
    # -------------------------------------------------------------------------
    print("\n[17] Preparing audit metrics")

    metrics = {
        "rows": int(len(df)),
        "rfqs": int(df["rfq_id"].nunique()),
        "candidate_configuration_ids": int(
            df["canonical_configuration_id"].nunique()
        ),
        "positive_rows": positive_rows,
        "negative_rows": negative_rows,
        "positive_rate": float(
            positive_rows / len(df)
        ),
        "realistic_baseline_feature_count": int(
            len(realistic_features)
        ),
        "provenance_aware_feature_count": int(
            len(provenance_features)
        ),
        "mechanistic_upper_bound_feature_count": int(
            len(mechanistic_features)
        ),
        "forbidden_columns_present": int(
            len(set(df.columns) & FORBIDDEN)
        ),
        "evaluation_only_columns_present": int(
            len(set(df.columns) & EVALUATION_ONLY)
        ),
        "constant_feature_count": int(
            len(constant_features)
        ),
        "unclassified_feature_count": int(
            len(unknown_features)
        ),
        "high_association_feature_count": int(
            len(high_association)
        ),
        "high_numeric_correlation_pair_count": int(
            high_correlation_pair_count
        ),
        "duplicate_rfq_candidate_pairs": int(
            duplicate_pairs
        ),
        "mean_missing_rate_all_features": float(
            audit["missing_rate"].mean()
        ),
        "max_missing_rate_all_features": float(
            audit["missing_rate"].max()
        ),
        "potential_target_copy_feature_count": int(
            len(target_copy_features)
        ),
    }

    # -------------------------------------------------------------------------
    # [18] REGIME SUMMARY
    # -------------------------------------------------------------------------
    print("\n[18] Building regime summary")

    regime_summary = pd.DataFrame(
        [
            {
                "feature_regime": "REALISTIC_BASELINE",
                "feature_count": len(realistic_features),
                "interpretation": (
                    "Primary thesis feature set using technical "
                    "candidate attributes and requirement-derived "
                    "features available before engineer choice."
                ),
                "thesis_role": "MAIN_EXPERIMENT",
            },
            {
                "feature_regime": "PROVENANCE_AWARE",
                "feature_count": len(provenance_features),
                "interpretation": (
                    "Adds synthetic provenance/process metadata. "
                    "Use as sensitivity/upper-bound analysis."
                ),
                "thesis_role": "SENSITIVITY_EXPERIMENT",
            },
            {
                "feature_regime": "MECHANISTIC_UPPER_BOUND",
                "feature_count": len(mechanistic_features),
                "interpretation": (
                    "Adds variables directly involved in the "
                    "synthetic engineer preference mechanism."
                ),
                "thesis_role": "MECHANISM_RECOVERY_BENCHMARK",
            },
            {
                "feature_regime": "FORBIDDEN",
                "feature_count": len(
                    set(df.columns) & FORBIDDEN
                ),
                "interpretation": (
                    "Direct target, post-choice or ground-truth "
                    "information."
                ),
                "thesis_role": "NEVER_MODEL",
            },
        ]
    )

    # -------------------------------------------------------------------------
    # [19] WRITE OUTPUTS
    # -------------------------------------------------------------------------
    print("\n[19] Writing outputs")

    ML_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        OUTPUT_AUDIT,
        index=False,
    )

    regime_summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    profile.to_csv(
        OUTPUT_PROFILE,
        index=False,
    )

    correlations.to_csv(
        OUTPUT_CORRELATIONS,
        index=False,
    )

    target_relationships.to_csv(
        OUTPUT_TARGET_RELATIONSHIPS,
        index=False,
    )

    manifest.to_csv(
        OUTPUT_MANIFEST,
        index=False,
    )

    metrics_df = pd.DataFrame(
        [
            {
                "metric": key,
                "value": (
                    json.dumps(value)
                    if isinstance(value, (list, dict))
                    else value
                ),
            }
            for key, value in metrics.items()
        ]
    )

    metrics_df.to_csv(
        OUTPUT_METRICS,
        index=False,
    )

    # -------------------------------------------------------------------------
    # [20] TEXT REPORT
    # -------------------------------------------------------------------------
    print("\n[20] Writing text report")

    report_lines = [
        "731 V4.2 FEATURE & LEAKAGE AUDIT",
        "=" * 80,
        "",
        f"Rows: {len(df):,}",
        f"RFQs: {df['rfq_id'].nunique():,}",
        (
            "Candidate configuration IDs: "
            f"{df['canonical_configuration_id'].nunique():,}"
        ),
        "",
        "FEATURE REGIMES",
        f"REALISTIC_BASELINE: {len(realistic_features)}",
        f"PROVENANCE_AWARE: {len(provenance_features)}",
        f"MECHANISTIC_UPPER_BOUND: {len(mechanistic_features)}",
        "",
        "REALISTIC BASELINE FEATURES",
    ]

    report_lines.extend(
        f"  - {feature}"
        for feature in realistic_features
    )

    report_lines.extend(
        [
            "",
            "PROVENANCE / PROCESS FEATURES",
        ]
    )

    report_lines.extend(
        f"  - {feature}"
        for feature in sorted(
            PROVENANCE & set(df.columns)
        )
    )

    report_lines.extend(
        [
            "",
            "SYNTHETIC MECHANISM FEATURES",
        ]
    )

    report_lines.extend(
        f"  - {feature}"
        for feature in sorted(
            MECHANISTIC & set(df.columns)
        )
    )

    report_lines.extend(
        [
            "",
            "FORBIDDEN FEATURES PRESENT IN DATASET",
        ]
    )

    report_lines.extend(
        f"  - {feature}"
        for feature in sorted(
            FORBIDDEN & set(df.columns)
        )
    )

    report_lines.extend(
        [
            "",
            "UNCLASSIFIED FEATURES",
        ]
    )

    if unknown_features:
        report_lines.extend(
            f"  - {feature}"
            for feature in unknown_features
        )
    else:
        report_lines.append("  None")

    report_lines.extend(
        [
            "",
            "HIGH TARGET-ASSOCIATION FEATURES",
        ]
    )

    if len(high_association) > 0:
        for _, row in (
            high_association
            .sort_values("feature")
            .iterrows()
        ):
            report_lines.append(
                "  - "
                f"{row['feature']} | "
                f"regime={row['regime']} | "
                f"pearson={row['pearson_with_target']} | "
                f"spearman={row['spearman_with_target']} | "
                f"categorical={row['categorical_association']}"
            )
    else:
        report_lines.append("  None")

    report_lines.extend(
        [
            "",
            "HIGH NUMERIC CORRELATION PAIRS",
        ]
    )

    if high_correlation_pair_count > 0:
        for _, row in (
            correlations[
                correlations["high_correlation_flag"]
            ]
            .head(50)
            .iterrows()
        ):
            report_lines.append(
                "  - "
                f"{row['feature_a']} <-> "
                f"{row['feature_b']} | "
                f"spearman={row['spearman_correlation']}"
            )
    else:
        report_lines.append("  None")

    report_lines.extend(
        [
            "",
            "LEAKAGE VALIDATION",
            (
                "Forbidden in realistic baseline: "
                f"{forbidden_in_realistic}"
            ),
            (
                "Forbidden in provenance-aware: "
                f"{forbidden_in_provenance}"
            ),
            (
                "Forbidden in mechanistic upper bound: "
                f"{forbidden_in_mechanistic}"
            ),
            (
                "Evaluation-only fields in realistic baseline: "
                f"{evaluation_in_realistic}"
            ),
            "",
            "METHODOLOGICAL INTERPRETATION",
            (
                "REALISTIC_BASELINE is the primary thesis feature regime."
            ),
            (
                "PROVENANCE_AWARE is a sensitivity/upper-bound experiment."
            ),
            (
                "MECHANISTIC_UPPER_BOUND evaluates recovery of the known "
                "synthetic preference-generation mechanism."
            ),
            (
                "Synthetic engineer labels do not represent historical "
                "engineer decisions."
            ),
            (
                "Strong target association is a screening signal, not "
                "automatic proof of leakage."
            ),
            (
                "Features directly used by the synthetic label generator "
                "must not be interpreted as independent evidence of "
                "real engineer behaviour."
            ),
        ]
    )

    OUTPUT_REPORT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # [21] FINAL OUTPUT
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("V4.2 FEATURE & LEAKAGE AUDIT COMPLETE")
    print("=" * 80)

    print(
        f"\nRows:                         {len(df):,}"
    )
    print(
        f"RFQs:                         {df['rfq_id'].nunique():,}"
    )
    print(
        f"Realistic baseline features:  {len(realistic_features)}"
    )
    print(
        f"Provenance-aware features:    {len(provenance_features)}"
    )
    print(
        f"Mechanistic upper-bound:      {len(mechanistic_features)}"
    )
    print(
        f"Unclassified features:        {len(unknown_features)}"
    )
    print(
        f"High-association features:    {len(high_association)}"
    )
    print(
        f"High-correlation pairs:       {high_correlation_pair_count}"
    )
    print(
        f"Potential target-copy fields: {len(target_copy_features)}"
    )

    print("\nOutputs:")
    print(f"  {OUTPUT_AUDIT}")
    print(f"  {OUTPUT_SUMMARY}")
    print(f"  {OUTPUT_PROFILE}")
    print(f"  {OUTPUT_CORRELATIONS}")
    print(f"  {OUTPUT_TARGET_RELATIONSHIPS}")
    print(f"  {OUTPUT_MANIFEST}")
    print(f"  {OUTPUT_METRICS}")
    print(f"  {OUTPUT_REPORT}")

    if (
        len(forbidden_in_realistic) == 0
        and len(forbidden_in_provenance) == 0
        and len(forbidden_in_mechanistic) == 0
        and len(evaluation_in_realistic) == 0
        and len(target_copy_features) == 0
    ):
        print(
            "\nLeakage checks: PASS"
        )
    else:
        print(
            "\nLeakage checks: FAIL"
        )
        sys.exit(2)

    print(
        "\nV4.2 complete."
    )
    print(
        "Review the feature manifest and target-association results "
        "before V4.3 model training."
    )


if __name__ == "__main__":
    main()
