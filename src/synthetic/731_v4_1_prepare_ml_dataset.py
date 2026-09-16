#!/usr/bin/env python3
"""
731 V4.1 - Build leakage-safe ML-ready candidate dataset

Purpose
-------
Build the modelling table for the next thesis stage from the already-generated
731 synthetic data. The modelling unit is:

    one RFQ x one technically valid candidate configuration

Primary target:

    is_synthetic_engineer_choice

This script does NOT generate new synthetic observations.

Design principles
-----------------
1. Ground-truth/evaluation fields are excluded from ML features.
2. RFQ-level grouping is preserved so train/validation/test splitting can be
   performed by RFQ, preventing candidate-level leakage.
3. Existing synthetic engineer labels remain explicitly SYNTHETIC.
4. Requirement compliance is represented only through information that would
   be available before the engineer choice.
5. Columns that may encode the synthetic-generation process are retained in
   an audit file and explicitly classified rather than silently treating them
   as historical engineer signals.

Expected project root
---------------------
Run from:

    ~/thesis-configurator

Expected inputs
---------------
data/synthetic/engineer_preferences/731_engineer_candidate_preferences_v1.csv
data/synthetic/engineer_preferences/731_engineer_rankings_v1.csv
data/synthetic/engineer_preferences/731_engineer_pairwise_preferences_v1.csv
data/synthetic/rfqs/731_rfq_ground_truth_v2.csv
data/synthetic/rfqs/731_rfq_requirements_v2.csv
data/synthetic/configurations/731_synthetic_configurations_v5.csv

Outputs
-------
data/processed/ml/
    731_v4_1_ml_candidate_dataset.csv
    731_v4_1_ml_feature_dictionary.csv
    731_v4_1_ml_leakage_audit.csv
    731_v4_1_ml_split_summary.csv
    731_v4_1_ml_preparation_metrics.csv
    731_v4_1_ml_pairwise_dataset.csv
    731_v4_1_ml_rfq_summary.csv
"""

from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CANDIDATE_PATH = (
    PROJECT_ROOT
    / "data/synthetic/engineer_preferences"
    / "731_engineer_candidate_preferences_v1.csv"
)
RANKING_PATH = (
    PROJECT_ROOT
    / "data/synthetic/engineer_preferences"
    / "731_engineer_rankings_v1.csv"
)
PAIRWISE_PATH = (
    PROJECT_ROOT
    / "data/synthetic/engineer_preferences"
    / "731_engineer_pairwise_preferences_v1.csv"
)
GT_PATH = PROJECT_ROOT / "data/synthetic/rfqs/731_rfq_ground_truth_v2.csv"
REQ_PATH = PROJECT_ROOT / "data/synthetic/rfqs/731_rfq_requirements_v2.csv"
CONFIG_PATH = (
    PROJECT_ROOT
    / "data/synthetic/configurations/731_synthetic_configurations_v5.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data/processed/ml"


# ---------------------------------------------------------------------
# Expected schemas
# ---------------------------------------------------------------------

REQUIRED_CANDIDATE = {
    "rfq_id",
    "target_package",
    "canonical_configuration_id",
    "package_context",
    "record_type",
    "generation_method",
    "evidence_level",
    "source_frequency",
    "nearest_real_distance",
    "num_changed_characteristics",
    "evidence_score",
    "mandatory_total",
    "mandatory_satisfied",
    "mandatory_violations",
    "preferred_total",
    "preferred_satisfied",
    "preferred_differences",
    "unmentioned_features",
    "mandatory_valid",
    "mandatory_preferred_valid",
    "source_frequency_score",
    "real_proximity_score",
    "low_change_score",
    "observed_configuration_score",
    "evidence_score_normalized",
    "synthetic_engineer_utility",
    "tie_break_jitter",
    "synthetic_engineer_score",
    "preference_label_provenance",
    "preference_generation_method",
    "is_ground_truth_configuration",
    "engineer_rank",
    "candidate_count",
    "is_synthetic_engineer_choice",
    "pairwise_comparison_count",
}

REQUIRED_RANKING = {
    "rfq_id",
    "target_package",
    "candidate_count",
    "synthetic_engineer_choice",
    "synthetic_engineer_top_score",
    "ground_truth_configuration_id",
    "ground_truth_engineer_rank",
    "ground_truth_engineer_score",
    "ground_truth_selected_by_engineer",
    "preference_label_provenance",
    "preference_generation_method",
}

REQUIRED_PAIRWISE = {
    "rfq_id",
    "target_package",
    "preferred_configuration_id",
    "rejected_configuration_id",
    "preferred_rank",
    "rejected_rank",
    "preferred_score",
    "rejected_score",
    "score_margin",
    "preferred_unmentioned_features",
    "rejected_unmentioned_features",
    "preferred_nearest_real_distance",
    "rejected_nearest_real_distance",
    "preferred_num_changed_characteristics",
    "rejected_num_changed_characteristics",
    "preferred_record_type",
    "rejected_record_type",
    "label",
    "preference_label_provenance",
    "preference_generation_method",
}

REQUIRED_GT = {
    "rfq_id",
    "customer_id",
    "canonical_configuration_id",
    "target_package",
    "record_type",
    "evidence_level",
    "nearest_real_distance",
    "source_canonical_configuration_id",
    "ground_truth_hidden_from_customer",
}

REQUIRED_REQ = {
    "rfq_id",
}

REQUIRED_CONFIG = {
    "canonical_configuration_id",
    "package_context",
    "record_type",
    "generation_method",
    "source_canonical_configuration_id",
    "source_frequency",
    "nearest_real_distance",
    "num_changed_characteristics",
    "evidence_level",
    "evidence_score",
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


TECHNICAL_FEATURES = [
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
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def fail(message):
    raise ValueError(message)


def require_columns(df, required, name):
    missing = sorted(required - set(df.columns))
    if missing:
        fail(f"{name} is missing required columns: {missing}")


def norm_text(series):
    return (
        series.astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    )


def norm_bool(series):
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
    }
    s = norm_text(series)
    return s.str.lower().map(mapping)


def norm_id(series):
    return norm_text(series)


def normalize_package(series):
    """
    Canonical package representation for comparisons.

    Package strings in the source datasets are semantically identical but
    differ in capitalization, e.g. X731/x731 and
    F731WD_DUALCHANNEL/F731WD_DualChannel.
    """
    return (
        series.astype("string")
        .str.strip()
        .str.upper()
        .replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA})
    )


def numeric(df, columns):
    for c in columns:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def hash_rfq(value):
    # Stable deterministic hash. The seed is not used as a feature.
    return int(hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16], 16)


def deterministic_split(rfq_id):
    """
    RFQ-level 70/15/15 split.

    A hash-based split is deterministic and does not depend on row order.
    The split is performed on RFQ IDs, not candidate rows.
    """
    u = hash_rfq(rfq_id) / float(2**64)
    if u < 0.70:
        return "TRAIN"
    if u < 0.85:
        return "VALIDATION"
    return "TEST"


def normalize_requirement_columns(req):
    """
    Preserve the original requirement representation.

    We identify likely value/type columns from the known V2 dataset rather
    than inventing new requirement semantics. If the requirement file already
    contains precomputed counts, they are retained.

    This function returns a compact RFQ-level summary that can safely be
    joined to every candidate.
    """
    req = req.copy()
    req["rfq_id"] = norm_id(req["rfq_id"])

    # Detect likely type/value columns from existing data.
    type_candidates = [
        "requirement_type",
        "requirement_level",
        "preference_type",
        "priority",
        "type",
    ]
    value_candidates = [
        "requirement_value",
        "value",
        "requested_value",
        "target_value",
        "characteristic_value",
    ]
    char_candidates = [
        "characteristic",
        "requirement_characteristic",
        "feature",
        "field",
    ]

    type_col = next((c for c in type_candidates if c in req.columns), None)
    value_col = next((c for c in value_candidates if c in req.columns), None)
    char_col = next((c for c in char_candidates if c in req.columns), None)

    rows = []

    for rfq_id, g in req.groupby("rfq_id", sort=False):
        out = {
            "rfq_requirement_count": len(g),
            "rfq_mandatory_requirement_count": np.nan,
            "rfq_preferred_requirement_count": np.nan,
            "rfq_unique_required_characteristics": np.nan,
        }

        if type_col:
            t = norm_text(g[type_col]).str.upper()
            mandatory = t.isin(["MANDATORY", "REQUIRED", "MUST"])
            preferred = t.isin(["PREFERRED", "PREFERENCE", "PREFER"])
            out["rfq_mandatory_requirement_count"] = int(mandatory.sum())
            out["rfq_preferred_requirement_count"] = int(preferred.sum())

        if char_col:
            out["rfq_unique_required_characteristics"] = int(
                norm_text(g[char_col]).nunique(dropna=True)
            )

        rows.append((rfq_id, out))

    summary = pd.DataFrame(
        [{"rfq_id": k, **v} for k, v in rows]
    )

    # If the source already has useful aggregate fields, keep them.
    # Avoid adding potentially duplicate columns.
    return summary


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("=" * 80)
    print("731 V4.1 ML-READY DATASET CONSTRUCTION")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = {
        "candidate_preferences": CANDIDATE_PATH,
        "rankings": RANKING_PATH,
        "pairwise_preferences": PAIRWISE_PATH,
        "ground_truth": GT_PATH,
        "requirements": REQ_PATH,
        "configurations": CONFIG_PATH,
    }

    print("\n[1] Checking input files")
    for name, path in paths.items():
        print(f"{name:25s} {path}")
        if not path.exists():
            fail(f"Missing input file: {path}")

    print("\n[2] Loading existing datasets")
    candidates = pd.read_csv(CANDIDATE_PATH)
    rankings = pd.read_csv(RANKING_PATH)
    pairwise = pd.read_csv(PAIRWISE_PATH)
    gt = pd.read_csv(GT_PATH)
    req = pd.read_csv(REQ_PATH)
    configs = pd.read_csv(CONFIG_PATH)

    print(f"Candidate rows:       {len(candidates):,}")
    print(f"Ranking rows:         {len(rankings):,}")
    print(f"Pairwise rows:        {len(pairwise):,}")
    print(f"Ground-truth rows:    {len(gt):,}")
    print(f"Requirement rows:     {len(req):,}")
    print(f"Configuration rows:   {len(configs):,}")

    print("\n[3] Schema validation")
    require_columns(candidates, REQUIRED_CANDIDATE, "Candidate preferences")
    require_columns(rankings, REQUIRED_RANKING, "Rankings")
    require_columns(pairwise, REQUIRED_PAIRWISE, "Pairwise preferences")
    require_columns(gt, REQUIRED_GT, "Ground truth")
    require_columns(req, REQUIRED_REQ, "Requirements")
    require_columns(configs, REQUIRED_CONFIG, "Configurations")
    print("Schema validation passed.")

    print("\n[4] Normalizing identifiers and flags")

    for df in [candidates, rankings, pairwise, gt, req, configs]:
        if "rfq_id" in df.columns:
            df["rfq_id"] = norm_id(df["rfq_id"])

    for df in [candidates, rankings, pairwise, gt, configs]:
        for c in [
            "target_package",
            "package_context",
            "canonical_configuration_id",
            "ground_truth_configuration_id",
            "preferred_configuration_id",
            "rejected_configuration_id",
            "record_type",
            "evidence_level",
            "generation_method",
            "preference_label_provenance",
            "preference_generation_method",
        ]:
            if c in df.columns:
                df[c] = norm_text(df[c])

    for c in [
        "is_ground_truth_configuration",
        "is_synthetic_engineer_choice",
    ]:
        if c in candidates.columns:
            candidates[c] = norm_bool(candidates[c])

    if "synthetic_engineer_choice" in rankings.columns:
        rankings["synthetic_engineer_choice"] = norm_text(
            rankings["synthetic_engineer_choice"]
        )

    if "ground_truth_selected_by_engineer" in rankings.columns:
        rankings["ground_truth_selected_by_engineer"] = norm_bool(
            rankings["ground_truth_selected_by_engineer"]
        )

    gt["ground_truth_hidden_from_customer"] = norm_bool(
        gt["ground_truth_hidden_from_customer"]
    )

    numeric(
        candidates,
        [
            "source_frequency",
            "nearest_real_distance",
            "num_changed_characteristics",
            "evidence_score",
            "mandatory_total",
            "mandatory_satisfied",
            "mandatory_violations",
            "preferred_total",
            "preferred_satisfied",
            "preferred_differences",
            "unmentioned_features",
            "source_frequency_score",
            "real_proximity_score",
            "low_change_score",
            "observed_configuration_score",
            "evidence_score_normalized",
            "synthetic_engineer_utility",
            "tie_break_jitter",
            "synthetic_engineer_score",
            "engineer_rank",
            "candidate_count",
            "pairwise_comparison_count",
        ],
    )

    numeric(
        configs,
        [
            "source_frequency",
            "nearest_real_distance",
            "num_changed_characteristics",
            "evidence_score",
        ],
    )

    print("Normalization complete.")

    print("\n[5] Structural integrity checks")

    candidate_rfqs = set(candidates["rfq_id"].dropna())
    ranking_rfqs = set(rankings["rfq_id"].dropna())
    pairwise_rfqs = set(pairwise["rfq_id"].dropna())
    gt_rfqs = set(gt["rfq_id"].dropna())
    req_rfqs = set(req["rfq_id"].dropna())

    if candidate_rfqs != ranking_rfqs:
        fail("Candidate and ranking RFQ sets differ.")

    if not pairwise_rfqs.issubset(candidate_rfqs):
        fail("Pairwise dataset contains RFQs absent from candidate preferences.")

    if candidate_rfqs != gt_rfqs:
        missing_gt = sorted(candidate_rfqs - gt_rfqs)[:10]
        missing_candidate = sorted(gt_rfqs - candidate_rfqs)[:10]
        fail(
            "Candidate and ground-truth RFQ sets differ. "
            f"Missing GT examples={missing_gt}; missing candidate examples={missing_candidate}"
        )

    duplicate_candidate_pairs = candidates.duplicated(
        ["rfq_id", "canonical_configuration_id"]
    ).sum()

    if duplicate_candidate_pairs:
        fail(
            f"Found {duplicate_candidate_pairs:,} duplicate RFQ/configuration "
            "candidate rows."
        )

    if candidates["candidate_count"].isna().any():
        fail("Candidate count contains missing values.")

    print(f"RFQs in candidate data: {len(candidate_rfqs):,}")
    print(f"RFQs in pairwise data:  {len(pairwise_rfqs):,}")
    print(f"RFQs in requirements:   {len(req_rfqs):,}")
    print(f"Duplicate RFQ/candidate pairs: {duplicate_candidate_pairs}")

    print("\n[6] Building RFQ-level context")

    req_summary = normalize_requirement_columns(req)

    gt_context = gt[
        [
            "rfq_id",
            "customer_id",
            "target_package",
            "record_type",
            "evidence_level",
            "nearest_real_distance",
            "source_canonical_configuration_id",
        ]
    ].copy()

    gt_context = gt_context.rename(
        columns={
            "record_type": "rfq_ground_truth_record_type",
            "evidence_level": "rfq_ground_truth_evidence_level",
            "nearest_real_distance": "rfq_ground_truth_nearest_real_distance",
        }
    )

    # Ensure one RFQ-level row.
    if gt_context["rfq_id"].duplicated().any():
        fail("Ground-truth RFQ context is not one row per RFQ.")

    rfq_context = gt_context.merge(req_summary, on="rfq_id", how="left")

    # Candidate's target package should agree with GT package.
    package_check = candidates[
        ["rfq_id", "target_package"]
    ].drop_duplicates()

    package_counts = package_check.groupby("rfq_id")["target_package"].nunique()
    if (package_counts > 1).any():
        fail("An RFQ has multiple target packages in candidate data.")

    package_check = package_check.rename(
        columns={"target_package": "candidate_target_package"}
    )

    rfq_context = rfq_context.merge(package_check, on="rfq_id", how="left")

    # Package labels use different capitalization across source files
    # (e.g. x731 vs X731). Compare canonicalized representations.
    target_package_norm = normalize_package(rfq_context["target_package"])
    candidate_package_norm = normalize_package(
        rfq_context["candidate_target_package"]
    )

    mismatch = (
        target_package_norm != candidate_package_norm
    ) & candidate_package_norm.notna()

    if mismatch.any():
        mismatch_examples = rfq_context.loc[
            mismatch,
            ["rfq_id", "target_package", "candidate_target_package"],
        ].head(10)

        fail(
            "Target-package mismatches remain after canonical normalization "
            f"for {int(mismatch.sum())} RFQs. Examples:\n"
            f"{mismatch_examples.to_string(index=False)}"
        )

    print(
        "Package consistency check: "
        f"{int((~mismatch).sum()):,}/{len(rfq_context):,} RFQs agree "
        "after package normalization."
    )

    rfq_context = rfq_context.drop(columns=["candidate_target_package"])

    print(f"RFQ context rows: {len(rfq_context):,}")

    print("\n[7] Constructing modelling table")

    # Candidate preferences are already the correct observation unit:
    # RFQ x technically valid candidate.
    ml = candidates.copy()

    # Join RFQ-level context.
    context_columns = [
        "rfq_id",
        "customer_id",
        "rfq_ground_truth_record_type",
        "rfq_ground_truth_evidence_level",
        "rfq_ground_truth_nearest_real_distance",
        "rfq_requirement_count",
        "rfq_mandatory_requirement_count",
        "rfq_preferred_requirement_count",
        "rfq_unique_required_characteristics",
    ]

    ml = ml.merge(
        rfq_context[context_columns],
        on="rfq_id",
        how="left",
        validate="many_to_one",
    )

    # Join technical configuration attributes.
    config_feature_columns = [
        "canonical_configuration_id",
        "package_context",
        *TECHNICAL_FEATURES,
    ]

    config_unique = configs[config_feature_columns].copy()

    # There are duplicate canonical IDs in V5. We need one deterministic
    # technical record per ID for this candidate-level feature table.
    config_unique = config_unique.drop_duplicates(
        subset=["canonical_configuration_id"], keep="first"
    )

    if config_unique["canonical_configuration_id"].duplicated().any():
        fail("Configuration ID deduplication failed.")

    config_unique = config_unique.rename(
        columns={
            c: f"config_{c}" for c in config_unique.columns
            if c != "canonical_configuration_id"
        }
    )

    ml = ml.merge(
        config_unique,
        on="canonical_configuration_id",
        how="left",
        validate="many_to_one",
    )

    missing_config = ml["config_package_context"].isna().sum()
    if missing_config:
        fail(
            f"{missing_config:,} candidate rows could not be matched to "
            "a configuration record."
        )

    # The candidate preference file already contains package context and
    # technical metadata. We retain both the candidate-level values and the
    # configuration-table technical values so consistency can be audited.
    for left, right in [
        ("package_context", "config_package_context"),
        ("record_type", "config_record_type"),
        ("generation_method", "config_generation_method"),
        ("evidence_level", "config_evidence_level"),
    ]:
        if left in ml.columns and right in ml.columns:
            comparison = (
                ml[left].astype("string").str.strip()
                == ml[right].astype("string").str.strip()
            )
            mismatch_count = int((~comparison.fillna(False)).sum())
            if mismatch_count:
                print(
                    f"WARNING: {mismatch_count:,} rows differ between "
                    f"{left} and {right}."
                )

    # Stable RFQ-level split.
    # Canonicalize package values in the final modelling table.
    if "target_package" in ml.columns:
        ml["target_package"] = normalize_package(ml["target_package"])

    if "package_context" in ml.columns:
        ml["package_context"] = normalize_package(ml["package_context"])

    ml["ml_split"] = ml["rfq_id"].map(deterministic_split)

    # Primary target.
    ml["ml_target"] = ml["is_synthetic_engineer_choice"].astype("Int64")

    print(f"ML rows: {len(ml):,}")
    print(
        "Target distribution:\n"
        + ml["ml_target"].value_counts(dropna=False).sort_index().to_string()
    )

    print("\n[8] Leakage audit and feature classification")

    # Fields explicitly forbidden because they directly identify the outcome
    # or ground truth used for evaluation.
    forbidden = {
        "is_synthetic_engineer_choice": "TARGET",
        "ml_target": "TARGET",
        "engineer_rank": "POST_OUTCOME",
        "synthetic_engineer_score": "POST_OUTCOME",
        "synthetic_engineer_utility": "POST_OUTCOME",
        "tie_break_jitter": "POST_OUTCOME",
        "is_ground_truth_configuration": "GROUND_TRUTH_LEAKAGE",
        "ground_truth_configuration_id": "GROUND_TRUTH_LEAKAGE",
        "ground_truth_engineer_rank": "GROUND_TRUTH_LEAKAGE",
        "ground_truth_engineer_score": "GROUND_TRUTH_LEAKAGE",
        "ground_truth_selected_by_engineer": "GROUND_TRUTH_LEAKAGE",
        "synthetic_engineer_choice": "POST_OUTCOME",
        "synthetic_engineer_top_score": "POST_OUTCOME",
    }

    # Identifiers / administrative columns are not modelling features.
    identifiers = {
        "rfq_id": "IDENTIFIER",
        "customer_id": "IDENTIFIER",
        "canonical_configuration_id": "IDENTIFIER",
        "source_canonical_configuration_id": "IDENTIFIER",
        "package_context": "CONTEXT_IDENTIFIER",
        "target_package": "CONTEXT",
        "preference_label_provenance": "PROVENANCE",
        "preference_generation_method": "PROVENANCE",
        "ml_split": "SPLIT",
    }

    # Candidate features available before the synthetic engineer decision.
    candidate_feature_names = [
        "target_package",
        "package_context",
        "record_type",
        "generation_method",
        "evidence_level",
        "source_frequency",
        "nearest_real_distance",
        "num_changed_characteristics",
        "evidence_score",
        "mandatory_total",
        "mandatory_satisfied",
        "mandatory_violations",
        "preferred_total",
        "preferred_satisfied",
        "preferred_differences",
        "unmentioned_features",
        "mandatory_valid",
        "mandatory_preferred_valid",
        "source_frequency_score",
        "real_proximity_score",
        "low_change_score",
        "observed_configuration_score",
        "evidence_score_normalized",
        *[f"config_{c}" for c in TECHNICAL_FEATURES],
        "rfq_ground_truth_record_type",
        "rfq_ground_truth_evidence_level",
        "rfq_ground_truth_nearest_real_distance",
        "rfq_requirement_count",
        "rfq_mandatory_requirement_count",
        "rfq_preferred_requirement_count",
        "rfq_unique_required_characteristics",
    ]

    # IMPORTANT:
    # Several candidate fields above are synthetic-generation metadata.
    # They are useful for baseline modelling only if explicitly justified.
    # We therefore assign a separate "SYNTHETIC_PROCESS_FEATURE" category.
    synthetic_process_features = {
        "record_type",
        "generation_method",
        "source_frequency",
        "nearest_real_distance",
        "num_changed_characteristics",
        "evidence_level",
        "evidence_score",
        "source_frequency_score",
        "real_proximity_score",
        "low_change_score",
        "observed_configuration_score",
        "evidence_score_normalized",
        "rfq_ground_truth_record_type",
        "rfq_ground_truth_evidence_level",
        "rfq_ground_truth_nearest_real_distance",
    }

    # These fields are based on requirement-candidate comparison and are
    # available before the preference decision. They are legitimate for the
    # preference-ranking task, but should be described carefully because the
    # current candidate pool was generated using requirement validity.
    requirement_derived_features = {
        "mandatory_total",
        "mandatory_satisfied",
        "mandatory_violations",
        "preferred_total",
        "preferred_satisfied",
        "preferred_differences",
        "unmentioned_features",
        "mandatory_valid",
        "mandatory_preferred_valid",
        "rfq_requirement_count",
        "rfq_mandatory_requirement_count",
        "rfq_preferred_requirement_count",
        "rfq_unique_required_characteristics",
    }

    feature_rows = []

    all_columns = list(ml.columns)

    for c in all_columns:
        if c in forbidden:
            role = forbidden[c]
            use = "NO"
            rationale = "Direct target, post-outcome, or ground-truth evaluation field."
        elif c in identifiers:
            role = identifiers[c]
            use = "NO"
            rationale = "Identifier/context/provenance field; not a raw predictive feature."
        elif c in candidate_feature_names:
            role = (
                "SYNTHETIC_PROCESS_FEATURE"
                if c in synthetic_process_features
                else (
                    "REQUIREMENT_DERIVED_FEATURE"
                    if c in requirement_derived_features
                    else "CANDIDATE_FEATURE"
                )
            )
            use = "YES"
            rationale = (
                "Available before engineer selection, subject to synthetic-data caveat."
            )
        else:
            role = "UNCLASSIFIED"
            use = "NO"
            rationale = "Not approved until manually reviewed."

        feature_rows.append(
            {
                "column": c,
                "role": role,
                "use_in_baseline_ml": use,
                "rationale": rationale,
                "dtype": str(ml[c].dtype),
                "missing_count": int(ml[c].isna().sum()),
                "missing_rate": float(ml[c].isna().mean()),
                "nunique": int(ml[c].nunique(dropna=True)),
            }
        )

    feature_dictionary = pd.DataFrame(feature_rows)

    unclassified = feature_dictionary[
        feature_dictionary["role"] == "UNCLASSIFIED"
    ]["column"].tolist()

    if unclassified:
        print("UNCLASSIFIED columns excluded from modelling:")
        for c in unclassified:
            print(f"  - {c}")

    print("\nApproved baseline feature count:",
          int((feature_dictionary["use_in_baseline_ml"] == "YES").sum()))

    print("\n[9] Building baseline-safe modelling dataset")

    approved_features = feature_dictionary.loc[
        feature_dictionary["use_in_baseline_ml"] == "YES", "column"
    ].tolist()

    final_columns = (
        ["rfq_id", "canonical_configuration_id", "ml_split", "ml_target"]
        + approved_features
    )

    # Remove duplicate columns while preserving order.
    final_columns = list(dict.fromkeys(final_columns))

    ml_final = ml[final_columns].copy()

    # Validate target completeness.
    if ml_final["ml_target"].isna().any():
        fail("ML target contains missing values.")

    # Validate split integrity.
    split_counts = (
        ml_final[["rfq_id", "ml_split"]]
        .drop_duplicates()
        .groupby("rfq_id")["ml_split"]
        .nunique()
    )

    if (split_counts != 1).any():
        fail("An RFQ appears in more than one ML split.")

    print(
        ml_final.groupby("ml_split")["rfq_id"]
        .nunique()
        .reindex(["TRAIN", "VALIDATION", "TEST"])
        .fillna(0)
        .astype(int)
        .to_string()
    )

    print("\n[10] Candidate-level target balance by split")

    balance = (
        ml_final.groupby(["ml_split", "ml_target"])
        .size()
        .unstack(fill_value=0)
        .reindex(["TRAIN", "VALIDATION", "TEST"])
    )

    print(balance)

    print("\n[11] Building pairwise ML dataset")

    # Pairwise labels are already constructed as preferred > rejected.
    # Keep this dataset separate from the pointwise candidate table.
    pairwise_ml = pairwise.copy()
    pairwise_ml["ml_split"] = pairwise_ml["rfq_id"].map(deterministic_split)

    # Do not include pairwise target-construction artifacts as candidate
    # features in a pointwise model. This file is for future ranking models.
    pairwise_output_columns = [
        "rfq_id",
        "target_package",
        "preferred_configuration_id",
        "rejected_configuration_id",
        "preferred_rank",
        "rejected_rank",
        "preferred_score",
        "rejected_score",
        "score_margin",
        "preferred_unmentioned_features",
        "rejected_unmentioned_features",
        "preferred_nearest_real_distance",
        "rejected_nearest_real_distance",
        "preferred_num_changed_characteristics",
        "rejected_num_changed_characteristics",
        "preferred_record_type",
        "rejected_record_type",
        "label",
        "ml_split",
        "preference_label_provenance",
        "preference_generation_method",
    ]

    pairwise_ml = pairwise_ml[pairwise_output_columns]

    # Pairwise train/validation/test split must also be RFQ-level.
    pairwise_rfq_split = (
        pairwise_ml.groupby("rfq_id")["ml_split"].nunique()
    )
    if (pairwise_rfq_split != 1).any():
        fail("Pairwise RFQ appears in multiple splits.")

    print(
        pairwise_ml.groupby("ml_split")["rfq_id"]
        .nunique()
        .reindex(["TRAIN", "VALIDATION", "TEST"])
        .fillna(0)
        .astype(int)
        .to_string()
    )

    print("\n[12] RFQ-level summary")

    rfq_summary = (
        ml.groupby(["rfq_id", "ml_split"], as_index=False)
        .agg(
            candidate_count=("canonical_configuration_id", "nunique"),
            positive_candidates=("ml_target", "sum"),
            target_rate=("ml_target", "mean"),
            package=("target_package", "first"),
        )
    )

    rfq_summary["has_positive_target"] = (
        rfq_summary["positive_candidates"] == 1
    )

    if not rfq_summary["has_positive_target"].all():
        bad = rfq_summary.loc[
            ~rfq_summary["has_positive_target"], "rfq_id"
        ].head(10).tolist()
        fail(
            "Some RFQs do not have exactly one positive engineer choice. "
            f"Examples: {bad}"
        )

    print(
        rfq_summary.groupby("ml_split").agg(
            rfq_count=("rfq_id", "nunique"),
            mean_candidates=("candidate_count", "mean"),
            median_candidates=("candidate_count", "median"),
            min_candidates=("candidate_count", "min"),
            max_candidates=("candidate_count", "max"),
        )
    )

    print("\n[13] Preparing metrics")

    leakage_rows = feature_dictionary.copy()

    metrics = {
        "candidate_rows": len(ml_final),
        "candidate_rfqs": ml_final["rfq_id"].nunique(),
        "pairwise_rows": len(pairwise_ml),
        "pairwise_rfqs": pairwise_ml["rfq_id"].nunique(),
        "unique_candidate_configuration_ids": ml_final[
            "canonical_configuration_id"
        ].nunique(),
        "approved_baseline_feature_count": int(
            (feature_dictionary["use_in_baseline_ml"] == "YES").sum()
        ),
        "unclassified_feature_count": int(
            (feature_dictionary["role"] == "UNCLASSIFIED").sum()
        ),
        "train_rfqs": int(
            rfq_summary.loc[rfq_summary["ml_split"] == "TRAIN", "rfq_id"].nunique()
        ),
        "validation_rfqs": int(
            rfq_summary.loc[
                rfq_summary["ml_split"] == "VALIDATION", "rfq_id"
            ].nunique()
        ),
        "test_rfqs": int(
            rfq_summary.loc[rfq_summary["ml_split"] == "TEST", "rfq_id"].nunique()
        ),
        "positive_target_rows": int(ml_final["ml_target"].sum()),
        "negative_target_rows": int((ml_final["ml_target"] == 0).sum()),
        "missing_target_rows": int(ml_final["ml_target"].isna().sum()),
        "missing_rate_all_approved_features_mean": float(
            feature_dictionary.loc[
                feature_dictionary["use_in_baseline_ml"] == "YES",
                "missing_rate",
            ].mean()
        ),
    }

    metrics_df = pd.DataFrame(
        [{"metric": k, "value": v} for k, v in metrics.items()]
    )

    print(metrics_df.to_string(index=False))

    print("\n[14] Writing outputs")

    ml_path = OUTPUT_DIR / "731_v4_1_ml_candidate_dataset.csv"
    feature_path = OUTPUT_DIR / "731_v4_1_ml_feature_dictionary.csv"
    leakage_path = OUTPUT_DIR / "731_v4_1_ml_leakage_audit.csv"
    split_path = OUTPUT_DIR / "731_v4_1_ml_split_summary.csv"
    metrics_path = OUTPUT_DIR / "731_v4_1_ml_preparation_metrics.csv"
    pairwise_path = OUTPUT_DIR / "731_v4_1_ml_pairwise_dataset.csv"
    rfq_summary_path = OUTPUT_DIR / "731_v4_1_ml_rfq_summary.csv"

    ml_final.to_csv(ml_path, index=False)
    feature_dictionary.to_csv(feature_path, index=False)
    leakage_rows.to_csv(leakage_path, index=False)
    rfq_summary.to_csv(split_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    pairwise_ml.to_csv(pairwise_path, index=False)
    rfq_summary.to_csv(rfq_summary_path, index=False)

    print(f"Candidate ML dataset: {ml_path}")
    print(f"Feature dictionary:   {feature_path}")
    print(f"Leakage audit:        {leakage_path}")
    print(f"Split summary:        {split_path}")
    print(f"Metrics:              {metrics_path}")
    print(f"Pairwise dataset:     {pairwise_path}")
    print(f"RFQ summary:          {rfq_summary_path}")

    print("\n" + "=" * 80)
    print("V4.1 ML DATASET PREPARATION COMPLETE")
    print("=" * 80)

    print(
        "\nImportant methodological note:\n"
        "This dataset contains synthetic engineer labels. The resulting ML "
        "model estimates the behaviour of the synthetic preference mechanism; "
        "it must not be presented as a model trained on historical engineer "
        "decisions. Ground-truth identifiers and direct post-outcome fields "
        "are excluded from the baseline feature set. RFQ-level ground-truth "
        "context is retained only for audit/evaluation and is not approved "
        "as a baseline predictive feature."
    )


if __name__ == "__main__":
    main()
