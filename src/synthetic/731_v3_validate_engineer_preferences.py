#!/usr/bin/env python3

"""
731 V3 — Synthetic Engineer Preference Validation

Purpose
-------
Validate the synthetic engineer-preference layer before ML ranking.

The validator is intentionally dependency-light and does not require SciPy.
Spearman correlations are calculated from pandas rank-transformed values.

This script checks:

1. Data integrity
2. Candidate-pool coverage
3. Candidate-count distribution
4. Ground-truth ranking distribution
5. Synthetic engineer Top-1 / Top-3 / Top-5 / Top-10
6. Pairwise preference consistency
7. Package-level behaviour
8. Observed vs derived behaviour
9. Preference-feature behaviour
10. Potential ground-truth leakage
11. Provenance
12. Whether the synthetic engineer choice is overly deterministic

Inputs
------
data/synthetic/engineer_preferences/
    731_engineer_candidate_preferences_v1.csv
    731_engineer_rankings_v1.csv
    731_engineer_pairwise_preferences_v1.csv

data/synthetic/rfqs/
    731_rfq_ground_truth_v2.csv

data/synthetic/configurations/
    731_synthetic_configurations_v5.csv

Outputs
-------
data/synthetic/engineer_preferences/validation/
    731_engineer_preference_validation_metrics_v1.csv
    731_engineer_preference_candidate_distribution_v1.csv
    731_engineer_preference_rank_distribution_v1.csv
    731_engineer_preference_pairwise_balance_v1.csv
    731_engineer_preference_package_summary_v1.csv
    731_engineer_preference_record_type_summary_v1.csv
    731_engineer_preference_feature_behavior_v1.csv
    731_engineer_preference_leakage_audit_v1.csv
    731_engineer_preference_provenance_audit_v1.csv
    731_engineer_preference_validation_summary_v1.txt

Provenance
----------
SYNTHETIC
Generation method:
synthetic_engineer_preference_validation_v1
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

PREF_DIR = ROOT / "data" / "synthetic" / "engineer_preferences"
RFQ_DIR = ROOT / "data" / "synthetic" / "rfqs"
CONFIG_DIR = ROOT / "data" / "synthetic" / "configurations"

OUTPUT_DIR = PREF_DIR / "validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


CANDIDATE_FILE = (
    PREF_DIR /
    "731_engineer_candidate_preferences_v1.csv"
)

RANKING_FILE = (
    PREF_DIR /
    "731_engineer_rankings_v1.csv"
)

PAIRWISE_FILE = (
    PREF_DIR /
    "731_engineer_pairwise_preferences_v1.csv"
)

GT_FILE = (
    RFQ_DIR /
    "731_rfq_ground_truth_v2.csv"
)

CONFIG_FILE = (
    CONFIG_DIR /
    "731_synthetic_configurations_v5.csv"
)


# =============================================================================
# CONSTANTS
# =============================================================================

EXPECTED_RFQ_COUNT = 5000

EXPECTED_PROVENANCE = "SYNTHETIC"
EXPECTED_GENERATION_METHOD = "synthetic_engineer_v1"


# =============================================================================
# HELPERS
# =============================================================================

def print_header(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_section(number, title):
    print()
    print(f"[{number}] {title}")
    print("-" * 80)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_bool_series(series):
    """Convert common CSV boolean representations to real booleans."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    normalized = series.map(normalize_text).str.lower()

    true_values = {"true", "1", "yes", "y", "t"}
    false_values = {"false", "0", "no", "n", "f", ""}

    result = pd.Series(np.nan, index=series.index, dtype="object")
    result[normalized.isin(true_values)] = True
    result[normalized.isin(false_values)] = False

    # Preserve unexpected non-null values as NaN so they can be audited
    # instead of silently turning every non-empty string into True.
    return result.astype("boolean")


def numeric(df, column):
    if column not in df.columns:
        return pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

    return pd.to_numeric(
        df[column],
        errors="coerce",
    )


def pct(value):
    if pd.isna(value):
        return "NA"

    return f"{value * 100:.2f}%"


def rate(condition):
    if len(condition) == 0:
        return np.nan

    return condition.mean()


def validate_schema(df, required, name):

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name} is missing required columns:\n"
            f"{missing}"
        )


# =============================================================================
# LOAD DATA
# =============================================================================

print_header(
    "731 V3 SYNTHETIC ENGINEER PREFERENCE VALIDATION"
)

print_section(
    1,
    "Loading files"
)

for path in [
    CANDIDATE_FILE,
    RANKING_FILE,
    PAIRWISE_FILE,
    GT_FILE,
    CONFIG_FILE,
]:
    require_file(path)

candidates = pd.read_csv(
    CANDIDATE_FILE
)

rankings = pd.read_csv(
    RANKING_FILE
)

pairwise = pd.read_csv(
    PAIRWISE_FILE
)

ground_truth = pd.read_csv(
    GT_FILE
)

configs = pd.read_csv(
    CONFIG_FILE
)

print(
    f"Candidate rows:       {len(candidates):,}"
)

print(
    f"Ranking rows:         {len(rankings):,}"
)

print(
    f"Pairwise rows:        {len(pairwise):,}"
)

print(
    f"Ground-truth rows:    {len(ground_truth):,}"
)

print(
    f"Configuration rows:   {len(configs):,}"
)


# =============================================================================
# SCHEMA VALIDATION
# =============================================================================

print_section(
    2,
    "Validating schemas"
)

candidate_required = [
    "rfq_id",
    "target_package",
    "canonical_configuration_id",
    "ground_truth_configuration_id",
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
]


ranking_required = [
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
]


pairwise_required = [
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
]


gt_required = [
    "rfq_id",
    "canonical_configuration_id",
    "target_package",
]


config_required = [
    "canonical_configuration_id",
    "package_context",
    "record_type",
    "source_frequency",
    "nearest_real_distance",
    "num_changed_characteristics",
    "evidence_score",
]


validate_schema(
    candidates,
    candidate_required,
    "Candidate preference file",
)

validate_schema(
    rankings,
    ranking_required,
    "Ranking file",
)

validate_schema(
    pairwise,
    pairwise_required,
    "Pairwise preference file",
)

validate_schema(
    ground_truth,
    gt_required,
    "Ground-truth file",
)

validate_schema(
    configs,
    config_required,
    "Configuration file",
)

print(
    "Schema validation passed."
)


# =============================================================================
# NORMALIZATION
# =============================================================================

print_section(
    3,
    "Normalizing identifiers and numeric fields"
)

id_columns = [
    "rfq_id",
    "target_package",
    "canonical_configuration_id",
    "ground_truth_configuration_id",
    "package_context",
]

for column in id_columns:

    if column in candidates.columns:
        candidates[column] = (
            candidates[column]
            .map(normalize_text)
        )

for column in [
    "rfq_id",
    "target_package",
    "synthetic_engineer_choice",
    "ground_truth_configuration_id",
]:

    if column in rankings.columns:
        rankings[column] = (
            rankings[column]
            .map(normalize_text)
        )

for column in [
    "rfq_id",
    "target_package",
    "preferred_configuration_id",
    "rejected_configuration_id",
]:

    if column in pairwise.columns:
        pairwise[column] = (
            pairwise[column]
            .map(normalize_text)
        )

for column in [
    "rfq_id",
    "canonical_configuration_id",
    "target_package",
]:

    if column in ground_truth.columns:
        ground_truth[column] = (
            ground_truth[column]
            .map(normalize_text)
        )

for column in [
    "canonical_configuration_id",
    "package_context",
    "record_type",
]:

    if column in configs.columns:
        configs[column] = (
            configs[column]
            .map(normalize_text)
        )


numeric_candidate_columns = [
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
]

for column in numeric_candidate_columns:

    if column in candidates.columns:
        candidates[column] = numeric(
            candidates,
            column,
        )


numeric_ranking_columns = [
    "candidate_count",
    "synthetic_engineer_top_score",
    "ground_truth_engineer_rank",
    "ground_truth_engineer_score",
]

for column in numeric_ranking_columns:

    if column in rankings.columns:
        rankings[column] = numeric(
            rankings,
            column,
        )


numeric_pairwise_columns = [
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
]

for column in numeric_pairwise_columns:

    if column in pairwise.columns:
        pairwise[column] = numeric(
            pairwise,
            column,
        )


# Normalize boolean-like fields explicitly. This avoids the common
# pandas pitfall where bool("False") evaluates to True.
for column in [
    "is_ground_truth_configuration",
    "is_synthetic_engineer_choice",
]:
    if column in candidates.columns:
        candidates[column] = normalize_bool_series(candidates[column])

if "ground_truth_selected_by_engineer" in rankings.columns:
    rankings["ground_truth_selected_by_engineer"] = normalize_bool_series(
        rankings["ground_truth_selected_by_engineer"]
    )

print(
    "Normalization complete."
)


# =============================================================================
# DATA INTEGRITY
# =============================================================================

print_section(
    4,
    "Checking data integrity"
)

integrity_rows = []


def add_integrity(
    metric_name,
    value,
    status,
    note="",
):

    integrity_rows.append(
        {
            "metric": metric_name,
            "value": value,
            "status": status,
            "note": note,
        }
    )


candidate_rfqs = candidates[
    "rfq_id"
].nunique()

ranking_rfqs = rankings[
    "rfq_id"
].nunique()

pairwise_rfqs = pairwise[
    "rfq_id"
].nunique()


add_integrity(
    "candidate_rfq_count",
    candidate_rfqs,
    "PASS"
    if candidate_rfqs == EXPECTED_RFQ_COUNT
    else "CHECK",
)


add_integrity(
    "ranking_rfq_count",
    ranking_rfqs,
    "PASS"
    if ranking_rfqs == EXPECTED_RFQ_COUNT
    else "CHECK",
)


add_integrity(
    "pairwise_rfq_count",
    pairwise_rfqs,
    "PASS"
    if 0 <= pairwise_rfqs <= candidate_rfqs
    else "FAIL",
    "Pairwise data need not contain every RFQ; RFQs with fewer than "
    "two candidates legitimately produce zero pairwise comparisons.",
)


pairwise_unknown_rfqs = len(
    set(pairwise["rfq_id"].dropna().unique())
    - set(candidates["rfq_id"].dropna().unique())
)

add_integrity(
    "pairwise_unknown_rfq_count",
    pairwise_unknown_rfqs,
    "PASS" if pairwise_unknown_rfqs == 0 else "FAIL",
    "Every pairwise RFQ should exist in the candidate-preference dataset.",
)


candidate_duplicates = candidates.duplicated(
    subset=[
        "rfq_id",
        "canonical_configuration_id",
    ]
).sum()


add_integrity(
    "duplicate_rfq_configuration_candidates",
    int(candidate_duplicates),
    "PASS"
    if candidate_duplicates == 0
    else "FAIL",
)


ranking_duplicates = rankings.duplicated(
    subset=["rfq_id"]
).sum()


add_integrity(
    "duplicate_rfq_rows",
    int(ranking_duplicates),
    "PASS"
    if ranking_duplicates == 0
    else "FAIL",
)


invalid_ranks = (
    candidates["engineer_rank"].isna()
    | (candidates["engineer_rank"] < 1)
)


add_integrity(
    "invalid_engineer_rank_rows",
    int(invalid_ranks.sum()),
    "PASS"
    if invalid_ranks.sum() == 0
    else "FAIL",
)


invalid_candidate_counts = (
    candidates["candidate_count"].isna()
    | (candidates["candidate_count"] < 1)
)


add_integrity(
    "invalid_candidate_count_rows",
    int(invalid_candidate_counts.sum()),
    "PASS"
    if invalid_candidate_counts.sum() == 0
    else "FAIL",
)


integrity_df = pd.DataFrame(
    integrity_rows
)

print(
    integrity_df.to_string(
        index=False
    )
)


# =============================================================================
# CANDIDATE DISTRIBUTION
# =============================================================================

print_section(
    5,
    "Analysing candidate distribution"
)

candidate_counts = (
    candidates[
        [
            "rfq_id",
            "candidate_count",
        ]
    ]
    .drop_duplicates("rfq_id")
    .copy()
)

candidate_distribution = pd.DataFrame(
    {
        "metric": [
            "rfq_count",
            "mean_candidates",
            "median_candidates",
            "min_candidates",
            "max_candidates",
            "p25_candidates",
            "p75_candidates",
            "p90_candidates",
            "p95_candidates",
        ],
        "value": [
            candidate_counts["rfq_id"].nunique(),
            candidate_counts["candidate_count"].mean(),
            candidate_counts["candidate_count"].median(),
            candidate_counts["candidate_count"].min(),
            candidate_counts["candidate_count"].max(),
            candidate_counts["candidate_count"].quantile(0.25),
            candidate_counts["candidate_count"].quantile(0.75),
            candidate_counts["candidate_count"].quantile(0.90),
            candidate_counts["candidate_count"].quantile(0.95),
        ],
    }
)


candidate_buckets = pd.cut(
    candidate_counts["candidate_count"],
    bins=[
        0,
        1,
        2,
        5,
        10,
        20,
        50,
        100,
        np.inf,
    ],
    labels=[
        "1",
        "2",
        "3-5",
        "6-10",
        "11-20",
        "21-50",
        "51-100",
        "101+",
    ],
)


candidate_bucket_summary = (
    candidate_counts
    .assign(
        candidate_bucket=candidate_buckets
    )
    .groupby(
        "candidate_bucket",
        observed=False,
    )
    .agg(
        rfq_count=(
            "rfq_id",
            "count",
        ),
        mean_candidates=(
            "candidate_count",
            "mean",
        ),
    )
    .reset_index()
)


candidate_bucket_summary["share"] = (
    candidate_bucket_summary["rfq_count"]
    / len(candidate_counts)
)


candidate_distribution.to_csv(
    OUTPUT_DIR /
    "731_engineer_preference_candidate_distribution_v1.csv",
    index=False,
)


candidate_bucket_summary.to_csv(
    OUTPUT_DIR /
    "731_engineer_preference_candidate_buckets_v1.csv",
    index=False,
)


print(
    f"Mean candidates/RFQ: "
    f"{candidate_counts['candidate_count'].mean():.2f}"
)

print(
    f"Median candidates/RFQ: "
    f"{candidate_counts['candidate_count'].median():.2f}"
)

print(
    f"Minimum candidates: "
    f"{int(candidate_counts['candidate_count'].min())}"
)

print(
    f"Maximum candidates: "
    f"{int(candidate_counts['candidate_count'].max())}"
)


# =============================================================================
# RANK CONSISTENCY
# =============================================================================

print_section(
    6,
    "Checking ranking consistency"
)

rank_check = (
    candidates
    .groupby("rfq_id")
    .agg(
        candidate_rows=(
            "canonical_configuration_id",
            "count",
        ),
        unique_configurations=(
            "canonical_configuration_id",
            "nunique",
        ),
        candidate_count_reported=(
            "candidate_count",
            "first",
        ),
        minimum_rank=(
            "engineer_rank",
            "min",
        ),
        maximum_rank=(
            "engineer_rank",
            "max",
        ),
        rank_unique_count=(
            "engineer_rank",
            "nunique",
        ),
        choice_count=(
            "is_synthetic_engineer_choice",
            "sum",
        ),
    )
    .reset_index()
)


rank_check["candidate_count_matches"] = (
    rank_check["candidate_rows"]
    == rank_check["candidate_count_reported"]
)

rank_check["candidate_unique_id_count_matches"] = (
    rank_check["unique_configurations"]
    == rank_check["candidate_count_reported"]
)


rank_check["rank_sequence_valid"] = (
    rank_check["minimum_rank"] == 1
) & (
    rank_check["maximum_rank"]
    == rank_check["candidate_count_reported"]
) & (
    rank_check["rank_unique_count"]
    == rank_check["candidate_count_reported"]
)


rank_check["single_engineer_choice"] = (
    rank_check["choice_count"] == 1
)


print(
    "Candidate-count mismatches:",
    int(
        (~rank_check["candidate_count_matches"])
        .sum()
    ),
)

print(
    "Candidate unique-ID mismatches:",
    int(
        (~rank_check["candidate_unique_id_count_matches"])
        .sum()
    ),
)

print(
    "Invalid rank sequences:",
    int(
        (~rank_check["rank_sequence_valid"])
        .sum()
    ),
)

print(
    "RFQs without exactly one engineer choice:",
    int(
        (~rank_check["single_engineer_choice"])
        .sum()
    ),
)


# =============================================================================
# GROUND-TRUTH RANK ANALYSIS
# =============================================================================

print_section(
    7,
    "Analysing ground-truth engineer ranking"
)

gt_rank = (
    rankings[
        [
            "rfq_id",
            "ground_truth_configuration_id",
            "ground_truth_engineer_rank",
            "ground_truth_engineer_score",
            "ground_truth_selected_by_engineer",
            "candidate_count",
            "target_package",
        ]
    ]
    .drop_duplicates("rfq_id")
    .copy()
)


missing_gt_rank = (
    gt_rank[
        "ground_truth_engineer_rank"
    ]
    .isna()
    .sum()
)


gt_top1 = rate(
    gt_rank["ground_truth_engineer_rank"] == 1
)

gt_top3 = rate(
    gt_rank["ground_truth_engineer_rank"] <= 3
)

gt_top5 = rate(
    gt_rank["ground_truth_engineer_rank"] <= 5
)

gt_top10 = rate(
    gt_rank["ground_truth_engineer_rank"] <= 10
)


mean_gt_rank = gt_rank[
    "ground_truth_engineer_rank"
].mean()


median_gt_rank = gt_rank[
    "ground_truth_engineer_rank"
].median()


print(
    f"RFQs with GT rank: "
    f"{gt_rank['ground_truth_engineer_rank'].notna().sum():,}"
)

print(
    f"RFQs missing GT rank: "
    f"{int(missing_gt_rank):,}"
)

print(
    f"Ground-truth Top-1: "
    f"{pct(gt_top1)}"
)

print(
    f"Ground-truth Top-3: "
    f"{pct(gt_top3)}"
)

print(
    f"Ground-truth Top-5: "
    f"{pct(gt_top5)}"
)

print(
    f"Ground-truth Top-10: "
    f"{pct(gt_top10)}"
)

print(
    f"Ground-truth mean rank: "
    f"{mean_gt_rank:.2f}"
)

print(
    f"Ground-truth median rank: "
    f"{median_gt_rank:.2f}"
)


rank_distribution = (
    gt_rank[
        "ground_truth_engineer_rank"
    ]
    .value_counts()
    .sort_index()
    .rename_axis(
        "ground_truth_engineer_rank"
    )
    .reset_index(
        name="rfq_count"
    )
)


rank_distribution["share"] = (
    rank_distribution["rfq_count"]
    / len(gt_rank)
)


rank_distribution.to_csv(
    OUTPUT_DIR /
    "731_engineer_preference_rank_distribution_v1.csv",
    index=False,
)


# =============================================================================
# PAIRWISE CONSISTENCY
# =============================================================================

print_section(
    8,
    "Analysing pairwise preference consistency"
)

pairwise_rows = []

pairwise_rows.append(
    {
        "metric": "pairwise_rows",
        "value": len(pairwise),
    }
)

pairwise_rows.append(
    {
        "metric": "unique_rfqs",
        "value": pairwise["rfq_id"].nunique(),
    }
)


if len(pairwise) > 0:

    rank_consistent = (
        pairwise["preferred_rank"]
        < pairwise["rejected_rank"]
    )

    score_consistent = (
        pairwise["preferred_score"]
        >= pairwise["rejected_score"]
    )

    positive_label_rate = rate(
        pairwise["label"] == 1
    )

    pairwise_rows.append(
        {
            "metric": "preferred_rank_is_better_rate",
            "value": rank_consistent.mean(),
        }
    )

    pairwise_rows.append(
        {
            "metric": "preferred_score_ge_rejected_score_rate",
            "value": score_consistent.mean(),
        }
    )

    pairwise_rows.append(
        {
            "metric": "positive_label_rate",
            "value": positive_label_rate,
        }
    )

    pairwise_rows.append(
        {
            "metric": "mean_score_margin",
            "value": pairwise["score_margin"].mean(),
        }
    )

    pairwise_rows.append(
        {
            "metric": "median_score_margin",
            "value": pairwise["score_margin"].median(),
        }


    )

    pairwise_rows.append(
        {
            "metric": "zero_score_margin_rate",
            "value": rate(
                pairwise["score_margin"] == 0
            ),
        }
    )


pairwise_balance = pd.DataFrame(
    pairwise_rows
)


pairwise_balance.to_csv(
    OUTPUT_DIR /
    "731_engineer_preference_pairwise_balance_v1.csv",
    index=False,
)


print(
    f"Pairwise rows: "
    f"{len(pairwise):,}"
)

print(
    f"Pairwise RFQs: "
    f"{pairwise['rfq_id'].nunique():,}"
)


if len(pairwise) > 0:

    print(
        "Preferred rank better rate: "
        f"{pct(rank_consistent.mean())}"
    )

    print(
        "Preferred score >= rejected score: "
        f"{pct(score_consistent.mean())}"
    )

    print(
        "Positive-label rate: "
        f"{pct(positive_label_rate)}"
    )

    print(
        "Mean score margin: "
        f"{pairwise['score_margin'].mean():.6f}"
    )

    print(
        "Median score margin: "
        f"{pairwise['score_margin'].median():.6f}"
    )


# =============================================================================
# PACKAGE ANALYSIS
# =============================================================================

print_section(
    9,
    "Analysing package behaviour"
)

package_summary = (
    gt_rank
    .groupby("target_package")
    .agg(
        rfq_count=(
            "rfq_id",
            "count",
        ),
        top1_rate=(
            "ground_truth_engineer_rank",
            lambda x: (x == 1).mean(),
        ),
        top3_rate=(
            "ground_truth_engineer_rank",
            lambda x: (x <= 3).mean(),
        ),
        top5_rate=(
            "ground_truth_engineer_rank",
            lambda x: (x <= 5).mean(),
        ),
        top10_rate=(
            "ground_truth_engineer_rank",
            lambda x: (x <= 10).mean(),
        ),
        mean_gt_rank=(
            "ground_truth_engineer_rank",
            "mean",
        ),
        median_gt_rank=(
            "ground_truth_engineer_rank",
            "median",
        ),
        mean_candidate_count=(
            "candidate_count",
            "mean",
        ),
        median_candidate_count=(
            "candidate_count",
            "median",
        ),
    )
    .reset_index()
)


package_summary.to_csv(
    OUTPUT_DIR /
    "731_engineer_preference_package_summary_v1.csv",
    index=False,
)


print(
    package_summary.to_string(
        index=False
    )
)


# =============================================================================
# RECORD TYPE ANALYSIS
# =============================================================================

print_section(
    10,
    "Analysing observed vs derived configurations"
)

record_summary = (
    candidates
    .groupby("record_type")
    .agg(
        candidate_rows=(
            "canonical_configuration_id",
            "count",
        ),
        unique_rfqs=(
            "rfq_id",
            "nunique",
        ),
        mean_utility=(
            "synthetic_engineer_utility",
            "mean",
        ),
        median_utility=(
            "synthetic_engineer_utility",
            "median",
        ),
        mean_score=(
            "synthetic_engineer_score",
            "mean",
        ),
        mean_rank=(
            "engineer_rank",
            "mean",
        ),
        median_rank=(
            "engineer_rank",
            "median",
        ),
        rank1_rate=(
            "engineer_rank",
            lambda x: (x == 1).mean(),
        ),
    )
    .reset_index()
)


record_summary.to_csv(
    OUTPUT_DIR /
    "731_engineer_preference_record_type_summary_v1.csv",
    index=False,
)


print(
    record_summary.to_string(
        index=False
    )
)


# =============================================================================
# FEATURE BEHAVIOUR
# =============================================================================

print_section(
    11,
    "Analysing preference-feature behaviour"
)

feature_columns = [
    "unmentioned_features",
    "source_frequency_score",
    "real_proximity_score",
    "low_change_score",
    "observed_configuration_score",
    "evidence_score_normalized",
    "synthetic_engineer_utility",
    "synthetic_engineer_score",
    "engineer_rank",
]


feature_rows = []


for feature in feature_columns:

    if feature not in candidates.columns:
        continue

    values = numeric(
        candidates,
        feature,
    )

    feature_rows.append(
        {
            "feature": feature,
            "count": int(values.notna().sum()),
            "mean": values.mean(),
            "median": values.median(),
            "std": values.std(),
            "min": values.min(),
            "max": values.max(),
        }
    )


feature_behavior = pd.DataFrame(
    feature_rows
)


# Correlations with synthetic utility/rank

correlation_rows = []


for feature in [
    "unmentioned_features",
    "source_frequency_score",
    "real_proximity_score",
    "low_change_score",
    "observed_configuration_score",
    "evidence_score_normalized",
]:

    if feature not in candidates.columns:
        continue

    temp = candidates[
        [
            feature,
            "synthetic_engineer_utility",
            "synthetic_engineer_score",
            "engineer_rank",
        ]
    ].copy()

    temp = temp.apply(
        pd.to_numeric,
        errors="coerce",
    ).dropna()

    if len(temp) < 3:
        continue

    correlation_rows.append(
        {
            "feature": feature,
            "utility_pearson": temp[
                feature
            ].corr(
                temp[
                    "synthetic_engineer_utility"
                ]
            ),
            "score_pearson": temp[
                feature
            ].corr(
                temp[
                    "synthetic_engineer_score"
                ]
            ),
            "rank_spearman": temp[
                feature
            ].rank().corr(
                temp["engineer_rank"].rank(),
            )
        }
    )


if correlation_rows:

    feature_behavior = feature_behavior.merge(
        pd.DataFrame(
            correlation_rows
        ),
        on="feature",
        how="left",
    )


feature_behavior.to_csv(
    OUTPUT_DIR /
    "731_engineer_preference_feature_behavior_v1.csv",
    index=False,
)


print(
    feature_behavior.to_string(
        index=False
    )
)


# =============================================================================
# ENGINEER CHOICE ANALYSIS
# =============================================================================

print_section(
    12,
    "Analysing synthetic engineer choices"
)

choice_rows = candidates[
    candidates[
        "is_synthetic_engineer_choice"
    ].fillna(False).astype(bool)
].copy()


choice_count = (
    choice_rows[
        "rfq_id"
    ].nunique()
)


print(
    f"RFQs with synthetic engineer choice: "
    f"{choice_count:,}"
)


print(
    f"Expected RFQs: "
    f"{EXPECTED_RFQ_COUNT:,}"
)


choice_record_summary = (
    choice_rows
    .groupby("record_type")
    .agg(
        choice_count=(
            "rfq_id",
            "count",
        ),
        mean_utility=(
            "synthetic_engineer_utility",
            "mean",
        ),
        mean_unmentioned=(
            "unmentioned_features",
            "mean",
        ),
        mean_real_distance=(
            "nearest_real_distance",
            "mean",
        ),
        mean_changes=(
            "num_changed_characteristics",
            "mean",
        ),
    )
    .reset_index()
)


print(
    choice_record_summary.to_string(
        index=False
    )
)


# =============================================================================
# GROUND-TRUTH LEAKAGE AUDIT
# =============================================================================

print_section(
    13,
    "Running ground-truth leakage audit"
)

leakage_rows = []


def leakage(
    check,
    status,
    details,
):

    leakage_rows.append(
        {
            "check": check,
            "status": status,
            "details": details,
        }
    )


# -------------------------------------------------------------------------
# Check 1: GT ID is present because it is intentionally retained as
# evaluation metadata. It must NOT be used as a feature.
# -------------------------------------------------------------------------

if (
    "ground_truth_configuration_id"
    in candidates.columns
):

    leakage(
        "ground_truth_id_present",
        "CHECK",
        (
            "Ground-truth ID is retained in candidate output "
            "for evaluation. It must be excluded from ML features."
        ),
    )

else:

    leakage(
        "ground_truth_id_present",
        "PASS",
        "Ground-truth ID is not present.",
    )


# -------------------------------------------------------------------------
# Check 2: engineer utility should not explicitly reference GT.
# -------------------------------------------------------------------------

if (
    "is_ground_truth_configuration"
    in candidates.columns
):

    gt_flag_count = candidates[
        "is_ground_truth_configuration"
    ].astype(bool).sum()

    leakage(
        "ground_truth_flag_present",
        "CHECK",
        (
            f"{gt_flag_count:,} candidate rows are explicitly marked "
            "as GT for evaluation. Exclude this column from ML features."
        ),
    )

else:

    leakage(
        "ground_truth_flag_present",
        "PASS",
        "No GT flag found.",
    )


# -------------------------------------------------------------------------
# Check 3: Check whether GT is always engineer choice.
# -------------------------------------------------------------------------

gt_always_choice = (
    gt_rank[
        "ground_truth_selected_by_engineer"
    ]
    .fillna(False).astype(bool)
    .mean()
)


if gt_always_choice >= 0.999:

    leakage(
        "gt_forced_as_engineer_choice",
        "CHECK",
        (
            "Ground truth is selected almost always. "
            "This may indicate label leakage or an overly strong rule."
        ),
    )

else:

    leakage(
        "gt_forced_as_engineer_choice",
        "PASS",
        (
            f"Ground truth selected by synthetic engineer in "
            f"{pct(gt_always_choice)} of RFQs."
        ),
    )


# -------------------------------------------------------------------------
# Check 4: Check whether synthetic engineer choice is literally
# the GT configuration.
# -------------------------------------------------------------------------

choice_merge = rankings.merge(
    gt_rank[
        [
            "rfq_id",
            "ground_truth_configuration_id",
        ]
    ],
    on="rfq_id",
    how="left",
    suffixes=(
        "",
        "_gt_check",
    ),
)


choice_is_gt = (
    choice_merge[
        "synthetic_engineer_choice"
    ]
    ==
    choice_merge[
        "ground_truth_configuration_id_gt_check"
    ]
)


choice_gt_rate = choice_is_gt.mean()


leakage(
    "engineer_choice_equals_ground_truth",
    "CHECK"
    if choice_gt_rate > 0.90
    else "PASS",
    (
        f"Synthetic engineer choice equals GT in "
        f"{pct(choice_gt_rate)} of RFQs."
    ),
)


# -------------------------------------------------------------------------
# Check 5: Candidate utility has no direct GT-derived feature.
# -------------------------------------------------------------------------

gt_related_candidate_columns = [
    column
    for column in candidates.columns
    if (
        "ground_truth"
        in column.lower()
        or "true_configuration"
        in column.lower()
    )
]


leakage(
    "gt_related_candidate_columns",
    "CHECK"
    if gt_related_candidate_columns
    else "PASS",
    str(gt_related_candidate_columns),
)


leakage_audit = pd.DataFrame(
    leakage_rows
)


leakage_audit.to_csv(
    OUTPUT_DIR /
    "731_engineer_preference_leakage_audit_v1.csv",
    index=False,
)


print(
    leakage_audit.to_string(
        index=False
    )
)


# =============================================================================
# PROVENANCE AUDIT
# =============================================================================

print_section(
    14,
    "Checking synthetic provenance"
)

provenance_rows = []


for column, expected in [
    (
        "preference_label_provenance",
        EXPECTED_PROVENANCE,
    ),
    (
        "preference_generation_method",
        EXPECTED_GENERATION_METHOD,
    ),
]:

    if column not in candidates.columns:

        provenance_rows.append(
            {
                "dataset": "candidates",
                "column": column,
                "status": "MISSING",
                "unique_values": "",
            }
        )

        continue

    values = sorted(
        {
            normalize_text(value)
            for value in candidates[column]
            if normalize_text(value)
        }
    )

    status = (
        "PASS"
        if values == [expected]
        else "CHECK"
    )

    provenance_rows.append(
        {
            "dataset": "candidates",
            "column": column,
            "status": status,
            "unique_values": "|".join(values),
        }
    )


for column, expected in [
    (
        "preference_label_provenance",
        EXPECTED_PROVENANCE,
    ),
    (
        "preference_generation_method",
        EXPECTED_GENERATION_METHOD,
    ),
]:

    if column not in rankings.columns:

        provenance_rows.append(
            {
                "dataset": "rankings",
                "column": column,
                "status": "MISSING",
                "unique_values": "",
            }
        )

        continue

    values = sorted(
        {
            normalize_text(value)
            for value in rankings[column]
            if normalize_text(value)
        }
    )

    status = (
        "PASS"
        if values == [expected]
        else "CHECK"
    )

    provenance_rows.append(
        {
            "dataset": "rankings",
            "column": column,
            "status": status,
            "unique_values": "|".join(values),
        }
    )


for column, expected in [
    (
        "preference_label_provenance",
        EXPECTED_PROVENANCE,
    ),
    (
        "preference_generation_method",
        EXPECTED_GENERATION_METHOD,
    ),
]:

    if column not in pairwise.columns:

        provenance_rows.append(
            {
                "dataset": "pairwise",
                "column": column,
                "status": "MISSING",
                "unique_values": "",
            }
        )

        continue

    values = sorted(
        {
            normalize_text(value)
            for value in pairwise[column]
            if normalize_text(value)
        }
    )

    status = (
        "PASS"
        if values == [expected]
        else "CHECK"
    )

    provenance_rows.append(
        {
            "dataset": "pairwise",
            "column": column,
            "status": status,
            "unique_values": "|".join(values),
        }
    )


provenance_audit = pd.DataFrame(
    provenance_rows
)


provenance_audit.to_csv(
    OUTPUT_DIR /
    "731_engineer_preference_provenance_audit_v1.csv",
    index=False,
)


print(
    provenance_audit.to_string(
        index=False
    )
)


# =============================================================================
# OVERALL METRICS
# =============================================================================

print_section(
    15,
    "Building validation metrics"
)

metrics = []


def add_metric(
    name,
    value,
    interpretation,
):

    metrics.append(
        {
            "metric": name,
            "value": value,
            "interpretation": interpretation,
        }
    )


add_metric(
    "rfq_count",
    candidates["rfq_id"].nunique(),
    "Number of RFQs represented.",
)


add_metric(
    "candidate_rows",
    len(candidates),
    "Total candidate-preference rows.",
)


add_metric(
    "ranking_rows",
    len(rankings),
    "One synthetic engineer ranking summary per RFQ.",
)


add_metric(
    "pairwise_rows",
    len(pairwise),
    "Synthetic pairwise preference examples.",
)


add_metric(
    "mean_candidates_per_rfq",
    candidate_counts["candidate_count"].mean(),
    "Average valid candidate pool size.",
)


add_metric(
    "median_candidates_per_rfq",
    candidate_counts["candidate_count"].median(),
    "Median valid candidate pool size.",
)


add_metric(
    "gt_top1_rate",
    gt_top1,
    "GT configuration ranked first by synthetic engineer.",
)


add_metric(
    "gt_top3_rate",
    gt_top3,
    "GT configuration ranked in Top-3.",
)


add_metric(
    "gt_top5_rate",
    gt_top5,
    "GT configuration ranked in Top-5.",
)


add_metric(
    "gt_top10_rate",
    gt_top10,
    "GT configuration ranked in Top-10.",
)


add_metric(
    "gt_mean_rank",
    mean_gt_rank,
    "Mean synthetic engineer rank of GT.",
)


add_metric(
    "gt_median_rank",
    median_gt_rank,
    "Median synthetic engineer rank of GT.",
)


add_metric(
    "missing_gt_rank_count",
    int(missing_gt_rank),
    "RFQs where GT has no engineer rank.",
)


add_metric(
    "candidate_duplicate_count",
    int(candidate_duplicates),
    "Should be zero.",
)


add_metric(
    "ranking_duplicate_count",
    int(ranking_duplicates),
    "Should be zero.",
)


add_metric(
    "invalid_rank_count",
    int(invalid_ranks.sum()),
    "Should be zero.",
)


add_metric(
    "invalid_candidate_count_rows",
    int(invalid_candidate_counts.sum()),
    "Should be zero.",
)


add_metric(
    "rank_sequence_invalid_rfqs",
    int(
        (~rank_check["rank_sequence_valid"])
        .sum()
    ),
    "Should be zero.",
)


add_metric(
    "candidate_unique_id_mismatch_rfqs",
    int(
        (~rank_check["candidate_unique_id_count_matches"])
        .sum()
    ),
    "Should be zero when candidate_count represents unique configurations.",
)


add_metric(
    "rfqs_without_single_engineer_choice",
    int(
        (~rank_check["single_engineer_choice"])
        .sum()
    ),
    "Should be zero.",
)


add_metric(
    "engineer_choice_equals_gt_rate",
    choice_gt_rate,
    "Rate at which synthetic engineer chooses GT exactly.",
)


add_metric(
    "gt_selected_by_engineer_rate",
    gt_always_choice,
    "Rate at which GT is selected by synthetic engineer.",
)


if len(pairwise) > 0:

    add_metric(
        "pairwise_rank_consistency",
        rank_consistent.mean(),
        "Preferred configuration should have better rank.",
    )

    add_metric(
        "pairwise_score_consistency",
        score_consistent.mean(),
        "Preferred score should be >= rejected score.",
    )

    add_metric(
        "pairwise_positive_label_rate",
        positive_label_rate,
        "Share of pairwise labels equal to 1.",
    )

    add_metric(
        "pairwise_zero_margin_rate",
        (
            pairwise["score_margin"] == 0
        ).mean(),
        "Rate of exact score ties.",
    )


metrics_df = pd.DataFrame(
    metrics
)


metrics_df.to_csv(
    OUTPUT_DIR /
    "731_engineer_preference_validation_metrics_v1.csv",
    index=False,
)


# =============================================================================
# TEXT SUMMARY
# =============================================================================

summary_lines = []

summary_lines.append(
    "731 V3 SYNTHETIC ENGINEER PREFERENCE VALIDATION"
)

summary_lines.append(
    "=" * 70
)

summary_lines.append("")

summary_lines.append(
    f"RFQs: {candidates['rfq_id'].nunique():,}"
)

summary_lines.append(
    f"Candidate rows: {len(candidates):,}"
)

summary_lines.append(
    f"Ranking rows: {len(rankings):,}"
)

summary_lines.append(
    f"Pairwise rows: {len(pairwise):,}"
)

summary_lines.append("")

summary_lines.append(
    "CANDIDATE POOL"
)

summary_lines.append(
    f"Mean candidates/RFQ: "
    f"{candidate_counts['candidate_count'].mean():.2f}"
)

summary_lines.append(
    f"Median candidates/RFQ: "
    f"{candidate_counts['candidate_count'].median():.2f}"
)

summary_lines.append(
    f"Min candidates/RFQ: "
    f"{int(candidate_counts['candidate_count'].min())}"
)

summary_lines.append(
    f"Max candidates/RFQ: "
    f"{int(candidate_counts['candidate_count'].max())}"
)

summary_lines.append("")

summary_lines.append(
    "GROUND-TRUTH ENGINEER RANK"
)

summary_lines.append(
    f"Top-1: {pct(gt_top1)}"
)

summary_lines.append(
    f"Top-3: {pct(gt_top3)}"
)

summary_lines.append(
    f"Top-5: {pct(gt_top5)}"
)

summary_lines.append(
    f"Top-10: {pct(gt_top10)}"
)

summary_lines.append(
    f"Mean rank: {mean_gt_rank:.2f}"
)

summary_lines.append(
    f"Median rank: {median_gt_rank:.2f}"
)

summary_lines.append("")

summary_lines.append(
    "DATA INTEGRITY"
)

summary_lines.append(
    f"Duplicate candidate pairs: "
    f"{candidate_duplicates:,}"
)

summary_lines.append(
    f"Duplicate RFQ ranking rows: "
    f"{ranking_duplicates:,}"
)

summary_lines.append(
    f"Invalid ranks: "
    f"{int(invalid_ranks.sum()):,}"
)

summary_lines.append(
    f"Invalid candidate-count rows: "
    f"{int(invalid_candidate_counts.sum()):,}"
)

summary_lines.append(
    f"Invalid rank sequences: "
    f"{int((~rank_check['rank_sequence_valid']).sum()):,}"
)

summary_lines.append(
    f"Candidate unique-ID mismatches: "
    f"{int((~rank_check['candidate_unique_id_count_matches']).sum()):,}"
)

summary_lines.append(
    f"RFQs without one engineer choice: "
    f"{int((~rank_check['single_engineer_choice']).sum()):,}"
)

summary_lines.append("")

summary_lines.append(
    "LEAKAGE CHECK"
)

summary_lines.append(
    f"GT selected by engineer: "
    f"{pct(gt_always_choice)}"
)

summary_lines.append(
    f"Engineer choice equals GT: "
    f"{pct(choice_gt_rate)}"
)

summary_lines.append("")

summary_lines.append(
    "PROVENANCE"
)

summary_lines.append(
    "Preference labels: SYNTHETIC"
)

summary_lines.append(
    "Generation method: synthetic_engineer_v1"
)

summary_lines.append("")

summary_lines.append(
    "IMPORTANT INTERPRETATION"
)

summary_lines.append(
    "These are synthetic engineer-preference labels."
)

summary_lines.append(
    "They are not historical engineer decisions."
)

summary_lines.append(
    "They should be used as a reproducible baseline for "
    "ranking-model development and evaluation."
)

summary_lines.append(
    "Ground-truth configuration identity should not be used "
    "as an ML feature."
)


summary_file = (
    OUTPUT_DIR /
    "731_engineer_preference_validation_summary_v1.txt"
)


summary_file.write_text(
    "\n".join(summary_lines),
    encoding="utf-8",
)


# =============================================================================
# FINAL OUTPUT
# =============================================================================

print_header(
    "VALIDATION COMPLETE"
)

print(
    f"RFQs:                         "
    f"{candidates['rfq_id'].nunique():,}"
)

print(
    f"Candidate rows:               "
    f"{len(candidates):,}"
)

print(
    f"Ranking rows:                 "
    f"{len(rankings):,}"
)

print(
    f"Pairwise rows:                "
    f"{len(pairwise):,}"
)

print()

print(
    f"Mean candidates/RFQ:          "
    f"{candidate_counts['candidate_count'].mean():.2f}"
)

print(
    f"Median candidates/RFQ:        "
    f"{candidate_counts['candidate_count'].median():.2f}"
)

print()

print(
    f"GT engineer Top-1:            "
    f"{pct(gt_top1)}"
)

print(
    f"GT engineer Top-3:            "
    f"{pct(gt_top3)}"
)

print(
    f"GT engineer Top-5:            "
    f"{pct(gt_top5)}"
)

print(
    f"GT engineer Top-10:           "
    f"{pct(gt_top10)}"
)

print(
    f"GT engineer mean rank:        "
    f"{mean_gt_rank:.2f}"
)

print(
    f"GT engineer median rank:      "
    f"{median_gt_rank:.2f}"
)

print()

print(
    f"Engineer choice = GT:         "
    f"{pct(choice_gt_rate)}"
)

print(
    f"GT selected by engineer:      "
    f"{pct(gt_always_choice)}"
)

print()

print(
    f"Duplicate candidate pairs:    "
    f"{candidate_duplicates:,}"
)

print(
    f"Duplicate ranking rows:       "
    f"{ranking_duplicates:,}"
)

print(
    f"Invalid ranks:                "
    f"{int(invalid_ranks.sum()):,}"
)

print(
    f"Invalid rank sequences:       "
    f"{int((~rank_check['rank_sequence_valid']).sum()):,}"
)

print(
    f"RFQs without one choice:      "
    f"{int((~rank_check['single_engineer_choice']).sum()):,}"
)

print()

print(
    "Validation outputs:"
)

print(
    OUTPUT_DIR
)

print()

print(
    "Validation complete."
)