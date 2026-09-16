#!/usr/bin/env python3

"""
731 V3 - Synthetic Engineer Preference Generation

Purpose
-------
Generate synthetic engineer preferences among technically valid
731-series configurations.

IMPORTANT METHODOLOGICAL PRINCIPLES
-----------------------------------
1. Only package-compatible configurations are considered.
2. Mandatory requirements must be satisfied.
3. Preferred requirements must also be satisfied for the primary
   preference pool.
4. The synthetic engineer does NOT simply select the ground-truth
   configuration.
5. Ground-truth configuration IDs are retained only for later
   evaluation and are NOT used to calculate preference scores.
6. All generated preference labels are explicitly SYNTHETIC.

Pipeline
--------
RFQ requirements
    |
    v
Package-constrained configuration pool
    |
    v
Mandatory + preferred requirement validation
    |
    v
Engineering preference features
    |
    v
Synthetic engineer utility
    |
    +--> ranked candidates
    +--> selected candidate
    +--> pairwise preferences

Outputs
-------
data/synthetic/engineer_preferences/
    731_engineer_candidate_preferences_v1.csv
    731_engineer_rankings_v1.csv
    731_engineer_pairwise_preferences_v1.csv
    731_engineer_preference_metrics_v1.csv
    731_engineer_preference_feature_summary_v1.csv
"""

from pathlib import Path
import hashlib
import math

import numpy as np
import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = (
    ROOT
    / "data"
    / "synthetic"
    / "configurations"
    / "731_synthetic_configurations_v5.csv"
)

GT_PATH = (
    ROOT
    / "data"
    / "synthetic"
    / "rfqs"
    / "731_rfq_ground_truth_v2.csv"
)

REQ_PATH = (
    ROOT
    / "data"
    / "synthetic"
    / "rfqs"
    / "731_rfq_requirements_v2.csv"
)

OUT_DIR = (
    ROOT
    / "data"
    / "synthetic"
    / "engineer_preferences"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CONFIGURATION
# ============================================================================

TECHNICAL_CHARACTERISTICS = [
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

REQUIRED_CONFIG_COLUMNS = [
    "canonical_configuration_id",
    "package_context",
    "record_type",
    "generation_method",
    "source_frequency",
    "nearest_real_distance",
    "num_changed_characteristics",
    "constraint_valid",
    "evidence_level",
    "evidence_score",
] + TECHNICAL_CHARACTERISTICS

REQUIRED_GT_COLUMNS = [
    "rfq_id",
    "canonical_configuration_id",
    "target_package",
    "record_type",
    "evidence_level",
]

REQUIRED_REQ_COLUMNS = [
    "rfq_id",
    "characteristic",
    "internal_value",
    "requirement_type",
]


# ============================================================================
# HELPERS
# ============================================================================

def normalize_text(value):
    """Normalize scalar values for deterministic comparisons."""
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    if text == "":
        return None

    return text.upper()


def normalize_package(value):
    return normalize_text(value)


def normalize_requirement_type(value):
    """
    Normalize requirement labels.

    MANDATORY:
        MANDATORY, REQUIRED, MUST

    PREFERRED:
        PREFERRED, PREFERENCE, PREFER
    """
    value = normalize_text(value)

    if value in {"MANDATORY", "REQUIRED", "MUST"}:
        return "MANDATORY"

    if value in {"PREFERRED", "PREFERENCE", "PREFER"}:
        return "PREFERRED"

    return value


def deterministic_jitter(*values):
    """
    Small deterministic tie-breaking value.

    This is NOT a meaningful engineering preference.
    It only prevents identical utility scores from creating
    arbitrary unstable ranking ties.
    """
    text = "||".join("" if v is None else str(v) for v in values)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    integer = int(digest[:12], 16)

    return (integer % 1_000_000) / 1_000_000_000_000


def safe_numeric(series, default=0.0):
    result = pd.to_numeric(series, errors="coerce")
    return result.fillna(default)


def minmax_normalize(series, reverse=False):
    """
    Normalize a numeric series to [0, 1].

    If reverse=True, lower original values receive higher scores.
    """
    values = pd.to_numeric(series, errors="coerce")

    if values.notna().sum() == 0:
        result = pd.Series(0.5, index=series.index, dtype=float)
        return result

    median = values.median()
    values = values.fillna(median)

    min_value = values.min()
    max_value = values.max()

    if max_value == min_value:
        result = pd.Series(0.5, index=series.index, dtype=float)
    else:
        result = (values - min_value) / (max_value - min_value)

    if reverse:
        result = 1.0 - result

    return result


def build_requirement_lookup(requirements):
    """
    Build:

        rfq_id -> characteristic -> requirement information

    Multiple requirements for the same characteristic are retained.
    """
    lookup = {}

    for rfq_id, group in requirements.groupby("rfq_id", sort=False):
        characteristic_lookup = {}

        for characteristic, char_group in group.groupby(
            "characteristic",
            sort=False
        ):
            entries = []

            for _, row in char_group.iterrows():
                value = normalize_text(row["internal_value"])
                req_type = normalize_requirement_type(
                    row["requirement_type"]
                )

                if value is None:
                    continue

                entries.append(
                    {
                        "value": value,
                        "requirement_type": req_type,
                    }
                )

            characteristic_lookup[characteristic] = entries

        lookup[rfq_id] = characteristic_lookup

    return lookup


def candidate_satisfies_requirement(
    candidate_value,
    required_value,
):
    """
    Exact normalized-value satisfaction.

    This follows the same strict requirement-equivalence principle
    used in the previous V2 analysis.
    """
    candidate_value = normalize_text(candidate_value)
    required_value = normalize_text(required_value)

    if candidate_value is None or required_value is None:
        return False

    return candidate_value == required_value


def evaluate_candidate(
    candidate,
    rfq_requirements,
):
    """
    Evaluate one candidate against the RFQ.

    Returns requirement-level statistics and the number of
    configured features not explicitly mentioned by the RFQ.
    """

    mandatory_total = 0
    mandatory_satisfied = 0

    preferred_total = 0
    preferred_satisfied = 0

    mentioned_characteristics = set()

    for characteristic, entries in rfq_requirements.items():

        if characteristic not in TECHNICAL_CHARACTERISTICS:
            continue

        if not entries:
            continue

        mentioned_characteristics.add(characteristic)

        for requirement in entries:

            required_value = requirement["value"]
            requirement_type = requirement["requirement_type"]

            candidate_value = candidate.get(characteristic)

            satisfied = candidate_satisfies_requirement(
                candidate_value,
                required_value,
            )

            if requirement_type == "MANDATORY":
                mandatory_total += 1

                if satisfied:
                    mandatory_satisfied += 1

            elif requirement_type == "PREFERRED":
                preferred_total += 1

                if satisfied:
                    preferred_satisfied += 1

    mandatory_violations = (
        mandatory_total - mandatory_satisfied
    )

    preferred_differences = (
        preferred_total - preferred_satisfied
    )

    # A feature is "unmentioned" if the RFQ contains no requirement
    # for it and the configuration nevertheless has a concrete value.
    unmentioned_features = 0

    for characteristic in TECHNICAL_CHARACTERISTICS:

        if characteristic in mentioned_characteristics:
            continue

        value = normalize_text(candidate.get(characteristic))

        if value is not None:
            unmentioned_features += 1

    mandatory_valid = (
        mandatory_violations == 0
    )

    mandatory_preferred_valid = (
        mandatory_violations == 0
        and preferred_differences == 0
    )

    return {
        "mandatory_total": mandatory_total,
        "mandatory_satisfied": mandatory_satisfied,
        "mandatory_violations": mandatory_violations,
        "preferred_total": preferred_total,
        "preferred_satisfied": preferred_satisfied,
        "preferred_differences": preferred_differences,
        "unmentioned_features": unmentioned_features,
        "mandatory_valid": mandatory_valid,
        "mandatory_preferred_valid": mandatory_preferred_valid,
    }


# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 80)
print("731 V3 SYNTHETIC ENGINEER PREFERENCE GENERATION")
print("=" * 80)

print("\n[1/9] Loading files...")

configs = pd.read_csv(CONFIG_PATH)
ground_truth = pd.read_csv(GT_PATH)
requirements = pd.read_csv(REQ_PATH)

print(f"Configuration rows: {len(configs):,}")
print(f"Ground-truth rows:  {len(ground_truth):,}")
print(f"Requirement rows:   {len(requirements):,}")


# ============================================================================
# SCHEMA VALIDATION
# ============================================================================

print("\n[2/9] Validating schemas...")

missing_config = [
    col for col in REQUIRED_CONFIG_COLUMNS
    if col not in configs.columns
]

missing_gt = [
    col for col in REQUIRED_GT_COLUMNS
    if col not in ground_truth.columns
]

missing_req = [
    col for col in REQUIRED_REQ_COLUMNS
    if col not in requirements.columns
]

if missing_config:
    raise ValueError(
        "Configuration file is missing required columns: "
        + str(missing_config)
    )

if missing_gt:
    raise ValueError(
        "Ground-truth file is missing required columns: "
        + str(missing_gt)
    )

if missing_req:
    raise ValueError(
        "Requirements file is missing required columns: "
        + str(missing_req)
    )

print("Schema validation passed.")


# ============================================================================
# NORMALIZATION
# ============================================================================

print("\n[3/9] Normalizing data...")

configs["package_context_normalized"] = (
    configs["package_context"]
    .map(normalize_package)
)

ground_truth["target_package_normalized"] = (
    ground_truth["target_package"]
    .map(normalize_package)
)

requirements["characteristic"] = (
    requirements["characteristic"]
    .astype(str)
    .str.strip()
)

requirements["requirement_type"] = (
    requirements["requirement_type"]
    .map(normalize_requirement_type)
)

requirements["internal_value"] = (
    requirements["internal_value"]
    .map(normalize_text)
)

configs["record_type_normalized"] = (
    configs["record_type"]
    .map(normalize_text)
)

configs["evidence_level_normalized"] = (
    configs["evidence_level"]
    .map(normalize_text)
)

configs["source_frequency_numeric"] = safe_numeric(
    configs["source_frequency"],
    default=0.0,
)

configs["nearest_real_distance_numeric"] = safe_numeric(
    configs["nearest_real_distance"],
    default=0.0,
)

configs["num_changed_characteristics_numeric"] = safe_numeric(
    configs["num_changed_characteristics"],
    default=0.0,
)

configs["evidence_score_numeric"] = safe_numeric(
    configs["evidence_score"],
    default=0.0,
)


# ============================================================================
# REQUIREMENT LOOKUP
# ============================================================================

print("\n[4/9] Building RFQ requirement lookup...")

requirement_lookup = build_requirement_lookup(requirements)

print(
    f"RFQs with requirements: {len(requirement_lookup):,}"
)


# ============================================================================
# NORMALIZATION OF CONFIGURATION-LEVEL PREFERENCE FEATURES
# ============================================================================

print("\n[5/9] Preparing engineering preference features...")

# These features are configuration-level properties and do NOT use
# the ground-truth configuration.
#
# Higher is preferred:
#   source_frequency_score
#   real_proximity_score
#   low_change_score
#   observed_score
#   evidence_score_normalized
#
# Lower is preferred:
#   unmentioned_features
#
# All continuous features are normalized globally for reproducibility.

configs["source_frequency_score"] = minmax_normalize(
    configs["source_frequency_numeric"],
    reverse=False,
)

configs["real_proximity_score"] = minmax_normalize(
    configs["nearest_real_distance_numeric"],
    reverse=True,
)

configs["low_change_score"] = minmax_normalize(
    configs["num_changed_characteristics_numeric"],
    reverse=True,
)

configs["evidence_score_normalized"] = minmax_normalize(
    configs["evidence_score_numeric"],
    reverse=False,
)

configs["observed_configuration_score"] = (
    configs["record_type_normalized"]
    .eq("OBSERVED")
    .astype(float)
)

print("Preference features prepared.")


# ============================================================================
# CONFIGURATION POOLS
# ============================================================================

print("\n[6/9] Building package-constrained configuration pools...")

package_pools = {}

for package, package_group in configs.groupby(
    "package_context_normalized",
    sort=False,
):
    # Deduplicate canonical IDs while retaining the first configuration
    # record for each identity.
    package_group = (
        package_group
        .sort_values(
            [
                "canonical_configuration_id",
                "record_type_normalized",
                "source_frequency_numeric",
            ],
            ascending=[True, True, False],
        )
        .drop_duplicates(
            subset=["canonical_configuration_id"],
            keep="first",
        )
        .copy()
    )

    package_pools[package] = package_group

print(
    f"Package contexts prepared: {len(package_pools):,}"
)


# ============================================================================
# GENERATE CANDIDATE PREFERENCE DATA
# ============================================================================

print("\n[7/9] Generating synthetic engineer preferences...")
print("This evaluates package-constrained valid configurations.")

candidate_rows = []
ranking_rows = []
pairwise_rows = []

# --------------------------------------------------------------------------
# SYNTHETIC ENGINEER UTILITY WEIGHTS
# --------------------------------------------------------------------------
#
# The utility deliberately emphasizes:
#
# 1. fewer unmentioned/configured features
# 2. proximity to real configurations
# 3. fewer synthetic changes
# 4. observed configuration provenance
# 5. source frequency
# 6. evidence strength
#
# These are transparent synthetic assumptions and should be described
# as such in the thesis.
#
# IMPORTANT:
# Ground-truth configuration identity is NOT included.

WEIGHT_UNMENTIONED = 0.30
WEIGHT_REAL_PROXIMITY = 0.20
WEIGHT_LOW_CHANGE = 0.15
WEIGHT_OBSERVED = 0.15
WEIGHT_FREQUENCY = 0.10
WEIGHT_EVIDENCE = 0.10

assert math.isclose(
    sum(
        [
            WEIGHT_UNMENTIONED,
            WEIGHT_REAL_PROXIMITY,
            WEIGHT_LOW_CHANGE,
            WEIGHT_OBSERVED,
            WEIGHT_FREQUENCY,
            WEIGHT_EVIDENCE,
        ]
    ),
    1.0,
)


rfq_ids = ground_truth["rfq_id"].tolist()

for index, gt_row in enumerate(
    ground_truth.itertuples(index=False),
    start=1,
):

    if index == 1 or index % 500 == 0 or index == len(rfq_ids):
        print(
            f"  Processing {index:,}/{len(rfq_ids):,}"
        )

    rfq_id = gt_row.rfq_id
    target_package = gt_row.target_package_normalized
    ground_truth_configuration_id = (
        gt_row.canonical_configuration_id
    )

    rfq_requirements = requirement_lookup.get(
        rfq_id,
        {},
    )

    if target_package not in package_pools:
        continue

    package_candidates = package_pools[target_package]

    # --------------------------------------------------------------
    # Evaluate every candidate in the package
    # --------------------------------------------------------------

    evaluated = []

    for _, candidate in package_candidates.iterrows():

        evaluation = evaluate_candidate(
            candidate,
            rfq_requirements,
        )

        # Only configurations satisfying BOTH mandatory and preferred
        # requirements enter the primary engineer preference pool.
        if not evaluation["mandatory_preferred_valid"]:
            continue

        row = {
            "rfq_id": rfq_id,
            "target_package": target_package,
            "canonical_configuration_id": (
                candidate["canonical_configuration_id"]
            ),
            "ground_truth_configuration_id": (
                ground_truth_configuration_id
            ),

            "package_context": candidate["package_context"],
            "record_type": candidate["record_type"],
            "generation_method": candidate["generation_method"],
            "evidence_level": candidate["evidence_level"],

            "source_frequency": candidate["source_frequency"],
            "nearest_real_distance": candidate[
                "nearest_real_distance"
            ],
            "num_changed_characteristics": candidate[
                "num_changed_characteristics"
            ],
            "evidence_score": candidate["evidence_score"],

            "mandatory_total": evaluation["mandatory_total"],
            "mandatory_satisfied": evaluation[
                "mandatory_satisfied"
            ],
            "mandatory_violations": evaluation[
                "mandatory_violations"
            ],
            "preferred_total": evaluation["preferred_total"],
            "preferred_satisfied": evaluation[
                "preferred_satisfied"
            ],
            "preferred_differences": evaluation[
                "preferred_differences"
            ],
            "unmentioned_features": evaluation[
                "unmentioned_features"
            ],

            "mandatory_valid": evaluation[
                "mandatory_valid"
            ],
            "mandatory_preferred_valid": evaluation[
                "mandatory_preferred_valid"
            ],

            "source_frequency_score": candidate[
                "source_frequency_score"
            ],
            "real_proximity_score": candidate[
                "real_proximity_score"
            ],
            "low_change_score": candidate[
                "low_change_score"
            ],
            "observed_configuration_score": candidate[
                "observed_configuration_score"
            ],
            "evidence_score_normalized": candidate[
                "evidence_score_normalized"
            ],
        }

        # ----------------------------------------------------------
        # Synthetic engineer utility
        # ----------------------------------------------------------

        utility = (
            WEIGHT_UNMENTIONED
            * (1.0 - min(
                evaluation["unmentioned_features"]
                / len(TECHNICAL_CHARACTERISTICS),
                1.0,
            ))
            +
            WEIGHT_REAL_PROXIMITY
            * candidate["real_proximity_score"]
            +
            WEIGHT_LOW_CHANGE
            * candidate["low_change_score"]
            +
            WEIGHT_OBSERVED
            * candidate["observed_configuration_score"]
            +
            WEIGHT_FREQUENCY
            * candidate["source_frequency_score"]
            +
            WEIGHT_EVIDENCE
            * candidate["evidence_score_normalized"]
        )

        # Tiny deterministic tie-breaker.
        jitter = deterministic_jitter(
            rfq_id,
            candidate["canonical_configuration_id"],
        )

        utility_with_jitter = utility + jitter

        row["synthetic_engineer_utility"] = utility
        row["tie_break_jitter"] = jitter
        row["synthetic_engineer_score"] = utility_with_jitter

        # Explicit provenance.
        row["preference_label_provenance"] = "SYNTHETIC"
        row["preference_generation_method"] = (
            "synthetic_engineer_v1"
        )

        # Ground-truth comparison ONLY for later analysis.
        row["is_ground_truth_configuration"] = (
            candidate["canonical_configuration_id"]
            == ground_truth_configuration_id
        )

        evaluated.append(row)

    if not evaluated:
        continue

    candidate_df = pd.DataFrame(evaluated)

    # --------------------------------------------------------------
    # Rank candidates
    # --------------------------------------------------------------

    candidate_df = candidate_df.sort_values(
        [
            "synthetic_engineer_score",
            "canonical_configuration_id",
        ],
        ascending=[False, True],
    ).reset_index(drop=True)

    candidate_df["engineer_rank"] = (
        np.arange(len(candidate_df)) + 1
    )

    candidate_df["candidate_count"] = len(candidate_df)

    candidate_df["is_synthetic_engineer_choice"] = (
        candidate_df["engineer_rank"] == 1
    )

    candidate_df["pairwise_comparison_count"] = (
        candidate_df["engineer_rank"].apply(
            lambda rank: len(candidate_df) - rank
        )
    )

    candidate_rows.extend(
        candidate_df.to_dict("records")
    )

    # --------------------------------------------------------------
    # RFQ-level ranking record
    # --------------------------------------------------------------

    top_candidate = candidate_df.iloc[0]

    ground_truth_matches = candidate_df[
        candidate_df["is_ground_truth_configuration"]
    ]

    if len(ground_truth_matches) > 0:
        gt_rank = int(
            ground_truth_matches.iloc[0]["engineer_rank"]
        )

        gt_score = float(
            ground_truth_matches.iloc[0][
                "synthetic_engineer_score"
            ]
        )
    else:
        gt_rank = np.nan
        gt_score = np.nan

    ranking_rows.append(
        {
            "rfq_id": rfq_id,
            "target_package": target_package,

            "candidate_count": len(candidate_df),

            "synthetic_engineer_choice": (
                top_candidate[
                    "canonical_configuration_id"
                ]
            ),

            "synthetic_engineer_top_score": (
                top_candidate[
                    "synthetic_engineer_score"
                ]
            ),

            "ground_truth_configuration_id": (
                ground_truth_configuration_id
            ),

            "ground_truth_engineer_rank": gt_rank,
            "ground_truth_engineer_score": gt_score,

            "ground_truth_selected_by_engineer": (
                gt_rank == 1
                if not pd.isna(gt_rank)
                else False
            ),

            "preference_label_provenance": "SYNTHETIC",
            "preference_generation_method": (
                "synthetic_engineer_v1"
            ),
        }
    )

    # --------------------------------------------------------------
    # Pairwise preferences
    # --------------------------------------------------------------
    #
    # Candidate A is preferred to candidate B whenever A has a
    # higher synthetic engineer utility.
    #
    # Since the candidates are already ranked, we generate all
    # ordered preference pairs.
    # --------------------------------------------------------------

    records = candidate_df.to_dict("records")

    for i in range(len(records)):

        preferred = records[i]

        # Full pairwise generation.
        for j in range(i + 1, len(records)):

            rejected = records[j]

            pairwise_rows.append(
                {
                    "rfq_id": rfq_id,
                    "target_package": target_package,

                    "preferred_configuration_id": (
                        preferred[
                            "canonical_configuration_id"
                        ]
                    ),

                    "rejected_configuration_id": (
                        rejected[
                            "canonical_configuration_id"
                        ]
                    ),

                    "preferred_rank": (
                        preferred["engineer_rank"]
                    ),

                    "rejected_rank": (
                        rejected["engineer_rank"]
                    ),

                    "preferred_score": (
                        preferred[
                            "synthetic_engineer_score"
                        ]
                    ),

                    "rejected_score": (
                        rejected[
                            "synthetic_engineer_score"
                        ]
                    ),

                    "score_margin": (
                        preferred[
                            "synthetic_engineer_score"
                        ]
                        -
                        rejected[
                            "synthetic_engineer_score"
                        ]
                    ),

                    "preferred_unmentioned_features": (
                        preferred[
                            "unmentioned_features"
                        ]
                    ),

                    "rejected_unmentioned_features": (
                        rejected[
                            "unmentioned_features"
                        ]
                    ),

                    "preferred_nearest_real_distance": (
                        preferred[
                            "nearest_real_distance"
                        ]
                    ),

                    "rejected_nearest_real_distance": (
                        rejected[
                            "nearest_real_distance"
                        ]
                    ),

                    "preferred_num_changed_characteristics": (
                        preferred[
                            "num_changed_characteristics"
                        ]
                    ),

                    "rejected_num_changed_characteristics": (
                        rejected[
                            "num_changed_characteristics"
                        ]
                    ),

                    "preferred_record_type": (
                        preferred["record_type"]
                    ),

                    "rejected_record_type": (
                        rejected["record_type"]
                    ),

                    "label": 1,

                    "preference_label_provenance": (
                        "SYNTHETIC"
                    ),

                    "preference_generation_method": (
                        "synthetic_engineer_v1"
                    ),
                }
            )


# ============================================================================
# SAVE CANDIDATE DATA
# ============================================================================

print("\n[8/9] Building outputs...")

candidate_output = pd.DataFrame(candidate_rows)
ranking_output = pd.DataFrame(ranking_rows)
pairwise_output = pd.DataFrame(pairwise_rows)

candidate_path = (
    OUT_DIR
    / "731_engineer_candidate_preferences_v1.csv"
)

ranking_path = (
    OUT_DIR
    / "731_engineer_rankings_v1.csv"
)

pairwise_path = (
    OUT_DIR
    / "731_engineer_pairwise_preferences_v1.csv"
)

candidate_output.to_csv(
    candidate_path,
    index=False,
)

ranking_output.to_csv(
    ranking_path,
    index=False,
)

pairwise_output.to_csv(
    pairwise_path,
    index=False,
)

print(f"  Saved: {candidate_path}")
print(f"  Saved: {ranking_path}")
print(f"  Saved: {pairwise_path}")


# ============================================================================
# METRICS
# ============================================================================

print("\n[9/9] Calculating preference-generation metrics...")

if len(ranking_output) > 0:

    engineer_selected_gt = (
        ranking_output[
            "ground_truth_selected_by_engineer"
        ]
        .mean()
    )

    gt_available = (
        ranking_output["ground_truth_engineer_rank"]
        .notna()
    )

    gt_rank_mean = (
        ranking_output.loc[
            gt_available,
            "ground_truth_engineer_rank",
        ].mean()
    )

    gt_rank_median = (
        ranking_output.loc[
            gt_available,
            "ground_truth_engineer_rank",
        ].median()
    )

else:

    engineer_selected_gt = np.nan
    gt_rank_mean = np.nan
    gt_rank_median = np.nan


if len(candidate_output) > 0:

    mean_candidates = (
        candidate_output
        .groupby("rfq_id")
        .size()
        .mean()
    )

    median_candidates = (
        candidate_output
        .groupby("rfq_id")
        .size()
        .median()
    )

else:

    mean_candidates = np.nan
    median_candidates = np.nan


metrics = pd.DataFrame(
    [
        {
            "metric": "rfqs_with_valid_preference_pool",
            "value": ranking_output["rfq_id"].nunique()
            if len(ranking_output) > 0
            else 0,
        },
        {
            "metric": "candidate_preference_rows",
            "value": len(candidate_output),
        },
        {
            "metric": "pairwise_preference_rows",
            "value": len(pairwise_output),
        },
        {
            "metric": "mean_valid_preference_candidates",
            "value": mean_candidates,
        },
        {
            "metric": "median_valid_preference_candidates",
            "value": median_candidates,
        },
        {
            "metric": "ground_truth_selected_by_synthetic_engineer",
            "value": engineer_selected_gt,
        },
        {
            "metric": "ground_truth_mean_engineer_rank",
            "value": gt_rank_mean,
        },
        {
            "metric": "ground_truth_median_engineer_rank",
            "value": gt_rank_median,
        },
        {
            "metric": "weight_unmentioned_features",
            "value": WEIGHT_UNMENTIONED,
        },
        {
            "metric": "weight_real_proximity",
            "value": WEIGHT_REAL_PROXIMITY,
        },
        {
            "metric": "weight_low_change",
            "value": WEIGHT_LOW_CHANGE,
        },
        {
            "metric": "weight_observed",
            "value": WEIGHT_OBSERVED,
        },
        {
            "metric": "weight_frequency",
            "value": WEIGHT_FREQUENCY,
        },
        {
            "metric": "weight_evidence",
            "value": WEIGHT_EVIDENCE,
        },
        {
            "metric": "preference_label_provenance",
            "value": "SYNTHETIC",
        },
        {
            "metric": "preference_generation_method",
            "value": "synthetic_engineer_v1",
        },
    ]
)

metrics_path = (
    OUT_DIR
    / "731_engineer_preference_metrics_v1.csv"
)

metrics.to_csv(
    metrics_path,
    index=False,
)

print(f"  Saved: {metrics_path}")


# ============================================================================
# FEATURE SUMMARY
# ============================================================================

feature_summary = []

if len(candidate_output) > 0:

    numeric_preference_features = [
        "unmentioned_features",
        "nearest_real_distance",
        "num_changed_characteristics",
        "source_frequency",
        "evidence_score",
        "source_frequency_score",
        "real_proximity_score",
        "low_change_score",
        "observed_configuration_score",
        "evidence_score_normalized",
        "synthetic_engineer_utility",
        "synthetic_engineer_score",
    ]

    for feature in numeric_preference_features:

        values = pd.to_numeric(
            candidate_output[feature],
            errors="coerce",
        )

        feature_summary.append(
            {
                "feature": feature,
                "count": values.notna().sum(),
                "mean": values.mean(),
                "median": values.median(),
                "std": values.std(),
                "min": values.min(),
                "max": values.max(),
            }
        )

feature_summary_output = pd.DataFrame(
    feature_summary
)

feature_summary_path = (
    OUT_DIR
    / "731_engineer_preference_feature_summary_v1.csv"
)

feature_summary_output.to_csv(
    feature_summary_path,
    index=False,
)

print(f"  Saved: {feature_summary_path}")


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print(
    f"\nRFQs with preference pools: "
    f"{ranking_output['rfq_id'].nunique():,}"
)

print(
    f"Candidate preference rows: "
    f"{len(candidate_output):,}"
)

print(
    f"Pairwise preference rows: "
    f"{len(pairwise_output):,}"
)

print(
    f"Mean candidates per RFQ: "
    f"{mean_candidates:.2f}"
)

print(
    f"Median candidates per RFQ: "
    f"{median_candidates:.2f}"
)

print(
    f"\nGround-truth selected by synthetic engineer: "
    f"{engineer_selected_gt:.4f}"
)

print(
    f"Ground-truth mean engineer rank: "
    f"{gt_rank_mean:.2f}"
)

print(
    f"Ground-truth median engineer rank: "
    f"{gt_rank_median:.2f}"
)

print("\nSynthetic engineer weights:")
print(
    f"  Unmentioned features:      "
    f"{WEIGHT_UNMENTIONED:.2f}"
)
print(
    f"  Real proximity:            "
    f"{WEIGHT_REAL_PROXIMITY:.2f}"
)
print(
    f"  Low configuration changes: "
    f"{WEIGHT_LOW_CHANGE:.2f}"
)
print(
    f"  Observed configuration:    "
    f"{WEIGHT_OBSERVED:.2f}"
)
print(
    f"  Source frequency:          "
    f"{WEIGHT_FREQUENCY:.2f}"
)
print(
    f"  Evidence strength:         "
    f"{WEIGHT_EVIDENCE:.2f}"
)

print("\nProvenance: SYNTHETIC")
print("Generation method: synthetic_engineer_v1")

print("\nAnalysis complete.")