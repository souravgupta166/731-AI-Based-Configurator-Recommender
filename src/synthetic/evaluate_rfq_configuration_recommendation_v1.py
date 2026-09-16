#!/usr/bin/env python3

"""
731 RFQ CONFIGURATION RECOMMENDATION — V1
=========================================

End-to-end recommendation layer:

    RFQ
      ↓
    V8 extracted requirements
      ↓
    Requirement-aware candidate scoring
      ↓
    Ranked 731 configurations

The system evaluates whether the correct technical configuration
can be recovered from extracted customer requirements.

Important:
- Unknown requirements are NOT treated as contradictions.
- V8 steamApplication abstentions are ignored for value matching.
- Explicit requirement/value matches receive positive scores.
- Explicit contradictions receive stronger negative scores.
- Package context is used as a hard compatibility constraint where
  available from the V5 configuration.
"""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

V8_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "731_rfq_extraction_results_v8.csv"
)

V8_ABSTENTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "731_rfq_extraction_abstentions_v8.csv"
)

V5_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "configurations"
    / "731_synthetic_configurations_v5.csv"
)

GT_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "rfqs"
    / "731_rfq_ground_truth_v2.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

RESULT_FILE = (
    OUTPUT_DIR
    / "731_rfq_configuration_recommendations_v1.csv"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "731_rfq_configuration_recommendation_metrics_v1.csv"
)

TOPK_FILE = (
    OUTPUT_DIR
    / "731_rfq_configuration_topk_v1.csv"
)

ERROR_FILE = (
    OUTPUT_DIR
    / "731_rfq_configuration_recommendation_errors_v1.csv"
)


# =============================================================================
# TECHNICAL CHARACTERISTICS
# =============================================================================

TECHNICAL_COLUMNS = [
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


# =============================================================================
# SCORING PARAMETERS
# =============================================================================

MATCH_SCORE = 3.0
STRONG_MATCH_SCORE = 4.0
CONTRADICTION_SCORE = -6.0

OPTIONAL_MATCH_SCORE = 2.0

UNKNOWN_VALUE = "NOVALUE"


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize(value):

    if pd.isna(value):
        return UNKNOWN_VALUE

    text = str(value).strip()

    if not text:
        return UNKNOWN_VALUE

    return text


# =============================================================================
# LOAD DATA
# =============================================================================

def load_data():

    print("=" * 100)
    print("LOADING RECOMMENDATION DATA")
    print("=" * 100)

    print()
    print("Loading V8 extracted requirements...")

    v8 = pd.read_csv(
        V8_FILE,
        low_memory=False,
    )

    print(
        f"V8 predictions: {len(v8):,}"
    )

    print()
    print("Loading V5 technical configurations...")

    configs = pd.read_csv(
        V5_FILE,
        low_memory=False,
    )

    print(
        f"V5 configurations: {len(configs):,}"
    )

    print()
    print("Loading RFQ ground truth...")

    gt = pd.read_csv(
        GT_FILE,
        low_memory=False,
    )

    print(
        f"Ground-truth RFQs: {len(gt):,}"
    )

    return v8, configs, gt


# =============================================================================
# PREPARE CONFIGURATIONS
# =============================================================================

def prepare_configurations(configs):

    configs = configs.copy()

    required = [
        "canonical_configuration_id",
        "package_context",
    ]

    missing = [
        c
        for c in required
        if c not in configs.columns
    ]

    if missing:

        raise ValueError(
            "Missing configuration columns: "
            + ", ".join(missing)
        )

    for column in TECHNICAL_COLUMNS:

        if column not in configs.columns:

            configs[column] = UNKNOWN_VALUE

        configs[column] = (
            configs[column]
            .map(normalize)
        )

    configs["canonical_configuration_id"] = (
        configs["canonical_configuration_id"]
        .astype(str)
        .str.strip()
    )

    configs["package_context"] = (
        configs["package_context"]
        .astype(str)
        .str.strip()
    )

    configs = configs.drop_duplicates(
        subset=[
            "canonical_configuration_id"
        ]
    )

    return configs


# =============================================================================
# PREPARE REQUIREMENTS
# =============================================================================

def prepare_requirements(v8):

    v8 = v8.copy()

    required = [
        "rfq_id",
        "characteristic",
        "internal_value",
    ]

    missing = [
        c
        for c in required
        if c not in v8.columns
    ]

    if missing:

        raise ValueError(
            "Missing V8 columns: "
            + ", ".join(missing)
        )

    for column in required:

        v8[column] = (
            v8[column]
            .astype(str)
            .str.strip()
        )

    # Remove steamApplication because V8 deliberately abstains
    # from assigning H/N.

    v8 = v8[
        v8["characteristic"]
        != "steamApplication"
    ].copy()

    # Remove duplicate requirements.

    v8 = v8.drop_duplicates(
        subset=[
            "rfq_id",
            "characteristic",
            "internal_value",
        ]
    )

    return v8


# =============================================================================
# BUILD REQUIREMENT MAP
# =============================================================================

def build_requirement_map(v8):

    requirement_map = defaultdict(list)

    for _, row in v8.iterrows():

        rfq_id = row["rfq_id"]

        characteristic = (
            row["characteristic"]
        )

        value = (
            row["internal_value"]
        )

        confidence = float(
            row.get(
                "confidence",
                1.0,
            )
        )

        requirement_map[rfq_id].append(
            {
                "characteristic":
                    characteristic,
                "value":
                    value,
                "confidence":
                    confidence,
            }
        )

    return requirement_map


# =============================================================================
# BUILD GROUND TRUTH
# =============================================================================

def build_ground_truth(gt):

    required = [
        "rfq_id",
        "canonical_configuration_id",
        "target_package",
    ]

    missing = [
        c
        for c in required
        if c not in gt.columns
    ]

    if missing:

        raise ValueError(
            "Missing ground-truth columns: "
            + ", ".join(missing)
        )

    gt = gt[
        required
    ].copy()

    for column in required:

        gt[column] = (
            gt[column]
            .astype(str)
            .str.strip()
        )

    gt = gt.drop_duplicates(
        subset=["rfq_id"]
    )

    return gt


# =============================================================================
# SCORE ONE CONFIGURATION
# =============================================================================

def score_configuration(
    requirements,
    candidate,
):
    """
    Score one candidate configuration.

    Match:
        +3

    Strong match:
        +4

    Contradiction:
        -6

    Unknown candidate value:
        0

    The candidate is NOT penalized for characteristics
    that are absent from the extracted RFQ requirements.
    """

    score = 0.0

    matches = 0
    contradictions = 0
    unknowns = 0

    matched_characteristics = []
    contradictory_characteristics = []

    for requirement in requirements:

        characteristic = (
            requirement["characteristic"]
        )

        required_value = normalize(
            requirement["value"]
        )

        if characteristic not in candidate:

            unknowns += 1

            continue

        candidate_value = normalize(
            candidate[characteristic]
        )

        # -------------------------------------------------------------
        # Unknown candidate value
        # -------------------------------------------------------------

        if candidate_value == UNKNOWN_VALUE:

            unknowns += 1

            continue

        # -------------------------------------------------------------
        # Match
        # -------------------------------------------------------------

        if candidate_value == required_value:

            # Higher confidence receives slightly higher weight.

            confidence = float(
                requirement.get(
                    "confidence",
                    1.0,
                )
            )

            if confidence >= 1.5:

                contribution = (
                    STRONG_MATCH_SCORE
                )

            else:

                contribution = (
                    MATCH_SCORE
                )

            score += contribution

            matches += 1

            matched_characteristics.append(
                characteristic
            )

        # -------------------------------------------------------------
        # Contradiction
        # -------------------------------------------------------------

        else:

            score += (
                CONTRADICTION_SCORE
            )

            contradictions += 1

            contradictory_characteristics.append(
                characteristic
            )

    # -----------------------------------------------------------------
    # Coverage
    # -----------------------------------------------------------------

    total_requirements = len(
        requirements
    )

    if total_requirements:

        coverage = (
            matches
            / total_requirements
        )

    else:

        coverage = 0.0

    # -----------------------------------------------------------------
    # Final adjusted score
    # -----------------------------------------------------------------

    # Small coverage bonus encourages configurations that
    # satisfy more extracted requirements.

    score += (
        coverage * 2.0
    )

    return {
        "score":
            score,
        "matches":
            matches,
        "contradictions":
            contradictions,
        "unknowns":
            unknowns,
        "coverage":
            coverage,
        "matched_characteristics":
            "|".join(
                sorted(
                    set(
                        matched_characteristics
                    )
                )
            ),
        "contradictory_characteristics":
            "|".join(
                sorted(
                    set(
                        contradictory_characteristics
                    )
                )
            ),
    }


# =============================================================================
# RANK ONE RFQ
# =============================================================================

def rank_rfq(
    rfq_id,
    requirements,
    candidates,
):

    rows = []

    for _, candidate in candidates.iterrows():

        result = score_configuration(
            requirements,
            candidate,
        )

        rows.append(
            {
                "rfq_id":
                    rfq_id,

                "canonical_configuration_id":
                    candidate[
                        "canonical_configuration_id"
                    ],

                "package_context":
                    candidate[
                        "package_context"
                    ],

                "score":
                    result["score"],

                "matches":
                    result["matches"],

                "contradictions":
                    result["contradictions"],

                "unknowns":
                    result["unknowns"],

                "coverage":
                    result["coverage"],

                "matched_characteristics":
                    result[
                        "matched_characteristics"
                    ],

                "contradictory_characteristics":
                    result[
                        "contradictory_characteristics"
                    ],
            }
        )

    ranked = pd.DataFrame(
        rows
    )

    ranked = ranked.sort_values(
        by=[
            "score",
            "matches",
            "coverage",
            "contradictions",
        ],
        ascending=[
            False,
            False,
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )

    ranked["rank"] = (
        np.arange(
            len(ranked)
        )
        + 1
    )

    return ranked


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def main():

    print("=" * 100)
    print("731 RFQ CONFIGURATION RECOMMENDATION — V1")
    print("=" * 100)

    # -----------------------------------------------------------------
    # LOAD
    # -----------------------------------------------------------------

    v8, configs, gt = load_data()

    # -----------------------------------------------------------------
    # PREPARE
    # -----------------------------------------------------------------

    print()
    print("=" * 100)
    print("PREPARING DATA")
    print("=" * 100)

    configs = prepare_configurations(
        configs
    )

    requirements = prepare_requirements(
        v8
    )

    ground_truth = build_ground_truth(
        gt
    )

    requirement_map = build_requirement_map(
        requirements
    )

    print()
    print(
        "Unique configurations:",
        len(configs),
    )

    print(
        "RFQs with extracted requirements:",
        len(requirement_map),
    )

    # -----------------------------------------------------------------
    # EVALUATION
    # -----------------------------------------------------------------

    print()
    print("=" * 100)
    print("RANKING CONFIGURATIONS")
    print("=" * 100)

    recommendation_rows = []
    topk_rows = []
    error_rows = []

    total_rfqs = len(
        ground_truth
    )

    for index, (_, gt_row) in enumerate(
        ground_truth.iterrows(),
        start=1,
    ):

        rfq_id = gt_row[
            "rfq_id"
        ]

        true_configuration = (
            gt_row[
                "canonical_configuration_id"
            ]
        )

        target_package = (
            gt_row[
                "target_package"
            ]
        )

        rfq_requirements = (
            requirement_map.get(
                rfq_id,
                [],
            )
        )

        # -------------------------------------------------------------
        # Package-aware candidate pool
        # -------------------------------------------------------------

        package_candidates = configs[
            configs["package_context"]
            == target_package
        ].copy()

        # If no package candidates exist,
        # fall back to all configurations.

        if len(package_candidates) == 0:

            package_candidates = configs.copy()

        ranked = rank_rfq(
            rfq_id,
            rfq_requirements,
            package_candidates,
        )

        # -------------------------------------------------------------
        # Top K
        # -------------------------------------------------------------

        top10 = ranked.head(10)

        true_match = ranked[
            ranked[
                "canonical_configuration_id"
            ]
            == true_configuration
        ]

        if len(true_match):

            true_rank = int(
                true_match.iloc[0]["rank"]
            )

        else:

            true_rank = np.nan

        # -------------------------------------------------------------
        # Recommendation row
        # -------------------------------------------------------------

        if len(ranked):

            top = ranked.iloc[0]

            recommendation_rows.append(
                {
                    "rfq_id":
                        rfq_id,

                    "true_configuration_id":
                        true_configuration,

                    "recommended_configuration_id":
                        top[
                            "canonical_configuration_id"
                        ],

                    "true_package":
                        target_package,

                    "recommended_package":
                        top[
                            "package_context"
                        ],

                    "top1_score":
                        top["score"],

                    "top1_matches":
                        top["matches"],

                    "top1_contradictions":
                        top["contradictions"],

                    "top1_coverage":
                        top["coverage"],

                    "true_rank":
                        true_rank,

                    "requirements_used":
                        len(rfq_requirements),
                }
            )

        # -------------------------------------------------------------
        # Top 10 records
        # -------------------------------------------------------------

        for _, candidate in top10.iterrows():

            topk_rows.append(
                {
                    "rfq_id":
                        rfq_id,

                    "rank":
                        candidate["rank"],

                    "canonical_configuration_id":
                        candidate[
                            "canonical_configuration_id"
                        ],

                    "package_context":
                        candidate[
                            "package_context"
                        ],

                    "score":
                        candidate["score"],

                    "matches":
                        candidate["matches"],

                    "contradictions":
                        candidate[
                            "contradictions"
                        ],

                    "coverage":
                        candidate["coverage"],

                    "is_ground_truth":
                        candidate[
                            "canonical_configuration_id"
                        ]
                        == true_configuration,
                }
            )

        # -------------------------------------------------------------
        # Errors
        # -------------------------------------------------------------

        if (
            not len(true_match)
            or true_rank > 10
        ):

            error_rows.append(
                {
                    "rfq_id":
                        rfq_id,

                    "true_configuration_id":
                        true_configuration,

                    "true_package":
                        target_package,

                    "true_rank":
                        true_rank,

                    "recommended_configuration_id":
                        ranked.iloc[0][
                            "canonical_configuration_id"
                        ]
                        if len(ranked)
                        else "",

                    "recommended_package":
                        ranked.iloc[0][
                            "package_context"
                        ]
                        if len(ranked)
                        else "",

                    "requirements_used":
                        len(rfq_requirements),
                }
            )

        # -------------------------------------------------------------
        # Progress
        # -------------------------------------------------------------

        if (
            index % 500 == 0
            or index == total_rfqs
        ):

            print(
                f"Processed "
                f"{index:,} / "
                f"{total_rfqs:,}"
            )

    # -----------------------------------------------------------------
    # RESULTS DATAFRAMES
    # -----------------------------------------------------------------

    recommendations = pd.DataFrame(
        recommendation_rows
    )

    topk = pd.DataFrame(
        topk_rows
    )

    errors = pd.DataFrame(
        error_rows
    )

    # -----------------------------------------------------------------
    # TOP-K METRICS
    # -----------------------------------------------------------------

    print()
    print("=" * 100)
    print("RECOMMENDATION RESULTS")
    print("=" * 100)

    top1 = (
        recommendations["true_rank"]
        <= 1
    ).mean()

    top3 = (
        recommendations["true_rank"]
        <= 3
    ).mean()

    top5 = (
        recommendations["true_rank"]
        <= 5
    ).mean()

    top10 = (
        recommendations["true_rank"]
        <= 10
    ).mean()

    reciprocal_rank = (
        1
        / recommendations[
            "true_rank"
        ]
    )

    reciprocal_rank = (
        reciprocal_rank
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    mrr = reciprocal_rank.mean()

    # -----------------------------------------------------------------
    # PACKAGE ACCURACY
    # -----------------------------------------------------------------

    package_accuracy = (
        recommendations[
            "true_package"
        ]
        ==
        recommendations[
            "recommended_package"
        ]
    ).mean()

    # -----------------------------------------------------------------
    # TRUE RANK
    # -----------------------------------------------------------------

    mean_true_rank = (
        recommendations[
            "true_rank"
        ]
        .dropna()
        .mean()
    )

    median_true_rank = (
        recommendations[
            "true_rank"
        ]
        .dropna()
        .median()
    )

    # -----------------------------------------------------------------
    # PRINT
    # -----------------------------------------------------------------

    print()
    print(
        f"RFQs evaluated: "
        f"{len(recommendations):,}"
    )

    print()
    print(
        f"Top-1 accuracy: "
        f"{top1:.4f}"
    )

    print(
        f"Top-3 accuracy: "
        f"{top3:.4f}"
    )

    print(
        f"Top-5 accuracy: "
        f"{top5:.4f}"
    )

    print(
        f"Top-10 accuracy: "
        f"{top10:.4f}"
    )

    print(
        f"MRR: "
        f"{mrr:.4f}"
    )

    print()
    print(
        f"Package accuracy: "
        f"{package_accuracy:.4f}"
    )

    print(
        f"Mean true rank: "
        f"{mean_true_rank:.2f}"
    )

    print(
        f"Median true rank: "
        f"{median_true_rank:.2f}"
    )

    # -----------------------------------------------------------------
    # METRICS TABLE
    # -----------------------------------------------------------------

    metrics = pd.DataFrame(
        [
            {
                "metric":
                    "top1_accuracy",
                "value":
                    top1,
            },
            {
                "metric":
                    "top3_accuracy",
                "value":
                    top3,
            },
            {
                "metric":
                    "top5_accuracy",
                "value":
                    top5,
            },
            {
                "metric":
                    "top10_accuracy",
                "value":
                    top10,
            },
            {
                "metric":
                    "MRR",
                "value":
                    mrr,
            },
            {
                "metric":
                    "package_accuracy",
                "value":
                    package_accuracy,
            },
            {
                "metric":
                    "mean_true_rank",
                "value":
                    mean_true_rank,
            },
            {
                "metric":
                    "median_true_rank",
                "value":
                    median_true_rank,
            },
        ]
    )

    # -----------------------------------------------------------------
    # SAVE
    # -----------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    recommendations.to_csv(
        RESULT_FILE,
        index=False,
    )

    metrics.to_csv(
        METRICS_FILE,
        index=False,
    )

    topk.to_csv(
        TOPK_FILE,
        index=False,
    )

    errors.to_csv(
        ERROR_FILE,
        index=False,
    )

    # -----------------------------------------------------------------
    # FINAL REPORT
    # -----------------------------------------------------------------

    print()
    print("=" * 100)
    print("V1 RECOMMENDATION FILES SAVED")
    print("=" * 100)

    print()
    print(
        "Recommendations:"
    )

    print(
        RESULT_FILE
    )

    print()
    print(
        "Metrics:"
    )

    print(
        METRICS_FILE
    )

    print()
    print(
        "Top-K rankings:"
    )

    print(
        TOPK_FILE
    )

    print()
    print(
        "Recommendation errors:"
    )

    print(
        ERROR_FILE
    )

    print()
    print("=" * 100)
    print(
        "731 RFQ CONFIGURATION RECOMMENDATION — V1 COMPLETE"
    )
    print("=" * 100)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()