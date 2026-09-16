#!/usr/bin/env python3

"""
731 RFQ CONFIGURATION RECOMMENDATION — V2
=========================================

Purpose
-------
Evaluate configuration recommendation using:

1. V8 extracted RFQ requirements
2. Oracle / ground-truth requirements

The comparison separates:

    RFQ extraction quality
            from
    configuration ranking quality

V2 improvements over V1
-----------------------
- Evidence-weighted requirement scoring
- Value discriminativeness
- Characteristic discriminativeness
- Confidence-aware extracted requirements
- Explicit contradiction penalties
- Package-aware and package-agnostic evaluation
- Oracle benchmark
- Per-package metrics
- True-rank diagnostics

Important
---------
Ground truth is NEVER used by the V8 recommendation mode.

Ground truth is used only for:
    - evaluation
    - oracle benchmark
    - diagnostic statistics

"""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import math

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

CONFIG_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "configurations"
    / "731_synthetic_configurations_v5.csv"
)

GROUND_TRUTH_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "rfqs"
    / "731_rfq_ground_truth_v2.csv"
)

REQUIREMENTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "rfqs"
    / "731_rfq_requirements_v2.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# OUTPUT FILES
# =============================================================================

RECOMMENDATION_FILE = (
    OUTPUT_DIR
    / "731_rfq_configuration_recommendations_v2.csv"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "731_rfq_configuration_recommendation_metrics_v2.csv"
)

TOPK_FILE = (
    OUTPUT_DIR
    / "731_rfq_configuration_topk_v2.csv"
)

ERROR_FILE = (
    OUTPUT_DIR
    / "731_rfq_configuration_recommendation_errors_v2.csv"
)

PACKAGE_METRICS_FILE = (
    OUTPUT_DIR
    / "731_rfq_configuration_package_metrics_v2.csv"
)

SCORING_STATS_FILE = (
    OUTPUT_DIR
    / "731_rfq_configuration_scoring_statistics_v2.csv"
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
# NORMALIZATION
# =============================================================================

def normalize(value) -> str:

    if pd.isna(value):
        return "NOVALUE"

    text = str(value).strip()

    if not text:
        return "NOVALUE"

    return text


def normalize_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    for column in TECHNICAL_COLUMNS:

        if column in result.columns:

            result[column] = (
                result[column]
                .map(normalize)
            )

    return result


# =============================================================================
# COLUMN DETECTION
# =============================================================================

def detect_column(
    df: pd.DataFrame,
    candidates: List[str],
    required: bool = True,
):

    for column in candidates:

        if column in df.columns:
            return column

    if required:

        raise ValueError(
            "Could not find required column. "
            f"Tried: {candidates}. "
            f"Available: {df.columns.tolist()}"
        )

    return None


# =============================================================================
# CONFIGURATION IDENTIFIER
# =============================================================================

def configuration_key(
    row: pd.Series,
) -> Tuple[str, ...]:

    return tuple(
        normalize(
            row.get(
                column,
                "NOVALUE",
            )
        )
        for column in TECHNICAL_COLUMNS
    )


# =============================================================================
# BUILD UNIQUE CONFIGURATION SPACE
# =============================================================================

def build_configuration_space(
    configurations: pd.DataFrame,
) -> pd.DataFrame:

    df = normalize_dataframe(
        configurations
    )

    id_column = detect_column(
        df,
        [
            "canonical_configuration_id",
            "configuration_id",
            "config_id",
        ],
    )

    package_column = detect_column(
        df,
        [
            "package_context",
            "target_package",
            "package",
        ],
        required=False,
    )

    if package_column is None:

        package_column = "package_context"

        df[package_column] = "UNKNOWN"

    keep_columns = [
        id_column,
        package_column,
    ] + [
        column
        for column in TECHNICAL_COLUMNS
        if column in df.columns
    ]

    df = df[
        keep_columns
    ].copy()

    df = (
        df
        .drop_duplicates(
            subset=[id_column]
        )
        .reset_index(drop=True)
    )

    df = df.rename(
        columns={
            id_column:
                "canonical_configuration_id",
            package_column:
                "package_context",
        }
    )

    return df


# =============================================================================
# REQUIREMENT TABLE
# =============================================================================

def prepare_requirement_table(
    requirements: pd.DataFrame,
) -> pd.DataFrame:

    required_columns = [
        "rfq_id",
        "characteristic",
        "internal_value",
    ]

    missing = [
        column
        for column in required_columns
        if column not in requirements.columns
    ]

    if missing:

        raise ValueError(
            f"Requirement file missing columns: {missing}"
        )

    df = requirements[
        required_columns
    ].copy()

    df["rfq_id"] = (
        df["rfq_id"]
        .astype(str)
        .str.strip()
    )

    df["characteristic"] = (
        df["characteristic"]
        .astype(str)
        .str.strip()
    )

    df["internal_value"] = (
        df["internal_value"]
        .map(normalize)
    )

    return (
        df
        .drop_duplicates()
        .reset_index(drop=True)
    )


# =============================================================================
# BUILD EVIDENCE WEIGHTS
# =============================================================================

def build_evidence_weights(
    configurations: pd.DataFrame,
) -> Dict[Tuple[str, str], float]:

    """
    Calculate discriminative weights.

    A value occurring in many configurations gets a smaller
    weight.

    A rare value gets a larger weight.

    This prevents ubiquitous values from dominating the ranking.
    """

    weights = {}

    n_configurations = len(
        configurations
    )

    if n_configurations == 0:
        return weights

    for characteristic in TECHNICAL_COLUMNS:

        if characteristic not in configurations.columns:
            continue

        frequencies = (
            configurations[
                characteristic
            ]
            .value_counts()
            .to_dict()
        )

        for value, frequency in frequencies.items():

            frequency = max(
                int(frequency),
                1,
            )

            # Smoothed inverse frequency.
            idf = math.log(
                (n_configurations + 1)
                /
                (frequency + 1)
            ) + 1.0

            weights[
                (
                    characteristic,
                    normalize(value),
                )
            ] = idf

    return weights


# =============================================================================
# CHARACTERISTIC WEIGHTS
# =============================================================================

def build_characteristic_weights(
    configurations: pd.DataFrame,
) -> Dict[str, float]:

    weights = {}

    n_configurations = len(
        configurations
    )

    if n_configurations == 0:
        return weights

    for characteristic in TECHNICAL_COLUMNS:

        if characteristic not in configurations.columns:
            continue

        unique_values = (
            configurations[
                characteristic
            ]
            .nunique()
        )

        # Characteristics with more variation are
        # more discriminative.
        variation_weight = math.log(
            unique_values + 1
        )

        weights[
            characteristic
        ] = max(
            variation_weight,
            1.0,
        )

    return weights


# =============================================================================
# EXTRACTED REQUIREMENT CONFIDENCE
# =============================================================================

def get_prediction_confidence(
    row: pd.Series,
) -> float:

    value = row.get(
        "confidence",
        1.0,
    )

    try:
        value = float(value)
    except Exception:
        value = 1.0

    if not np.isfinite(value):
        value = 1.0

    return max(
        value,
        0.05,
    )


def confidence_weight(
    confidence: float,
) -> float:

    """
    Convert extraction confidence into a bounded
    recommendation weight.

    Avoids allowing extremely high V8 confidence
    values to dominate the ranking.
    """

    return min(
        max(
            confidence / 1.5,
            0.25,
        ),
        1.50,
    )


# =============================================================================
# BUILD REQUIREMENT DICTIONARIES
# =============================================================================

def build_extracted_requirements(
    predictions: pd.DataFrame,
) -> Dict[str, List[dict]]:

    result = defaultdict(list)

    if len(predictions) == 0:
        return result

    required = [
        "rfq_id",
        "characteristic",
        "internal_value",
    ]

    missing = [
        column
        for column in required
        if column not in predictions.columns
    ]

    if missing:

        raise ValueError(
            "V8 prediction file missing columns: "
            f"{missing}"
        )

    for _, row in predictions.iterrows():

        rfq_id = str(
            row["rfq_id"]
        ).strip()

        characteristic = str(
            row["characteristic"]
        ).strip()

        value = normalize(
            row["internal_value"]
        )

        confidence = (
            get_prediction_confidence(
                row
            )
        )

        result[rfq_id].append(
            {
                "characteristic":
                    characteristic,
                "internal_value":
                    value,
                "confidence":
                    confidence,
            }
        )

    return result


def build_oracle_requirements(
    requirements: pd.DataFrame,
) -> Dict[str, List[dict]]:

    result = defaultdict(list)

    for _, row in requirements.iterrows():

        rfq_id = str(
            row["rfq_id"]
        ).strip()

        result[rfq_id].append(
            {
                "characteristic":
                    str(
                        row["characteristic"]
                    ).strip(),

                "internal_value":
                    normalize(
                        row["internal_value"]
                    ),

                "confidence":
                    1.0,
            }
        )

    return result


# =============================================================================
# SCORE CONFIGURATION
# =============================================================================

def score_configuration(
    configuration: pd.Series,
    requirements: List[dict],
    evidence_weights: Dict[Tuple[str, str], float],
    characteristic_weights: Dict[str, float],
) -> Tuple[float, float, float, int]:

    score = 0.0

    positive_score = 0.0

    negative_score = 0.0

    matched = 0

    for requirement in requirements:

        characteristic = (
            requirement[
                "characteristic"
            ]
        )

        requested_value = normalize(
            requirement[
                "internal_value"
            ]
        )

        if (
            characteristic
            not in configuration.index
        ):
            continue

        candidate_value = normalize(
            configuration[
                characteristic
            ]
        )

        evidence_weight = (
            evidence_weights.get(
                (
                    characteristic,
                    requested_value,
                ),
                1.0,
            )
        )

        characteristic_weight = (
            characteristic_weights.get(
                characteristic,
                1.0,
            )
        )

        confidence = (
            confidence_weight(
                requirement.get(
                    "confidence",
                    1.0,
                )
            )
        )

        weight = (
            evidence_weight
            *
            characteristic_weight
            *
            confidence
        )

        # ------------------------------------------------------------
        # Exact match
        # ------------------------------------------------------------

        if candidate_value == requested_value:

            contribution = weight

            positive_score += contribution

            score += contribution

            matched += 1

        # ------------------------------------------------------------
        # Contradiction
        # ------------------------------------------------------------

        else:

            # NOVALUE is treated as a real technical state.
            #
            # A requested value vs NOVALUE is therefore a
            # meaningful contradiction.
            contradiction_penalty = (
                weight * 1.25
            )

            negative_score += (
                contradiction_penalty
            )

            score -= (
                contradiction_penalty
            )

    return (
        score,
        positive_score,
        negative_score,
        matched,
    )


# =============================================================================
# RANK CONFIGURATIONS
# =============================================================================

def rank_configurations(
    requirements: List[dict],
    configurations: pd.DataFrame,
    evidence_weights: Dict[Tuple[str, str], float],
    characteristic_weights: Dict[str, float],
) -> pd.DataFrame:

    rows = []

    for _, configuration in configurations.iterrows():

        (
            score,
            positive_score,
            negative_score,
            matched,
        ) = score_configuration(
            configuration,
            requirements,
            evidence_weights,
            characteristic_weights,
        )

        rows.append(
            {
                "canonical_configuration_id":
                    configuration[
                        "canonical_configuration_id"
                    ],

                "package_context":
                    configuration[
                        "package_context"
                    ],

                "score":
                    score,

                "positive_score":
                    positive_score,

                "negative_score":
                    negative_score,

                "matched_requirements":
                    matched,
            }
        )

    ranked = pd.DataFrame(
        rows
    )

    if len(ranked) == 0:
        return ranked

    ranked = (
        ranked
        .sort_values(
            [
                "score",
                "positive_score",
                "matched_requirements",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    ranked["rank"] = (
        np.arange(
            len(ranked)
        ) + 1
    )

    return ranked


# =============================================================================
# PACKAGE-AWARE RANKING
# =============================================================================

def get_package_candidates(
    configurations: pd.DataFrame,
    target_package: str,
) -> pd.DataFrame:

    package = str(
        target_package
    ).strip()

    candidates = configurations[
        configurations[
            "package_context"
        ]
        .astype(str)
        .str.strip()
        == package
    ].copy()

    # Fallback to all configurations if the package
    # is absent from the configuration file.
    if len(candidates) == 0:

        return configurations.copy()

    return candidates.reset_index(
        drop=True
    )


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(
    results: List[dict],
) -> dict:

    if not results:

        return {
            "rfqs_evaluated": 0,
            "top1_accuracy": 0.0,
            "top3_accuracy": 0.0,
            "top5_accuracy": 0.0,
            "top10_accuracy": 0.0,
            "mrr": 0.0,
            "mean_true_rank": 0.0,
            "median_true_rank": 0.0,
        }

    ranks = np.array(
        [
            row["true_rank"]
            for row in results
            if row["true_rank"] > 0
        ],
        dtype=float,
    )

    n = len(results)

    top1 = sum(
        row["true_rank"] == 1
        for row in results
    ) / n

    top3 = sum(
        0 < row["true_rank"] <= 3
        for row in results
    ) / n

    top5 = sum(
        0 < row["true_rank"] <= 5
        for row in results
    ) / n

    top10 = sum(
        0 < row["true_rank"] <= 10
        for row in results
    ) / n

    reciprocal_ranks = [
        1.0 / row["true_rank"]
        if row["true_rank"] > 0
        else 0.0
        for row in results
    ]

    return {
        "rfqs_evaluated":
            n,

        "top1_accuracy":
            top1,

        "top3_accuracy":
            top3,

        "top5_accuracy":
            top5,

        "top10_accuracy":
            top10,

        "mrr":
            float(
                np.mean(
                    reciprocal_ranks
                )
            ),

        "mean_true_rank":
            float(
                np.mean(ranks)
            )
            if len(ranks)
            else 0.0,

        "median_true_rank":
            float(
                np.median(ranks)
            )
            if len(ranks)
            else 0.0,
    }


# =============================================================================
# EVALUATE MODE
# =============================================================================

def evaluate_mode(
    mode_name: str,
    requirement_map: Dict[str, List[dict]],
    ground_truth: pd.DataFrame,
    configurations: pd.DataFrame,
    evidence_weights: Dict[Tuple[str, str], float],
    characteristic_weights: Dict[str, float],
    package_aware: bool,
) -> Tuple[List[dict], List[dict]]:

    true_config_map = (
        ground_truth
        .set_index("rfq_id")
        [
            "canonical_configuration_id"
        ]
        .to_dict()
    )

    package_map = (
        ground_truth
        .set_index("rfq_id")
        [
            "target_package"
        ]
        .to_dict()
    )

    recommendation_rows = []

    error_rows = []

    rfq_ids = list(
        true_config_map.keys()
    )

    print()
    print(
        f"Mode: {mode_name}"
    )

    for index, rfq_id in enumerate(
        rfq_ids,
        start=1,
    ):

        requirements = (
            requirement_map.get(
                rfq_id,
                [],
            )
        )

        target_config = str(
            true_config_map[
                rfq_id
            ]
        ).strip()

        target_package = str(
            package_map[
                rfq_id
            ]
        ).strip()

        if package_aware:

            candidates = (
                get_package_candidates(
                    configurations,
                    target_package,
                )
            )

        else:

            candidates = (
                configurations
                .copy()
            )

        ranked = rank_configurations(
            requirements,
            candidates,
            evidence_weights,
            characteristic_weights,
        )

        if len(ranked) == 0:

            true_rank = 0

            top_id = ""

            top_package = ""

            top_score = 0.0

        else:

            matches = ranked[
                ranked[
                    "canonical_configuration_id"
                ].astype(str).str.strip()
                == target_config
            ]

            if len(matches):

                true_rank = int(
                    matches.iloc[0]["rank"]
                )

            else:

                true_rank = 0

            top = ranked.iloc[0]

            top_id = str(
                top[
                    "canonical_configuration_id"
                ]
            )

            top_package = str(
                top[
                    "package_context"
                ]
            )

            top_score = float(
                top["score"]
            )

        recommendation_rows.append(
            {
                "rfq_id":
                    rfq_id,

                "mode":
                    mode_name,

                "package_aware":
                    package_aware,

                "target_package":
                    target_package,

                "true_configuration_id":
                    target_config,

                "recommended_configuration_id":
                    top_id,

                "recommended_package":
                    top_package,

                "top_score":
                    top_score,

                "true_rank":
                    true_rank,

                "candidate_count":
                    len(candidates),

                "requirement_count":
                    len(requirements),
            }
        )

        if (
            true_rank == 0
            or true_rank > 10
        ):

            error_rows.append(
                {
                    "rfq_id":
                        rfq_id,

                    "mode":
                        mode_name,

                    "package_aware":
                        package_aware,

                    "target_package":
                        target_package,

                    "true_configuration_id":
                        target_config,

                    "recommended_configuration_id":
                        top_id,

                    "true_rank":
                        true_rank,

                    "candidate_count":
                        len(candidates),

                    "requirement_count":
                        len(requirements),
                }
            )

        if index % 500 == 0:

            print(
                f"Processed "
                f"{index:,} / "
                f"{len(rfq_ids):,}"
            )

    return (
        recommendation_rows,
        error_rows,
    )


# =============================================================================
# TOP-K TABLE
# =============================================================================

def build_topk_table(
    recommendation_rows: List[dict],
    requirement_map: Dict[str, List[dict]],
    configurations: pd.DataFrame,
    evidence_weights: Dict[Tuple[str, str], float],
    characteristic_weights: Dict[str, float],
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:

    package_map = (
        ground_truth
        .set_index("rfq_id")
        ["target_package"]
        .to_dict()
    )

    rows = []

    for row in recommendation_rows:

        rfq_id = row["rfq_id"]

        # Only produce top-k for package-aware rows.
        if not row["package_aware"]:
            continue

        target_package = str(
            package_map[
                rfq_id
            ]
        ).strip()

        candidates = (
            get_package_candidates(
                configurations,
                target_package,
            )
        )

        ranked = rank_configurations(
            requirement_map.get(
                rfq_id,
                [],
            ),
            candidates,
            evidence_weights,
            characteristic_weights,
        )

        for _, candidate in (
            ranked.head(10).iterrows()
        ):

            rows.append(
                {
                    "rfq_id":
                        rfq_id,

                    "mode":
                        row["mode"],

                    "rank":
                        int(
                            candidate[
                                "rank"
                            ]
                        ),

                    "canonical_configuration_id":
                        candidate[
                            "canonical_configuration_id"
                        ],

                    "package_context":
                        candidate[
                            "package_context"
                        ],

                    "score":
                        candidate[
                            "score"
                        ],

                    "positive_score":
                        candidate[
                            "positive_score"
                        ],

                    "negative_score":
                        candidate[
                            "negative_score"
                        ],

                    "matched_requirements":
                        candidate[
                            "matched_requirements"
                        ],
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# PACKAGE METRICS
# =============================================================================

def build_package_metrics(
    recommendation_rows: List[dict],
) -> pd.DataFrame:

    df = pd.DataFrame(
        recommendation_rows
    )

    if len(df) == 0:
        return pd.DataFrame()

    rows = []

    for (
        mode,
        package_aware,
        package,
    ), group in df.groupby(
        [
            "mode",
            "package_aware",
            "target_package",
        ]
    ):

        records = group.to_dict(
            "records"
        )

        metrics = calculate_metrics(
            records
        )

        metrics.update(
            {
                "mode":
                    mode,

                "package_aware":
                    package_aware,

                "target_package":
                    package,
            }
        )

        rows.append(
            metrics
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 100)
    print(
        "731 RFQ CONFIGURATION "
        "RECOMMENDATION — V2"
    )
    print("=" * 100)

    # -----------------------------------------------------------------
    # LOAD V8
    # -----------------------------------------------------------------

    print()
    print(
        "Loading V8 extracted requirements..."
    )

    v8 = pd.read_csv(
        V8_FILE,
        low_memory=False,
    )

    print(
        f"V8 predictions: "
        f"{len(v8):,}"
    )

    # -----------------------------------------------------------------
    # LOAD CONFIGURATIONS
    # -----------------------------------------------------------------

    print()
    print(
        "Loading V5 technical configurations..."
    )

    configurations_raw = pd.read_csv(
        CONFIG_FILE,
        low_memory=False,
    )

    configurations = (
        build_configuration_space(
            configurations_raw
        )
    )

    print(
        f"Unique configurations: "
        f"{len(configurations):,}"
    )

    # -----------------------------------------------------------------
    # LOAD GROUND TRUTH
    # -----------------------------------------------------------------

    print()
    print(
        "Loading RFQ ground truth..."
    )

    ground_truth = pd.read_csv(
        GROUND_TRUTH_FILE,
        low_memory=False,
    )

    print(
        f"Ground-truth RFQs: "
        f"{len(ground_truth):,}"
    )

    # -----------------------------------------------------------------
    # LOAD STRUCTURED REQUIREMENTS
    # -----------------------------------------------------------------

    print()
    print(
        "Loading structured requirements..."
    )

    requirements = pd.read_csv(
        REQUIREMENTS_FILE,
        low_memory=False,
    )

    requirements = (
        prepare_requirement_table(
            requirements
        )
    )

    print(
        f"Ground-truth requirements: "
        f"{len(requirements):,}"
    )

    # -----------------------------------------------------------------
    # NORMALIZE GROUND TRUTH
    # -----------------------------------------------------------------

    ground_truth = ground_truth.copy()

    ground_truth["rfq_id"] = (
        ground_truth["rfq_id"]
        .astype(str)
        .str.strip()
    )

    ground_truth[
        "canonical_configuration_id"
    ] = (
        ground_truth[
            "canonical_configuration_id"
        ]
        .astype(str)
        .str.strip()
    )

    ground_truth[
        "target_package"
    ] = (
        ground_truth[
            "target_package"
        ]
        .astype(str)
        .str.strip()
    )

    ground_truth = (
        ground_truth
        .drop_duplicates(
            subset=["rfq_id"]
        )
        .reset_index(drop=True)
    )

    # -----------------------------------------------------------------
    # BUILD WEIGHTS
    # -----------------------------------------------------------------

    print()
    print(
        "Building evidence-weighted "
        "scoring model..."
    )

    evidence_weights = (
        build_evidence_weights(
            configurations
        )
    )

    characteristic_weights = (
        build_characteristic_weights(
            configurations
        )
    )

    print(
        f"Value evidence weights: "
        f"{len(evidence_weights):,}"
    )

    print(
        f"Characteristic weights: "
        f"{len(characteristic_weights):,}"
    )

    # -----------------------------------------------------------------
    # REQUIREMENT MAPS
    # -----------------------------------------------------------------

    extracted_map = (
        build_extracted_requirements(
            v8
        )
    )

    oracle_map = (
        build_oracle_requirements(
            requirements
        )
    )

    print()
    print(
        f"RFQs with extracted requirements: "
        f"{len(extracted_map):,}"
    )

    print(
        f"RFQs with oracle requirements: "
        f"{len(oracle_map):,}"
    )

    # -----------------------------------------------------------------
    # RUN FOUR BENCHMARKS
    # -----------------------------------------------------------------

    all_recommendations = []

    all_errors = []

    benchmark_definitions = [

        (
            "V8_EXTRACTED",
            extracted_map,
            True,
        ),

        (
            "V8_EXTRACTED",
            extracted_map,
            False,
        ),

        (
            "ORACLE",
            oracle_map,
            True,
        ),

        (
            "ORACLE",
            oracle_map,
            False,
        ),
    ]

    for (
        mode,
        requirement_map,
        package_aware,
    ) in benchmark_definitions:

        recommendations, errors = (
            evaluate_mode(
                mode,
                requirement_map,
                ground_truth,
                configurations,
                evidence_weights,
                characteristic_weights,
                package_aware,
            )
        )

        all_recommendations.extend(
            recommendations
        )

        all_errors.extend(
            errors
        )

    # -----------------------------------------------------------------
    # METRICS
    # -----------------------------------------------------------------

    recommendation_df = pd.DataFrame(
        all_recommendations
    )

    error_df = pd.DataFrame(
        all_errors
    )

    metrics_rows = []

    print()
    print("=" * 100)
    print(
        "731 RFQ CONFIGURATION "
        "RECOMMENDATION V2 RESULTS"
    )
    print("=" * 100)

    for (
        mode,
        package_aware,
    ), group in recommendation_df.groupby(
        [
            "mode",
            "package_aware",
        ]
    ):

        records = group.to_dict(
            "records"
        )

        metrics = calculate_metrics(
            records
        )

        metrics[
            "mode"
        ] = mode

        metrics[
            "package_aware"
        ] = package_aware

        metrics_rows.append(
            metrics
        )

        print()
        print(
            f"{mode} | "
            f"package_aware={package_aware}"
        )

        print(
            f"RFQs evaluated:       "
            f"{metrics['rfqs_evaluated']:,}"
        )

        print(
            f"Top-1 accuracy:       "
            f"{metrics['top1_accuracy']:.4f}"
        )

        print(
            f"Top-3 accuracy:       "
            f"{metrics['top3_accuracy']:.4f}"
        )

        print(
            f"Top-5 accuracy:       "
            f"{metrics['top5_accuracy']:.4f}"
        )

        print(
            f"Top-10 accuracy:      "
            f"{metrics['top10_accuracy']:.4f}"
        )

        print(
            f"MRR:                  "
            f"{metrics['mrr']:.4f}"
        )

        print(
            f"Mean true rank:       "
            f"{metrics['mean_true_rank']:.2f}"
        )

        print(
            f"Median true rank:     "
            f"{metrics['median_true_rank']:.2f}"
        )

    metrics_df = pd.DataFrame(
        metrics_rows
    )

    # -----------------------------------------------------------------
    # TOP-K
    # -----------------------------------------------------------------

    print()
    print(
        "Building top-K rankings..."
    )

    topk_df = build_topk_table(
        [
            row
            for row in all_recommendations
            if row["package_aware"]
        ],
        extracted_map,
        configurations,
        evidence_weights,
        characteristic_weights,
        ground_truth,
    )

    # Add oracle top-k separately.
    oracle_topk = build_topk_table(
        [
            row
            for row in all_recommendations
            if (
                row["package_aware"]
                and row["mode"]
                == "ORACLE"
            )
        ],
        oracle_map,
        configurations,
        evidence_weights,
        characteristic_weights,
        ground_truth,
    )

    # The first top-k call may contain V8 only.
    # Build both explicitly to avoid ambiguity.
    extracted_topk = build_topk_table(
        [
            row
            for row in all_recommendations
            if (
                row["package_aware"]
                and row["mode"]
                == "V8_EXTRACTED"
            )
        ],
        extracted_map,
        configurations,
        evidence_weights,
        characteristic_weights,
        ground_truth,
    )

    topk_df = pd.concat(
        [
            extracted_topk,
            oracle_topk,
        ],
        ignore_index=True,
    )

    # -----------------------------------------------------------------
    # PACKAGE METRICS
    # -----------------------------------------------------------------

    package_metrics = (
        build_package_metrics(
            [
                row
                for row in all_recommendations
                if row["package_aware"]
            ]
        )
    )

    # -----------------------------------------------------------------
    # SCORING STATISTICS
    # -----------------------------------------------------------------

    scoring_rows = []

    for characteristic, weight in (
        characteristic_weights.items()
    ):

        scoring_rows.append(
            {
                "weight_type":
                    "characteristic",

                "characteristic":
                    characteristic,

                "value":
                    "",

                "weight":
                    weight,
            }
        )

    for (
        characteristic,
        value,
    ), weight in (
        evidence_weights.items()
    ):

        scoring_rows.append(
            {
                "weight_type":
                    "value",

                "characteristic":
                    characteristic,

                "value":
                    value,

                "weight":
                    weight,
            }
        )

    scoring_df = pd.DataFrame(
        scoring_rows
    )

    # -----------------------------------------------------------------
    # SAVE
    # -----------------------------------------------------------------

    recommendation_df.to_csv(
        RECOMMENDATION_FILE,
        index=False,
    )

    metrics_df.to_csv(
        METRICS_FILE,
        index=False,
    )

    topk_df.to_csv(
        TOPK_FILE,
        index=False,
    )

    error_df.to_csv(
        ERROR_FILE,
        index=False,
    )

    package_metrics.to_csv(
        PACKAGE_METRICS_FILE,
        index=False,
    )

    scoring_df.to_csv(
        SCORING_STATS_FILE,
        index=False,
    )

    # -----------------------------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "V2 FILES SAVED"
    )
    print("=" * 100)

    print()
    print(
        "Recommendations:"
    )
    print(
        RECOMMENDATION_FILE
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
    print(
        "Package metrics:"
    )
    print(
        PACKAGE_METRICS_FILE
    )

    print()
    print(
        "Scoring statistics:"
    )
    print(
        SCORING_STATS_FILE
    )

    print()
    print("=" * 100)
    print(
        "731 RFQ CONFIGURATION "
        "RECOMMENDATION V2 COMPLETE"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()