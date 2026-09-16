#!/usr/bin/env python3

"""
731 SYNTHETIC CONFIGURATION GENERATOR — V5
==========================================

Purpose
-------
Sample a realistic synthetic technical configuration dataset from:

    1. Real canonical 731 configurations
    2. V4 feasible derived configurations

V5 does NOT invent new technical values.

Every synthetic configuration must already exist in either:

    - the real canonical source pool, or
    - the V4 constraint-validated feasible space.

Sampling principles
-------------------
1. Preserve empirical package distribution.
2. Prefer observed real configurations.
3. Prefer distance-1 derived configurations over distance-2.
4. Prefer dependency-supported configurations.
5. Prefer frequently observed mutation pairs.
6. Do not force packages to generate configurations they do not support.
7. Preserve provenance and evidence metadata.

Output
------
data/synthetic/configurations/731_synthetic_configurations_v5.csv

data/processed/731_v5_sampling_summary.csv

data/processed/731_v5_package_statistics.csv

data/processed/731_v5_evidence_statistics.csv
"""

from __future__ import annotations

import hashlib
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "x731_allgemein_Rev24.xlsx"
)

V4_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "configurations"
    / "731_feasible_configuration_space_v4.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "configurations"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_SEED = 7312026

TARGET_RECORDS = 5000

# -------------------------------------------------------------------------
# Sampling weights
#
# Real configurations are intentionally preferred.
# -------------------------------------------------------------------------

REAL_BASE_WEIGHT = 5.0

DERIVED_BASE_WEIGHT = 1.0

DISTANCE_1_MULTIPLIER = 2.5

DISTANCE_2_MULTIPLIER = 1.0

DEPENDENCY_MULTIPLIER = 2.0

ONE_CHARACTERISTIC_MULTIPLIER = 1.8

TWO_CHARACTERISTIC_MULTIPLIER = 1.0

# Maximum times one exact canonical configuration may appear.
#
# This prevents a small number of popular configurations from dominating
# the synthetic dataset.
MAX_CONFIGURATION_FREQUENCY = 50


# =============================================================================
# TECHNICAL COLUMNS
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
# STRONG DEPENDENCY PAIRS
# =============================================================================

DEPENDENCY_PAIRS = {
    frozenset(
        [
            "stromEingaenge",
            "temperaturEingaenge",
        ]
    ),
    frozenset(
        [
            "explosionApproval",
            "certification",
        ]
    ),
    frozenset(
        [
            "stromSchaltbar",
            "stromEingaenge",
        ]
    ),
    frozenset(
        [
            "stromSchaltbar",
            "temperaturEingaenge",
        ]
    ),
    frozenset(
        [
            "powerSupply",
            "certification",
        ]
    ),
    frozenset(
        [
            "housing",
            "powerSupply",
        ]
    ),
    frozenset(
        [
            "binaerDigitalOpenColl_MN",
            "binaerOpenColl_MP",
        ]
    ),
    frozenset(
        [
            "protectionArea",
            "certification",
        ]
    ),
    frozenset(
        [
            "explosionApproval",
            "protectionArea",
        ]
    ),
    frozenset(
        [
            "dataInterface",
            "certification",
        ]
    ),
    frozenset(
        [
            "dataInterface",
            "explosionApproval",
        ]
    ),
}


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize_value(value):

    if pd.isna(value):
        return "NOVALUE"

    text = str(value).strip()

    if not text:
        return "NOVALUE"

    if text.upper() == "NOVALUE":
        return "NOVALUE"

    # Configurator expressions:
    #
    # in {NOVALUE, 'x731'}
    #
    # -> x731

    if text.lower().startswith("in"):

        start = text.find("{")
        end = text.rfind("}")

        if (
            start >= 0
            and end > start
        ):

            content = (
                text[start + 1:end]
            )

            parts = [
                part.strip()
                for part in content.split(",")
            ]

            cleaned = []

            for part in parts:

                part = (
                    part
                    .strip("'")
                    .strip('"')
                    .strip()
                )

                if (
                    part
                    and part.upper()
                    != "NOVALUE"
                ):
                    cleaned.append(
                        part
                    )

            if cleaned:
                return cleaned[0]

            return "NOVALUE"

    if (
        len(text) >= 2
        and (
            (
                text[0] == "'"
                and text[-1] == "'"
            )
            or
            (
                text[0] == '"'
                and text[-1] == '"'
            )
        )
    ):
        text = text[1:-1]

    return text.strip()


# =============================================================================
# CANONICAL ID
# =============================================================================

def make_canonical_id(row):

    values = [
        normalize_value(
            row.get(
                column,
                "NOVALUE",
            )
        )
        for column in TECHNICAL_COLUMNS
    ]

    representation = "||".join(
        values
    )

    digest = hashlib.sha256(
        representation.encode(
            "utf-8"
        )
    ).hexdigest()

    return (
        "CFG731_"
        + digest[:16]
    )


# =============================================================================
# LOAD REAL DATA
# =============================================================================

def load_real_data():

    print("=" * 100)
    print("LOADING REAL 731 DATA")
    print("=" * 100)

    df = pd.read_excel(
        REAL_FILE
    )

    if "Unnamed: 0" in df.columns:

        df = df.drop(
            columns=[
                "Unnamed: 0"
            ]
        )

    df[
        "package_context"
    ] = (
        df[
            "packageFixed_dev"
        ]
        .apply(normalize_value)
    )

    df[
        "canonical_configuration_id"
    ] = df.apply(
        make_canonical_id,
        axis=1,
    )

    print(
        f"\nRows: "
        f"{len(df):,}"
    )

    print(
        f"Canonical configurations: "
        f"{df['canonical_configuration_id'].nunique():,}"
    )

    return df


# =============================================================================
# BUILD REAL CANONICAL POOL
# =============================================================================

def build_real_pool(
    real_df
):

    records = []

    for (
        config_id,
        group,
    ) in real_df.groupby(
        "canonical_configuration_id"
    ):

        row = group.iloc[0]

        package = (
            row[
                "package_context"
            ]
        )

        frequency = len(
            group
        )

        record = {
            "canonical_configuration_id":
                config_id,

            "package_context":
                package,

            "record_type":
                "OBSERVED",

            "generation_method":
                "REAL_OBSERVED",

            "source_canonical_configuration_id":
                config_id,

            "source_frequency":
                frequency,

            "nearest_real_distance":
                0,

            "num_changed_characteristics":
                0,

            "changed_characteristics":
                "",

            "constraint_valid":
                True,

            "evidence_level":
                "OBSERVED",

            "evidence_score":
                100.0,
        }

        for column in TECHNICAL_COLUMNS:

            record[column] = (
                normalize_value(
                    row[column]
                )
            )

        records.append(
            record
        )

    return pd.DataFrame(
        records
    )


# =============================================================================
# BUILD EMPIRICAL MUTATION FREQUENCY
# =============================================================================

def build_mutation_frequency(
    v4_df
):

    frequency = Counter()

    if len(v4_df) == 0:
        return frequency

    derived = v4_df[
        v4_df[
            "num_changed_characteristics"
        ] > 0
    ]

    for pair in derived[
        "changed_characteristics"
    ].dropna():

        columns = [
            x.strip()
            for x in str(pair).split("|")
            if x.strip()
        ]

        if not columns:
            continue

        key = frozenset(
            columns
        )

        frequency[key] += 1

    return frequency


# =============================================================================
# EVIDENCE SCORE
# =============================================================================

def calculate_evidence_score(
    row,
    mutation_frequency,
):

    score = 0.0

    generation_method = str(
        row.get(
            "generation_method",
            ""
        )
    )

    distance = int(
        row.get(
            "nearest_real_distance",
            99
        )
    )

    changed_count = int(
        row.get(
            "num_changed_characteristics",
            0
        )
    )

    # -------------------------------------------------------------------------
    # Base evidence
    # -------------------------------------------------------------------------

    if (
        generation_method
        == "REAL_OBSERVED"
    ):

        return 100.0

    if (
        generation_method
        == "ONE_CHARACTERISTIC"
    ):

        score += 70.0

    elif (
        generation_method
        == "DEPENDENCY_TWO_CHARACTERISTIC"
    ):

        score += 65.0

    elif (
        generation_method
        == "TWO_CHARACTERISTIC"
    ):

        score += 45.0

    # -------------------------------------------------------------------------
    # Distance
    # -------------------------------------------------------------------------

    if distance == 1:

        score += 20.0

    elif distance == 2:

        score += 5.0

    # -------------------------------------------------------------------------
    # Dependency support
    # -------------------------------------------------------------------------

    changed = [
        x.strip()
        for x in str(
            row.get(
                "changed_characteristics",
                ""
            )
        ).split("|")
        if x.strip()
    ]

    if (
        frozenset(changed)
        in DEPENDENCY_PAIRS
    ):

        score += 15.0

    # -------------------------------------------------------------------------
    # Empirical frequency
    # -------------------------------------------------------------------------

    if changed:

        frequency = mutation_frequency[
            frozenset(changed)
        ]

        if frequency > 0:

            # Capped contribution.
            score += min(
                10.0,
                frequency / 10.0,
            )

    # -------------------------------------------------------------------------
    # Complexity penalty
    # -------------------------------------------------------------------------

    if changed_count >= 2:

        score -= 5.0

    return round(
        max(
            0.0,
            min(
                99.0,
                score,
            ),
        ),
        3,
    )


# =============================================================================
# EVIDENCE LEVEL
# =============================================================================

def evidence_level(
    row
):

    if (
        row["record_type"]
        == "OBSERVED"
    ):

        return "OBSERVED"

    score = float(
        row[
            "evidence_score"
        ]
    )

    if score >= 85:

        return "VERY_STRONG"

    if score >= 70:

        return "STRONG"

    if score >= 55:

        return "MODERATE"

    return "LOW"


# =============================================================================
# BUILD V5 CANDIDATE POOL
# =============================================================================

def build_candidate_pool(
    real_df,
    v4_df,
):

    real_pool = build_real_pool(
        real_df
    )

    real_ids = set(
        real_pool[
            "canonical_configuration_id"
        ]
    )

    if len(v4_df):

        derived = v4_df.copy()

        # Make sure only genuine derived configurations
        # are considered.

        derived = derived[
            ~derived[
                "canonical_configuration_id"
            ].isin(real_ids)
        ].copy()

        derived[
            "record_type"
        ] = "DERIVED"

        derived[
            "source_frequency"
        ] = 0

        derived[
            "constraint_valid"
        ] = True

        mutation_frequency = (
            build_mutation_frequency(
                derived
            )
        )

        derived[
            "evidence_score"
        ] = derived.apply(
            lambda row:
                calculate_evidence_score(
                    row,
                    mutation_frequency,
                ),
            axis=1,
        )

        derived[
            "evidence_level"
        ] = derived.apply(
            evidence_level,
            axis=1,
        )

        # Keep only columns needed by final model.
        derived = derived[
            [
                "canonical_configuration_id",
                "package_context",
                "record_type",
                "generation_method",
                "source_canonical_configuration_id",
                "source_frequency",
                "nearest_real_distance",
                "num_changed_characteristics",
                "changed_characteristics",
                "constraint_valid",
                "evidence_level",
                "evidence_score",
            ]
            + TECHNICAL_COLUMNS
        ]

    else:

        derived = pd.DataFrame(
            columns=real_pool.columns
        )

        mutation_frequency = Counter()

    candidates = pd.concat(
        [
            real_pool,
            derived,
        ],
        ignore_index=True,
    )

    return (
        candidates,
        mutation_frequency,
    )


# =============================================================================
# PACKAGE TARGETS
# =============================================================================

def build_package_targets(
    real_pool,
    target_records,
):

    counts = (
        real_pool[
            "package_context"
        ]
        .value_counts()
        .sort_index()
    )

    probabilities = (
        counts
        / counts.sum()
    )

    raw_targets = (
        probabilities
        * target_records
    )

    targets = (
        raw_targets
        .round()
        .astype(int)
    )

    # Fix rounding difference.
    difference = (
        target_records
        - targets.sum()
    )

    if difference != 0:

        order = (
            probabilities
            .sort_values(
                ascending=False
            )
            .index
            .tolist()
        )

        index = 0

        while difference != 0:

            package = order[
                index % len(order)
            ]

            if difference > 0:

                targets[
                    package
                ] += 1

                difference -= 1

            else:

                if targets[
                    package
                ] > 1:

                    targets[
                        package
                    ] -= 1

                    difference += 1

            index += 1

    return targets


# =============================================================================
# SAMPLING WEIGHT
# =============================================================================

def sampling_weight(
    row
):

    if (
        row["record_type"]
        == "OBSERVED"
    ):

        weight = (
            REAL_BASE_WEIGHT
        )

        # Give configurations that actually occurred more
        # probability, but cap the effect.

        frequency = min(
            int(
                row.get(
                    "source_frequency",
                    1,
                )
            ),
            10,
        )

        weight *= (
            1.0
            + 0.10 * frequency
        )

        return weight

    # -------------------------------------------------------------------------
    # Derived configuration
    # -------------------------------------------------------------------------

    weight = (
        DERIVED_BASE_WEIGHT
    )

    distance = int(
        row[
            "nearest_real_distance"
        ]
    )

    if distance == 1:

        weight *= (
            DISTANCE_1_MULTIPLIER
        )

    else:

        weight *= (
            DISTANCE_2_MULTIPLIER
        )

    method = str(
        row[
            "generation_method"
        ]
    )

    if method == (
        "DEPENDENCY_TWO_CHARACTERISTIC"
    ):

        weight *= (
            DEPENDENCY_MULTIPLIER
        )

    elif method == (
        "ONE_CHARACTERISTIC"
    ):

        weight *= (
            ONE_CHARACTERISTIC_MULTIPLIER
        )

    else:

        weight *= (
            TWO_CHARACTERISTIC_MULTIPLIER
        )

    # Evidence score provides a smooth adjustment.
    weight *= (
        0.5
        + float(
            row[
                "evidence_score"
            ]
        )
        / 100.0
    )

    return max(
        weight,
        0.001,
    )


# =============================================================================
# SAMPLE ONE PACKAGE
# =============================================================================

def sample_package(
    candidates,
    package,
    target,
    rng,
):

    pool = candidates[
        candidates[
            "package_context"
        ] == package
    ].copy()

    if len(pool) == 0:

        return pd.DataFrame()

    pool[
        "_sampling_weight"
    ] = pool.apply(
        sampling_weight,
        axis=1,
    )

    selected = []

    frequencies = Counter()

    # -------------------------------------------------------------------------
    # Weighted sampling with replacement, but capped per configuration.
    # -------------------------------------------------------------------------

    values = pool.index.tolist()

    weights = pool[
        "_sampling_weight"
    ].tolist()

    attempts = 0

    max_attempts = (
        max(
            10000,
            target * 100,
        )
    )

    while (
        len(selected) < target
        and attempts < max_attempts
    ):

        attempts += 1

        index = rng.choices(
            values,
            weights=weights,
            k=1,
        )[0]

        config_id = pool.loc[
            index,
            "canonical_configuration_id"
        ]

        if (
            frequencies[
                config_id
            ]
            >= MAX_CONFIGURATION_FREQUENCY
        ):

            continue

        frequencies[
            config_id
        ] += 1

        selected.append(
            index
        )

    # -------------------------------------------------------------------------
    # If cap prevents reaching target, fill from remaining configurations.
    # -------------------------------------------------------------------------

    if len(selected) < target:

        remaining = []

        for index in values:

            config_id = pool.loc[
                index,
                "canonical_configuration_id"
            ]

            if (
                frequencies[
                    config_id
                ]
                < MAX_CONFIGURATION_FREQUENCY
            ):

                remaining.append(
                    index
                )

        while (
            len(selected) < target
            and remaining
        ):

            index = rng.choice(
                remaining
            )

            config_id = pool.loc[
                index,
                "canonical_configuration_id"
            ]

            frequencies[
                config_id
            ] += 1

            selected.append(
                index
            )

            if (
                frequencies[
                    config_id
                ]
                >= MAX_CONFIGURATION_FREQUENCY
            ):

                remaining.remove(
                    index
                )

    result = pool.loc[
        selected
    ].copy()

    result = result.drop(
        columns=[
            "_sampling_weight"
        ]
    )

    return result.reset_index(
        drop=True
    )


# =============================================================================
# MAIN SAMPLER
# =============================================================================

def main():

    rng = random.Random(
        RANDOM_SEED
    )

    np.random.seed(
        RANDOM_SEED
    )

    print(
        "\n" + "=" * 100
    )

    print(
        "731 SYNTHETIC CONFIGURATION GENERATOR — V5"
    )

    print(
        "=" * 100
    )

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    real_df = load_real_data()

    print(
        "\nLoading V4 feasible configuration space..."
    )

    v4_df = pd.read_csv(
        V4_FILE
    )

    print(
        f"V4 feasible configurations: "
        f"{len(v4_df):,}"
    )

    # -------------------------------------------------------------------------
    # Candidate pool
    # -------------------------------------------------------------------------

    print(
        "\nBuilding evidence-weighted candidate pool..."
    )

    candidates, mutation_frequency = (
        build_candidate_pool(
            real_df,
            v4_df,
        )
    )

    print(
        f"Total candidates: "
        f"{len(candidates):,}"
    )

    print(
        f"Observed: "
        f"{(
            candidates['record_type']
            == 'OBSERVED'
        ).sum():,}"
    )

    print(
        f"Derived: "
        f"{(
            candidates['record_type']
            == 'DERIVED'
        ).sum():,}"
    )

    # -------------------------------------------------------------------------
    # Evidence distribution
    # -------------------------------------------------------------------------

    print(
        "\nEvidence levels:"
    )

    print(
        candidates[
            "evidence_level"
        ]
        .value_counts()
        .to_string()
    )

    # -------------------------------------------------------------------------
    # Package targets
    # -------------------------------------------------------------------------

    real_pool = candidates[
        candidates[
            "record_type"
        ] == "OBSERVED"
    ]

    targets = build_package_targets(
        real_pool,
        TARGET_RECORDS,
    )

    print(
        "\n" + "=" * 100
    )

    print(
        "V5 PACKAGE TARGETS"
    )

    print(
        "=" * 100
    )

    for package, target in (
        targets.items()
    ):

        real_count = int(
            (
                real_pool[
                    "package_context"
                ]
                == package
            ).sum()
        )

        print(
            f"{package:25s}"
            f" real_canonical={real_count:4d}"
            f" target={target:5d}"
            f" ({target / TARGET_RECORDS * 100:6.2f}%)"
        )

    # -------------------------------------------------------------------------
    # Sample packages
    # -------------------------------------------------------------------------

    print(
        "\n" + "=" * 100
    )

    print(
        "SAMPLING V5 CONFIGURATIONS"
    )

    print(
        "=" * 100
    )

    samples = []

    package_stats = []

    for package, target in (
        targets.items()
    ):

        print(
            f"\nPackage: {package}"
        )

        print(
            f"Target:  {target}"
        )

        sampled = sample_package(
            candidates,
            package,
            int(target),
            rng,
        )

        print(
            f"Generated: "
            f"{len(sampled)}"
        )

        if len(sampled):

            samples.append(
                sampled
            )

            package_stats.append(
                {
                    "package":
                        package,

                    "target":
                        int(target),

                    "generated":
                        len(sampled),

                    "observed":
                        int(
                            (
                                sampled[
                                    "record_type"
                                ]
                                == "OBSERVED"
                            ).sum()
                        ),

                    "derived":
                        int(
                            (
                                sampled[
                                    "record_type"
                                ]
                                == "DERIVED"
                            ).sum()
                        ),

                    "distance_1":
                        int(
                            (
                                sampled[
                                    "nearest_real_distance"
                                ]
                                == 1
                            ).sum()
                        ),

                    "distance_2":
                        int(
                            (
                                sampled[
                                    "nearest_real_distance"
                                ]
                                == 2
                            ).sum()
                        ),
                }
            )

    # -------------------------------------------------------------------------
    # Combine
    # -------------------------------------------------------------------------

    if samples:

        result = pd.concat(
            samples,
            ignore_index=True,
        )

    else:

        result = pd.DataFrame()

    # -------------------------------------------------------------------------
    # Add synthetic record ID
    # -------------------------------------------------------------------------

    if len(result):

        result.insert(
            0,
            "synthetic_record_id",
            [
                f"SYN731_{i:06d}"
                for i in range(
                    1,
                    len(result) + 1,
                )
            ],
        )

        # Randomize ordering while preserving reproducibility.

        result = result.sample(
            frac=1.0,
            random_state=RANDOM_SEED,
        ).reset_index(
            drop=True
        )

        # Re-number after shuffle.

        result[
            "synthetic_record_id"
        ] = [
            f"SYN731_{i:06d}"
            for i in range(
                1,
                len(result) + 1,
            )
        ]

    # -------------------------------------------------------------------------
    # Add final evidence metadata
    # -------------------------------------------------------------------------

    if len(result):

        result[
            "synthetic_dataset"
        ] = "731_V5"

        result[
            "technical_configuration_status"
        ] = result.apply(
            lambda row:
                (
                    "REAL_OBSERVED"
                    if row[
                        "record_type"
                    ] == "OBSERVED"
                    else
                    "SYNTHETIC_DERIVED"
                ),
            axis=1,
        )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print(
        "\n" + "=" * 100
    )

    print(
        "V5 GENERATION SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"\nRequested: "
        f"{TARGET_RECORDS:,}"
    )

    print(
        f"Generated: "
        f"{len(result):,}"
    )

    if len(result):

        print(
            f"\nRecord type:"
        )

        print(
            result[
                "record_type"
            ]
            .value_counts()
            .to_string()
        )

        print(
            "\nEvidence level:"
        )

        print(
            result[
                "evidence_level"
            ]
            .value_counts()
            .to_string()
        )

        print(
            "\nNearest-real distance:"
        )

        print(
            result[
                "nearest_real_distance"
            ]
            .value_counts()
            .sort_index()
            .to_string()
        )

        print(
            "\nGeneration method:"
        )

        print(
            result[
                "generation_method"
            ]
            .value_counts()
            .to_string()
        )

        print(
            "\nPackage distribution:"
        )

        package_distribution = (
            result[
                "package_context"
            ]
            .value_counts()
        )

        for package, count in (
            package_distribution.items()
        ):

            print(
                f"{package:25s}"
                f"{count:6d}"
                f" "
                f"{count / len(result) * 100:7.2f}%"
            )

    # =========================================================================
    # SAVE MAIN DATASET
    # =========================================================================

    output_file = (
        OUTPUT_DIR
        / "731_synthetic_configurations_v5.csv"
    )

    result.to_csv(
        output_file,
        index=False,
    )

    # =========================================================================
    # SAVE PACKAGE STATISTICS
    # =========================================================================

    package_stats_df = pd.DataFrame(
        package_stats
    )

    if len(package_stats_df):

        package_stats_df[
            "actual_percentage"
        ] = (
            package_stats_df[
                "generated"
            ]
            / len(result)
            * 100
        )

    package_file = (
        PROCESSED_DIR
        / "731_v5_package_statistics.csv"
    )

    package_stats_df.to_csv(
        package_file,
        index=False,
    )

    # =========================================================================
    # EVIDENCE STATISTICS
    # =========================================================================

    if len(result):

        evidence_stats = (
            result.groupby(
                [
                    "record_type",
                    "evidence_level",
                    "generation_method",
                    "nearest_real_distance",
                ]
            )
            .size()
            .reset_index(
                name="count"
            )
        )

    else:

        evidence_stats = pd.DataFrame()

    evidence_file = (
        PROCESSED_DIR
        / "731_v5_evidence_statistics.csv"
    )

    evidence_stats.to_csv(
        evidence_file,
        index=False,
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    summary_records = [
        {
            "metric":
                "requested_records",
            "value":
                TARGET_RECORDS,
        },
        {
            "metric":
                "generated_records",
            "value":
                len(result),
        },
        {
            "metric":
                "candidate_pool_size",
            "value":
                len(candidates),
        },
        {
            "metric":
                "real_observed_candidates",
            "value":
                int(
                    (
                        candidates[
                            "record_type"
                        ]
                        == "OBSERVED"
                    ).sum()
                ),
        },
        {
            "metric":
                "derived_candidates",
            "value":
                int(
                    (
                        candidates[
                            "record_type"
                        ]
                        == "DERIVED"
                    ).sum()
                ),
        },
    ]

    if len(result):

        summary_records.extend(
            [
                {
                    "metric":
                        "final_observed_records",
                    "value":
                        int(
                            (
                                result[
                                    "record_type"
                                ]
                                == "OBSERVED"
                            ).sum()
                        ),
                },
                {
                    "metric":
                        "final_derived_records",
                    "value":
                        int(
                            (
                                result[
                                    "record_type"
                                ]
                                == "DERIVED"
                            ).sum()
                        ),
                },
                {
                    "metric":
                        "final_distance_1",
                    "value":
                        int(
                            (
                                result[
                                    "nearest_real_distance"
                                ]
                                == 1
                            ).sum()
                        ),
                },
                {
                    "metric":
                        "final_distance_2",
                    "value":
                        int(
                            (
                                result[
                                    "nearest_real_distance"
                                ]
                                == 2
                            ).sum()
                        ),
                },
            ]
        )

    summary_df = pd.DataFrame(
        summary_records
    )

    summary_file = (
        PROCESSED_DIR
        / "731_v5_sampling_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    # =========================================================================
    # FINAL
    # =========================================================================

    print(
        "\n" + "=" * 100
    )

    print(
        "V5 FILES SAVED"
    )

    print(
        "=" * 100
    )

    print(
        f"\nSynthetic configurations:"
        f"\n{output_file}"
    )

    print(
        f"\nPackage statistics:"
        f"\n{package_file}"
    )

    print(
        f"\nEvidence statistics:"
        f"\n{evidence_file}"
    )

    print(
        f"\nSampling summary:"
        f"\n{summary_file}"
    )

    print(
        "\n" + "=" * 100
    )

    print(
        "V5 TECHNICAL CONFIGURATION SAMPLING COMPLETE"
    )

    print(
        "=" * 100
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()