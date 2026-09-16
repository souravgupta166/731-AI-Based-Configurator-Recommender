#!/usr/bin/env python3

"""
731 SYNTHETIC CONFIGURATION GENERATOR — V2
==========================================

Purpose
-------
Generate synthetic 731 technical configurations using:

1. Real 731 source data
2. Canonical technical configuration identities
3. Package distribution based on canonical technical diversity
4. Empirical values observed within each package
5. Controlled mutations of real configurations
6. Existing validated 731 ConstraintEngine
7. Canonical duplicate detection
8. Novelty scoring
9. Evidence-aware generation statistics

IMPORTANT
---------
This generator does NOT invent arbitrary technical values.

All mutation values originate from values observed in the
real 731 dataset.

The existing 731 ConstraintEngine remains the final hard
technical validation gate.

V2 differences from V1
-----------------------
V1:
    raw-record package sampling
    -> mutation
    -> pairwise filtering
    -> constraint engine

V2:
    canonical package distribution
    -> package quota
    -> empirical source configuration
    -> controlled mutation
    -> ConstraintEngine
    -> canonical identity
    -> novelty classification

Output
------
data/synthetic/configurations/731_synthetic_configurations_v2.csv

Additional diagnostics
----------------------
data/processed/731_v2_generation_summary.csv
data/processed/731_v2_package_generation_stats.csv
data/processed/731_v2_rejection_stats.csv
"""


from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "x731_allgemein_Rev24.xlsx"
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
# IMPORT PROJECT MODULES
# =============================================================================

SYNTHETIC_DIR = (
    PROJECT_ROOT
    / "src"
    / "synthetic"
)

CONSTRAINT_DIR = (
    PROJECT_ROOT
    / "src"
    / "constraints"
)

sys.path.insert(
    0,
    str(SYNTHETIC_DIR),
)

sys.path.insert(
    0,
    str(CONSTRAINT_DIR),
)


from canonical_configuration import (  # noqa: E402
    TECHNICAL_COLUMNS,
    add_canonical_identity,
    canonical_id,
    canonical_string,
    normalize_value,
)


from build_731_constraint_engine import (  # noqa: E402
    ConstraintEngine,
    build_rule_table,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

TARGET_SYNTHETIC_ROWS = 5000

MAX_ATTEMPTS = 250000

RANDOM_SEED = 7312026

# Mutation distribution.
#
# This deliberately favours conservative mutations because the real
# configuration space is highly structured.
MUTATION_COUNT_WEIGHTS = {
    1: 0.70,
    2: 0.20,
    3: 0.08,
    4: 0.02,
}


# Maximum number of attempts for a package before we consider
# its quota difficult to satisfy.
PACKAGE_MAX_ATTEMPTS = 50000


# =============================================================================
# HELPERS
# =============================================================================

def package_from_value(value) -> str:
    """
    Convert raw packageFixed_dev values to canonical package names.
    """

    return normalize_value(value)


def choose_weighted_mutation_count() -> int:
    """
    Select the number of characteristics to mutate.
    """

    counts = list(
        MUTATION_COUNT_WEIGHTS.keys()
    )

    weights = list(
        MUTATION_COUNT_WEIGHTS.values()
    )

    return random.choices(
        counts,
        weights=weights,
        k=1,
    )[0]


def make_configuration_key(
    configuration: Dict[str, str],
) -> str:
    """
    Canonical technical identity.
    """

    return canonical_string(
        configuration,
        TECHNICAL_COLUMNS,
    )


def make_configuration_id(
    configuration: Dict[str, str],
) -> str:
    """
    Deterministic canonical configuration ID.
    """

    return canonical_id(
        configuration,
        TECHNICAL_COLUMNS,
    )


# =============================================================================
# LOAD DATA
# =============================================================================

def load_real_data() -> pd.DataFrame:

    print("=" * 100)
    print("LOADING REAL 731 DATA")
    print("=" * 100)

    df = pd.read_excel(
        REAL_FILE
    )

    if "Unnamed: 0" in df.columns:

        df = df.drop(
            columns=["Unnamed: 0"]
        )

    print(
        f"\nRows:    {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns):,}"
    )

    # -------------------------------------------------------------------------
    # Canonicalize technical columns.
    # -------------------------------------------------------------------------

    df = add_canonical_identity(
        df,
        TECHNICAL_COLUMNS,
    )

    # -------------------------------------------------------------------------
    # Canonical package.
    # -------------------------------------------------------------------------

    df["package_context"] = (
        df["packageFixed_dev"]
        .apply(
            package_from_value
        )
    )

    return df


# =============================================================================
# BUILD PACKAGE EVIDENCE
# =============================================================================

def build_package_evidence(
    df: pd.DataFrame,
):
    """
    Build package-level empirical evidence.

    Each package gets:
        - source record count
        - canonical configuration count
        - canonical diversity
        - source configurations
        - observed values per characteristic
    """

    package_info = {}

    for package, package_df in df.groupby(
        "package_context"
    ):

        configurations = (
            package_df[
                "canonical_configuration_id"
            ]
            .drop_duplicates()
            .tolist()
        )

        observed_values = {}

        for column in TECHNICAL_COLUMNS:

            values = (
                package_df[column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            observed_values[column] = sorted(
                set(values)
            )

        package_info[package] = {
            "source_records":
                len(package_df),

            "canonical_configurations":
                len(configurations),

            "configurations":
                configurations,

            "observed_values":
                observed_values,

            "data":
                package_df.copy(),
        }

    return package_info


# =============================================================================
# PACKAGE QUOTAS
# =============================================================================

def calculate_package_quotas(
    package_info,
    target_rows: int,
):
    """
    Calculate synthetic package quotas from canonical technical diversity.

    This is intentionally different from V1.

    Example:

        x731
        240 / 412 canonical configurations

    receives approximately:

        240 / 412 * target_rows
    """

    packages = list(
        package_info.keys()
    )

    total_canonical = sum(
        package_info[p]["canonical_configurations"]
        for p in packages
    )

    raw_targets = {}

    for package in packages:

        diversity = package_info[
            package
        ][
            "canonical_configurations"
        ]

        raw_targets[package] = (
            diversity
            /
            total_canonical
            *
            target_rows
        )

    # -------------------------------------------------------------------------
    # Largest remainder allocation.
    # -------------------------------------------------------------------------

    quotas = {
        package: int(
            raw_targets[package]
        )
        for package in packages
    }

    remaining = (
        target_rows
        -
        sum(quotas.values())
    )

    remainders = sorted(
        packages,
        key=lambda p:
        raw_targets[p]
        -
        quotas[p],
        reverse=True,
    )

    for package in remainders[
        :remaining
    ]:

        quotas[package] += 1

    return quotas


# =============================================================================
# SOURCE CONFIGURATION SELECTION
# =============================================================================

def select_source_configuration(
    package_df: pd.DataFrame,
) -> Dict[str, str]:
    """
    Select one real canonical configuration from a package.
    """

    row = package_df.sample(
        n=1,
        random_state=random.randint(
            0,
            2_000_000_000,
        ),
    ).iloc[0]

    configuration = {}

    for column in TECHNICAL_COLUMNS:

        configuration[column] = (
            normalize_value(
                row[column]
            )
        )

    return configuration


# =============================================================================
# MUTATION
# =============================================================================

def mutate_configuration(
    source_configuration: Dict[str, str],
    package_info: Dict,
) -> Tuple[
    Dict[str, str],
    List[str],
]:
    """
    Mutate a real configuration using only values observed
    within the same package.
    """

    candidate = dict(
        source_configuration
    )

    mutation_count = (
        choose_weighted_mutation_count()
    )

    possible_columns = []

    for column in TECHNICAL_COLUMNS:

        values = package_info[
            "observed_values"
        ].get(
            column,
            [],
        )

        if len(values) <= 1:
            continue

        current = (
            source_configuration.get(
                column,
                "NOVALUE",
            )
        )

        alternatives = [
            value
            for value in values
            if value != current
        ]

        if alternatives:
            possible_columns.append(
                column
            )

    if not possible_columns:
        return (
            candidate,
            [],
        )

    mutation_count = min(
        mutation_count,
        len(possible_columns),
    )

    selected_columns = random.sample(
        possible_columns,
        mutation_count,
    )

    changed = []

    for column in selected_columns:

        current = candidate[
            column
        ]

        alternatives = [
            value
            for value in package_info[
                "observed_values"
            ][column]
            if value != current
        ]

        if not alternatives:
            continue

        candidate[column] = random.choice(
            alternatives
        )

        changed.append(
            column
        )

    return (
        candidate,
        changed,
    )


# =============================================================================
# BUILD REAL CONFIGURATION INDEX
# =============================================================================

def build_real_configuration_index(
    df: pd.DataFrame,
):
    """
    Map canonical IDs to their package/source information.
    """

    index = {}

    for _, row in df.iterrows():

        configuration = {
            column:
            normalize_value(
                row[column]
            )
            for column in TECHNICAL_COLUMNS
        }

        config_id = (
            make_configuration_id(
                configuration
            )
        )

        if config_id not in index:

            index[config_id] = {
                "configuration":
                    configuration,

                "package":
                    row[
                        "package_context"
                    ],
            }

    return index


# =============================================================================
# NOVELTY DISTANCE
# =============================================================================

def calculate_nearest_real_distance(
    candidate: Dict[str, str],
    real_configurations: List[
        Dict[str, str]
    ],
) -> int:
    """
    Calculate Hamming distance from candidate to closest
    observed real configuration.
    """

    minimum = len(
        TECHNICAL_COLUMNS
    )

    for real_configuration in real_configurations:

        distance = sum(
            candidate[column]
            != real_configuration[column]
            for column in TECHNICAL_COLUMNS
        )

        if distance < minimum:

            minimum = distance

            if minimum == 0:
                break

    return minimum


# =============================================================================
# BUILD CONSTRAINT ENGINE
# =============================================================================

def build_constraint_engine():

    print(
        "\nBuilding validated 731 ConstraintEngine..."
    )

    raw_df = pd.read_excel(
        REAL_FILE
    )

    if "Unnamed: 0" in raw_df.columns:

        raw_df = raw_df.drop(
            columns=["Unnamed: 0"]
        )

    rules = build_rule_table(
        raw_df
    )

    engine = ConstraintEngine(
        rules
    )

    print(
        f"Constraint rules: "
        f"{len(rules):,}"
    )

    return engine


# =============================================================================
# MAIN GENERATION
# =============================================================================

def main():

    print("=" * 100)
    print("731 SYNTHETIC CONFIGURATION GENERATOR — V2")
    print("=" * 100)

    random.seed(
        RANDOM_SEED
    )

    # -------------------------------------------------------------------------
    # Load real data.
    # -------------------------------------------------------------------------

    df = load_real_data()

    # -------------------------------------------------------------------------
    # Package evidence.
    # -------------------------------------------------------------------------

    print(
        "\nBuilding package-level empirical evidence..."
    )

    package_info = (
        build_package_evidence(
            df
        )
    )

    # -------------------------------------------------------------------------
    # Package quotas.
    # -------------------------------------------------------------------------

    quotas = calculate_package_quotas(
        package_info,
        TARGET_SYNTHETIC_ROWS,
    )

    print(
        "\n" + "=" * 100
    )

    print(
        "V2 PACKAGE TARGETS"
    )

    print(
        "=" * 100
    )

    quota_records = []

    for package in sorted(
        quotas,
        key=lambda p:
        quotas[p],
        reverse=True,
    ):

        info = package_info[
            package
        ]

        percentage = (
            quotas[package]
            /
            TARGET_SYNTHETIC_ROWS
            *
            100
        )

        print(
            f"{package:25s}"
            f" canonical={info['canonical_configurations']:4d}"
            f" source={info['source_records']:4d}"
            f" target={quotas[package]:5d}"
            f" ({percentage:6.2f}%)"
        )

        quota_records.append(
            {
                "package":
                    package,

                "source_records":
                    info[
                        "source_records"
                    ],

                "canonical_configurations":
                    info[
                        "canonical_configurations"
                    ],

                "source_record_percentage":
                    info[
                        "source_records"
                    ]
                    /
                    len(df)
                    *
                    100,

                "canonical_percentage":
                    info[
                        "canonical_configurations"
                    ]
                    /
                    df[
                        "canonical_configuration_id"
                    ].nunique()
                    *
                    100,

                "target_synthetic_rows":
                    quotas[package],
            }
        )

    quota_df = pd.DataFrame(
        quota_records
    )

    quota_df.to_csv(
        PROCESSED_DIR
        / "731_v2_package_generation_stats.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Constraint engine.
    # -------------------------------------------------------------------------

    engine = (
        build_constraint_engine()
    )

    # -------------------------------------------------------------------------
    # Real canonical index.
    # -------------------------------------------------------------------------

    real_index = (
        build_real_configuration_index(
            df
        )
    )

    real_configurations = [
        record[
            "configuration"
        ]
        for record in real_index.values()
    ]

    real_configuration_ids = set(
        real_index.keys()
    )

    print(
        f"\nReal canonical configurations: "
        f"{len(real_configuration_ids):,}"
    )

    # -------------------------------------------------------------------------
    # Generation state.
    # -------------------------------------------------------------------------

    results = []

    generated_by_package = Counter()

    attempts_by_package = Counter()

    rejected_by_package = Counter()

    rejection_reasons = Counter()

    generated_configuration_ids = set()

    # We track the number of attempts made per package.
    package_attempts = Counter()

    # -------------------------------------------------------------------------
    # Main quota-controlled loop.
    # -------------------------------------------------------------------------

    print(
        "\n" + "=" * 100
    )

    print(
        "GENERATING V2 CONFIGURATIONS"
    )

    print(
        "=" * 100
    )

    total_attempts = 0

    # -------------------------------------------------------------------------
    # Process each package independently.
    # -------------------------------------------------------------------------

    for package in sorted(
        quotas,
        key=lambda p:
        quotas[p],
        reverse=True,
    ):

        target = quotas[
            package
        ]

        if target <= 0:
            continue

        print(
            f"\nPackage: {package}"
        )

        print(
            f"Target:  {target:,}"
        )

        package_df = (
            package_info[
                package
            ][
                "data"
            ]
        )

        package_generated = 0

        local_attempts = 0

        # ------------------------------------------------------------
        # Generate until package quota is reached.
        # ------------------------------------------------------------

        while (
            package_generated < target
            and local_attempts
            < PACKAGE_MAX_ATTEMPTS
            and total_attempts
            < MAX_ATTEMPTS
        ):

            local_attempts += 1
            total_attempts += 1

            attempts_by_package[
                package
            ] += 1

            # --------------------------------------------------------
            # Select source configuration.
            # --------------------------------------------------------

            source_configuration = (
                select_source_configuration(
                    package_df
                )
            )

            source_id = (
                make_configuration_id(
                    source_configuration
                )
            )

            # --------------------------------------------------------
            # Mutate.
            # --------------------------------------------------------

            (
                candidate,
                changed,
            ) = mutate_configuration(
                source_configuration,
                package_info[
                    package
                ],
            )

            if not changed:

                rejected_by_package[
                    package
                ] += 1

                rejection_reasons[
                    "NO_MUTATABLE_CHARACTERISTICS"
                ] += 1

                continue

            # --------------------------------------------------------
            # Add package context for ConstraintEngine.
            # --------------------------------------------------------

            engine_candidate = dict(
                candidate
            )

            engine_candidate[
                "packageFixed_dev"
            ] = package

            # --------------------------------------------------------
            # Hard ConstraintEngine.
            # --------------------------------------------------------

            validation = (
                engine.validate(
                    engine_candidate
                )
            )

            if not validation[
                "valid"
            ]:

                rejected_by_package[
                    package
                ] += 1

                rejection_reasons[
                    "CONSTRAINT_ENGINE"
                ] += 1

                continue

            # --------------------------------------------------------
            # Canonical identity.
            # --------------------------------------------------------

            candidate_id = (
                make_configuration_id(
                    candidate
                )
            )

            # --------------------------------------------------------
            # Exact synthetic duplicate.
            # --------------------------------------------------------

            if candidate_id in (
                generated_configuration_ids
            ):

                rejected_by_package[
                    package
                ] += 1

                rejection_reasons[
                    "SYNTHETIC_DUPLICATE"
                ] += 1

                continue

            # --------------------------------------------------------
            # Novelty relative to real data.
            # --------------------------------------------------------

            if candidate_id in (
                real_configuration_ids
            ):

                novelty_type = (
                    "EXISTING_CONFIGURATION"
                )

                novelty_level = (
                    "NONE"
                )

            else:

                novelty_type = (
                    "NOVEL_CONFIGURATION"
                )

                novelty_distance = (
                    calculate_nearest_real_distance(
                        candidate,
                        real_configurations,
                    )
                )

                if novelty_distance <= 1:

                    novelty_level = (
                        "CONSERVATIVE"
                    )

                elif novelty_distance == 2:

                    novelty_level = (
                        "MODERATE"
                    )

                else:

                    novelty_level = (
                        "HIGHER_NOVELTY"
                    )

            # --------------------------------------------------------
            # Distance.
            # --------------------------------------------------------

            nearest_distance = (
                0
                if candidate_id
                in real_configuration_ids
                else calculate_nearest_real_distance(
                    candidate,
                    real_configurations,
                )
            )

            # --------------------------------------------------------
            # Accept.
            # --------------------------------------------------------

            generated_configuration_ids.add(
                candidate_id
            )

            package_generated += 1

            generated_by_package[
                package
            ] += 1

            results.append(
                {
                    "synthetic_id":
                        f"SYN731_V2_{len(results)+1:06d}",

                    "canonical_configuration_id":
                        candidate_id,

                    "package_context":
                        package,

                    "source_configuration_id":
                        source_id,

                    "generation_method":
                        "EMPIRICAL_MUTATION",

                    "num_changed_characteristics":
                        len(changed),

                    "changed_characteristics":
                        "|".join(changed),

                    "nearest_real_distance":
                        nearest_distance,

                    "novelty_type":
                        novelty_type,

                    "novelty_level":
                        novelty_level,

                    "constraint_valid":
                        True,

                    "matching_rule_count":
                        len(
                            validation[
                                "matching_rule_rows"
                            ]
                        ),

                    "source_match_rows":
                        "|".join(
                            str(x)
                            for x in validation[
                                "matching_rule_rows"
                            ]
                        ),

                    **candidate,
                }
            )

            # --------------------------------------------------------
            # Progress.
            # --------------------------------------------------------

            if (
                package_generated % 50
                == 0
            ):

                print(
                    f"  Generated "
                    f"{package_generated:,} / "
                    f"{target:,}"
                    f" | attempts={local_attempts:,}"
                )

        print(
            f"  FINAL: generated="
            f"{package_generated:,}"
            f" / target={target:,}"
            f" | attempts={local_attempts:,}"
        )

    # =========================================================================
    # BUILD DATAFRAME
    # =========================================================================

    synthetic = pd.DataFrame(
        results
    )

    # =========================================================================
    # GENERATION SUMMARY
    # =========================================================================

    print(
        "\n" + "=" * 100
    )

    print(
        "V2 GENERATION SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"\nRequested:       "
        f"{TARGET_SYNTHETIC_ROWS:,}"
    )

    print(
        f"Generated:       "
        f"{len(synthetic):,}"
    )

    print(
        f"Attempts:        "
        f"{total_attempts:,}"
    )

    acceptance_rate = (
        len(synthetic)
        /
        total_attempts
        *
        100
        if total_attempts
        else 0
    )

    print(
        f"Acceptance rate: "
        f"{acceptance_rate:.2f}%"
    )

    # =========================================================================
    # PACKAGE DISTRIBUTION
    # =========================================================================

    print(
        "\nPackage distribution:"
    )

    package_summary_records = []

    for package in sorted(
        quotas,
        key=lambda p:
        quotas[p],
        reverse=True,
    ):

        generated = (
            generated_by_package[
                package
            ]
        )

        target = quotas[
            package
        ]

        attempts = (
            attempts_by_package[
                package
            ]
        )

        percentage = (
            generated
            /
            len(synthetic)
            *
            100
            if len(synthetic)
            else 0
        )

        target_percentage = (
            target
            /
            TARGET_SYNTHETIC_ROWS
            *
            100
        )

        package_summary_records.append(
            {
                "package":
                    package,

                "target":
                    target,

                "generated":
                    generated,

                "shortfall":
                    max(
                        target
                        -
                        generated,
                        0,
                    ),

                "attempts":
                    attempts,

                "acceptance_rate_percent":
                    (
                        generated
                        /
                        attempts
                        *
                        100
                        if attempts
                        else 0
                    ),

                "target_percentage":
                    target_percentage,

                "actual_percentage":
                    percentage,

                "difference_percentage_points":
                    percentage
                    -
                    target_percentage,
            }
        )

        print(
            f"{package:25s}"
            f" target={target:5d}"
            f" generated={generated:5d}"
            f" actual={percentage:6.2f}%"
            f" target={target_percentage:6.2f}%"
        )

    package_summary = pd.DataFrame(
        package_summary_records
    )

    # =========================================================================
    # NOVELTY SUMMARY
    # =========================================================================

    print(
        "\nNovelty:"
    )

    if len(synthetic):

        print(
            synthetic[
                "novelty_type"
            ]
            .value_counts()
            .to_string()
        )

        print(
            "\nNovelty level:"
        )

        print(
            synthetic[
                "novelty_level"
            ]
            .value_counts()
            .to_string()
        )

        print(
            "\nNearest-real distance:"
        )

        print(
            synthetic[
                "nearest_real_distance"
            ]
            .value_counts()
            .sort_index()
            .to_string()
        )

        print(
            "\nChanged characteristics:"
        )

        print(
            synthetic[
                "num_changed_characteristics"
            ]
            .value_counts()
            .sort_index()
            .to_string()
        )

    # =========================================================================
    # SAVE SYNTHETIC DATA
    # =========================================================================

    output_file = (
        OUTPUT_DIR
        / "731_synthetic_configurations_v2.csv"
    )

    synthetic.to_csv(
        output_file,
        index=False,
    )

    # =========================================================================
    # SAVE PACKAGE STATISTICS
    # =========================================================================

    package_output = (
        PROCESSED_DIR
        / "731_v2_package_generation_stats.csv"
    )

    package_summary.to_csv(
        package_output,
        index=False,
    )

    # =========================================================================
    # SAVE REJECTION STATISTICS
    # =========================================================================

    rejection_records = []

    for reason, count in (
        rejection_reasons.items()
    ):

        rejection_records.append(
            {
                "rejection_reason":
                    reason,

                "count":
                    count,

                "percentage_of_attempts":
                    (
                        count
                        /
                        total_attempts
                        *
                        100
                        if total_attempts
                        else 0
                    ),
            }
        )

    rejection_df = (
        pd.DataFrame(
            rejection_records
        )
        .sort_values(
            "count",
            ascending=False,
        )
    )

    rejection_output = (
        PROCESSED_DIR
        / "731_v2_rejection_stats.csv"
    )

    rejection_df.to_csv(
        rejection_output,
        index=False,
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    summary = pd.DataFrame(
        [
            {
                "metric":
                    "source_records",

                "value":
                    len(df),
            },
            {
                "metric":
                    "real_canonical_configurations",

                "value":
                    len(
                        real_configuration_ids
                    ),
            },
            {
                "metric":
                    "target_synthetic_records",

                "value":
                    TARGET_SYNTHETIC_ROWS,
            },
            {
                "metric":
                    "generated_synthetic_records",

                "value":
                    len(synthetic),
            },
            {
                "metric":
                    "generation_attempts",

                "value":
                    total_attempts,
            },
            {
                "metric":
                    "acceptance_rate_percent",

                "value":
                    acceptance_rate,
            },
            {
                "metric":
                    "synthetic_unique_canonical_ids",

                "value":
                    synthetic[
                        "canonical_configuration_id"
                    ].nunique()
                    if len(synthetic)
                    else 0,
            },
            {
                "metric":
                    "existing_configuration_records",

                "value":
                    (
                        synthetic[
                            "novelty_type"
                        ]
                        ==
                        "EXISTING_CONFIGURATION"
                    ).sum()
                    if len(synthetic)
                    else 0,
            },
            {
                "metric":
                    "novel_configuration_records",

                "value":
                    (
                        synthetic[
                            "novelty_type"
                        ]
                        ==
                        "NOVEL_CONFIGURATION"
                    ).sum()
                    if len(synthetic)
                    else 0,
            },
        ]
    )

    summary_output = (
        PROCESSED_DIR
        / "731_v2_generation_summary.csv"
    )

    summary.to_csv(
        summary_output,
        index=False,
    )

    # =========================================================================
    # FINAL
    # =========================================================================

    print(
        "\n" + "=" * 100
    )

    print(
        "V2 FILES SAVED"
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
        f"\n{package_output}"
    )

    print(
        f"\nRejection statistics:"
        f"\n{rejection_output}"
    )

    print(
        f"\nGeneration summary:"
        f"\n{summary_output}"
    )

    print(
        "\n" + "=" * 100
    )

    print(
        "V2 GENERATION COMPLETE"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()