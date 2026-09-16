#!/usr/bin/env python3

"""
731 SYNTHETIC CONFIGURATION GENERATOR — V3
==========================================

Purpose
-------
Generate synthetic 731 technical configurations from complete,
real canonical configurations.

Design principles
-----------------
1. Real 731 configurations are the source of truth.
2. Generation starts from complete observed configurations.
3. Mutations use values observed within the same package.
4. Every candidate is validated by the existing 731 ConstraintEngine.
5. Existing canonical configurations are never counted as novel.
6. Synthetic duplicates are rejected.
7. Full provenance is retained for every generated record.

Output
------
data/synthetic/configurations/731_synthetic_configurations_v3.csv

Diagnostics
-----------
data/processed/731_v3_generation_summary.csv
data/processed/731_v3_package_generation_stats.csv
data/processed/731_v3_rejection_stats.csv
"""


from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

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

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src" / "synthetic"),
)

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src" / "constraints"),
)


from canonical_configuration import (  # noqa: E402
    TECHNICAL_COLUMNS,
    add_canonical_identity,
    canonical_id,
    normalize_value,
    technical_distance,
)


from build_731_constraint_engine import (  # noqa: E402
    ConstraintEngine,
    build_rule_table,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

TARGET_SYNTHETIC_ROWS = 5000

MAX_TOTAL_ATTEMPTS = 300000

MAX_PACKAGE_ATTEMPTS = 75000

RANDOM_SEED = 7312026


# Conservative mutation distribution.
#
# Most synthetic configurations should be close to an observed
# configuration.
MUTATION_WEIGHTS = {
    1: 0.70,
    2: 0.25,
    3: 0.05,
}


# =============================================================================
# NORMALIZATION HELPERS
# =============================================================================

def canonical_package(value) -> str:
    """
    Convert packageFixed_dev to its canonical selected package.
    """

    return normalize_value(value)


def canonical_configuration_from_row(
    row,
) -> Dict[str, str]:
    """
    Extract a complete canonical technical configuration.
    """

    return {
        column: normalize_value(
            row[column]
        )
        for column in TECHNICAL_COLUMNS
    }


def configuration_id(
    configuration: Dict[str, str],
) -> str:
    """
    Use the project's canonical ID implementation.
    """

    return canonical_id(
        configuration,
        TECHNICAL_COLUMNS,
    )


# =============================================================================
# LOAD REAL DATA
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

    # Add the project's canonical identity.
    df = add_canonical_identity(
        df
    )

    df["package_context"] = (
        df["packageFixed_dev"]
        .apply(
            canonical_package
        )
    )

    return df


# =============================================================================
# BUILD CANONICAL SOURCE POOL
# =============================================================================

def build_canonical_source_pool(
    df: pd.DataFrame,
):
    """
    Build exactly one source configuration for each canonical
    configuration ID.

    This removes duplicate source records while retaining
    the original row information.
    """

    canonical_rows = {}

    for row_index, row in df.iterrows():

        configuration = (
            canonical_configuration_from_row(
                row
            )
        )

        config_id = (
            configuration_id(
                configuration
            )
        )

        if config_id not in canonical_rows:

            canonical_rows[config_id] = {
                "canonical_configuration_id":
                    config_id,

                "source_row":
                    row_index,

                "package":
                    row[
                        "package_context"
                    ],

                "configuration":
                    configuration,
            }

    return canonical_rows


# =============================================================================
# PACKAGE EVIDENCE
# =============================================================================

def build_package_evidence(
    df: pd.DataFrame,
):
    """
    Build package-specific empirical value distributions.

    Values are taken only from real configurations belonging
    to that package.
    """

    package_evidence = {}

    for package, package_df in df.groupby(
        "package_context"
    ):

        values_by_characteristic = {}

        for column in TECHNICAL_COLUMNS:

            values = (
                package_df[column]
                .map(normalize_value)
                .dropna()
                .unique()
                .tolist()
            )

            values_by_characteristic[
                column
            ] = sorted(
                set(values)
            )

        canonical_ids = (
            package_df[
                "canonical_configuration_id"
            ]
            .drop_duplicates()
            .tolist()
        )

        package_evidence[package] = {
            "source_records":
                len(package_df),

            "canonical_configurations":
                len(canonical_ids),

            "canonical_ids":
                canonical_ids,

            "values":
                values_by_characteristic,
        }

    return package_evidence


# =============================================================================
# PACKAGE QUOTAS
# =============================================================================

def calculate_package_quotas(
    package_evidence,
    target_rows: int,
):
    """
    Allocate synthetic records according to the proportion of
    canonical configurations represented by each package.

    Largest-remainder allocation guarantees the final quotas
    sum exactly to target_rows.
    """

    packages = list(
        package_evidence.keys()
    )

    total_canonical = sum(
        package_evidence[p][
            "canonical_configurations"
        ]
        for p in packages
    )

    raw = {}

    quotas = {}

    for package in packages:

        canonical_count = (
            package_evidence[
                package
            ][
                "canonical_configurations"
            ]
        )

        raw[package] = (
            canonical_count
            /
            total_canonical
            *
            target_rows
        )

        quotas[package] = int(
            raw[package]
        )

    remaining = (
        target_rows
        -
        sum(quotas.values())
    )

    order = sorted(
        packages,
        key=lambda p:
        raw[p] - quotas[p],
        reverse=True,
    )

    for package in order[
        :remaining
    ]:

        quotas[package] += 1

    return quotas


# =============================================================================
# SOURCE SELECTION
# =============================================================================

def select_source(
    canonical_pool: Dict,
    package: str,
) -> Tuple[str, Dict[str, str], int]:
    """
    Select one canonical real configuration belonging
    to the requested package.
    """

    candidates = [
        item
        for item in canonical_pool.values()
        if item["package"] == package
    ]

    selected = random.choice(
        candidates
    )

    return (
        selected[
            "canonical_configuration_id"
        ],
        dict(
            selected[
                "configuration"
            ]
        ),
        selected[
            "source_row"
        ],
    )


# =============================================================================
# MUTATION COUNT
# =============================================================================

def choose_mutation_count() -> int:

    counts = list(
        MUTATION_WEIGHTS.keys()
    )

    weights = list(
        MUTATION_WEIGHTS.values()
    )

    return random.choices(
        counts,
        weights=weights,
        k=1,
    )[0]


# =============================================================================
# MUTATION
# =============================================================================

def mutate_configuration(
    source_configuration: Dict[str, str],
    package: str,
    package_evidence: Dict,
) -> Tuple[
    Dict[str, str],
    List[str],
]:
    """
    Create a candidate by changing characteristics using
    values observed within the same package.

    The complete source configuration is preserved first.
    """

    candidate = dict(
        source_configuration
    )

    possible_columns = []

    for column in TECHNICAL_COLUMNS:

        current = (
            source_configuration[
                column
            ]
        )

        observed_values = (
            package_evidence[
                "values"
            ].get(
                column,
                [],
            )
        )

        alternatives = [
            value
            for value in observed_values
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
        choose_mutation_count(),
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
            for value in package_evidence[
                "values"
            ][column]
            if value != current
        ]

        if not alternatives:
            continue

        candidate[
            column
        ] = random.choice(
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
# BUILD CONSTRAINT ENGINE
# =============================================================================

def build_engine():

    print(
        "\n" + "=" * 100
    )

    print(
        "BUILDING VALIDATED 731 CONSTRAINT ENGINE"
    )

    print(
        "=" * 100
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
        f"\nConstraint rules: "
        f"{len(rules):,}"
    )

    return engine


# =============================================================================
# DISTANCE
# =============================================================================

def nearest_real_distance(
    candidate: Dict[str, str],
    real_configurations: List[
        Dict[str, str]
    ],
) -> int:

    best = len(
        TECHNICAL_COLUMNS
    )

    for real_configuration in (
        real_configurations
    ):

        distance = technical_distance(
            candidate,
            real_configuration,
            TECHNICAL_COLUMNS,
        )

        if distance < best:

            best = distance

            if best == 0:
                return 0

    return best


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 100)
    print("731 SYNTHETIC CONFIGURATION GENERATOR — V3")
    print("=" * 100)

    random.seed(
        RANDOM_SEED
    )

    # -------------------------------------------------------------------------
    # Load source data.
    # -------------------------------------------------------------------------

    df = load_real_data()

    # -------------------------------------------------------------------------
    # Canonical source pool.
    # -------------------------------------------------------------------------

    print(
        "\nBuilding canonical source pool..."
    )

    canonical_pool = (
        build_canonical_source_pool(
            df
        )
    )

    real_configuration_ids = set(
        canonical_pool.keys()
    )

    print(
        f"Real source records: "
        f"{len(df):,}"
    )

    print(
        f"Real canonical configurations: "
        f"{len(canonical_pool):,}"
    )

    # -------------------------------------------------------------------------
    # Package evidence.
    # -------------------------------------------------------------------------

    print(
        "\nBuilding package-specific empirical evidence..."
    )

    package_evidence = (
        build_package_evidence(
            df
        )
    )

    # -------------------------------------------------------------------------
    # Package quotas.
    # -------------------------------------------------------------------------

    quotas = calculate_package_quotas(
        package_evidence,
        TARGET_SYNTHETIC_ROWS,
    )

    print(
        "\n" + "=" * 100
    )

    print(
        "PACKAGE GENERATION TARGETS"
    )

    print(
        "=" * 100
    )

    quota_records = []

    total_canonical = len(
        canonical_pool
    )

    for package in sorted(
        quotas,
        key=lambda p:
        quotas[p],
        reverse=True,
    ):

        info = package_evidence[
            package
        ]

        canonical_count = (
            info[
                "canonical_configurations"
            ]
        )

        target = quotas[
            package
        ]

        canonical_percentage = (
            canonical_count
            /
            total_canonical
            *
            100
        )

        target_percentage = (
            target
            /
            TARGET_SYNTHETIC_ROWS
            *
            100
        )

        print(
            f"{package:25s}"
            f" canonical={canonical_count:4d}"
            f" target={target:5d}"
            f" ({target_percentage:6.2f}%)"
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
                    canonical_count,

                "canonical_percentage":
                    canonical_percentage,

                "target_synthetic_rows":
                    target,

                "target_percentage":
                    target_percentage,
            }
        )

    # -------------------------------------------------------------------------
    # Engine.
    # -------------------------------------------------------------------------

    engine = build_engine()

    # -------------------------------------------------------------------------
    # Real configurations.
    # -------------------------------------------------------------------------

    real_configurations = [
        item[
            "configuration"
        ]
        for item in canonical_pool.values()
    ]

    # -------------------------------------------------------------------------
    # Generation state.
    # -------------------------------------------------------------------------

    generated_ids = set()

    results = []

    generated_by_package = Counter()

    attempts_by_package = Counter()

    rejection_reasons = Counter()

    total_attempts = 0

    # -------------------------------------------------------------------------
    # Generate package by package.
    # -------------------------------------------------------------------------

    print(
        "\n" + "=" * 100
    )

    print(
        "GENERATING V3 CONFIGURATIONS"
    )

    print(
        "=" * 100
    )

    for package in sorted(
        quotas,
        key=lambda p:
        quotas[p],
        reverse=True,
    ):

        target = quotas[
            package
        ]

        generated = 0

        attempts = 0

        print(
            f"\nPackage: {package}"
        )

        print(
            f"Target:  {target:,}"
        )

        while (
            generated < target
            and attempts < MAX_PACKAGE_ATTEMPTS
            and total_attempts < MAX_TOTAL_ATTEMPTS
        ):

            attempts += 1
            total_attempts += 1

            attempts_by_package[
                package
            ] += 1

            # ---------------------------------------------------------------
            # Select complete canonical source.
            # ---------------------------------------------------------------

            (
                source_id,
                source_configuration,
                source_row,
            ) = select_source(
                canonical_pool,
                package,
            )

            # ---------------------------------------------------------------
            # Mutate complete configuration.
            # ---------------------------------------------------------------

            (
                candidate,
                changed,
            ) = mutate_configuration(
                source_configuration,
                package,
                package_evidence[
                    package
                ],
            )

            if not changed:

                rejection_reasons[
                    "NO_MUTATABLE_CHARACTERISTIC"
                ] += 1

                continue

            # ---------------------------------------------------------------
            # Important:
            #
            # The engine requires the complete source-style candidate.
            # packageFixed_dev is supplied in the raw package representation
            # expected by the engine.
            # ---------------------------------------------------------------

            engine_candidate = dict(
                candidate
            )

            engine_candidate[
                "packageFixed_dev"
            ] = package

            # ----------------------------------------------------------------
            # Preserve contextual fields from the source row where the engine
            # may use them.
            # ----------------------------------------------------------------

            source_row_data = df.loc[
                source_row
            ]

            for column in df.columns:

                if column in engine_candidate:
                    continue

                if column == "package_context":
                    continue

                if column == "canonical_configuration_id":
                    continue

                engine_candidate[
                    column
                ] = normalize_value(
                    source_row_data[
                        column
                    ]
                )

            # ---------------------------------------------------------------
            # Constraint engine.
            # ---------------------------------------------------------------

            validation = engine.validate(
                engine_candidate
            )

            if not validation[
                "valid"
            ]:

                rejection_reasons[
                    "CONSTRAINT_ENGINE"
                ] += 1

                continue

            # ---------------------------------------------------------------
            # Canonical candidate ID.
            # ---------------------------------------------------------------

            candidate_id = (
                configuration_id(
                    candidate
                )
            )

            # ---------------------------------------------------------------
            # Existing real configuration.
            # ---------------------------------------------------------------

            if candidate_id in (
                real_configuration_ids
            ):

                rejection_reasons[
                    "EXISTING_REAL_CONFIGURATION"
                ] += 1

                continue

            # ---------------------------------------------------------------
            # Synthetic duplicate.
            # ---------------------------------------------------------------

            if candidate_id in (
                generated_ids
            ):

                rejection_reasons[
                    "SYNTHETIC_DUPLICATE"
                ] += 1

                continue

            # ---------------------------------------------------------------
            # Calculate novelty.
            # ---------------------------------------------------------------

            distance = (
                nearest_real_distance(
                    candidate,
                    real_configurations,
                )
            )

            if distance == 1:

                novelty_level = (
                    "VERY_CLOSE"
                )

            elif distance == 2:

                novelty_level = (
                    "CLOSE"
                )

            else:

                novelty_level = (
                    "MODERATE"
                )

            # ---------------------------------------------------------------
            # Accept.
            # ---------------------------------------------------------------

            generated_ids.add(
                candidate_id
            )

            generated += 1

            generated_by_package[
                package
            ] += 1

            results.append(
                {
                    "synthetic_id":
                        f"SYN731_V3_{len(results)+1:06d}",

                    "canonical_configuration_id":
                        candidate_id,

                    "package_context":
                        package,

                    "source_canonical_configuration_id":
                        source_id,

                    "source_row":
                        source_row,

                    "generation_method":
                        "COMPLETE_REAL_CONFIGURATION_MUTATION",

                    "num_changed_characteristics":
                        len(changed),

                    "changed_characteristics":
                        "|".join(changed),

                    "nearest_real_distance":
                        distance,

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

                    "matching_rule_rows":
                        "|".join(
                            str(x)
                            for x in validation[
                                "matching_rule_rows"
                            ]
                        ),

                    **candidate,
                }
            )

            if (
                generated % 50 == 0
            ):

                print(
                    f"  Generated "
                    f"{generated:,} / "
                    f"{target:,}"
                    f" | attempts={attempts:,}"
                )

        print(
            f"  FINAL:"
            f" generated={generated:,}"
            f" / target={target:,}"
            f" | attempts={attempts:,}"
        )

    # =========================================================================
    # OUTPUT DATAFRAME
    # =========================================================================

    synthetic = pd.DataFrame(
        results
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print(
        "\n" + "=" * 100
    )

    print(
        "V3 GENERATION SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"\nRequested: "
        f"{TARGET_SYNTHETIC_ROWS:,}"
    )

    print(
        f"Generated: "
        f"{len(synthetic):,}"
    )

    print(
        f"Attempts: "
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
    # PACKAGE STATISTICS
    # =========================================================================

    package_stats = []

    print(
        "\nPackage distribution:"
    )

    for package in sorted(
        quotas,
        key=lambda p:
        quotas[p],
        reverse=True,
    ):

        target = quotas[
            package
        ]

        generated = (
            generated_by_package[
                package
            ]
        )

        attempts = (
            attempts_by_package[
                package
            ]
        )

        target_pct = (
            target
            /
            TARGET_SYNTHETIC_ROWS
            *
            100
        )

        actual_pct = (
            generated
            /
            len(synthetic)
            *
            100
            if len(synthetic)
            else 0
        )

        print(
            f"{package:25s}"
            f" target={target:5d}"
            f" generated={generated:5d}"
            f" target%={target_pct:6.2f}"
            f" actual%={actual_pct:6.2f}"
        )

        package_stats.append(
            {
                "package":
                    package,

                "target":
                    target,

                "generated":
                    generated,

                "shortfall":
                    target - generated,

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
                    target_pct,

                "actual_percentage":
                    actual_pct,

                "difference_percentage_points":
                    actual_pct
                    -
                    target_pct,
            }
        )

    package_stats_df = pd.DataFrame(
        package_stats
    )

    # =========================================================================
    # NOVELTY STATISTICS
    # =========================================================================

    if len(synthetic):

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
        / "731_synthetic_configurations_v3.csv"
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
        / "731_v3_package_generation_stats.csv"
    )

    package_stats_df.to_csv(
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

    rejection_df = pd.DataFrame(
        rejection_records
    )

    if len(rejection_df):

        rejection_df = (
            rejection_df
            .sort_values(
                "count",
                ascending=False,
            )
        )

    rejection_output = (
        PROCESSED_DIR
        / "731_v3_rejection_stats.csv"
    )

    rejection_df.to_csv(
        rejection_output,
        index=False,
    )

    # =========================================================================
    # SAVE OVERALL SUMMARY
    # =========================================================================

    summary = pd.DataFrame(
        [
            {
                "metric":
                    "real_source_records",

                "value":
                    len(df),
            },
            {
                "metric":
                    "real_canonical_configurations",

                "value":
                    len(
                        canonical_pool
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
                    "unique_synthetic_canonical_configurations",

                "value":
                    (
                        synthetic[
                            "canonical_configuration_id"
                        ].nunique()
                        if len(synthetic)
                        else 0
                    ),
            },
            {
                "metric":
                    "constraint_valid_records",

                "value":
                    (
                        synthetic[
                            "constraint_valid"
                        ].sum()
                        if len(synthetic)
                        else 0
                    ),
            },
        ]
    )

    summary_output = (
        PROCESSED_DIR
        / "731_v3_generation_summary.csv"
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
        "V3 FILES SAVED"
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
        "V3 GENERATION COMPLETE"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()