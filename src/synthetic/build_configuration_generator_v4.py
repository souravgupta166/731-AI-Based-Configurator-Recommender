#!/usr/bin/env python3

"""
731 SYNTHETIC CONFIGURATION GENERATOR — V4
==========================================

V4 strategy
-----------
Discover the feasible technical configuration space rather than
randomly generating configurations.

Starting point:
    412 real canonical configurations

Candidate generation:
    1. One-characteristic mutations
    2. Two-characteristic mutations
    3. Dependency-aware mutations
    4. Same-package observed values only

Every candidate is checked against the real 731 ConstraintEngine.

Important:
    Existing real canonical configurations are retained as observed
    configurations but are NOT classified as synthetic.

Output:
    data/synthetic/configurations/731_feasible_configuration_space_v4.csv
    data/processed/731_v4_space_summary.csv
    data/processed/731_v4_package_space_stats.csv
    data/processed/731_v4_rejection_stats.csv
"""

from __future__ import annotations

import itertools
import random
import sys
from collections import Counter
from pathlib import Path

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

RANDOM_SEED = 7312026

MAX_TWO_WAY_CANDIDATES_PER_SOURCE = 5000

# Characteristics strongly supported by our previous dependency analysis.
DEPENDENCY_PAIRS = [
    (
        "stromEingaenge",
        "temperaturEingaenge",
    ),
    (
        "explosionApproval",
        "certification",
    ),
    (
        "stromSchaltbar",
        "stromEingaenge",
    ),
    (
        "stromSchaltbar",
        "temperaturEingaenge",
    ),
    (
        "powerSupply",
        "certification",
    ),
    (
        "housing",
        "powerSupply",
    ),
    (
        "binaerDigitalOpenColl_MN",
        "binaerOpenColl_MP",
    ),
    (
        "protectionArea",
        "certification",
    ),
    (
        "explosionApproval",
        "protectionArea",
    ),
    (
        "dataInterface",
        "certification",
    ),
    (
        "dataInterface",
        "explosionApproval",
    ),
]


# =============================================================================
# NORMALIZATION
# =============================================================================

def package_value(value):
    return normalize_value(value)


def canonical_configuration(row):
    return {
        column: normalize_value(row[column])
        for column in TECHNICAL_COLUMNS
    }


# =============================================================================
# LOAD DATA
# =============================================================================

def load_data():

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

    df = add_canonical_identity(
        df
    )

    df["package_context"] = (
        df["packageFixed_dev"]
        .apply(normalize_value)
    )

    print(
        f"\nRows:    {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns):,}"
    )

    return df


# =============================================================================
# CANONICAL SOURCE POOL
# =============================================================================

def build_source_pool(df):

    pool = {}

    for index, row in df.iterrows():

        configuration = (
            canonical_configuration(row)
        )

        config_id = canonical_id(
            configuration,
            TECHNICAL_COLUMNS,
        )

        if config_id not in pool:

            pool[config_id] = {
                "configuration":
                    configuration,

                "package":
                    row[
                        "package_context"
                    ],

                "source_row":
                    index,
            }

    return pool


# =============================================================================
# PACKAGE VALUE SPACE
# =============================================================================

def build_package_values(df):

    result = {}

    for package, group in (
        df.groupby("package_context")
    ):

        result[package] = {}

        for column in TECHNICAL_COLUMNS:

            values = (
                group[column]
                .map(normalize_value)
                .drop_duplicates()
                .tolist()
            )

            result[package][column] = (
                sorted(
                    set(values)
                )
            )

    return result


# =============================================================================
# ENGINE
# =============================================================================

def build_engine():

    print(
        "\n" + "=" * 100
    )

    print(
        "BUILDING VALIDATED CONSTRAINT ENGINE"
    )

    print(
        "=" * 100
    )

    raw = pd.read_excel(
        REAL_FILE
    )

    if "Unnamed: 0" in raw.columns:

        raw = raw.drop(
            columns=["Unnamed: 0"]
        )

    rules = build_rule_table(
        raw
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
# ENGINE CANDIDATE
# =============================================================================

def make_engine_candidate(
    configuration,
    package,
    source_row_data,
):
    """
    Reconstruct a complete candidate in the representation
    expected by the constraint engine.
    """

    candidate = dict(
        configuration
    )

    candidate[
        "packageFixed_dev"
    ] = package

    # Preserve contextual / administrative source fields.
    for column in source_row_data.index:

        if column in candidate:
            continue

        if column in {
            "package_context",
            "canonical_configuration_id",
        }:
            continue

        candidate[
            column
        ] = normalize_value(
            source_row_data[
                column
            ]
        )

    return candidate


# =============================================================================
# VALIDATE
# =============================================================================

def validate_candidate(
    engine,
    configuration,
    package,
    source_row_data,
):

    engine_candidate = (
        make_engine_candidate(
            configuration,
            package,
            source_row_data,
        )
    )

    result = engine.validate(
        engine_candidate
    )

    return result


# =============================================================================
# ONE-CHARACTERISTIC SPACE
# =============================================================================

def generate_one_mutations(
    source_configuration,
    package,
    package_values,
):

    for column in TECHNICAL_COLUMNS:

        current = (
            source_configuration[
                column
            ]
        )

        values = package_values[
            package
        ][column]

        for value in values:

            if value == current:
                continue

            candidate = dict(
                source_configuration
            )

            candidate[
                column
            ] = value

            yield (
                candidate,
                [column],
                "ONE_CHARACTERISTIC",
            )


# =============================================================================
# TWO-CHARACTERISTIC SPACE
# =============================================================================

def generate_two_mutations(
    source_configuration,
    package,
    package_values,
    dependency_pairs,
):

    emitted = 0

    # -------------------------------------------------------------------------
    # First: dependency-aware pairs.
    # -------------------------------------------------------------------------

    for column_a, column_b in (
        dependency_pairs
    ):

        if (
            column_a not in TECHNICAL_COLUMNS
            or
            column_b not in TECHNICAL_COLUMNS
        ):
            continue

        values_a = package_values[
            package
        ].get(
            column_a,
            [],
        )

        values_b = package_values[
            package
        ].get(
            column_b,
            [],
        )

        current_a = (
            source_configuration[
                column_a
            ]
        )

        current_b = (
            source_configuration[
                column_b
            ]
        )

        alternatives_a = [
            value
            for value in values_a
            if value != current_a
        ]

        alternatives_b = [
            value
            for value in values_b
            if value != current_b
        ]

        for value_a in alternatives_a:

            for value_b in alternatives_b:

                candidate = dict(
                    source_configuration
                )

                candidate[
                    column_a
                ] = value_a

                candidate[
                    column_b
                ] = value_b

                yield (
                    candidate,
                    [
                        column_a,
                        column_b,
                    ],
                    "DEPENDENCY_TWO_CHARACTERISTIC",
                )

                emitted += 1

                if emitted >= (
                    MAX_TWO_WAY_CANDIDATES_PER_SOURCE
                ):
                    return

    # -------------------------------------------------------------------------
    # Then: general two-characteristic combinations.
    # -------------------------------------------------------------------------

    mutable = []

    for column in TECHNICAL_COLUMNS:

        current = (
            source_configuration[
                column
            ]
        )

        alternatives = [
            value
            for value in package_values[
                package
            ][column]
            if value != current
        ]

        if alternatives:

            mutable.append(
                (
                    column,
                    alternatives,
                )
            )

    for (
        (column_a, values_a),
        (column_b, values_b),
    ) in itertools.combinations(
        mutable,
        2,
    ):

        for value_a in values_a:

            for value_b in values_b:

                candidate = dict(
                    source_configuration
                )

                candidate[
                    column_a
                ] = value_a

                candidate[
                    column_b
                ] = value_b

                yield (
                    candidate,
                    [
                        column_a,
                        column_b,
                    ],
                    "TWO_CHARACTERISTIC",
                )

                emitted += 1

                if emitted >= (
                    MAX_TWO_WAY_CANDIDATES_PER_SOURCE
                ):
                    return


# =============================================================================
# DISTANCE
# =============================================================================

def nearest_real_distance(
    configuration,
    real_configurations,
):

    best = len(
        TECHNICAL_COLUMNS
    )

    for real in real_configurations:

        distance = technical_distance(
            configuration,
            real,
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

    random.seed(
        RANDOM_SEED
    )

    df = load_data()

    source_pool = (
        build_source_pool(
            df
        )
    )

    real_ids = set(
        source_pool.keys()
    )

    real_configurations = [
        item["configuration"]
        for item in source_pool.values()
    ]

    print(
        "\nReal canonical configurations: "
        f"{len(source_pool):,}"
    )

    package_values = (
        build_package_values(
            df
        )
    )

    engine = build_engine()

    # -------------------------------------------------------------------------
    # Source row lookup.
    # -------------------------------------------------------------------------

    source_rows = {}

    for index, row in df.iterrows():

        package = row[
            "package_context"
        ]

        source_rows[
            index
        ] = row

    # -------------------------------------------------------------------------
    # Candidate sets.
    # -------------------------------------------------------------------------

    feasible = {}

    rejected = Counter()

    candidate_counter = Counter()

    # -------------------------------------------------------------------------
    # Process each real canonical configuration.
    # -------------------------------------------------------------------------

    print(
        "\n" + "=" * 100
    )

    print(
        "EXPLORING FEASIBLE CONFIGURATION SPACE"
    )

    print(
        "=" * 100
    )

    for number, (
        source_id,
        source,
    ) in enumerate(
        source_pool.items(),
        start=1,
    ):

        package = source[
            "package"
        ]

        configuration = source[
            "configuration"
        ]

        source_row = source[
            "source_row"
        ]

        source_row_data = source_rows[
            source_row
        ]

        # =====================================================================
        # ONE MUTATIONS
        # =====================================================================

        for (
            candidate,
            changed,
            method,
        ) in generate_one_mutations(
            configuration,
            package,
            package_values,
        ):

            candidate_counter[
                "ONE_CHARACTERISTIC"
            ] += 1

            candidate_id = canonical_id(
                candidate,
                TECHNICAL_COLUMNS,
            )

            if candidate_id in real_ids:

                rejected[
                    "EXISTING_REAL_CONFIGURATION"
                ] += 1

                continue

            if candidate_id in feasible:

                rejected[
                    "DUPLICATE_CANDIDATE"
                ] += 1

                continue

            validation = (
                validate_candidate(
                    engine,
                    candidate,
                    package,
                    source_row_data,
                )
            )

            if not validation[
                "valid"
            ]:

                rejected[
                    "CONSTRAINT_ENGINE"
                ] += 1

                continue

            distance = (
                nearest_real_distance(
                    candidate,
                    real_configurations,
                )
            )

            feasible[
                candidate_id
            ] = {
                "canonical_configuration_id":
                    candidate_id,

                "package_context":
                    package,

                "source_canonical_configuration_id":
                    source_id,

                "source_row":
                    source_row,

                "generation_method":
                    method,

                "num_changed_characteristics":
                    len(changed),

                "changed_characteristics":
                    "|".join(changed),

                "nearest_real_distance":
                    distance,

                "constraint_valid":
                    True,

                **candidate,
            }

        # =====================================================================
        # TWO MUTATIONS
        # =====================================================================

        for (
            candidate,
            changed,
            method,
        ) in generate_two_mutations(
            configuration,
            package,
            package_values,
            DEPENDENCY_PAIRS,
        ):

            candidate_counter[
                method
            ] += 1

            candidate_id = canonical_id(
                candidate,
                TECHNICAL_COLUMNS,
            )

            if candidate_id in real_ids:

                rejected[
                    "EXISTING_REAL_CONFIGURATION"
                ] += 1

                continue

            if candidate_id in feasible:

                rejected[
                    "DUPLICATE_CANDIDATE"
                ] += 1

                continue

            validation = (
                validate_candidate(
                    engine,
                    candidate,
                    package,
                    source_row_data,
                )
            )

            if not validation[
                "valid"
            ]:

                rejected[
                    "CONSTRAINT_ENGINE"
                ] += 1

                continue

            distance = (
                nearest_real_distance(
                    candidate,
                    real_configurations,
                )
            )

            feasible[
                candidate_id
            ] = {
                "canonical_configuration_id":
                    candidate_id,

                "package_context":
                    package,

                "source_canonical_configuration_id":
                    source_id,

                "source_row":
                    source_row,

                "generation_method":
                    method,

                "num_changed_characteristics":
                    len(changed),

                "changed_characteristics":
                    "|".join(changed),

                "nearest_real_distance":
                    distance,

                "constraint_valid":
                    True,

                **candidate,
            }

        if (
            number % 25 == 0
            or
            number == len(source_pool)
        ):

            print(
                f"Processed "
                f"{number:,} / "
                f"{len(source_pool):,}"
                f" | feasible={len(feasible):,}"
            )

    # =========================================================================
    # DATAFRAME
    # =========================================================================

    feasible_df = pd.DataFrame(
        list(
            feasible.values()
        )
    )

    # =========================================================================
    # PACKAGE STATISTICS
    # =========================================================================

    package_records = []

    for package, group in (
        feasible_df.groupby(
            "package_context"
        )
        if len(feasible_df)
        else []
    ):

        package_records.append(
            {
                "package":
                    package,

                "feasible_configurations":
                    len(group),

                "one_characteristic":
                    int(
                        (
                            group[
                                "num_changed_characteristics"
                            ] == 1
                        ).sum()
                    ),

                "two_characteristic":
                    int(
                        (
                            group[
                                "num_changed_characteristics"
                            ] == 2
                        ).sum()
                    ),

                "minimum_distance":
                    group[
                        "nearest_real_distance"
                    ].min(),

                "maximum_distance":
                    group[
                        "nearest_real_distance"
                    ].max(),
            }
        )

    package_df = pd.DataFrame(
        package_records
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print(
        "\n" + "=" * 100
    )

    print(
        "V4 FEASIBLE SPACE SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"\nReal canonical configurations:"
        f" {len(real_ids):,}"
    )

    print(
        f"Feasible derived configurations:"
        f" {len(feasible_df):,}"
    )

    print(
        "\nCandidate generation:"
    )

    for key, value in (
        candidate_counter.items()
    ):

        print(
            f"{key:35s} "
            f"{value:,}"
        )

    print(
        "\nRejections:"
    )

    for key, value in (
        rejected.most_common()
    ):

        print(
            f"{key:35s} "
            f"{value:,}"
        )

    if len(feasible_df):

        print(
            "\nPackage feasible-space:"
        )

        print(
            package_df.to_string(
                index=False
            )
        )

        print(
            "\nDistance distribution:"
        )

        print(
            feasible_df[
                "nearest_real_distance"
            ]
            .value_counts()
            .sort_index()
            .to_string()
        )

    # =========================================================================
    # SAVE FEASIBLE SPACE
    # =========================================================================

    output_file = (
        OUTPUT_DIR
        / "731_feasible_configuration_space_v4.csv"
    )

    feasible_df.to_csv(
        output_file,
        index=False,
    )

    # =========================================================================
    # SAVE PACKAGE STATISTICS
    # =========================================================================

    package_file = (
        PROCESSED_DIR
        / "731_v4_package_space_stats.csv"
    )

    package_df.to_csv(
        package_file,
        index=False,
    )

    # =========================================================================
    # SAVE REJECTION STATISTICS
    # =========================================================================

    rejection_df = pd.DataFrame(
        [
            {
                "reason":
                    reason,

                "count":
                    count,
            }
            for reason, count
            in rejected.items()
        ]
    )

    rejection_file = (
        PROCESSED_DIR
        / "731_v4_rejection_stats.csv"
    )

    rejection_df.to_csv(
        rejection_file,
        index=False,
    )

    # =========================================================================
    # SAVE SUMMARY
    # =========================================================================

    summary_df = pd.DataFrame(
        [
            {
                "metric":
                    "real_canonical_configurations",

                "value":
                    len(real_ids),
            },
            {
                "metric":
                    "feasible_derived_configurations",

                "value":
                    len(feasible_df),
            },
            {
                "metric":
                    "total_candidate_attempts",

                "value":
                    sum(
                        candidate_counter.values()
                    ),
            },
            {
                "metric":
                    "constraint_rejections",

                "value":
                    rejected[
                        "CONSTRAINT_ENGINE"
                    ],
            },
            {
                "metric":
                    "existing_real_rejections",

                "value":
                    rejected[
                        "EXISTING_REAL_CONFIGURATION"
                    ],
            },
            {
                "metric":
                    "duplicate_rejections",

                "value":
                    rejected[
                        "DUPLICATE_CANDIDATE"
                    ],
            },
        ]
    )

    summary_file = (
        PROCESSED_DIR
        / "731_v4_space_summary.csv"
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
        "V4 FILES SAVED"
    )

    print(
        "=" * 100
    )

    print(
        f"\nFeasible configuration space:"
        f"\n{output_file}"
    )

    print(
        f"\nPackage statistics:"
        f"\n{package_file}"
    )

    print(
        f"\nRejection statistics:"
        f"\n{rejection_file}"
    )

    print(
        f"\nSummary:"
        f"\n{summary_file}"
    )

    print(
        "\n" + "=" * 100
    )

    print(
        "V4 SPACE DISCOVERY COMPLETE"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()