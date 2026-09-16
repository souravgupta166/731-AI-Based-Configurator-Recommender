#!/usr/bin/env python3

"""
731 SYNTHETIC CONFIGURATION GENERATOR
=====================================

Purpose
-------
Generate novel 731 configurations using ONLY values observed in
the real 731 configurator dataset.

The generator combines:

1. Real package distribution
2. Real package-specific value distributions
3. Real package-specific configurations
4. Observed pairwise compatibility
5. Controlled mutation
6. Existing validated ConstraintEngine
7. Exact-duplicate rejection
8. Novelty scoring
9. Full provenance tracking

IMPORTANT
---------
Synthetic values are never invented.

Every generated value must have been observed in the real dataset.

The resulting configuration is considered usable only if it passes
the existing 731 ConstraintEngine.
"""

from __future__ import annotations

import copy
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "x731_allgemein_Rev24.xlsx"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "configurations"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT_FILE = (
    OUTPUT_DIR
    / "731_synthetic_configurations.csv"
)


# =============================================================================
# IMPORT VALIDATED CONSTRAINT ENGINE
# =============================================================================

CONSTRAINT_DIR = (
    PROJECT_ROOT
    / "src"
    / "constraints"
)

if str(CONSTRAINT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(CONSTRAINT_DIR),
    )


from build_731_constraint_engine import (  # noqa: E402
    ConstraintEngine,
    build_rule_table,
    load_source_data,
    representative_value,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_SEED = 731

TARGET_SYNTHETIC_ROWS = 5000

MAX_ATTEMPTS = 100000

PACKAGE_COLUMN = "packageFixed_dev"


IMPORTANT_COLUMNS = [
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


IMPORTANT_PAIRS = [
    (
        "explosionApproval",
        "certification",
    ),
    (
        "stromEingaenge",
        "temperaturEingaenge",
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
        "housing",
        "powerSupply",
    ),
    (
        "numberOfChannels",
        "temperaturEingaenge",
    ),
    (
        "dataInterface",
        "certification",
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
        "binaerDigitalOpenColl_MN",
        "binaerOpenColl_MP",
    ),
]


# =============================================================================
# RANDOM GENERATOR
# =============================================================================

rng = random.Random(
    RANDOM_SEED
)


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize(value) -> str:

    if pd.isna(value):
        return "NOVALUE"

    text = str(value).strip()

    if not text:
        return "NOVALUE"

    # Handle configurator expressions such as:
    #
    # in {NOVALUE, 'G731ST-LT'}
    #
    # We use the first actual non-NOVALUE token.

    match = re.match(
        r"^in\s*\{(.*)\}$",
        text,
        flags=re.IGNORECASE,
    )

    if match:

        content = match.group(1)

        tokens = re.findall(
            r"'([^']*)'|\"([^\"]*)\"|([^,\s]+)",
            content,
        )

        values = []

        for single, double, unquoted in tokens:

            value = (
                single
                or double
                or unquoted
            )

            if value:

                values.append(
                    value.strip()
                )

        non_novalue = [
            value
            for value in values
            if value.upper() != "NOVALUE"
        ]

        if non_novalue:

            return non_novalue[0]

        return "NOVALUE"

    # Remove surrounding quotes.
    if len(text) >= 2:

        if (
            text[0] == "'"
            and text[-1] == "'"
        ):
            text = text[1:-1]

        elif (
            text[0] == '"'
            and text[-1] == '"'
        ):
            text = text[1:-1]

    return text.strip()


# =============================================================================
# LOAD AND NORMALIZE SOURCE
# =============================================================================

def load_normalized_source() -> pd.DataFrame:

    print("=" * 100)
    print("LOADING REAL 731 DATA")
    print("=" * 100)

    df = pd.read_excel(
        RAW_FILE
    )

    if "Unnamed: 0" in df.columns:

        df = df.drop(
            columns=["Unnamed: 0"]
        )

    for column in df.columns:

        df[column] = df[column].apply(
            normalize
        )

    print(
        f"\nRows:    {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns):,}"
    )

    return df


# =============================================================================
# REPRESENTATIVE CONFIGURATION
# =============================================================================

def row_to_configuration(
    row: pd.Series,
) -> Dict[str, str]:

    configuration = {}

    for column in row.index:

        configuration[
            column
        ] = normalize(
            row[column]
        )

    return configuration


# =============================================================================
# CONFIGURATION KEY
# =============================================================================

def configuration_key(
    configuration: Dict[str, str],
    columns: List[str],
) -> Tuple:

    return tuple(
        configuration.get(
            column,
            "NOVALUE",
        )
        for column in columns
    )


# =============================================================================
# PACKAGE DISTRIBUTION
# =============================================================================

def build_package_sampler(
    df: pd.DataFrame,
):

    counts = (
        df[
            PACKAGE_COLUMN
        ]
        .value_counts()
    )

    packages = list(
        counts.index
    )

    weights = [
        float(
            counts[package]
        )
        for package in packages
    ]

    return packages, weights


# =============================================================================
# PACKAGE VALUE DISTRIBUTIONS
# =============================================================================

def build_value_distributions(
    df: pd.DataFrame,
):

    distributions = {}

    for package, package_df in (
        df.groupby(
            PACKAGE_COLUMN
        )
    ):

        distributions[
            package
        ] = {}

        for column in IMPORTANT_COLUMNS:

            if column not in df.columns:
                continue

            counts = (
                package_df[
                    column
                ]
                .value_counts()
            )

            if len(counts) == 0:
                continue

            values = list(
                counts.index
            )

            weights = [
                float(
                    counts[value]
                )
                for value in values
            ]

            distributions[
                package
            ][
                column
            ] = (
                values,
                weights,
            )

    return distributions


# =============================================================================
# PAIRWISE COMPATIBILITY MODEL
# =============================================================================

def build_pair_model(
    df: pd.DataFrame,
):

    model = {}

    for package, package_df in (
        df.groupby(
            PACKAGE_COLUMN
        )
    ):

        model[
            package
        ] = {}

        for column_a, column_b in (
            IMPORTANT_PAIRS
        ):

            if (
                column_a not in df.columns
                or column_b not in df.columns
            ):
                continue

            grouped = (
                package_df
                .groupby(
                    [
                        column_a,
                        column_b,
                    ]
                )
                .size()
                .reset_index(
                    name="count"
                )
            )

            compatibility = set()

            for _, row in grouped.iterrows():

                value_a = normalize(
                    row[column_a]
                )

                value_b = normalize(
                    row[column_b]
                )

                compatibility.add(
                    (
                        value_a,
                        value_b,
                    )
                )

            model[
                package
            ][
                (
                    column_a,
                    column_b,
                )
            ] = compatibility

    return model


# =============================================================================
# SELECT STARTING CONFIGURATION
# =============================================================================

def select_starting_configuration(
    df: pd.DataFrame,
    package: str,
):

    package_df = df[
        df[
            PACKAGE_COLUMN
        ] == package
    ]

    row = package_df.sample(
        n=1,
        random_state=rng.randint(
            0,
            2_000_000_000,
        ),
    ).iloc[0]

    return row_to_configuration(
        row
    ), int(row.name)


# =============================================================================
# MUTATION COUNT
# =============================================================================

def choose_mutation_count(
    package_size: int,
) -> int:

    # Small packages have very limited evidence.
    # Therefore we mutate them conservatively.

    if package_size <= 10:
        return rng.choice(
            [1, 1, 1, 2]
        )

    if package_size <= 30:
        return rng.choice(
            [1, 1, 2, 2, 3]
        )

    return rng.choice(
        [1, 1, 2, 2, 3, 3, 4]
    )


# =============================================================================
# MUTATE CONFIGURATION
# =============================================================================

def mutate_configuration(
    configuration: Dict[str, str],
    package: str,
    value_distributions,
    pair_model,
    package_size: int,
):

    candidate = copy.deepcopy(
        configuration
    )

    changed = []

    mutation_count = choose_mutation_count(
        package_size
    )

    available_columns = [
        column
        for column in IMPORTANT_COLUMNS
        if column in candidate
        and column in value_distributions.get(
            package,
            {},
        )
    ]

    rng.shuffle(
        available_columns
    )

    # Try several columns, but never mutate everything.
    for column in available_columns:

        if len(changed) >= mutation_count:
            break

        current_value = (
            candidate.get(
                column,
                "NOVALUE",
            )
        )

        values, weights = (
            value_distributions[
                package
            ][
                column
            ]
        )

        alternatives = [
            value
            for value in values
            if value != current_value
        ]

        if not alternatives:
            continue

        alternative_weights = [
            weights[
                values.index(value)
            ]
            for value in alternatives
        ]

        new_value = rng.choices(
            alternatives,
            weights=alternative_weights,
            k=1,
        )[0]

        candidate[
            column
        ] = new_value

        changed.append(
            column
        )

    return candidate, changed


# =============================================================================
# PAIRWISE SCORE
# =============================================================================

def pairwise_score(
    candidate: Dict[str, str],
    package: str,
    pair_model,
):

    package_pairs = pair_model.get(
        package,
        {}
    )

    if not package_pairs:
        return 1.0

    scores = []

    for (
        column_a,
        column_b,
    ), compatibility in (
        package_pairs.items()
    ):

        value_a = candidate.get(
            column_a,
            "NOVALUE",
        )

        value_b = candidate.get(
            column_b,
            "NOVALUE",
        )

        # If the pair was never modeled,
        # don't penalize it.
        if not compatibility:
            continue

        if (
            value_a,
            value_b,
        ) in compatibility:

            scores.append(
                1.0
            )

        else:

            scores.append(
                0.0
            )

    if not scores:
        return 1.0

    return sum(scores) / len(
        scores
    )


# =============================================================================
# NOVELTY SCORE
# =============================================================================

def calculate_novelty(
    candidate: Dict[str, str],
    real_keys: set,
    columns: List[str],
):

    key = configuration_key(
        candidate,
        columns,
    )

    if key in real_keys:
        return 0.0, True

    # Hamming distance to the closest
    # real configuration.
    #
    # We sample the real keys rather than
    # calculating a full distance matrix.

    minimum_distance = len(
        columns
    )

    for real_key in real_keys:

        distance = sum(
            a != b
            for a, b in zip(
                key,
                real_key,
            )
        )

        if distance < minimum_distance:

            minimum_distance = (
                distance
            )

            if minimum_distance == 1:
                break

    normalized_distance = (
        minimum_distance
        / len(columns)
    )

    return (
        normalized_distance,
        False,
    )


# =============================================================================
# GENERATE ONE CANDIDATE
# =============================================================================

def generate_candidate(
    df: pd.DataFrame,
    packages,
    package_weights,
    value_distributions,
    pair_model,
):

    package = rng.choices(
        packages,
        weights=package_weights,
        k=1,
    )[0]

    package_size = len(
        df[
            df[
                PACKAGE_COLUMN
            ] == package
        ]
    )

    base_configuration, source_row = (
        select_starting_configuration(
            df,
            package,
        )
    )

    candidate, changed = (
        mutate_configuration(
            base_configuration,
            package,
            value_distributions,
            pair_model,
            package_size,
        )
    )

    return (
        candidate,
        package,
        source_row,
        changed,
    )


# =============================================================================
# MAIN GENERATOR
# =============================================================================

def main():

    print("=" * 100)
    print("731 SYNTHETIC CONFIGURATION GENERATOR")
    print("=" * 100)

    df = load_normalized_source()

    # -------------------------------------------------------------------------
    # Build models
    # -------------------------------------------------------------------------

    print(
        "\nBuilding package distribution..."
    )

    packages, package_weights = (
        build_package_sampler(
            df
        )
    )

    print(
        "Building package value distributions..."
    )

    value_distributions = (
        build_value_distributions(
            df
        )
    )

    print(
        "Building pairwise compatibility..."
    )

    pair_model = (
        build_pair_model(
            df
        )
    )

    # -------------------------------------------------------------------------
    # Existing hard constraint engine
    # -------------------------------------------------------------------------

    print(
        "\nBuilding validated 731 ConstraintEngine..."
    )

    raw_df = load_source_data()

    rules = build_rule_table(
        raw_df
    )

    engine = ConstraintEngine(
        rules
    )

    # -------------------------------------------------------------------------
    # Real configuration keys
    # -------------------------------------------------------------------------

    configuration_columns = list(
        df.columns
    )

    real_keys = set()

    for _, row in df.iterrows():

        configuration = (
            row_to_configuration(
                row
            )
        )

        real_keys.add(
            configuration_key(
                configuration,
                configuration_columns,
            )
        )

    print(
        f"\nReal configuration keys: "
        f"{len(real_keys):,}"
    )

    # -------------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------------

    results = []

    accepted = 0
    attempts = 0
    rejected_constraint = 0
    rejected_pair = 0
    rejected_duplicate = 0

    print(
        f"\nTarget synthetic configurations: "
        f"{TARGET_SYNTHETIC_ROWS:,}"
    )

    print(
        f"Maximum attempts: "
        f"{MAX_ATTEMPTS:,}"
    )

    while (
        accepted < TARGET_SYNTHETIC_ROWS
        and attempts < MAX_ATTEMPTS
    ):

        attempts += 1

        (
            candidate,
            package,
            source_row,
            changed,
        ) = generate_candidate(
            df,
            packages,
            package_weights,
            value_distributions,
            pair_model,
        )

        # ---------------------------------------------------------------------
        # Pairwise compatibility
        # ---------------------------------------------------------------------

        dependency_score = (
            pairwise_score(
                candidate,
                package,
                pair_model,
            )
        )

        # We require all modeled important pairs
        # to be observed for the selected package.
        if dependency_score < 1.0:

            rejected_pair += 1

            continue

        # ---------------------------------------------------------------------
        # Hard constraint engine
        # ---------------------------------------------------------------------

        validation = (
            engine.validate(
                candidate
            )
        )

        constraint_valid = bool(
            validation["valid"]
        )

        if not constraint_valid:

            rejected_constraint += 1

            continue

        # ---------------------------------------------------------------------
        # Novelty
        # ---------------------------------------------------------------------

        novelty_score, duplicate = (
            calculate_novelty(
                candidate,
                real_keys,
                configuration_columns,
            )
        )

        if duplicate:

            rejected_duplicate += 1

            continue

        # ---------------------------------------------------------------------
        # Build result
        # ---------------------------------------------------------------------

        record = {
            "synthetic_id":
                f"SYN_731_{accepted + 1:06d}",

            "data_origin":
                "SYNTHETIC",

            "generation_method":
                "constrained_empirical_recombination",

            "source_row":
                source_row,

            "package_context":
                package,

            "changed_characteristics":
                "|".join(changed),

            "num_changed_characteristics":
                len(changed),

            "pairwise_compatibility_score":
                dependency_score,

            "constraint_valid":
                constraint_valid,

            "exact_duplicate_of_real":
                False,

            "novelty_score":
                novelty_score,
        }

        # Add technical configuration.
        for column in configuration_columns:

            record[
                column
            ] = candidate.get(
                column,
                "NOVALUE",
            )

        results.append(
            record
        )

        accepted += 1

        # Progress indicator.
        if accepted % 250 == 0:

            print(
                f"Generated: "
                f"{accepted:,} / "
                f"{TARGET_SYNTHETIC_ROWS:,} "
                f"| attempts={attempts:,}"
            )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    result_df = pd.DataFrame(
        results
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print(
        "\n" + "=" * 100
    )

    print(
        "GENERATION SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"\nRequested:              "
        f"{TARGET_SYNTHETIC_ROWS:,}"
    )

    print(
        f"Generated:              "
        f"{len(result_df):,}"
    )

    print(
        f"Attempts:               "
        f"{attempts:,}"
    )

    print(
        f"Acceptance rate:        "
        f"{len(result_df) / attempts * 100:.2f}%"
        if attempts
        else "Acceptance rate:        0.00%"
    )

    print(
        f"\nRejected — pairwise:    "
        f"{rejected_pair:,}"
    )

    print(
        f"Rejected — constraints: "
        f"{rejected_constraint:,}"
    )

    print(
        f"Rejected — duplicates:  "
        f"{rejected_duplicate:,}"
    )

    if len(result_df):

        print(
            "\nSynthetic package distribution:"
        )

        package_summary = (
            result_df[
                "package_context"
            ]
            .value_counts()
            .rename_axis(
                "package"
            )
            .reset_index(
                name="count"
            )
        )

        package_summary[
            "percentage"
        ] = (
            package_summary["count"]
            / len(result_df)
            * 100
        )

        print(
            package_summary.to_string(
                index=False,
                formatters={
                    "percentage":
                        lambda x:
                        f"{x:.2f}%"
                },
            )
        )

        print(
            "\nNovelty statistics:"
        )

        print(
            result_df[
                "novelty_score"
            ].describe().to_string()
        )

        print(
            "\nChanged characteristics:"
        )

        print(
            result_df[
                "num_changed_characteristics"
            ]
            .value_counts()
            .sort_index()
            .to_string()
        )

    print(
        "\n" + "=" * 100
    )

    print(
        "SYNTHETIC CONFIGURATIONS SAVED"
    )

    print(
        "=" * 100
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()