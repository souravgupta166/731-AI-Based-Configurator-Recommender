#!/usr/bin/env python3

"""
731 REAL VS SYNTHETIC CONFIGURATION EVALUATION
==============================================

Purpose
-------
Compare the real 731 configuration space with the generated
synthetic population.

IMPORTANT
---------
Both datasets are converted into the SAME canonical representation
before comparison.

The real configurator contains expressions such as:

    in {NOVALUE, 'G731ST-LT'}

while the synthetic generator stores:

    G731ST-LT

Therefore the evaluation must normalize both representations
identically.

Measures
--------
1. Canonical uniqueness
2. Package distribution
3. Characteristic value distributions
4. Exact technical duplicates
5. Nearest-real configuration distance
6. Technical novelty
7. Changed characteristic distribution
"""

from __future__ import annotations

import re
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

SYNTHETIC_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "configurations"
    / "731_synthetic_configurations.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

PACKAGE_COLUMN = "packageFixed_dev"

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
# CANONICAL NORMALIZATION
# =============================================================================

def normalize(value):
    """
    Convert raw 731 configurator values into the same canonical
    representation used by the synthetic generator.

    Examples
    --------
    in {NOVALUE, 'x731'}
        -> x731

    in {'ST', 'AL'}
        -> ST

    'GP'
        -> GP

    NOVALUE
        -> NOVALUE
    """

    if pd.isna(value):
        return "NOVALUE"

    text = str(value).strip()

    if not text:
        return "NOVALUE"

    # -------------------------------------------------------------------------
    # Configurator expression
    # -------------------------------------------------------------------------

    match = re.match(
        r"^in\s*\{(.*)\}$",
        text,
        flags=re.IGNORECASE,
    )

    if match:

        content = match.group(1).strip()

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

        # Prefer actual value over NOVALUE.
        non_novalue = [
            value
            for value in values
            if value.upper() != "NOVALUE"
        ]

        if non_novalue:

            return non_novalue[0]

        return "NOVALUE"

    # -------------------------------------------------------------------------
    # Remove surrounding quotes
    # -------------------------------------------------------------------------

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
# LOAD REAL DATA
# =============================================================================

def load_real():

    df = pd.read_excel(
        REAL_FILE
    )

    if "Unnamed: 0" in df.columns:

        df = df.drop(
            columns=["Unnamed: 0"]
        )

    for column in df.columns:

        df[column] = df[column].apply(
            normalize
        )

    return df


# =============================================================================
# LOAD SYNTHETIC DATA
# =============================================================================

def load_synthetic():

    df = pd.read_csv(
        SYNTHETIC_FILE
    )

    # The synthetic generator has already canonicalized values,
    # but applying the same normalization again guarantees consistency.

    for column in df.columns:

        if column not in [
            "synthetic_id",
            "data_origin",
            "generation_method",
            "changed_characteristics",
        ]:

            df[column] = df[column].apply(
                normalize
            )

    return df


# =============================================================================
# CANONICAL KEY
# =============================================================================

def canonical_key(
    df,
    columns,
):

    return (
        df[columns]
        .astype(str)
        .agg(
            "||".join,
            axis=1,
        )
    )


# =============================================================================
# PACKAGE DISTRIBUTION
# =============================================================================

def compare_packages(
    real,
    synthetic,
):

    real_counts = (
        real[
            PACKAGE_COLUMN
        ]
        .value_counts(
            normalize=True
        )
        .rename(
            "real_percentage"
        )
    )

    synthetic_counts = (
        synthetic[
            "package_context"
        ]
        .value_counts(
            normalize=True
        )
        .rename(
            "synthetic_percentage"
        )
    )

    comparison = pd.concat(
        [
            real_counts,
            synthetic_counts,
        ],
        axis=1,
    ).fillna(0)

    comparison[
        "absolute_difference_percentage_points"
    ] = (
        (
            comparison[
                "synthetic_percentage"
            ]
            -
            comparison[
                "real_percentage"
            ]
        )
        * 100
    )

    return comparison.reset_index(
        names="package"
    )


# =============================================================================
# CHARACTERISTIC DISTRIBUTIONS
# =============================================================================

def compare_characteristics(
    real,
    synthetic,
):

    records = []

    for column in TECHNICAL_COLUMNS:

        if (
            column not in real.columns
            or column not in synthetic.columns
        ):
            continue

        real_counts = (
            real[column]
            .value_counts(
                normalize=True
            )
        )

        synthetic_counts = (
            synthetic[column]
            .value_counts(
                normalize=True
            )
        )

        values = sorted(
            set(
                real_counts.index
            )
            |
            set(
                synthetic_counts.index
            )
        )

        for value in values:

            real_probability = (
                real_counts.get(
                    value,
                    0,
                )
            )

            synthetic_probability = (
                synthetic_counts.get(
                    value,
                    0,
                )
            )

            records.append(
                {
                    "characteristic":
                        column,

                    "value":
                        value,

                    "real_probability":
                        real_probability,

                    "synthetic_probability":
                        synthetic_probability,

                    "absolute_difference":
                        abs(
                            real_probability
                            -
                            synthetic_probability
                        ),
                }
            )

    return pd.DataFrame(
        records
    )


# =============================================================================
# NOVELTY ANALYSIS
# =============================================================================

def calculate_novelty(
    real,
    synthetic,
):

    real_keys = set(
        canonical_key(
            real,
            TECHNICAL_COLUMNS,
        )
    )

    synthetic_keys = canonical_key(
        synthetic,
        TECHNICAL_COLUMNS,
    )

    synthetic[
        "exact_match_to_real"
    ] = synthetic_keys.isin(
        real_keys
    )

    synthetic[
        "technical_novelty"
    ] = (
        ~synthetic[
            "exact_match_to_real"
        ]
    )

    # -------------------------------------------------------------------------
    # Unique real technical configurations
    # -------------------------------------------------------------------------

    real_technical = (
        real[
            TECHNICAL_COLUMNS
        ]
        .drop_duplicates()
    )

    real_values = (
        real_technical
        .values
    )

    novelty_records = []

    for idx, row in synthetic[
        TECHNICAL_COLUMNS
    ].iterrows():

        candidate = row.values

        minimum_distance = (
            len(
                TECHNICAL_COLUMNS
            )
        )

        for real_row in real_values:

            distance = sum(
                a != b
                for a, b in zip(
                    candidate,
                    real_row,
                )
            )

            if distance < minimum_distance:

                minimum_distance = (
                    distance
                )

                if minimum_distance == 1:
                    break

                if minimum_distance == 0:
                    break

        novelty_records.append(
            {
                "index":
                    idx,

                "nearest_real_distance":
                    minimum_distance,

                "technical_novelty_score":
                    minimum_distance
                    /
                    len(
                        TECHNICAL_COLUMNS
                    ),
            }
        )

    novelty = (
        pd.DataFrame(
            novelty_records
        )
        .set_index(
            "index"
        )
    )

    synthetic = synthetic.join(
        novelty
    )

    return synthetic


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 100)
    print("731 REAL VS SYNTHETIC CONFIGURATION EVALUATION")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    real = load_real()

    synthetic = load_synthetic()

    print(
        f"\nReal records:       {len(real):,}"
    )

    print(
        f"Synthetic records:  {len(synthetic):,}"
    )

    # -------------------------------------------------------------------------
    # Canonical uniqueness
    # -------------------------------------------------------------------------

    real_keys = canonical_key(
        real,
        TECHNICAL_COLUMNS,
    )

    synthetic_keys = canonical_key(
        synthetic,
        TECHNICAL_COLUMNS,
    )

    print(
        "\n" + "=" * 100
    )

    print(
        "CANONICAL TECHNICAL CONFIGURATIONS"
    )

    print(
        "=" * 100
    )

    print(
        f"\nReal unique:       "
        f"{real_keys.nunique():,}"
    )

    print(
        f"Synthetic unique:  "
        f"{synthetic_keys.nunique():,}"
    )

    print(
        f"Synthetic duplicate rate: "
        f"{(
            1
            -
            synthetic_keys.nunique()
            /
            len(synthetic)
        ) * 100:.2f}%"
    )

    # -------------------------------------------------------------------------
    # Package distribution
    # -------------------------------------------------------------------------

    package_comparison = (
        compare_packages(
            real,
            synthetic,
        )
    )

    print(
        "\n" + "=" * 100
    )

    print(
        "PACKAGE DISTRIBUTION COMPARISON"
    )

    print(
        "=" * 100
    )

    print(
        package_comparison.to_string(
            index=False,
            formatters={
                "real_percentage":
                    lambda x:
                    f"{x * 100:.2f}%",

                "synthetic_percentage":
                    lambda x:
                    f"{x * 100:.2f}%",

                "absolute_difference_percentage_points":
                    lambda x:
                    f"{x:.2f}",
            },
        )
    )

    # -------------------------------------------------------------------------
    # Characteristic distribution
    # -------------------------------------------------------------------------

    characteristic_comparison = (
        compare_characteristics(
            real,
            synthetic,
        )
    )

    # -------------------------------------------------------------------------
    # Novelty
    # -------------------------------------------------------------------------

    synthetic = calculate_novelty(
        real,
        synthetic,
    )

    print(
        "\n" + "=" * 100
    )

    print(
        "TECHNICAL NOVELTY"
    )

    print(
        "=" * 100
    )

    print(
        synthetic[
            "technical_novelty_score"
        ]
        .describe()
        .to_string()
    )

    print(
        "\nNearest-real configuration distance:"
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
        "\nExact matches to real configurations:"
    )

    print(
        synthetic[
            "exact_match_to_real"
        ]
        .value_counts()
        .to_string()
    )

    # -------------------------------------------------------------------------
    # Changed characteristics
    # -------------------------------------------------------------------------

    if (
        "num_changed_characteristics"
        in synthetic.columns
    ):

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

    # -------------------------------------------------------------------------
    # Save package comparison
    # -------------------------------------------------------------------------

    package_output = (
        OUTPUT_DIR
        / "731_real_vs_synthetic_package_distribution.csv"
    )

    package_comparison.to_csv(
        package_output,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save characteristic comparison
    # -------------------------------------------------------------------------

    characteristic_output = (
        OUTPUT_DIR
        / "731_real_vs_synthetic_characteristics.csv"
    )

    characteristic_comparison.to_csv(
        characteristic_output,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save novelty
    # -------------------------------------------------------------------------

    novelty_output = (
        OUTPUT_DIR
        / "731_synthetic_novelty_analysis.csv"
    )

    synthetic[
        [
            "synthetic_id",
            "package_context",
            "nearest_real_distance",
            "technical_novelty_score",
            "exact_match_to_real",
            "technical_novelty",
        ]
    ].to_csv(
        novelty_output,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save full evaluated synthetic data
    # -------------------------------------------------------------------------

    full_output = (
        OUTPUT_DIR
        / "731_synthetic_evaluated.csv"
    )

    synthetic.to_csv(
        full_output,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print(
        "\n" + "=" * 100
    )

    print(
        "EVALUATION FILES SAVED"
    )

    print(
        "=" * 100
    )

    print(
        f"\n{package_output}"
    )

    print(
        f"\n{characteristic_output}"
    )

    print(
        f"\n{novelty_output}"
    )

    print(
        f"\n{full_output}"
    )


if __name__ == "__main__":
    main()