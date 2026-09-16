#!/usr/bin/env python3

"""
731 SYNTHETIC CONSTRAINT MODEL
==============================

Purpose
-------
Build an empirical model of the REAL 731 configuration space.

This model is NOT the final hard constraint engine.

The existing ConstraintEngine remains the authoritative validator.

This model provides:

1. Observed package distribution
2. Allowed values per package/characteristic
3. Package-specific value probabilities
4. Package-specific pairwise compatibility
5. Pairwise conditional probabilities
6. Statistics used by the synthetic generator

Design principle
----------------
We do NOT invent engineering values.

Every value appearing in this model must have been observed
in the real 731 configurator.

Novel configurations are created by recombining observed values
while respecting observed package-level and pairwise relationships.
"""

from __future__ import annotations

from pathlib import Path
from itertools import combinations

import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "x731_allgemein_Rev24.xlsx"
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


OUTPUT_VALUES = (
    OUTPUT_DIR
    / "731_synthetic_allowed_values.csv"
)

OUTPUT_PAIRS = (
    OUTPUT_DIR
    / "731_synthetic_pair_compatibility.csv"
)

OUTPUT_PACKAGES = (
    OUTPUT_DIR
    / "731_synthetic_package_distribution.csv"
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR
    / "731_synthetic_model_summary.csv"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

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


# These pairs are particularly important because previous dependency
# analysis showed strong relationships between them.
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
# NORMALIZATION
# =============================================================================

def normalize(value) -> str:

    if pd.isna(value):
        return "NOVALUE"

    text = str(value).strip()

    if not text:
        return "NOVALUE"

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
# LOAD DATA
# =============================================================================

def load_data():

    print("=" * 100)
    print("BUILDING 731 SYNTHETIC CONSTRAINT MODEL")
    print("=" * 100)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Could not find:\n{INPUT_FILE}"
        )

    df = pd.read_excel(
        INPUT_FILE
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
# PACKAGE DISTRIBUTION
# =============================================================================

def build_package_distribution(
    df: pd.DataFrame,
) -> pd.DataFrame:

    grouped = (
        df[PACKAGE_COLUMN]
        .value_counts()
        .rename_axis(
            "package"
        )
        .reset_index(
            name="count"
        )
    )

    grouped["probability"] = (
        grouped["count"]
        / grouped["count"].sum()
    )

    grouped["percentage"] = (
        grouped["probability"]
        * 100
    )

    return grouped


# =============================================================================
# PACKAGE + CHARACTERISTIC VALUE MODEL
# =============================================================================

def build_allowed_value_model(
    df: pd.DataFrame,
) -> pd.DataFrame:

    records = []

    for package, package_df in (
        df.groupby(
            PACKAGE_COLUMN
        )
    ):

        for characteristic in IMPORTANT_COLUMNS:

            if characteristic not in df.columns:
                continue

            counts = (
                package_df[
                    characteristic
                ]
                .value_counts()
            )

            total = counts.sum()

            for value, count in counts.items():

                records.append(
                    {
                        "package": package,
                        "characteristic": characteristic,
                        "value": value,
                        "count": int(count),
                        "package_characteristic_total": int(
                            total
                        ),
                        "probability": (
                            count / total
                            if total
                            else 0
                        ),
                    }
                )

    return pd.DataFrame(
        records
    )


# =============================================================================
# PAIRWISE COMPATIBILITY
# =============================================================================

def build_pairwise_model(
    df: pd.DataFrame,
) -> pd.DataFrame:

    records = []

    for package, package_df in (
        df.groupby(
            PACKAGE_COLUMN
        )
    ):

        for column_a, column_b in IMPORTANT_PAIRS:

            if (
                column_a not in df.columns
                or column_b not in df.columns
            ):
                continue

            pair_counts = (
                package_df
                .groupby(
                    [
                        column_a,
                        column_b,
                    ]
                )
                .size()
                .reset_index(
                    name="joint_count"
                )
            )

            total = len(
                package_df
            )

            # Marginal counts.
            counts_a = (
                package_df[
                    column_a
                ]
                .value_counts()
                .to_dict()
            )

            counts_b = (
                package_df[
                    column_b
                ]
                .value_counts()
                .to_dict()
            )

            for _, row in pair_counts.iterrows():

                value_a = row[
                    column_a
                ]

                value_b = row[
                    column_b
                ]

                joint_count = int(
                    row["joint_count"]
                )

                count_a = int(
                    counts_a.get(
                        value_a,
                        0,
                    )
                )

                count_b = int(
                    counts_b.get(
                        value_b,
                        0,
                    )
                )

                # P(B | A)
                conditional_b_given_a = (
                    joint_count / count_a
                    if count_a
                    else 0
                )

                # P(A | B)
                conditional_a_given_b = (
                    joint_count / count_b
                    if count_b
                    else 0
                )

                # P(A,B)
                joint_probability = (
                    joint_count / total
                    if total
                    else 0
                )

                records.append(
                    {
                        "package": package,
                        "characteristic_a": column_a,
                        "characteristic_b": column_b,
                        "value_a": value_a,
                        "value_b": value_b,
                        "joint_count": joint_count,
                        "joint_probability": joint_probability,
                        "conditional_b_given_a": conditional_b_given_a,
                        "conditional_a_given_b": conditional_a_given_b,
                    }
                )

    return pd.DataFrame(
        records
    )


# =============================================================================
# MODEL SUMMARY
# =============================================================================

def build_summary(
    df: pd.DataFrame,
    values: pd.DataFrame,
    pairs: pd.DataFrame,
) -> pd.DataFrame:

    records = []

    records.append(
        {
            "metric": "source_rows",
            "value": len(df),
        }
    )

    records.append(
        {
            "metric": "unique_configurations",
            "value": len(
                df.drop_duplicates()
            ),
        }
    )

    records.append(
        {
            "metric": "package_contexts",
            "value": df[
                PACKAGE_COLUMN
            ].nunique(),
        }
    )

    records.append(
        {
            "metric": "modeled_characteristics",
            "value": values[
                "characteristic"
            ].nunique(),
        }
    )

    records.append(
        {
            "metric": "package_characteristic_values",
            "value": len(values),
        }
    )

    records.append(
        {
            "metric": "observed_pair_combinations",
            "value": len(pairs),
        }
    )

    records.append(
        {
            "metric": "pair_types",
            "value": (
                pairs[
                    [
                        "characteristic_a",
                        "characteristic_b",
                    ]
                ]
                .drop_duplicates()
                .shape[0]
            ),
        }
    )

    return pd.DataFrame(
        records
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    df = load_data()

    # -------------------------------------------------------------------------
    # Package distribution
    # -------------------------------------------------------------------------

    package_distribution = (
        build_package_distribution(
            df
        )
    )

    # -------------------------------------------------------------------------
    # Allowed values
    # -------------------------------------------------------------------------

    allowed_values = (
        build_allowed_value_model(
            df
        )
    )

    # -------------------------------------------------------------------------
    # Pairwise compatibility
    # -------------------------------------------------------------------------

    pairwise = (
        build_pairwise_model(
            df
        )
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    summary = build_summary(
        df,
        allowed_values,
        pairwise,
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    package_distribution.to_csv(
        OUTPUT_PACKAGES,
        index=False,
    )

    allowed_values.to_csv(
        OUTPUT_VALUES,
        index=False,
    )

    pairwise.to_csv(
        OUTPUT_PAIRS,
        index=False,
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Console output
    # -------------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("SYNTHETIC CONSTRAINT MODEL SUMMARY")
    print("=" * 100)

    print(
        summary.to_string(
            index=False
        )
    )

    print("\nPackage distribution:")

    print(
        package_distribution.to_string(
            index=False,
            formatters={
                "probability": lambda x:
                f"{x:.4f}",
                "percentage": lambda x:
                f"{x:.2f}%",
            },
        )
    )

    print(
        "\n" + "=" * 100
    )
    print("FILES SAVED")
    print("=" * 100)

    print(
        f"\nPackage distribution:"
        f"\n{OUTPUT_PACKAGES}"
    )

    print(
        f"\nAllowed values:"
        f"\n{OUTPUT_VALUES}"
    )

    print(
        f"\nPairwise compatibility:"
        f"\n{OUTPUT_PAIRS}"
    )

    print(
        f"\nModel summary:"
        f"\n{OUTPUT_SUMMARY}"
    )

    print(
        "\n" + "=" * 100
    )
    print("SYNTHETIC CONSTRAINT MODEL COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()