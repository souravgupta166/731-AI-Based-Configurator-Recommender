#!/usr/bin/env python3

"""
731 CONFIGURATION SPACE ANALYSIS
================================

Purpose
-------
Understand the real configuration space before generating
synthetic configurations.

This script does NOT generate synthetic data.

It measures:

1. Exact duplicate configurations
2. Package distribution
3. Characteristic value distributions
4. Package-specific value distributions
5. Important pairwise combinations
6. Configuration diversity

The output will be used to design the synthetic generator.
"""

from __future__ import annotations

from pathlib import Path

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


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize(value):

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
# LOAD
# =============================================================================

def load_data():

    print("=" * 100)
    print("LOADING REAL 731 CONFIGURATION DATA")
    print("=" * 100)

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
# DUPLICATES
# =============================================================================

def analyze_duplicates(df):

    print("\n" + "=" * 100)
    print("EXACT CONFIGURATION DUPLICATES")
    print("=" * 100)

    # Exclude administrative package context from technical duplicate check
    # only if desired. For now we keep all columns.
    technical_columns = list(
        df.columns
    )

    duplicate_mask = df.duplicated(
        subset=technical_columns,
        keep=False,
    )

    duplicate_rows = df[
        duplicate_mask
    ]

    unique_count = (
        df.drop_duplicates(
            subset=technical_columns
        )
        .shape[0]
    )

    print(
        f"\nTotal rows:              {len(df):,}"
    )

    print(
        f"Unique configurations:   {unique_count:,}"
    )

    print(
        f"Duplicate rows:          {duplicate_mask.sum():,}"
    )

    print(
        f"Duplicate percentage:    "
        f"{duplicate_mask.mean() * 100:.2f}%"
    )

    duplicate_output = (
        OUTPUT_DIR
        / "731_duplicate_configurations.csv"
    )

    duplicate_rows.to_csv(
        duplicate_output,
        index=False,
    )

    print(
        f"\nSaved duplicate records:"
        f"\n{duplicate_output}"
    )


# =============================================================================
# PACKAGE DISTRIBUTION
# =============================================================================

def analyze_packages(df):

    print("\n" + "=" * 100)
    print("PACKAGE DISTRIBUTION")
    print("=" * 100)

    distribution = (
        df[PACKAGE_COLUMN]
        .value_counts()
        .rename_axis(
            PACKAGE_COLUMN
        )
        .reset_index(
            name="count"
        )
    )

    distribution["percentage"] = (
        distribution["count"]
        / len(df)
        * 100
    )

    print(
        distribution.to_string(
            index=False,
            formatters={
                "percentage": lambda x:
                f"{x:.2f}%"
            },
        )
    )

    output = (
        OUTPUT_DIR
        / "731_package_distribution.csv"
    )

    distribution.to_csv(
        output,
        index=False,
    )


# =============================================================================
# GLOBAL VALUE DISTRIBUTIONS
# =============================================================================

def analyze_global_values(df):

    print("\n" + "=" * 100)
    print("GLOBAL CHARACTERISTIC VALUE DISTRIBUTIONS")
    print("=" * 100)

    records = []

    for column in IMPORTANT_COLUMNS:

        if column not in df.columns:
            continue

        counts = (
            df[column]
            .value_counts()
        )

        for value, count in counts.items():

            records.append(
                {
                    "characteristic": column,
                    "value": value,
                    "count": int(count),
                    "percentage": (
                        count
                        / len(df)
                        * 100
                    ),
                }
            )

    output_df = pd.DataFrame(
        records
    )

    output = (
        OUTPUT_DIR
        / "731_global_value_distribution.csv"
    )

    output_df.to_csv(
        output,
        index=False,
    )

    print(
        f"\nCharacteristics analyzed: "
        f"{output_df['characteristic'].nunique()}"
    )

    print(
        f"Value records: "
        f"{len(output_df):,}"
    )

    print(
        f"\nSaved:"
        f"\n{output}"
    )


# =============================================================================
# PACKAGE-SPECIFIC VALUES
# =============================================================================

def analyze_package_values(df):

    print("\n" + "=" * 100)
    print("PACKAGE-SPECIFIC VALUE DISTRIBUTIONS")
    print("=" * 100)

    records = []

    for package, package_df in (
        df.groupby(
            PACKAGE_COLUMN
        )
    ):

        for column in IMPORTANT_COLUMNS:

            if column not in df.columns:
                continue

            counts = (
                package_df[column]
                .value_counts()
            )

            for value, count in counts.items():

                records.append(
                    {
                        "package": package,
                        "characteristic": column,
                        "value": value,
                        "count": int(count),
                        "package_size": len(
                            package_df
                        ),
                        "percentage": (
                            count
                            / len(package_df)
                            * 100
                        ),
                    }
                )

    output_df = pd.DataFrame(
        records
    )

    output = (
        OUTPUT_DIR
        / "731_package_value_distribution.csv"
    )

    output_df.to_csv(
        output,
        index=False,
    )

    print(
        f"\nRecords: "
        f"{len(output_df):,}"
    )

    print(
        f"Saved:"
        f"\n{output}"
    )


# =============================================================================
# IMPORTANT PAIRS
# =============================================================================

def analyze_pairs(df):

    print("\n" + "=" * 100)
    print("IMPORTANT PAIRWISE COMBINATIONS")
    print("=" * 100)

    pairs = [
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
            "binaerDigitalOpenColl_MN",
            "binaerOpenColl_MP",
        ),
    ]

    records = []

    for column_a, column_b in pairs:

        if (
            column_a not in df.columns
            or column_b not in df.columns
        ):
            continue

        grouped = (
            df.groupby(
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

        grouped[
            "percentage"
        ] = (
            grouped["count"]
            / len(df)
            * 100
        )

        grouped[
            "characteristic_a"
        ] = column_a

        grouped[
            "characteristic_b"
        ] = column_b

        records.append(
            grouped[
                [
                    "characteristic_a",
                    "characteristic_b",
                    column_a,
                    column_b,
                    "count",
                    "percentage",
                ]
            ]
        )

    if records:

        output_df = pd.concat(
            records,
            ignore_index=True,
        )

        output = (
            OUTPUT_DIR
            / "731_important_pair_combinations.csv"
        )

        output_df.to_csv(
            output,
            index=False,
        )

        print(
            f"\nPair combinations: "
            f"{len(output_df):,}"
        )

        print(
            f"Saved:"
            f"\n{output}"
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    df = load_data()

    analyze_duplicates(
        df
    )

    analyze_packages(
        df
    )

    analyze_global_values(
        df
    )

    analyze_package_values(
        df
    )

    analyze_pairs(
        df
    )

    print("\n" + "=" * 100)
    print("CONFIGURATION SPACE ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()