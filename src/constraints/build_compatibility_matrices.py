import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = (
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


# ============================================================
# LOAD
# ============================================================

df = pd.read_excel(
    RAW_FILE
)


# ============================================================
# REMOVE NON-INFORMATIVE COLUMN
# ============================================================

df = df.drop(
    columns=["Unnamed: 0"],
    errors="ignore"
)


# ============================================================
# CHARACTERISTIC PAIRS
# ============================================================

PAIRS = [

    (
        "stromEingaenge",
        "temperaturEingaenge"
    ),

    (
        "explosionApproval",
        "certification"
    ),

    (
        "stromSchaltbar",
        "stromEingaenge"
    ),

    (
        "binaerDigitalOpenColl_MN",
        "binaerOpenColl_MP"
    ),

    (
        "stromSchaltbar",
        "temperaturEingaenge"
    ),
]


# ============================================================
# CLEAN VALUES
# ============================================================

for column_a, column_b in PAIRS:

    df[column_a] = (
        df[column_a]
        .fillna("MISSING")
        .astype(str)
        .str.strip()
    )

    df[column_b] = (
        df[column_b]
        .fillna("MISSING")
        .astype(str)
        .str.strip()
    )


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# ANALYZE PAIRS
# ============================================================

for column_a, column_b in PAIRS:

    print("\n" + "=" * 100)

    print(
        f"COMPATIBILITY ANALYSIS"
    )

    print(
        f"{column_a}  ↔  {column_b}"
    )

    print("=" * 100)


    # --------------------------------------------------------
    # GLOBAL COMBINATIONS
    # --------------------------------------------------------

    combinations = (
        df.groupby(
            [
                column_a,
                column_b
            ]
        )
        .size()
        .reset_index(
            name="occurrences"
        )
    )


    combinations[
        "percentage"
    ] = (
        combinations["occurrences"]
        / len(df)
        * 100
    )


    combinations = (
        combinations
        .sort_values(
            "occurrences",
            ascending=False
        )
    )


    print(
        "\nGLOBAL OBSERVED COMBINATIONS:"
    )

    print(
        combinations
        .to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # PACKAGE-CONTEXT COMBINATIONS
    # --------------------------------------------------------

    package_combinations = (
        df.groupby(
            [
                "packageFixed_dev",
                column_a,
                column_b
            ]
        )
        .size()
        .reset_index(
            name="occurrences"
        )
    )


    print(
        "\nPACKAGE-CONTEXT COMBINATIONS:"
    )

    print(
        package_combinations
        .to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    safe_a = (
        column_a
        .replace(
            "/",
            "_"
        )
    )

    safe_b = (
        column_b
        .replace(
            "/",
            "_"
        )
    )


    global_file = (
        OUTPUT_DIR
        / f"731_compatibility_{safe_a}_{safe_b}.csv"
    )


    package_file = (
        OUTPUT_DIR
        / f"731_compatibility_package_{safe_a}_{safe_b}.csv"
    )


    combinations.to_csv(
        global_file,
        index=False
    )


    package_combinations.to_csv(
        package_file,
        index=False
    )


    print(
        f"\nSaved global matrix:"
        f"\n{global_file}"
    )


    print(
        f"\nSaved package matrix:"
        f"\n{package_file}"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 100)
print("COMPATIBILITY ANALYSIS COMPLETE")
print("=" * 100)