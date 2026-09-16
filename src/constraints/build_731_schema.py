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

df = pd.read_excel(RAW_FILE)


# ============================================================
# REMOVE NON-INFORMATIVE COLUMN
# ============================================================

NON_INFORMATIVE_COLUMNS = [
    "Unnamed: 0"
]

working_df = df.drop(
    columns=NON_INFORMATIVE_COLUMNS,
    errors="ignore"
)


# ============================================================
# COLUMN PROFILING
# ============================================================

schema = []

for column in working_df.columns:

    series = working_df[column]

    unique_count = series.nunique(
        dropna=False
    )

    novalue_count = (
        series
        .astype(str)
        .str.strip()
        .str.upper()
        .eq("NOVALUE")
        .sum()
    )

    missing_count = series.isna().sum()

    novalue_percentage = (
        novalue_count
        / len(series)
        * 100
    )

    # --------------------------------------------------------
    # Initial automatic classification
    # --------------------------------------------------------

    if unique_count == 1:

        column_type = "constant_metadata"

    elif novalue_percentage >= 95:

        column_type = "mostly_inactive"

    else:

        column_type = "variable_characteristic"


    schema.append({

        "column_name": column,

        "data_type":
            str(series.dtype),

        "unique_values":
            unique_count,

        "missing_count":
            missing_count,

        "novalue_count":
            novalue_count,

        "novalue_percentage":
            round(
                novalue_percentage,
                2
            ),

        "initial_type":
            column_type
    })


schema = pd.DataFrame(schema)


# ============================================================
# DISPLAY
# ============================================================

print("=" * 80)
print("731 CONFIGURATION SCHEMA")
print("=" * 80)

print(
    f"\nOriginal columns: "
    f"{len(df.columns)}"
)

print(
    f"After removing non-informative columns: "
    f"{len(working_df.columns)}"
)


print("\n" + "=" * 80)
print("COLUMN CLASSIFICATION")
print("=" * 80)

print(
    schema[
        [
            "column_name",
            "unique_values",
            "novalue_percentage",
            "initial_type"
        ]
    ]
    .to_string(index=False)
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("CLASSIFICATION SUMMARY")
print("=" * 80)

print(
    schema["initial_type"]
    .value_counts()
    .to_string()
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

output_file = (
    OUTPUT_DIR
    / "731_schema_initial.csv"
)

schema.to_csv(
    output_file,
    index=False
)

print("\nSaved to:")
print(output_file)