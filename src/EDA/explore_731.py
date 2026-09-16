import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"

FILE_PATH = RAW_DATA / "x731_allgemein_Rev24.xlsx"


# ============================================================
# 2. DISPLAY SETTINGS
# ============================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_colwidth", 120)


# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 80)
print("731 CONFIGURATOR — INITIAL EXPLORATION")
print("=" * 80)

print(f"\nInput file:")
print(FILE_PATH)

print(f"\nFile exists: {FILE_PATH.exists()}")

if not FILE_PATH.exists():
    raise FileNotFoundError(
        f"Could not find the Excel file: {FILE_PATH}"
    )


df = pd.read_excel(FILE_PATH)

print("\nDataset loaded successfully")
print(f"Rows:    {df.shape[0]:,}")
print(f"Columns: {df.shape[1]:,}")


# ============================================================
# 4. COLUMN INVENTORY
# ============================================================

print("\n" + "=" * 80)
print("COLUMN INVENTORY")
print("=" * 80)

for i, column in enumerate(df.columns, start=1):
    print(f"{i:02d}. {column}")


# ============================================================
# 5. DATA DICTIONARY
# ============================================================

data_dictionary = pd.DataFrame({
    "column_number": range(1, len(df.columns) + 1),
    "column_name": df.columns,
    "data_type": [
        str(df[col].dtype)
        for col in df.columns
    ],
    "unique_values": [
        df[col].nunique(dropna=False)
        for col in df.columns
    ],
    "missing_values": [
        df[col].isna().sum()
        for col in df.columns
    ]
})

PROCESSED_DATA.mkdir(
    parents=True,
    exist_ok=True
)

dictionary_path = (
    PROCESSED_DATA /
    "731_data_dictionary_initial.csv"
)

data_dictionary.to_csv(
    dictionary_path,
    index=False
)

print("\nData dictionary saved to:")
print(dictionary_path)


# ============================================================
# 6. CONSTANT COLUMNS
# ============================================================

constant_columns = [
    col
    for col in df.columns
    if df[col].nunique(dropna=False) <= 1
]

print("\n" + "=" * 80)
print("CONSTANT COLUMNS")
print("=" * 80)

print(
    f"Number of constant columns: "
    f"{len(constant_columns)}"
)

for col in constant_columns:
    print(
        f"{col}: "
        f"{df[col].drop_duplicates().tolist()}"
    )


# ============================================================
# 7. VARIATION
# ============================================================

variation = (
    df.nunique(dropna=False)
      .sort_values(ascending=False)
      .to_frame(
          name="unique_values"
      )
)

print("\n" + "=" * 80)
print("COLUMN VARIATION")
print("=" * 80)

print(variation)


# ============================================================
# 8. NOVALUE ANALYSIS
# ============================================================

novalue_summary = []

for column in df.columns:

    count = (
        df[column]
        .astype(str)
        .str.strip()
        .str.upper()
        .eq("NOVALUE")
        .sum()
    )

    if count > 0:

        novalue_summary.append({
            "column": column,
            "novalue_count": count,
            "novalue_percentage": round(
                count / len(df) * 100,
                2
            )
        })


novalue_summary = pd.DataFrame(
    novalue_summary
)

print("\n" + "=" * 80)
print("NOVALUE SUMMARY")
print("=" * 80)

if len(novalue_summary) > 0:

    print(
        novalue_summary.sort_values(
            "novalue_count",
            ascending=False
        ).to_string(index=False)
    )

else:

    print("No NOVALUE values found.")


# ============================================================
# 9. RULE EXPRESSION SEARCH
# ============================================================

rule_mask = df.astype(str).apply(
    lambda col: col.str.contains(
        r"\bin\s*\{",
        case=False,
        regex=True,
        na=False
    )
)

rule_counts = (
    rule_mask.sum()
    .sort_values(ascending=False)
)

print("\n" + "=" * 80)
print("COLUMNS CONTAINING 'in {...}' RULES")
print("=" * 80)

print(
    rule_counts[
        rule_counts > 0
    ]
)


# ============================================================
# 10. DISPLAY RULE EXAMPLES
# ============================================================

print("\n" + "=" * 80)
print("RULE EXAMPLES")
print("=" * 80)

for column in df.columns:

    matches = df.loc[
        rule_mask[column],
        column
    ].drop_duplicates()

    if len(matches) > 0:

        print("\n" + "-" * 80)
        print(f"COLUMN: {column}")

        for value in matches.head(20):

            print(
                repr(value)
            )


# ============================================================
# 11. SAMPLE DATA
# ============================================================

print("\n" + "=" * 80)
print("FIRST 10 ROWS")
print("=" * 80)

print(
    df.head(10).to_string()
)


# ============================================================
# 12. FINISHED
# ============================================================

print("\n" + "=" * 80)
print("EXPLORATION COMPLETE")
print("=" * 80)