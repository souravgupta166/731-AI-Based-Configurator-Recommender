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
# REMOVE NON-INFORMATIVE SOURCE COLUMN
# ============================================================

df = df.drop(
    columns=["Unnamed: 0"],
    errors="ignore"
)


# ============================================================
# CHARACTERISTICS TO PROFILE
# ============================================================

schema_file = (
    OUTPUT_DIR
    / "731_schema_initial.csv"
)

schema = pd.read_csv(
    schema_file
)

active_columns = (
    schema.loc[
        schema["initial_type"]
        == "variable_characteristic",
        "column_name"
    ]
    .tolist()
)


# ============================================================
# VALUE PROFILE
# ============================================================

records = []

for column in active_columns:

    series = df[column]

    counts = (
        series
        .astype(str)
        .str.strip()
        .value_counts(
            dropna=False
        )
    )

    for value, count in counts.items():

        records.append({

            "characteristic":
                column,

            "raw_value":
                value,

            "occurrences":
                count,

            "percentage":
                round(
                    count / len(df) * 100,
                    2
                )
        })


profile = pd.DataFrame(
    records
)


# ============================================================
# DISPLAY
# ============================================================

print("=" * 80)
print("731 ACTIVE CONFIGURATION VALUE PROFILE")
print("=" * 80)

for column in active_columns:

    print("\n" + "-" * 80)
    print(f"CHARACTERISTIC: {column}")
    print("-" * 80)

    subset = profile[
        profile["characteristic"]
        == column
    ]

    print(
        subset[
            [
                "raw_value",
                "occurrences",
                "percentage"
            ]
        ]
        .to_string(index=False)
    )


# ============================================================
# SAVE
# ============================================================

output_file = (
    OUTPUT_DIR
    / "731_configuration_value_profile.csv"
)

profile.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 80)
print("SAVED")
print("=" * 80)

print(output_file)