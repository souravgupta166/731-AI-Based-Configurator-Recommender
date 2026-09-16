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


# ============================================================
# LOAD
# ============================================================

df = pd.read_excel(RAW_FILE)


# ============================================================
# TARGET RULE-STATE COMBINATION
# ============================================================

strom_rule = "in {NOVALUE, 'IS1', 'IS2'}"

temperature_rule = (
    "in {NOVALUE, 'TT1', 'TT2'}"
)


mask = (
    df["stromEingaenge"].astype(str).str.strip()
    .eq(strom_rule)
    &
    df["temperaturEingaenge"].astype(str).str.strip()
    .eq(temperature_rule)
)


subset = df.loc[mask].copy()


print("=" * 100)
print("RULE-STATE CONTEXT INSPECTION")
print("=" * 100)

print(
    f"\nMatching rows: {len(subset)}"
)


# ============================================================
# SHOW VARIABLE COLUMNS
# ============================================================

for column in df.columns:

    unique_values = (
        subset[column]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    print("\n" + "-" * 100)
    print(f"COLUMN: {column}")
    print(f"Unique values: {len(unique_values)}")

    for value in unique_values[:20]:
        print("   ", value)

    if len(unique_values) > 20:
        print(
            f"    ... "
            f"{len(unique_values) - 20} more"
        )


# ============================================================
# SAVE MATCHING ROWS
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


output_file = (
    OUTPUT_DIR
    / "strom_temperature_context.csv"
)

subset.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 100)
print("SAVED")
print("=" * 100)

print(output_file)