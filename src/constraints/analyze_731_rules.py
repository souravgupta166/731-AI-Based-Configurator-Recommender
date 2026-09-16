import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RULE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "731_extracted_rules.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


# ============================================================
# LOAD
# ============================================================

rules = pd.read_csv(RULE_FILE)

print("=" * 80)
print("731 RULE STRUCTURE ANALYSIS")
print("=" * 80)

print(f"\nTotal extracted rule records: {len(rules):,}")

print("\nColumns:")
for column in rules.columns:
    print(f"  - {column}")


# ============================================================
# 1. DISTINCT RULE SETS BY CHARACTERISTIC
# ============================================================

summary = (
    rules
    .groupby("characteristic")
    .agg(
        rule_occurrences=("raw_rule", "count"),
        distinct_rule_sets=("allowed_values", "nunique"),
    )
    .sort_values(
        "distinct_rule_sets",
        ascending=False
    )
)

print("\n" + "=" * 80)
print("DISTINCT RULE SETS BY CHARACTERISTIC")
print("=" * 80)

print(summary.to_string())


# ============================================================
# 2. DETAILED RULE SETS
# ============================================================

print("\n" + "=" * 80)
print("DETAILED RULE SETS")
print("=" * 80)

for characteristic in summary.index:

    subset = rules[
        rules["characteristic"] == characteristic
    ]

    grouped = (
        subset
        .groupby(
            ["allowed_values", "allowed_count"]
        )
        .size()
        .reset_index(
            name="occurrences"
        )
        .sort_values(
            "occurrences",
            ascending=False
        )
    )

    print("\n" + "-" * 80)
    print(f"CHARACTERISTIC: {characteristic}")
    print("-" * 80)

    print(
        grouped.to_string(
            index=False
        )
    )


# ============================================================
# 3. SAVE SUMMARY
# ============================================================

summary_path = (
    OUTPUT_DIR
    / "731_rule_summary.csv"
)

summary.to_csv(summary_path)

print("\n" + "=" * 80)
print("SUMMARY SAVED")
print("=" * 80)

print(summary_path)