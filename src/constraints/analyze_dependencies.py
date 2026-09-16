import pandas as pd
import numpy as np
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

print("=" * 80)
print("731 CONFIGURATION DEPENDENCY ANALYSIS")
print("=" * 80)

print(f"\nRows: {len(df):,}")
print(f"Columns: {len(df.columns):,}")


# ============================================================
# CHARACTERISTICS
# ============================================================

characteristics = [
    "packageFixed_dev",
    "dataInterface",
    "stromSchaltbar",
    "stromEingaenge",
    "temperaturEingaenge",
    "binaerDigitalOpenColl_MN",
    "binaerOpenColl_MP",
    "certification",
    "waveInjector",
    "housing",
    "explosionApproval",
    "dynamicGasMaster",
    "dev_advMeterVerification",
    "customUserFluid",
    "numberOfChannels",
    "powerSupply",
]


# ============================================================
# NORMALISE RULE TEXT
# ============================================================

def clean_rule(value):

    if pd.isna(value):
        return "MISSING"

    return (
        str(value)
        .strip()
        .replace(" ", "")
    )


for column in characteristics:
    df[column] = df[column].apply(clean_rule)


# ============================================================
# ENTROPY
# ============================================================

def entropy(series):

    probabilities = (
        series
        .value_counts(normalize=True)
    )

    return -np.sum(
        probabilities *
        np.log2(probabilities)
    )


# ============================================================
# CONDITIONAL ENTROPY
# ============================================================

def conditional_entropy(
    target,
    condition
):

    total = len(target)

    result = 0.0

    for _, group in target.groupby(
        condition
    ):

        weight = len(group) / total

        result += (
            weight *
            entropy(group)
        )

    return result


# ============================================================
# MUTUAL INFORMATION
# ============================================================

def mutual_information(
    x,
    y
):

    h_x = entropy(x)

    h_y = entropy(y)

    h_xy = conditional_entropy(
        x,
        y
    )

    return h_x - h_xy


# ============================================================
# PAIRWISE ANALYSIS
# ============================================================

results = []

for i, col_a in enumerate(
    characteristics
):

    for col_b in characteristics[i + 1:]:

        mi = mutual_information(
            df[col_a],
            df[col_b]
        )

        results.append({
            "characteristic_a": col_a,
            "characteristic_b": col_b,
            "mutual_information": mi,
            "unique_a": df[col_a].nunique(),
            "unique_b": df[col_b].nunique(),
        })


results = pd.DataFrame(results)

results = results.sort_values(
    "mutual_information",
    ascending=False
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 80)
print("STRONGEST RELATIONSHIPS")
print("=" * 80)

print(
    results
    .head(30)
    .to_string(index=False)
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
    / "731_dependency_analysis.csv"
)

results.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 80)
print("SAVED")
print("=" * 80)

print(output_file)