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

print("=" * 80)
print("731 CONFIGURATION CONTEXT PROFILING")
print("=" * 80)

print(f"\nRows: {len(df):,}")
print(f"Columns: {len(df.columns):,}")


# ============================================================
# SELECT IMPORTANT CHARACTERISTICS
# ============================================================

focus_columns = [
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


missing_columns = [
    col
    for col in focus_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# ============================================================
# 1. PACKAGE DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("PACKAGE / CONFIGURATION CONTEXTS")
print("=" * 80)

package_distribution = (
    df["packageFixed_dev"]
    .value_counts(dropna=False)
)

print(package_distribution)


# ============================================================
# 2. PROFILE EACH PACKAGE
# ============================================================

for package in package_distribution.index:

    subset = df[
        df["packageFixed_dev"] == package
    ]

    print("\n" + "-" * 80)
    print(f"PACKAGE: {package}")
    print(f"ROWS: {len(subset)}")
    print("-" * 80)

    for column in focus_columns:

        if column == "packageFixed_dev":
            continue

        values = (
            subset[column]
            .drop_duplicates()
            .tolist()
        )

        print(
            f"{column}: "
            f"{len(values)} unique value(s)"
        )

        print(
            "   ",
            values[:10]
        )


# ============================================================
# 3. DISTINCT RULE SIGNATURE
# ============================================================

print("\n" + "=" * 80)
print("CONFIGURATION SIGNATURES")
print("=" * 80)

signature_columns = [
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


df["configuration_signature"] = (
    df[signature_columns]
    .astype(str)
    .agg(" || ".join, axis=1)
)


signature_counts = (
    df["configuration_signature"]
    .value_counts()
)

print(
    f"\nDistinct configuration signatures: "
    f"{len(signature_counts):,}"
)

print("\nMost frequent signatures:")

print(
    signature_counts
    .head(20)
)


# ============================================================
# 4. SAVE PACKAGE PROFILE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

package_profile = []

for package in package_distribution.index:

    subset = df[
        df["packageFixed_dev"] == package
    ]

    record = {
        "packageFixed_dev": package,
        "row_count": len(subset),
    }

    for column in focus_columns:

        if column == "packageFixed_dev":
            continue

        record[
            f"{column}_unique_count"
        ] = subset[column].nunique(
            dropna=False
        )

    package_profile.append(record)


package_profile = pd.DataFrame(
    package_profile
)

output_file = (
    OUTPUT_DIR
    / "731_package_context_profile.csv"
)

package_profile.to_csv(
    output_file,
    index=False
)

print("\nSaved:")
print(output_file)