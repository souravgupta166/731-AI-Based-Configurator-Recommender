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
# NORMALISE
# ============================================================

def clean(value):

    if pd.isna(value):
        return "MISSING"

    return (
        str(value)
        .strip()
    )


df["stromEingaenge"] = (
    df["stromEingaenge"]
    .apply(clean)
)

df["temperaturEingaenge"] = (
    df["temperaturEingaenge"]
    .apply(clean)
)


# ============================================================
# CROSS-TABULATION
# ============================================================

relationship = pd.crosstab(
    df["stromEingaenge"],
    df["temperaturEingaenge"]
)


print("=" * 100)
print("RELATIONSHIP: stromEingaenge ↔ temperaturEingaenge")
print("=" * 100)

print("\nRows = stromEingaenge")
print("Columns = temperaturEingaenge")

print(
    relationship.to_string()
)


# ============================================================
# COMBINATION FREQUENCIES
# ============================================================

combinations = (
    df[
        [
            "stromEingaenge",
            "temperaturEingaenge"
        ]
    ]
    .value_counts()
    .reset_index(
        name="occurrences"
    )
)


print("\n" + "=" * 100)
print("OBSERVED COMBINATIONS")
print("=" * 100)

print(
    combinations.to_string(
        index=False
    )
)


# ============================================================
# IMPOSSIBLE / UNOBSERVED COMBINATIONS
# ============================================================

all_current = (
    df["stromEingaenge"]
    .unique()
)

all_temperature = (
    df["temperaturEingaenge"]
    .unique()
)

observed = set(
    zip(
        df["stromEingaenge"],
        df["temperaturEingaenge"]
    )
)

unobserved = []

for current in all_current:

    for temperature in all_temperature:

        if (
            current,
            temperature
        ) not in observed:

            unobserved.append({
                "stromEingaenge":
                    current,

                "temperaturEingaenge":
                    temperature
            })


unobserved = pd.DataFrame(
    unobserved
)


print("\n" + "=" * 100)
print("UNOBSERVED COMBINATIONS")
print("=" * 100)

print(
    f"\nPossible combinations from "
    f"marginal values: "
    f"{len(all_current) * len(all_temperature)}"
)

print(
    f"Observed combinations: "
    f"{len(observed)}"
)

print(
    f"Unobserved combinations: "
    f"{len(unobserved)}"
)

print(
    unobserved.to_string(
        index=False
    )
)