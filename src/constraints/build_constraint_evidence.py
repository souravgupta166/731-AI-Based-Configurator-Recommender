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
# PACKAGE RULES
# ============================================================

PACKAGE_RULES = {

    "x731": "in {NOVALUE, 'x731'}",

    "G731ST-LT":
        "in {NOVALUE, 'G731ST-LT'}",

    "G731ST-HT":
        "in {NOVALUE, 'G731ST-HT'}",

    "F731TE":
        "in {NOVALUE, 'F731TE'}",

    "G731CA":
        "in {NOVALUE, 'G731CA'}",

    "F731PW":
        "in {NOVALUE, 'F731PW'}",

    "F731WD_DualChannel":
        "in {NOVALUE, 'F731WD_DualChannel'}",

    "F731WD_SingleChannel":
        "in {NOVALUE, 'F731WD_SingleChannel'}",

    "G731VG":
        "in {NOVALUE, 'G731VG'}",
}


# ============================================================
# LOAD
# ============================================================

df = pd.read_excel(
    RAW_FILE
)

df = df.drop(
    columns=["Unnamed: 0"],
    errors="ignore"
)


# ============================================================
# CORE CHARACTERISTICS
# ============================================================

CHARACTERISTICS = [

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
]


# ============================================================
# NORMALIZATION
# ============================================================

def clean(value):

    if pd.isna(value):
        return "MISSING"

    return str(value).strip()


# ============================================================
# EVIDENCE RECORDS
# ============================================================

records = []


for package_name, package_rule in PACKAGE_RULES.items():

    subset = df[
        df["packageFixed_dev"]
        == package_rule
    ].copy()

    sample_size = len(subset)

    if sample_size == 0:
        continue


    for characteristic in CHARACTERISTICS:

        values = (
            subset[characteristic]
            .map(clean)
        )

        counts = (
            values
            .value_counts()
        )

        # ----------------------------------------------------
        # Most common observed state
        # ----------------------------------------------------

        top_state = counts.index[0]

        top_count = counts.iloc[0]

        support_rate = (
            top_count
            / sample_size
        )


        # ----------------------------------------------------
        # Number of distinct states
        # ----------------------------------------------------

        distinct_states = (
            values.nunique()
        )


        # ----------------------------------------------------
        # Evidence classification
        # ----------------------------------------------------

        if support_rate == 1.0:

            if sample_size >= 30:

                evidence_level = (
                    "strong"
                )

            else:

                evidence_level = (
                    "strong_but_small_sample"
                )

        elif support_rate >= 0.90:

            evidence_level = (
                "high"
            )

        elif support_rate >= 0.75:

            evidence_level = (
                "moderate"
            )

        else:

            evidence_level = (
                "weak"
            )


        records.append({

            "package":
                package_name,

            "package_rule":
                package_rule,

            "characteristic":
                characteristic,

            "sample_size":
                sample_size,

            "distinct_states":
                distinct_states,

            "dominant_state":
                top_state,

            "dominant_count":
                top_count,

            "support_rate":
                round(
                    support_rate,
                    4
                ),

            "evidence_level":
                evidence_level
        })


# ============================================================
# CREATE DATAFRAME
# ============================================================

evidence = pd.DataFrame(
    records
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
    / "731_constraint_evidence.csv"
)

evidence.to_csv(
    output_file,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print("=" * 100)
print("731 CONSTRAINT EVIDENCE")
print("=" * 100)

print(
    f"\nEvidence records:"
    f" {len(evidence):,}"
)


# ------------------------------------------------------------
# Strong constraints
# ------------------------------------------------------------

strong = evidence[
    evidence["support_rate"]
    == 1.0
]

print("\n" + "=" * 100)
print("PERFECTLY CONSISTENT RELATIONSHIPS")
print("=" * 100)

print(
    strong[
        [
            "package",
            "characteristic",
            "sample_size",
            "dominant_state",
            "support_rate",
            "evidence_level"
        ]
    ]
    .to_string(index=False)
)


# ------------------------------------------------------------
# High support
# ------------------------------------------------------------

high = evidence[
    (evidence["support_rate"] >= 0.90)
    &
    (evidence["support_rate"] < 1.0)
]

print("\n" + "=" * 100)
print("HIGH-SUPPORT RELATIONSHIPS")
print("=" * 100)

print(
    high[
        [
            "package",
            "characteristic",
            "sample_size",
            "dominant_state",
            "support_rate",
            "evidence_level"
        ]
    ]
    .to_string(index=False)
)


print("\n" + "=" * 100)
print("SAVED")
print("=" * 100)

print(output_file)