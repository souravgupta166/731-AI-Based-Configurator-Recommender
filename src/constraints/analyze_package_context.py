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

df = df.drop(
    columns=["Unnamed: 0"],
    errors="ignore"
)


# ============================================================
# CORE CHARACTERISTICS
# ============================================================

core_characteristics = [

    "packageFixed_dev",
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
# PACKAGE CONTEXT SUMMARY
# ============================================================

package_summary = (
    df.groupby(
        "packageFixed_dev"
    )
    .size()
    .reset_index(
        name="rows"
    )
)

package_summary["percentage"] = (
    package_summary["rows"]
    / len(df)
    * 100
)


print("=" * 100)
print("731 PACKAGE CONTEXT ANALYSIS")
print("=" * 100)

print("\nPACKAGE DISTRIBUTION")

print(
    package_summary
    .sort_values(
        "rows",
        ascending=False
    )
    .to_string(
        index=False
    )
)


# ============================================================
# PACKAGE × CHARACTERISTIC
# ============================================================

records = []


for package in (
    df["packageFixed_dev"]
    .dropna()
    .unique()
):

    package_df = df[
        df["packageFixed_dev"]
        == package
    ]

    print("\n" + "=" * 100)
    print(
        f"PACKAGE: {package}"
    )
    print("=" * 100)

    print(
        f"Rows: {len(package_df)}"
    )


    for characteristic in (
        core_characteristics
    ):

        if characteristic == "packageFixed_dev":
            continue


        # ----------------------------------------------------
        # Number of distinct observed rule states
        # ----------------------------------------------------

        unique_states = (
            package_df[
                characteristic
            ]
            .astype(str)
            .nunique()
        )


        # ----------------------------------------------------
        # Most common states
        # ----------------------------------------------------

        states = (
            package_df[
                characteristic
            ]
            .astype(str)
            .value_counts()
            .head(5)
        )


        records.append({

            "package":
                package,

            "characteristic":
                characteristic,

            "rows":
                len(package_df),

            "distinct_rule_states":
                unique_states,

            "top_state":
                states.index[0]
                if len(states)
                else None,

            "top_state_count":
                states.iloc[0]
                if len(states)
                else 0,

            "top_state_percentage":
                round(
                    states.iloc[0]
                    / len(package_df)
                    * 100,
                    2
                )
                if len(states)
                else 0
        })


        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(
            f"\n{characteristic}"
        )

        print(
            states.to_string()
        )


# ============================================================
# SAVE
# ============================================================

result = pd.DataFrame(
    records
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


output_file = (
    OUTPUT_DIR
    / "731_package_characteristic_context.csv"
)


result.to_csv(
    output_file,
    index=False
)


print("\n" + "=" * 100)
print("SAVED")
print("=" * 100)

print(output_file)