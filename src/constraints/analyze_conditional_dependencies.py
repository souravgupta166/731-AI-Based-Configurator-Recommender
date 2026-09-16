import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PROJECT PATHS
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
# CONFIGURATION CHARACTERISTICS
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

    "binaerDigitalOpenColl_MN",

    "binaerOpenColl_MP",

    "steamApplication",

    "waveInjector",

    "dev_advMeterVerification",

    "dynamicGasMaster",

    "customUserFluid",
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 100)
print("LOADING 731 CONFIGURATOR")
print("=" * 100)

if not RAW_FILE.exists():

    raise FileNotFoundError(
        f"Could not find source file:\n{RAW_FILE}"
    )


df = pd.read_excel(
    RAW_FILE
)


print(
    f"\nOriginal dataset: "
    f"{df.shape[0]:,} rows × "
    f"{df.shape[1]} columns"
)


# ============================================================
# REMOVE NON-INFORMATIVE COLUMN
# ============================================================

if "Unnamed: 0" in df.columns:

    df = df.drop(
        columns=["Unnamed: 0"]
    )

    print(
        "\nRemoved non-informative column: "
        "Unnamed: 0"
    )


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

missing_columns = [
    column
    for column in CHARACTERISTICS
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "\nMissing expected characteristics:\n"
        + "\n".join(
            f"  - {column}"
            for column in missing_columns
        )
    )


# ============================================================
# CLEAN VALUES
# ============================================================

for column in CHARACTERISTICS:

    df[column] = (
        df[column]
        .fillna("MISSING")
        .astype(str)
        .str.strip()
    )


# ============================================================
# ENTROPY
# ============================================================

def entropy(series):
    """
    Calculate Shannon entropy of a categorical series.
    """

    probabilities = (
        series
        .value_counts(
            normalize=True
        )
    )

    probabilities = probabilities[
        probabilities > 0
    ]

    if len(probabilities) == 0:

        return 0.0

    return float(
        -np.sum(
            probabilities
            * np.log2(
                probabilities
            )
        )
    )


# ============================================================
# CONDITIONAL ENTROPY
# ============================================================

def conditional_entropy(
    target,
    condition
):
    """
    H(target | condition)
    """

    total = len(target)

    if total == 0:

        return 0.0

    temp = pd.DataFrame({
        "target": target,
        "condition": condition
    })

    result = 0.0

    for _, group in temp.groupby(
        "condition",
        dropna=False
    ):

        weight = (
            len(group)
            / total
        )

        result += (
            weight
            * entropy(
                group["target"]
            )
        )

    return result


# ============================================================
# MUTUAL INFORMATION
# ============================================================

def mutual_information(
    x,
    y
):
    """
    Calculate mutual information:

        MI(X,Y) = H(X) - H(X|Y)

    For categorical variables.
    """

    if len(x) == 0:

        return 0.0

    if x.nunique() <= 1:

        return 0.0

    if y.nunique() <= 1:

        return 0.0

    return (
        entropy(x)
        - conditional_entropy(
            x,
            y
        )
    )


# ============================================================
# GLOBAL MUTUAL INFORMATION
# ============================================================

print("\n" + "=" * 100)
print("CALCULATING GLOBAL DEPENDENCIES")
print("=" * 100)


global_results = []


for i, col_a in enumerate(
    CHARACTERISTICS
):

    for col_b in CHARACTERISTICS[
        i + 1:
    ]:

        subset = df[
            [
                col_a,
                col_b
            ]
        ].dropna()


        if len(subset) < 10:

            continue


        if (
            subset[col_a].nunique()
            <= 1
        ):

            continue


        if (
            subset[col_b].nunique()
            <= 1
        ):

            continue


        mi = mutual_information(
            subset[col_a],
            subset[col_b]
        )


        global_results.append({

            "characteristic_a":
                col_a,

            "characteristic_b":
                col_b,

            "global_mutual_information":
                mi,

            "sample_size":
                len(subset)
        })


global_results = pd.DataFrame(
    global_results
)


# ============================================================
# CONDITIONAL MI BY PACKAGE
# ============================================================

print(
    "\nCalculating dependencies "
    "within package contexts..."
)


conditional_records = []


package_values = (
    df[
        "packageFixed_dev"
    ]
    .dropna()
    .unique()
)


print(
    f"Package contexts found: "
    f"{len(package_values)}"
)


for _, pair in global_results.iterrows():

    col_a = pair[
        "characteristic_a"
    ]

    col_b = pair[
        "characteristic_b"
    ]

    global_mi = pair[
        "global_mutual_information"
    ]


    for package in package_values:

        package_df = df[
            df["packageFixed_dev"]
            == package
        ]


        subset = package_df[
            [
                col_a,
                col_b
            ]
        ].dropna()


        # ----------------------------------------------------
        # Ignore very small package groups
        # ----------------------------------------------------

        if len(subset) < 10:

            continue


        # ----------------------------------------------------
        # Ignore constant variables within package
        # ----------------------------------------------------

        if (
            subset[col_a].nunique()
            <= 1
        ):

            continue


        if (
            subset[col_b].nunique()
            <= 1
        ):

            continue


        mi = mutual_information(
            subset[col_a],
            subset[col_b]
        )


        conditional_records.append({

            "characteristic_a":
                col_a,

            "characteristic_b":
                col_b,

            "package":
                package,

            "package_sample_size":
                len(subset),

            "conditional_mutual_information":
                mi,

            "global_mutual_information":
                global_mi
        })


conditional = pd.DataFrame(
    conditional_records
)


# ============================================================
# CONDITIONAL DEPENDENCY SUMMARY
# ============================================================

summary_records = []


if not conditional.empty:

    grouped = conditional.groupby(
        [
            "characteristic_a",
            "characteristic_b"
        ]
    )


    for (
        col_a,
        col_b
    ), group in grouped:

        global_mi = (
            group[
                "global_mutual_information"
            ]
            .iloc[0]
        )


        mean_conditional_mi = (
            group[
                "conditional_mutual_information"
            ]
            .mean()
        )


        median_conditional_mi = (
            group[
                "conditional_mutual_information"
            ]
            .median()
        )


        max_conditional_mi = (
            group[
                "conditional_mutual_information"
            ]
            .max()
        )


        supporting_packages = (
            (
                group[
                    "conditional_mutual_information"
                ]
                > 0
            )
            .sum()
        )


        summary_records.append({

            "characteristic_a":
                col_a,

            "characteristic_b":
                col_b,

            "global_mi":
                global_mi,

            "mean_conditional_mi":
                mean_conditional_mi,

            "median_conditional_mi":
                median_conditional_mi,

            "max_conditional_mi":
                max_conditional_mi,

            "supporting_package_count":
                supporting_packages,

            "package_count_analyzed":
                len(group)
        })


summary = pd.DataFrame(
    summary_records
)


# ============================================================
# SORT RESULTS
# ============================================================

if not summary.empty:

    summary = summary.sort_values(
        [
            "mean_conditional_mi",
            "global_mi"
        ],
        ascending=False
    )


# ============================================================
# SAVE OUTPUTS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


summary_file = (
    OUTPUT_DIR
    / "731_conditional_dependency_summary.csv"
)


details_file = (
    OUTPUT_DIR
    / "731_conditional_dependency_details.csv"
)


summary.to_csv(
    summary_file,
    index=False
)


conditional.to_csv(
    details_file,
    index=False
)


# ============================================================
# TOP DEPENDENCIES
# ============================================================

print("\n" + "=" * 100)
print(
    "TOP RELATIONSHIPS AFTER CONDITIONING "
    "ON packageFixed_dev"
)
print("=" * 100)


if summary.empty:

    print(
        "\nNo conditional dependencies "
        "could be calculated."
    )

else:

    print(
        summary
        .head(30)
        .to_string(
            index=False
        )
    )


# ============================================================
# SPECIFIC RELATIONSHIPS
# ============================================================

interesting_pairs = [

    (
        "stromEingaenge",
        "temperaturEingaenge"
    ),

    (
        "certification",
        "explosionApproval"
    ),

    (
        "stromSchaltbar",
        "stromEingaenge"
    ),

    (
        "stromSchaltbar",
        "temperaturEingaenge"
    ),

    (
        "certification",
        "powerSupply"
    ),

    (
        "housing",
        "powerSupply"
    ),

    (
        "binaerDigitalOpenColl_MN",
        "binaerOpenColl_MP"
    ),

]


print("\n" + "=" * 100)
print(
    "INTERESTING RELATIONSHIPS"
)
print("=" * 100)


for col_a, col_b in interesting_pairs:

    if summary.empty:

        print(
            f"\n{col_a} ↔ {col_b}: "
            "not available"
        )

        continue


    match = summary[
        (
            (
                summary[
                    "characteristic_a"
                ]
                == col_a
            )
            &
            (
                summary[
                    "characteristic_b"
                ]
                == col_b
            )
        )
        |
        (
            (
                summary[
                    "characteristic_a"
                ]
                == col_b
            )
            &
            (
                summary[
                    "characteristic_b"
                ]
                == col_a
            )
        )
    ]


    if match.empty:

        print(
            f"\n{col_a} ↔ {col_b}: "
            "not found"
        )

    else:

        print(
            "\n"
            + match.to_string(
                index=False
            )
        )


# ============================================================
# SAVE CONFIRMATION
# ============================================================

print("\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)

print(
    f"\nGlobal dependency pairs: "
    f"{len(global_results):,}"
)

print(
    f"Conditional dependency records: "
    f"{len(conditional):,}"
)

print(
    f"\nSummary saved to:"
    f"\n{summary_file}"
)

print(
    f"\nDetailed results saved to:"
    f"\n{details_file}"
)