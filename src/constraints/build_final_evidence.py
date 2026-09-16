import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

CONSTRAINT_EVIDENCE_FILE = (
    PROCESSED_DIR
    / "731_constraint_evidence.csv"
)

CONDITIONAL_SUMMARY_FILE = (
    PROCESSED_DIR
    / "731_conditional_dependency_summary.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR
    / "731_final_evidence.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

# These thresholds are deliberately conservative.
#
# The objective is NOT to maximize the number of rules.
#
# The objective is to avoid introducing unsupported assumptions
# into the synthetic dataset.


# ------------------------------------------------------------
# SOURCE-OBSERVED RULES
# ------------------------------------------------------------

# Perfect support with at least this many observations can
# become an enforceable rule.
MIN_HARD_SAMPLE_SIZE = 20

# Perfect support with smaller samples will not become a hard
# rule. It can influence the generator, but will not reject
# configurations.
MIN_WEIGHT_SAMPLE_SIZE = 8


# ------------------------------------------------------------
# STATISTICAL DEPENDENCIES
# ------------------------------------------------------------

# Minimum number of package contexts in which a dependency
# should appear before we allow it to influence generation.
MIN_DEPENDENCY_PACKAGE_COVERAGE = 3

# Mean conditional mutual information above this threshold
# is considered meaningful enough to use as a soft dependency.
MIN_SOFT_MI = 0.50


# ============================================================
# LOAD SOURCE-OBSERVED EVIDENCE
# ============================================================

print("=" * 100)
print("BUILDING FINAL 731 EVIDENCE HIERARCHY")
print("=" * 100)


if not CONSTRAINT_EVIDENCE_FILE.exists():

    raise FileNotFoundError(
        f"\nCould not find:\n"
        f"{CONSTRAINT_EVIDENCE_FILE}\n\n"
        f"Run first:\n"
        f"python3 src/constraints/build_constraint_evidence.py"
    )


constraint_evidence = pd.read_csv(
    CONSTRAINT_EVIDENCE_FILE
)


print(
    f"\nSource-observed evidence records: "
    f"{len(constraint_evidence):,}"
)


# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

constraint_evidence.columns = (
    constraint_evidence.columns
    .str.strip()
)


# ============================================================
# BUILD SOURCE-OBSERVED EVIDENCE
# ============================================================

final_records = []


for _, row in constraint_evidence.iterrows():

    package = row.get(
        "package",
        ""
    )

    characteristic = row.get(
        "characteristic",
        ""
    )

    sample_size = int(
        row.get(
            "sample_size",
            0
        )
    )

    dominant_state = row.get(
        "dominant_state",
        ""
    )

    support_rate = float(
        row.get(
            "support_rate",
            0
        )
    )

    evidence_level = row.get(
        "evidence_level",
        ""
    )


    # --------------------------------------------------------
    # Perfectly supported and sufficiently large
    # --------------------------------------------------------

    if (
        support_rate >= 1.0
        and sample_size >= MIN_HARD_SAMPLE_SIZE
    ):

        action = "ENFORCE"

        evidence_strength = (
            "HARD_SOURCE_OBSERVED"
        )

        reason = (
            "Perfect observed support with "
            f"{sample_size} observations."
        )


    # --------------------------------------------------------
    # Perfect support but smaller sample
    # --------------------------------------------------------

    elif (
        support_rate >= 1.0
        and sample_size >= MIN_WEIGHT_SAMPLE_SIZE
    ):

        action = "WEIGHT"

        evidence_strength = (
            "STRONG_SOURCE_OBSERVED_SMALL_SAMPLE"
        )

        reason = (
            "Perfect observed support, but "
            f"sample size is only {sample_size}; "
            "not sufficient for hard enforcement."
        )


    # --------------------------------------------------------
    # Moderate observed support
    # --------------------------------------------------------

    elif support_rate >= 0.80:

        action = "WEIGHT"

        evidence_strength = (
            "MODERATE_SOURCE_OBSERVED"
        )

        reason = (
            f"Observed support rate = "
            f"{support_rate:.3f}; "
            "use as probabilistic evidence."
        )


    # --------------------------------------------------------
    # Weak source evidence
    # --------------------------------------------------------

    else:

        action = "IGNORE"

        evidence_strength = (
            "WEAK_SOURCE_OBSERVED"
        )

        reason = (
            f"Observed support rate = "
            f"{support_rate:.3f}; "
            "insufficient evidence."
        )


    final_records.append({

        "evidence_source":
            "source_observed_rule",

        "package":
            package,

        "characteristic_a":
            package,

        "characteristic_b":
            characteristic,

        "relationship_type":
            "package_to_characteristic",

        "observed_state":
            dominant_state,

        "sample_size":
            sample_size,

        "support_rate":
            support_rate,

        "package_coverage":
            1,

        "global_mi":
            None,

        "conditional_mi":
            None,

        "evidence_strength":
            evidence_strength,

        "action":
            action,

        "reason":
            reason
    })


# ============================================================
# LOAD CONDITIONAL DEPENDENCY EVIDENCE
# ============================================================

print(
    "\nLoading conditional dependency evidence..."
)


if not CONDITIONAL_SUMMARY_FILE.exists():

    raise FileNotFoundError(
        f"\nCould not find:\n"
        f"{CONDITIONAL_SUMMARY_FILE}\n\n"
        f"Run first:\n"
        f"python3 src/constraints/analyze_conditional_dependencies.py"
    )


conditional = pd.read_csv(
    CONDITIONAL_SUMMARY_FILE
)


print(
    f"Conditional dependency records: "
    f"{len(conditional):,}"
)


# ============================================================
# BUILD STATISTICAL EVIDENCE
# ============================================================

for _, row in conditional.iterrows():

    characteristic_a = row[
        "characteristic_a"
    ]

    characteristic_b = row[
        "characteristic_b"
    ]

    global_mi = float(
        row["global_mi"]
    )

    mean_conditional_mi = float(
        row["mean_conditional_mi"]
    )

    median_conditional_mi = float(
        row["median_conditional_mi"]
    )

    max_conditional_mi = float(
        row["max_conditional_mi"]
    )

    package_coverage = int(
        row[
            "supporting_package_count"
        ]
    )

    package_count = int(
        row[
            "package_count_analyzed"
        ]
    )


    # --------------------------------------------------------
    # Strong soft dependency
    # --------------------------------------------------------

    if (
        package_coverage
        >= MIN_DEPENDENCY_PACKAGE_COVERAGE
        and mean_conditional_mi
        >= MIN_SOFT_MI
    ):

        action = "WEIGHT"

        evidence_strength = (
            "STRONG_STATISTICAL_DEPENDENCY"
        )

        reason = (
            f"Conditional dependency remains "
            f"meaningful across "
            f"{package_coverage}/{package_count} "
            f"package contexts."
        )


    # --------------------------------------------------------
    # Moderate dependency
    # --------------------------------------------------------

    elif (
        package_coverage >= 2
        and mean_conditional_mi >= 0.30
    ):

        action = "WEIGHT"

        evidence_strength = (
            "MODERATE_STATISTICAL_DEPENDENCY"
        )

        reason = (
            f"Moderate conditional dependency "
            f"across {package_coverage}/"
            f"{package_count} package contexts."
        )


    # --------------------------------------------------------
    # Weak dependency
    # --------------------------------------------------------

    else:

        action = "IGNORE"

        evidence_strength = (
            "WEAK_STATISTICAL_DEPENDENCY"
        )

        reason = (
            f"Insufficient package coverage "
            f"or conditional MI."
        )


    final_records.append({

        "evidence_source":
            "conditional_dependency",

        "package":
            None,

        "characteristic_a":
            characteristic_a,

        "characteristic_b":
            characteristic_b,

        "relationship_type":
            "characteristic_to_characteristic",

        "observed_state":
            None,

        "sample_size":
            None,

        "support_rate":
            None,

        "package_coverage":
            package_coverage,

        "global_mi":
            global_mi,

        "conditional_mi":
            mean_conditional_mi,

        "evidence_strength":
            evidence_strength,

        "action":
            action,

        "reason":
            reason
    })


# ============================================================
# CREATE FINAL DATAFRAME
# ============================================================

final_evidence = pd.DataFrame(
    final_records
)


# ============================================================
# SORT BY IMPORTANCE
# ============================================================

action_order = {

    "ENFORCE": 0,

    "WEIGHT": 1,

    "IGNORE": 2
}


final_evidence[
    "_action_order"
] = final_evidence[
    "action"
].map(
    action_order
)


final_evidence = (
    final_evidence
    .sort_values(
        [
            "_action_order",
            "evidence_strength"
        ]
    )
    .drop(
        columns=[
            "_action_order"
        ]
    )
)


# ============================================================
# SAVE
# ============================================================

final_evidence.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 100)
print("FINAL EVIDENCE SUMMARY")
print("=" * 100)


print(
    "\nTotal evidence records: "
    f"{len(final_evidence):,}"
)


print(
    "\nBy action:"
)


print(
    final_evidence[
        "action"
    ]
    .value_counts()
    .to_string()
)


print(
    "\nBy evidence source:"
)


print(
    final_evidence[
        "evidence_source"
    ]
    .value_counts()
    .to_string()
)


print(
    "\nBy evidence strength:"
)


print(
    final_evidence[
        "evidence_strength"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# SHOW HARD RULES
# ============================================================

hard_rules = final_evidence[
    final_evidence[
        "action"
    ]
    == "ENFORCE"
]


print("\n" + "=" * 100)
print("HARD SOURCE-OBSERVED RULES")
print("=" * 100)


if hard_rules.empty:

    print(
        "\nNo rules qualified for hard enforcement."
    )

else:

    print(
        hard_rules[
            [
                "package",
                "characteristic_b",
                "observed_state",
                "sample_size",
                "support_rate",
                "evidence_strength"
            ]
        ]
        .to_string(
            index=False
        )
    )


# ============================================================
# SHOW STRONG SOFT DEPENDENCIES
# ============================================================

soft_dependencies = final_evidence[
    (
        final_evidence[
            "action"
        ]
        == "WEIGHT"
    )
    &
    (
        final_evidence[
            "evidence_source"
        ]
        == "conditional_dependency"
    )
]


print("\n" + "=" * 100)
print("STRONG SOFT DEPENDENCIES")
print("=" * 100)


if soft_dependencies.empty:

    print(
        "\nNo soft dependencies qualified."
    )

else:

    print(
        soft_dependencies[
            [
                "characteristic_a",
                "characteristic_b",
                "global_mi",
                "conditional_mi",
                "package_coverage",
                "evidence_strength"
            ]
        ]
        .head(30)
        .to_string(
            index=False
        )
    )


# ============================================================
# SAVE
# ============================================================

print("\n" + "=" * 100)
print("SAVED")
print("=" * 100)

print(
    f"\nFinal evidence file:\n"
    f"{OUTPUT_FILE}"
)