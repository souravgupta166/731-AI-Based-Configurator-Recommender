import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVIDENCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "731_constraint_evidence.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


# ============================================================
# LOAD EVIDENCE
# ============================================================

evidence = pd.read_csv(
    EVIDENCE_FILE
)


# ============================================================
# BUILD GRAPH EDGES
# ============================================================

edges = []


for _, row in evidence.iterrows():

    package = row["package"]

    characteristic = row[
        "characteristic"
    ]

    sample_size = int(
        row["sample_size"]
    )

    support_rate = float(
        row["support_rate"]
    )

    dominant_state = str(
        row["dominant_state"]
    )

    evidence_level = row[
        "evidence_level"
    ]


    # --------------------------------------------------------
    # Determine relationship type
    # --------------------------------------------------------

    if support_rate == 1.0:

        relationship_type = (
            "observed_consistent_rule"
        )

    elif support_rate >= 0.90:

        relationship_type = (
            "observed_high_support_rule"
        )

    elif support_rate >= 0.75:

        relationship_type = (
            "observed_moderate_support"
        )

    else:

        relationship_type = (
            "observed_weak_association"
        )


    # --------------------------------------------------------
    # Evidence strength
    # --------------------------------------------------------

    if sample_size >= 100:

        evidence_strength = (
            "high_sample"
        )

    elif sample_size >= 30:

        evidence_strength = (
            "medium_sample"
        )

    else:

        evidence_strength = (
            "small_sample"
        )


    # --------------------------------------------------------
    # Graph edge
    # --------------------------------------------------------

    edges.append({

        "source_node":
            package,

        "target_node":
            characteristic,

        "relationship":
            relationship_type,

        "observed_state":
            dominant_state,

        "support_rate":
            support_rate,

        "sample_size":
            sample_size,

        "evidence_level":
            evidence_level,

        "evidence_strength":
            evidence_strength,

        # IMPORTANT:
        # This is evidence, not yet a
        # confirmed engineering constraint.
        "constraint_status":
            "observed_association"

    })


# ============================================================
# CREATE GRAPH TABLE
# ============================================================

graph = pd.DataFrame(
    edges
)


# ============================================================
# SORT
# ============================================================

graph = graph.sort_values(
    [
        "constraint_status",
        "support_rate",
        "sample_size"
    ],
    ascending=[
        True,
        False,
        False
    ]
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
    / "731_constraint_graph_edges.csv"
)


graph.to_csv(
    output_file,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print("=" * 100)
print("731 CONSTRAINT GRAPH")
print("=" * 100)

print(
    f"\nGraph edges:"
    f" {len(graph):,}"
)


# ------------------------------------------------------------
# Strong evidence
# ------------------------------------------------------------

strong = graph[
    graph["support_rate"]
    == 1.0
]


print("\n" + "=" * 100)
print("OBSERVED 100% SUPPORT RELATIONSHIPS")
print("=" * 100)

print(
    strong[
        [
            "source_node",
            "target_node",
            "observed_state",
            "support_rate",
            "sample_size",
            "evidence_strength"
        ]
    ]
    .to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Large-sample relationships
# ------------------------------------------------------------

large_sample = graph[
    graph["sample_size"]
    >= 100
]


print("\n" + "=" * 100)
print("RELATIONSHIPS WITH >=100 SOURCE ROWS")
print("=" * 100)

print(
    large_sample[
        [
            "source_node",
            "target_node",
            "observed_state",
            "support_rate",
            "sample_size"
        ]
    ]
    .to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Evidence counts
# ------------------------------------------------------------

print("\n" + "=" * 100)
print("EVIDENCE STRENGTH")
print("=" * 100)

print(
    graph[
        "evidence_strength"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 100)
print("SAVED")
print("=" * 100)

print(output_file)