#!/usr/bin/env python3

"""
731 Knowledge Graph V1 Audit

Purpose:
    Audit the already-generated V1 Knowledge Graph before designing V2.

This script is READ-ONLY.
It does not modify the Knowledge Graph.

It checks:
    1. Node inventory
    2. Provenance
    3. Configuration families
    4. Technical characteristics
    5. Characteristic values
    6. Rule rows
    7. Configuration identity / duplicates
    8. Relationship structure
    9. Orphan / invalid references
   10. Synthetic configuration mapping

Outputs:
    graphs/731_knowledge_graph/v1_audit_report.txt
"""

from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRAPH_DIR = (
    PROJECT_ROOT
    / "graphs"
    / "731_knowledge_graph"
)

NODES_FILE = GRAPH_DIR / "nodes.csv"
EDGES_FILE = GRAPH_DIR / "relationships.csv"

REPORT_FILE = (
    GRAPH_DIR
    / "v1_audit_report.txt"
)


# ============================================================
# LOAD
# ============================================================

print("\n" + "=" * 70)
print("731 KNOWLEDGE GRAPH V1 AUDIT")
print("=" * 70)

print(f"\nGraph directory:")
print(GRAPH_DIR)

if not NODES_FILE.exists():
    raise FileNotFoundError(
        f"Missing nodes file: {NODES_FILE}"
    )

if not EDGES_FILE.exists():
    raise FileNotFoundError(
        f"Missing relationships file: {EDGES_FILE}"
    )


nodes = pd.read_csv(
    NODES_FILE,
    low_memory=False
).fillna("")

edges = pd.read_csv(
    EDGES_FILE,
    low_memory=False
).fillna("")


report = []


def section(title):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    report.append("\n" + "=" * 70)
    report.append(title)
    report.append("=" * 70)


def line(text=""):

    print(text)
    report.append(str(text))


# ============================================================
# 1. BASIC INVENTORY
# ============================================================

section("1. BASIC GRAPH INVENTORY")

line(f"Nodes:         {len(nodes):,}")
line(f"Relationships: {len(edges):,}")

line(
    f"Node columns:  {', '.join(nodes.columns)}"
)

line(
    f"Edge columns:  {', '.join(edges.columns)}"
)


# ============================================================
# 2. NODE TYPE INVENTORY
# ============================================================

section("2. NODE TYPE INVENTORY")

node_counts = (
    nodes["node_type"]
    .value_counts()
    .sort_index()
)

for node_type, count in node_counts.items():

    line(
        f"{node_type:25s} {count:>8,}"
    )


# ============================================================
# 3. PROVENANCE INVENTORY
# ============================================================

section("3. PROVENANCE INVENTORY")

prov_counts = (
    nodes["provenance"]
    .value_counts()
    .sort_index()
)

for provenance, count in prov_counts.items():

    line(
        f"{provenance:25s} {count:>8,}"
    )


line("\nProvenance by node type:")

prov_by_type = (
    nodes
    .groupby(
        [
            "node_type",
            "provenance"
        ]
    )
    .size()
    .reset_index(
        name="count"
    )
    .sort_values(
        [
            "node_type",
            "provenance"
        ]
    )
)

for _, row in prov_by_type.iterrows():

    line(
        f"  {row['node_type']:25s} "
        f"{row['provenance']:20s} "
        f"{row['count']:>8,}"
    )


# ============================================================
# 4. CONFIGURATION FAMILIES
# ============================================================

section("4. CONFIGURATION FAMILIES")

families = nodes[
    nodes["node_type"]
    == "ConfigurationFamily"
].copy()

line(
    f"Total configuration-family nodes: "
    f"{len(families):,}"
)

for _, row in families.sort_values(
    ["provenance", "label"]
).iterrows():

    line(
        f"  {row['label']:<35s} "
        f"| provenance={row['provenance']:<15s} "
        f"| source={row['source']}"
    )


# ============================================================
# 5. CHARACTERISTICS
# ============================================================

section("5. TECHNICAL CHARACTERISTICS")

characteristics = nodes[
    nodes["node_type"]
    == "Characteristic"
].copy()

line(
    f"Total characteristics: "
    f"{len(characteristics):,}"
)

for _, row in characteristics.sort_values(
    "label"
).iterrows():

    line(
        f"  {row['label']:<40s} "
        f"| {row['provenance']}"
    )


# ============================================================
# 6. CHARACTERISTIC VALUES
# ============================================================

section("6. CHARACTERISTIC VALUES")

values = nodes[
    nodes["node_type"]
    == "CharacteristicValue"
].copy()

line(
    f"Total characteristic-value nodes: "
    f"{len(values):,}"
)

if "label" in values.columns:

    duplicate_value_labels = (
        values["label"]
        .value_counts()
    )

    duplicate_value_labels = (
        duplicate_value_labels[
            duplicate_value_labels > 1
        ]
    )

    line(
        f"Value labels occurring more than once: "
        f"{len(duplicate_value_labels):,}"
    )

    if not duplicate_value_labels.empty:

        line("\nMost repeated value labels:")

        for label, count in (
            duplicate_value_labels
            .head(30)
            .items()
        ):

            line(
                f"  {label:<40s} {count:>6,}"
            )


# ============================================================
# 7. RULE ROWS
# ============================================================

section("7. REAL CONFIGURATOR RULE ROWS")

rules = nodes[
    nodes["node_type"]
    == "RuleRow"
].copy()

line(
    f"RuleRow nodes: {len(rules):,}"
)

line(
    "These represent rows from the real 731 "
    "configurator workbook."
)

line(
    "They must NOT be interpreted as historical "
    "customer orders or historical engineer decisions."
)

if "source" in rules.columns:

    unique_rule_sources = (
        rules["source"]
        .astype(str)
        .nunique()
    )

    line(
        f"Unique rule source labels: "
        f"{unique_rule_sources:,}"
    )


# ============================================================
# 8. CONFIGURATIONS
# ============================================================

section("8. SYNTHETIC CONFIGURATION NODES")

configs = nodes[
    nodes["node_type"]
    == "Configuration"
].copy()

line(
    f"Configuration nodes: "
    f"{len(configs):,}"
)

line(
    "These are canonical configuration entities "
    "represented in the synthetic configuration catalogue."
)


# ============================================================
# 9. CONFIGURATION NODE ID DUPLICATION
# ============================================================

section("9. CONFIGURATION IDENTITY CHECK")

if "node_id" in configs.columns:

    duplicate_node_ids = (
        configs["node_id"]
        .value_counts()
    )

    duplicate_node_ids = (
        duplicate_node_ids[
            duplicate_node_ids > 1
        ]
    )

    line(
        f"Duplicate Configuration node IDs: "
        f"{len(duplicate_node_ids):,}"
    )


# ============================================================
# 10. SOURCE ROW CHECK
# ============================================================

section("10. SOURCE ROW / PROVENANCE CHECK")

if "source_row" in nodes.columns:

    for node_type in sorted(
        nodes["node_type"]
        .unique()
    ):

        subset = nodes[
            nodes["node_type"]
            == node_type
        ]

        missing_source = (
            subset["source_row"]
            .astype(str)
            .str.strip()
            == ""
        ).sum()

        line(
            f"{node_type:25s} "
            f"missing source_row: "
            f"{missing_source:,} / "
            f"{len(subset):,}"
        )


# ============================================================
# 11. RELATIONSHIP INVENTORY
# ============================================================

section("11. RELATIONSHIP INVENTORY")

relation_counts = (
    edges["relation"]
    .value_counts()
    .sort_index()
)

for relation, count in relation_counts.items():

    line(
        f"{relation:35s} "
        f"{count:>10,}"
    )


# ============================================================
# 12. RELATIONSHIP PROVENANCE
# ============================================================

section("12. RELATIONSHIP PROVENANCE")

edge_prov = (
    edges["provenance"]
    .value_counts()
    .sort_index()
)

for provenance, count in edge_prov.items():

    line(
        f"{provenance:25s} "
        f"{count:>10,}"
    )


# ============================================================
# 13. INVALID REFERENCES
# ============================================================

section("13. RELATIONSHIP INTEGRITY")

valid_node_ids = set(
    nodes["node_id"]
    .astype(str)
)

source_ids = set(
    edges["source_id"]
    .astype(str)
)

target_ids = set(
    edges["target_id"]
    .astype(str)
)

invalid_sources = (
    source_ids
    - valid_node_ids
)

invalid_targets = (
    target_ids
    - valid_node_ids
)

line(
    f"Invalid source references: "
    f"{len(invalid_sources):,}"
)

line(
    f"Invalid target references: "
    f"{len(invalid_targets):,}"
)


# ============================================================
# 14. ORPHAN NODES
# ============================================================

section("14. ORPHAN NODE CHECK")

connected_ids = (
    source_ids
    | target_ids
)

orphan_nodes = nodes[
    ~nodes["node_id"]
    .astype(str)
    .isin(connected_ids)
].copy()

line(
    f"Orphan nodes: "
    f"{len(orphan_nodes):,}"
)

if not orphan_nodes.empty:

    line("\nOrphan node types:")

    orphan_counts = (
        orphan_nodes["node_type"]
        .value_counts()
    )

    for node_type, count in orphan_counts.items():

        line(
            f"  {node_type:<30s} "
            f"{count:>8,}"
        )


# ============================================================
# 15. RELATIONSHIP TYPE STRUCTURE
# ============================================================

section("15. RELATIONSHIP STRUCTURE")

for relation in sorted(
    edges["relation"]
    .astype(str)
    .unique()
):

    subset = edges[
        edges["relation"]
        .astype(str)
        == relation
    ]

    source_types = []

    target_types = []

    for _, edge in subset.head(5000).iterrows():

        source_node = nodes[
            nodes["node_id"]
            .astype(str)
            == str(edge["source_id"])
        ]

        target_node = nodes[
            nodes["node_id"]
            .astype(str)
            == str(edge["target_id"])
        ]

        if not source_node.empty:

            source_types.append(
                source_node.iloc[0]["node_type"]
            )

        if not target_node.empty:

            target_types.append(
                target_node.iloc[0]["node_type"]
            )


    source_types = sorted(
        set(source_types)
    )

    target_types = sorted(
        set(target_types)
    )

    line(
        f"\n{relation}"
    )

    line(
        f"  source types: "
        f"{source_types}"
    )

    line(
        f"  target types: "
        f"{target_types}"
    )


# ============================================================
# 16. CONFIGURATION CONNECTIVITY
# ============================================================

section("16. CONFIGURATION CONNECTIVITY")

config_ids = set(
    configs["node_id"]
    .astype(str)
)

config_edges = edges[
    (
        edges["source_id"]
        .astype(str)
        .isin(config_ids)
    )
    |
    (
        edges["target_id"]
        .astype(str)
        .isin(config_ids)
    )
]

line(
    f"Relationships touching Configuration nodes: "
    f"{len(config_edges):,}"
)

config_relation_counts = (
    config_edges["relation"]
    .value_counts()
)

for relation, count in (
    config_relation_counts.items()
):

    line(
        f"  {relation:<35s} "
        f"{count:>10,}"
    )


# ============================================================
# 17. CONFIGURATION CHARACTERISTIC COVERAGE
# ============================================================

section(
    "17. CONFIGURATION CHARACTERISTIC COVERAGE"
)

config_value_edges = edges[
    (
        edges["relation"]
        == "HAS_VALUE"
    )
    &
    (
        edges["source_id"]
        .astype(str)
        .isin(config_ids)
    )
]

line(
    f"Configuration → HAS_VALUE relationships: "
    f"{len(config_value_edges):,}"
)

if len(configs) > 0:

    coverage = (
        config_value_edges
        .groupby("source_id")
        .size()
    )

    line(
        f"Configurations with at least one "
        f"technical value: "
        f"{coverage.shape[0]:,} / "
        f"{len(configs):,}"
    )

    line(
        f"Mean values per configuration: "
        f"{coverage.mean():.2f}"
    )

    line(
        f"Median values per configuration: "
        f"{coverage.median():.2f}"
    )

    line(
        f"Maximum values per configuration: "
        f"{coverage.max():,}"
    )


# ============================================================
# 18. CHARACTERISTIC CONNECTIVITY
# ============================================================

section(
    "18. CHARACTERISTIC CONNECTIVITY"
)

characteristic_ids = set(
    characteristics["node_id"]
    .astype(str)
)

char_edges = edges[
    (
        edges["source_id"]
        .astype(str)
        .isin(characteristic_ids)
    )
    |
    (
        edges["target_id"]
        .astype(str)
        .isin(characteristic_ids)
    )
]

line(
    f"Relationships touching characteristics: "
    f"{len(char_edges):,}"
)


# ============================================================
# 19. FAMILY CONNECTIVITY
# ============================================================

section(
    "19. CONFIGURATION FAMILY CONNECTIVITY"
)

family_ids = set(
    families["node_id"]
    .astype(str)
)

family_edges = edges[
    (
        edges["source_id"]
        .astype(str)
        .isin(family_ids)
    )
    |
    (
        edges["target_id"]
        .astype(str)
        .isin(family_ids)
    )
]

line(
    f"Relationships touching family nodes: "
    f"{len(family_edges):,}"
)

family_relation_counts = (
    family_edges["relation"]
    .value_counts()
)

for relation, count in (
    family_relation_counts.items()
):

    line(
        f"  {relation:<35s} "
        f"{count:>10,}"
    )


# ============================================================
# 20. SUMMARY / V2 DESIGN FLAGS
# ============================================================

section(
    "20. V2 DESIGN FLAGS"
)

flags = []


if len(invalid_sources) == 0 and len(
    invalid_targets
) == 0:

    flags.append(
        "PASS: all relationship references "
        "point to existing nodes."
    )

else:

    flags.append(
        "REVIEW: invalid relationship references exist."
    )


if len(orphan_nodes) == 0:

    flags.append(
        "PASS: no orphan nodes."
    )

else:

    flags.append(
        "REVIEW: orphan nodes exist."
    )


if len(configs) < 5000:

    flags.append(
        "IMPORTANT: Configuration nodes are fewer "
        "than the 5,000 synthetic configuration rows. "
        "V2 should explicitly preserve row-level "
        "configuration-record provenance rather than "
        "silently collapsing records."
    )


if len(families) > 9:

    flags.append(
        "IMPORTANT: More configuration-family nodes "
        "exist in the KG than the previously observed "
        "set of real package expressions. V2 should "
        "separate observed family knowledge from "
        "synthetic package-context values."
    )


flags.append(
    "IMPORTANT: Do not assign undocumented expansions "
    "to internal technical codes without source evidence."
)

flags.append(
    "IMPORTANT: Real RuleRows represent configurator "
    "knowledge, not historical customer orders."
)

flags.append(
    "V2 goal: explicitly represent requirements, "
    "constraints, configuration records, provenance "
    "and explainable validity relationships."
)


for flag in flags:

    line(f"- {flag}")


# ============================================================
# SAVE REPORT
# ============================================================

REPORT_FILE.write_text(
    "\n".join(report),
    encoding="utf-8"
)


print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)

print(
    f"\nReport saved to:\n{REPORT_FILE}"
)

print("\nNo graph files were modified.")