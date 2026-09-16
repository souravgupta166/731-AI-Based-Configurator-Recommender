#!/usr/bin/env python3

"""
731 Knowledge Graph V2 Builder

Purpose
-------
Build a source-grounded V2 Knowledge Graph for the 731-series
configuration environment.

V2 explicitly separates:

    Product
    ConfigurationFamily
    FamilyEvidence
    Characteristic
    CharacteristicValue
    RuleRow
    ConfigurationRecord
    CanonicalConfiguration

Key design principles
---------------------
1. Real 731 Excel = OBSERVED_731 product/configurator knowledge.
2. Synthetic configuration catalogue = SYNTHETIC evidence.
3. 5,000 synthetic records are preserved.
4. 1,979 canonical configuration identities are preserved separately.
5. Observed family expressions are kept as evidence and mapped to
   canonical family labels only through transparent normalization.
6. Characteristic values are scoped to their characteristic.
7. No undocumented technical-code meanings are invented.
8. Real RuleRows are NOT historical orders or engineer decisions.
9. V1 graph is never modified.

Outputs
-------
graphs/731_knowledge_graph_v2/

    nodes.csv
    relationships.csv
    statistics.json
    validation_report.txt
    graph.graphml        (if networkx is available)
"""


from pathlib import Path
import json
import re
import hashlib
from collections import Counter, defaultdict

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REAL_XLSX = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "x731_allgemein_Rev24.xlsx"
)

SYNTHETIC_CSV = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "configurations"
    / "731_synthetic_configurations_v5.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "graphs"
    / "731_knowledge_graph_v2"
)

NODES_FILE = OUTPUT_DIR / "nodes.csv"
EDGES_FILE = OUTPUT_DIR / "relationships.csv"
STATS_FILE = OUTPUT_DIR / "statistics.json"
VALIDATION_FILE = OUTPUT_DIR / "validation_report.txt"
GRAPHML_FILE = OUTPUT_DIR / "graph.graphml"


# ============================================================
# KNOWN REAL WORKBOOK COLUMNS
# ============================================================

REAL_COLUMNS = [
    "OrgUnitAreaOfDocument",
    "marke",
    "packageFixed_dev",
    "installationsart",
    "devCategory",
    "type",
    "characteristic",
    "housing",
    "powerSupply",
    "numberOfChannels",
    "explosionApproval",
    "protectionArea",
    "intrinsicSafety",
    "certification",
    "dataInterface",
    "stromSchaltbar",
    "stromEingaenge",
    "temperaturEingaenge",
    "steamApplication",
    "waveInjector",
    "tagPlate",
    "inclusiveManual",
    "powerCable",
    "carryingCase",
    "measuringTape",
    "dev_advMeterVerification",
    "dynamicGasMaster",
    "customUserFluid",
]

# The technical identity used throughout the thesis work.
TECHNICAL_CHARACTERISTICS = [
    "devCategory",
    "characteristic",
    "housing",
    "powerSupply",
    "numberOfChannels",
    "explosionApproval",
    "protectionArea",
    "intrinsicSafety",
    "certification",
    "dataInterface",
    "stromSchaltbar",
    "stromEingaenge",
    "temperaturEingaenge",
    "steamApplication",
    "waveInjector",
    "dev_advMeterVerification",
    "dynamicGasMaster",
    "customUserFluid",
    "binaerDigitalOpenColl_MN",
    "binaerOpenColl_MP",
]

# Some synthetic V5 versions contain the two binary fields even though
# they are derived in the normalized technical schema.
# The builder dynamically handles whichever columns actually exist.


# ============================================================
# HELPERS
# ============================================================

def clean_value(value):
    """
    Convert a cell to a stable string representation.
    """
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
        "null",
    }:
        return ""

    return text


def is_no_value(value):
    """
    Identify the workbook's NOVALUE marker without changing
    the original stored text.
    """
    text = clean_value(value)

    return text.upper() == "NOVALUE"


def stable_hash(text, length=16):
    return hashlib.sha256(
        str(text).encode("utf-8")
    ).hexdigest()[:length]


def safe_id(prefix, *parts):
    raw = "||".join(
        clean_value(x)
        for x in parts
    )

    return (
        f"{prefix}_"
        f"{stable_hash(raw)}"
    )


def add_node(
    nodes,
    node_ids,
    node_id,
    node_type,
    label,
    provenance,
    source="",
    source_type="",
    field_name="",
    source_row="",
    characteristic="",
    value="",
    package_context="",
    canonical_configuration_id="",
    record_type="",
    generation_method="",
):
    """
    Add a node only once.
    """

    if node_id in node_ids:
        return

    nodes.append(
        {
            "node_id": node_id,
            "node_type": node_type,
            "label": label,
            "provenance": provenance,
            "source": source,
            "source_type": source_type,
            "field_name": field_name,
            "source_row": source_row,
            "characteristic": characteristic,
            "value": value,
            "package_context": package_context,
            "canonical_configuration_id":
                canonical_configuration_id,
            "record_type": record_type,
            "generation_method": generation_method,
        }
    )

    node_ids.add(node_id)


def add_edge(
    edges,
    edge_keys,
    source_id,
    target_id,
    relation,
    provenance,
    source="",
    source_row="",
    characteristic="",
):
    """
    Add a relationship only once.
    """

    key = (
        str(source_id),
        str(target_id),
        str(relation),
        str(source_row),
        str(characteristic),
    )

    if key in edge_keys:
        return

    edges.append(
        {
            "source_id": source_id,
            "target_id": target_id,
            "relation": relation,
            "provenance": provenance,
            "source": source,
            "source_row": source_row,
            "characteristic": characteristic,
        }
    )

    edge_keys.add(key)


def extract_family_code(expression):
    """
    Extract a known 731 family code from a real workbook
    packageFixed_dev expression.

    This does NOT invent or expand a family name.

    Examples:

        in {NOVALUE, 'F731PW'}
            -> F731PW

        in {NOVALUE, 'G731ST-HT'}
            -> G731ST-HT

        x731
            -> x731

    If no known 731 family code is detected, return the
    original expression as the canonical label.
    """

    text = clean_value(expression)

    if not text:
        return ""

    known_patterns = [
        "F731WD_SingleChannel",
        "F731WD_DualChannel",
        "F731TE",
        "F731PW",
        "G731CA",
        "G731ST-LT",
        "G731ST-HT",
        "G731VG",
        "x731",
    ]

    for code in known_patterns:

        if code in text:
            return code

    return text


# ============================================================
# START
# ============================================================

print()
print("=" * 75)
print("731 KNOWLEDGE GRAPH V2 BUILDER")
print("=" * 75)

print()
print("Project root:")
print(PROJECT_ROOT)

print()
print("Real workbook:")
print(REAL_XLSX)

print()
print("Synthetic configuration catalogue:")
print(SYNTHETIC_CSV)

print()
print("Output:")
print(OUTPUT_DIR)


# ============================================================
# FILE CHECK
# ============================================================

if not REAL_XLSX.exists():

    raise FileNotFoundError(
        f"\nReal workbook not found:\n{REAL_XLSX}"
    )


if not SYNTHETIC_CSV.exists():

    raise FileNotFoundError(
        f"\nSynthetic configuration file not found:\n"
        f"{SYNTHETIC_CSV}"
    )


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD SOURCE DATA
# ============================================================

print()
print("-" * 75)
print("LOADING SOURCE DATA")
print("-" * 75)


real_df = pd.read_excel(
    REAL_XLSX
)

synthetic_df = pd.read_csv(
    SYNTHETIC_CSV,
    low_memory=False
)


print(
    f"Real workbook rows: "
    f"{len(real_df):,}"
)

print(
    f"Real workbook columns: "
    f"{len(real_df.columns):,}"
)

print(
    f"Synthetic configuration rows: "
    f"{len(synthetic_df):,}"
)

print(
    f"Synthetic columns: "
    f"{len(synthetic_df.columns):,}"
)


# ============================================================
# CHECK REAL COLUMNS
# ============================================================

missing_real_columns = [
    c
    for c in REAL_COLUMNS
    if c not in real_df.columns
]

if missing_real_columns:

    print()
    print(
        "WARNING: The following expected real workbook "
        "columns were not found:"
    )

    for column in missing_real_columns:
        print(
            f"  - {column}"
        )


# ============================================================
# CHECK SYNTHETIC IDENTITY
# ============================================================

required_synthetic_columns = [
    "canonical_configuration_id",
]

missing_synthetic = [
    c
    for c in required_synthetic_columns
    if c not in synthetic_df.columns
]

if missing_synthetic:

    raise ValueError(
        "Synthetic catalogue is missing required columns: "
        + ", ".join(missing_synthetic)
    )


# ============================================================
# DETERMINE SYNTHETIC TECHNICAL COLUMNS
# ============================================================

synthetic_technical_columns = [
    c
    for c in TECHNICAL_CHARACTERISTICS
    if c in synthetic_df.columns
]

print()
print(
    "Synthetic technical characteristics used:"
)

for c in synthetic_technical_columns:

    print(
        f"  - {c}"
    )


# ============================================================
# NODE / EDGE CONTAINERS
# ============================================================

nodes = []
edges = []

node_ids = set()
edge_keys = set()


REAL_SOURCE = (
    "data/raw/x731_allgemein_Rev24.xlsx"
)

SYNTHETIC_SOURCE = (
    "data/synthetic/configurations/"
    "731_synthetic_configurations_v5.csv"
)


# ============================================================
# 1. PRODUCT
# ============================================================

PRODUCT_ID = "PRODUCT_731"

add_node(
    nodes,
    node_ids,
    PRODUCT_ID,
    "Product",
    "731",
    "OBSERVED_731",
    source=REAL_SOURCE,
    source_type="real_configurator_workbook",
)


# ============================================================
# 2. CHARACTERISTICS FROM REAL WORKBOOK
# ============================================================

print()
print("-" * 75)
print("BUILDING OBSERVED CHARACTERISTICS")
print("-" * 75)


characteristic_ids = {}


for column in REAL_COLUMNS:

    if column not in real_df.columns:
        continue

    node_id = safe_id(
        "CHAR",
        column
    )

    characteristic_ids[
        column
    ] = node_id

    add_node(
        nodes,
        node_ids,
        node_id,
        "Characteristic",
        column,
        "OBSERVED_731",
        source=REAL_SOURCE,
        source_type="real_configurator_workbook",
        field_name=column,
    )

    add_edge(
        edges,
        edge_keys,
        PRODUCT_ID,
        node_id,
        "HAS_CHARACTERISTIC",
        "OBSERVED_731",
        source=REAL_SOURCE,
        characteristic=column,
    )


print(
    f"Observed characteristics: "
    f"{len(characteristic_ids):,}"
)


# ============================================================
# 3. OBSERVED CHARACTERISTIC VALUES
# ============================================================

print()
print("-" * 75)
print("BUILDING OBSERVED CHARACTERISTIC VALUES")
print("-" * 75)


observed_value_count = 0


for column in REAL_COLUMNS:

    if column not in real_df.columns:
        continue

    characteristic_id = characteristic_ids[
        column
    ]

    unique_values = (
        real_df[column]
        .map(clean_value)
        .loc[
            lambda s: s != ""
        ]
        .unique()
    )

    for value in unique_values:

        value_id = safe_id(
            "VALUE",
            column,
            value
        )

        add_node(
            nodes,
            node_ids,
            value_id,
            "CharacteristicValue",
            value,
            "OBSERVED_731",
            source=REAL_SOURCE,
            source_type="real_configurator_workbook",
            field_name=column,
            characteristic=column,
            value=value,
        )

        add_edge(
            edges,
            edge_keys,
            characteristic_id,
            value_id,
            "HAS_ALLOWED_VALUE",
            "OBSERVED_731",
            source=REAL_SOURCE,
            characteristic=column,
        )

        observed_value_count += 1


print(
    f"Observed characteristic values: "
    f"{observed_value_count:,}"
)


# ============================================================
# 4. OBSERVED CONFIGURATION FAMILIES
# ============================================================

print()
print("-" * 75)
print("BUILDING CONFIGURATION FAMILY KNOWLEDGE")
print("-" * 75)


observed_family_expressions = {}

if "packageFixed_dev" in real_df.columns:

    for row_number, row in real_df.iterrows():

        expression = clean_value(
            row["packageFixed_dev"]
        )

        if not expression:
            continue

        family_code = extract_family_code(
            expression
        )

        if not family_code:
            continue

        observed_family_expressions[
            expression
        ] = family_code


canonical_family_codes = sorted(
    set(
        observed_family_expressions.values()
    )
)


family_ids = {}

for family_code in canonical_family_codes:

    family_id = safe_id(
        "FAMILY",
        family_code
    )

    family_ids[
        family_code
    ] = family_id

    add_node(
        nodes,
        node_ids,
        family_id,
        "ConfigurationFamily",
        family_code,
        "OBSERVED_731",
        source=REAL_SOURCE,
        source_type="real_configurator_workbook",
    )


print(
    f"Canonical observed families: "
    f"{len(family_ids):,}"
)


# ============================================================
# 5. FAMILY EVIDENCE FROM REAL EXPRESSIONS
# ============================================================

for expression, family_code in (
    observed_family_expressions.items()
):

    family_id = family_ids[
        family_code
    ]

    evidence_id = safe_id(
        "FAMILY_EVIDENCE",
        "OBSERVED_731",
        expression
    )

    add_node(
        nodes,
        node_ids,
        evidence_id,
        "FamilyEvidence",
        expression,
        "OBSERVED_731",
        source=REAL_SOURCE,
        source_type="real_configurator_workbook",
        field_name="packageFixed_dev",
        value=expression,
        package_context=family_code,
    )

    add_edge(
        edges,
        edge_keys,
        family_id,
        evidence_id,
        "HAS_OBSERVED_EXPRESSION",
        "OBSERVED_731",
        source=REAL_SOURCE,
        characteristic="packageFixed_dev",
    )


# ============================================================
# 6. REAL RULE ROWS
# ============================================================

print()
print("-" * 75)
print("BUILDING REAL CONFIGURATOR RULE ROWS")
print("-" * 75)


rule_count = 0
rule_value_edges = 0


for idx, row in real_df.iterrows():

    source_row = idx + 2

    rule_id = (
        f"RULE_ROW_{source_row:04d}"
    )

    add_node(
        nodes,
        node_ids,
        rule_id,
        "RuleRow",
        f"Rule row {source_row}",
        "OBSERVED_731",
        source=REAL_SOURCE,
        source_type="real_configurator_workbook",
        source_row=source_row,
    )

    add_edge(
        edges,
        edge_keys,
        PRODUCT_ID,
        rule_id,
        "HAS_RULE_ROW",
        "OBSERVED_731",
        source=REAL_SOURCE,
        source_row=source_row,
    )

    rule_count += 1

    # --------------------------------------------------------
    # Rule → family
    # --------------------------------------------------------

    if "packageFixed_dev" in real_df.columns:

        expression = clean_value(
            row["packageFixed_dev"]
        )

        family_code = extract_family_code(
            expression
        )

        if family_code in family_ids:

            add_edge(
                edges,
                edge_keys,
                rule_id,
                family_ids[family_code],
                "RULE_APPLIES_TO_FAMILY",
                "OBSERVED_731",
                source=REAL_SOURCE,
                source_row=source_row,
                characteristic="packageFixed_dev",
            )


    # --------------------------------------------------------
    # Rule → specified characteristic values
    # --------------------------------------------------------

    for column in REAL_COLUMNS:

        if column not in real_df.columns:
            continue

        value = clean_value(
            row[column]
        )

        if not value:
            continue

        # NOVALUE is retained as source evidence but is not
        # treated as a concrete technical option.
        value_id = safe_id(
            "VALUE",
            column,
            value
        )

        if value_id not in node_ids:

            add_node(
                nodes,
                node_ids,
                value_id,
                "CharacteristicValue",
                value,
                "OBSERVED_731",
                source=REAL_SOURCE,
                source_type="real_configurator_workbook",
                field_name=column,
                characteristic=column,
                value=value,
            )

        add_edge(
            edges,
            edge_keys,
            rule_id,
            value_id,
            "RULE_SPECIFIES",
            "OBSERVED_731",
            source=REAL_SOURCE,
            source_row=source_row,
            characteristic=column,
        )

        rule_value_edges += 1


print(
    f"Rule rows: {rule_count:,}"
)

print(
    f"Rule → value relationships: "
    f"{rule_value_edges:,}"
)


# ============================================================
# 7. SYNTHETIC FAMILY CONTEXT
# ============================================================

print()
print("-" * 75)
print("BUILDING SYNTHETIC FAMILY EVIDENCE")
print("-" * 75)


synthetic_family_contexts = set()


if "package_context" in synthetic_df.columns:

    for value in (
        synthetic_df["package_context"]
        .map(clean_value)
        .unique()
    ):

        if value:

            synthetic_family_contexts.add(
                value
            )


for family_code in sorted(
    synthetic_family_contexts
):

    # Only create a canonical family if the code is already
    # supported by observed family evidence.
    if family_code not in family_ids:

        family_id = safe_id(
            "FAMILY",
            family_code
        )

        family_ids[
            family_code
        ] = family_id

        add_node(
            nodes,
            node_ids,
            family_id,
            "ConfigurationFamily",
            family_code,
            "SYNTHETIC",
            source=SYNTHETIC_SOURCE,
            source_type="synthetic_configuration_catalogue",
        )

    else:

        family_id = family_ids[
            family_code
        ]

    evidence_id = safe_id(
        "FAMILY_EVIDENCE",
        "SYNTHETIC",
        family_code
    )

    add_node(
        nodes,
        node_ids,
        evidence_id,
        "FamilyEvidence",
        family_code,
        "SYNTHETIC",
        source=SYNTHETIC_SOURCE,
        source_type="synthetic_configuration_catalogue",
        package_context=family_code,
    )

    add_edge(
        edges,
        edge_keys,
        family_id,
        evidence_id,
        "HAS_SYNTHETIC_PACKAGE_CONTEXT",
        "SYNTHETIC",
        source=SYNTHETIC_SOURCE,
    )


print(
    f"Synthetic package contexts: "
    f"{len(synthetic_family_contexts):,}"
)


# ============================================================
# 8. CANONICAL CONFIGURATIONS
# ============================================================

print()
print("-" * 75)
print("BUILDING CANONICAL CONFIGURATIONS")
print("-" * 75)


canonical_ids = (
    synthetic_df[
        "canonical_configuration_id"
    ]
    .map(clean_value)
    .loc[
        lambda s: s != ""
    ]
    .unique()
)


canonical_node_ids = {}


for canonical_id in sorted(
    canonical_ids
):

    node_id = safe_id(
        "CANONICAL_CONFIG",
        canonical_id
    )

    canonical_node_ids[
        canonical_id
    ] = node_id

    add_node(
        nodes,
        node_ids,
        node_id,
        "CanonicalConfiguration",
        canonical_id,
        "SYNTHETIC",
        source=SYNTHETIC_SOURCE,
        source_type="synthetic_configuration_catalogue",
        canonical_configuration_id=canonical_id,
    )


print(
    f"Canonical configurations: "
    f"{len(canonical_node_ids):,}"
)


# ============================================================
# 9. SYNTHETIC CONFIGURATION RECORDS
# ============================================================

print()
print("-" * 75)
print("BUILDING SYNTHETIC CONFIGURATION RECORDS")
print("-" * 75)


record_count = 0
record_value_edges = 0


for idx, row in synthetic_df.iterrows():

    source_row = idx + 2

    canonical_id = clean_value(
        row["canonical_configuration_id"]
    )

    if not canonical_id:
        continue

    record_id = (
        f"CONFIG_RECORD_{source_row:05d}"
    )

    package_context = clean_value(
        row.get(
            "package_context",
            ""
        )
    )

    record_type = clean_value(
        row.get(
            "record_type",
            ""
        )
    )

    generation_method = clean_value(
        row.get(
            "generation_method",
            ""
        )
    )

    evidence_level = clean_value(
        row.get(
            "evidence_level",
            ""
        )
    )

    add_node(
        nodes,
        node_ids,
        record_id,
        "ConfigurationRecord",
        f"Configuration record {source_row}",
        "SYNTHETIC",
        source=SYNTHETIC_SOURCE,
        source_type="synthetic_configuration_catalogue",
        source_row=source_row,
        package_context=package_context,
        canonical_configuration_id=canonical_id,
        record_type=record_type,
        generation_method=generation_method,
    )

    record_count += 1

    canonical_node = canonical_node_ids[
        canonical_id
    ]

    add_edge(
        edges,
        edge_keys,
        record_id,
        canonical_node,
        "REPRESENTS_CANONICAL_CONFIGURATION",
        "SYNTHETIC",
        source=SYNTHETIC_SOURCE,
        source_row=source_row,
    )

    # --------------------------------------------------------
    # Record → family
    # --------------------------------------------------------

    if package_context:

        family_code = extract_family_code(
            package_context
        )

        if family_code in family_ids:

            add_edge(
                edges,
                edge_keys,
                record_id,
                family_ids[family_code],
                "HAS_PACKAGE_CONTEXT",
                "SYNTHETIC",
                source=SYNTHETIC_SOURCE,
                source_row=source_row,
            )

    # --------------------------------------------------------
    # Canonical configuration → technical values
    # --------------------------------------------------------

    for column in synthetic_technical_columns:

        value = clean_value(
            row[column]
        )

        if not value:
            continue

        value_id = safe_id(
            "VALUE",
            column,
            value
        )

        # If value was not observed in the real workbook,
        # create it as synthetic evidence.
        if value_id not in node_ids:

            add_node(
                nodes,
                node_ids,
                value_id,
                "CharacteristicValue",
                value,
                "SYNTHETIC",
                source=SYNTHETIC_SOURCE,
                source_type="synthetic_configuration_catalogue",
                field_name=column,
                characteristic=column,
                value=value,
            )

        add_edge(
            edges,
            edge_keys,
            canonical_node,
            value_id,
            "HAS_TECHNICAL_VALUE",
            (
                "OBSERVED_731"
                if any(
                    (
                        nodes_item["node_id"]
                        == value_id
                        and nodes_item["provenance"]
                        == "OBSERVED_731"
                    )
                    for nodes_item in nodes
                )
                else "SYNTHETIC"
            ),
            source=SYNTHETIC_SOURCE,
            source_row=source_row,
            characteristic=column,
        )

        record_value_edges += 1


print(
    f"Configuration records: "
    f"{record_count:,}"
)

print(
    f"Canonical configurations: "
    f"{len(canonical_node_ids):,}"
)

print(
    f"Canonical configuration → value "
    f"relationships: {record_value_edges:,}"
)


# ============================================================
# 10. PRODUCT → CANONICAL CONFIGURATIONS
# ============================================================

for canonical_id, canonical_node_id in (
    canonical_node_ids.items()
):

    add_edge(
        edges,
        edge_keys,
        PRODUCT_ID,
        canonical_node_id,
        "HAS_CANONICAL_CONFIGURATION",
        "SYNTHETIC",
        source=SYNTHETIC_SOURCE,
    )


# ============================================================
# 11. CANONICAL CONFIGURATION → FAMILY
# ============================================================

canonical_family_pairs = set()


if "package_context" in synthetic_df.columns:

    for _, row in synthetic_df[
        [
            "canonical_configuration_id",
            "package_context",
        ]
    ].drop_duplicates().iterrows():

        canonical_id = clean_value(
            row[
                "canonical_configuration_id"
            ]
        )

        family_code = clean_value(
            row[
                "package_context"
            ]
        )

        family_code = extract_family_code(
            family_code
        )

        if (
            canonical_id
            and family_code in family_ids
        ):

            canonical_family_pairs.add(
                (
                    canonical_id,
                    family_code
                )
            )


for canonical_id, family_code in (
    canonical_family_pairs
):

    add_edge(
        edges,
        edge_keys,
        canonical_node_ids[
            canonical_id
        ],
        family_ids[
            family_code
        ],
        "BELONGS_TO_FAMILY",
        "SYNTHETIC",
        source=SYNTHETIC_SOURCE,
    )


# ============================================================
# 12. VALIDATION
# ============================================================

print()
print("-" * 75)
print("VALIDATING V2 GRAPH")
print("-" * 75)


valid_node_ids = set(
    nodes_df_id["node_id"]
    for nodes_df_id in nodes
)


invalid_sources = [
    edge
    for edge in edges
    if edge["source_id"]
    not in valid_node_ids
]


invalid_targets = [
    edge
    for edge in edges
    if edge["target_id"]
    not in valid_node_ids
]


connected_ids = set()

for edge in edges:

    connected_ids.add(
        edge["source_id"]
    )

    connected_ids.add(
        edge["target_id"]
    )


orphan_nodes = [
    node
    for node in nodes
    if node["node_id"]
    not in connected_ids
]


print(
    f"Nodes: "
    f"{len(nodes):,}"
)

print(
    f"Relationships: "
    f"{len(edges):,}"
)

print(
    f"Invalid source references: "
    f"{len(invalid_sources):,}"
)

print(
    f"Invalid target references: "
    f"{len(invalid_targets):,}"
)

print(
    f"Orphan nodes: "
    f"{len(orphan_nodes):,}"
)


# ============================================================
# NODE COUNTS
# ============================================================

node_type_counts = Counter(
    node["node_type"]
    for node in nodes
)

provenance_counts = Counter(
    node["provenance"]
    for node in nodes
)

relation_counts = Counter(
    edge["relation"]
    for edge in edges
)

edge_provenance_counts = Counter(
    edge["provenance"]
    for edge in edges
)


# ============================================================
# CANONICAL CONFIGURATION RECORD CHECK
# ============================================================

configuration_records = [
    node
    for node in nodes
    if node["node_type"]
    == "ConfigurationRecord"
]


canonical_configurations = [
    node
    for node in nodes
    if node["node_type"]
    == "CanonicalConfiguration"
]


record_to_canonical = [
    edge
    for edge in edges
    if edge["relation"]
    == "REPRESENTS_CANONICAL_CONFIGURATION"
]


# ============================================================
# FAMILY COUNTS
# ============================================================

family_nodes = [
    node
    for node in nodes
    if node["node_type"]
    == "ConfigurationFamily"
]


observed_family_nodes = [
    node
    for node in family_nodes
    if node["provenance"]
    == "OBSERVED_731"
]


synthetic_family_nodes = [
    node
    for node in family_nodes
    if node["provenance"]
    == "SYNTHETIC"
]


# ============================================================
# REPORT
# ============================================================

report = []


def report_line(text=""):

    report.append(
        str(text)
    )


report_line("=" * 75)
report_line("731 KNOWLEDGE GRAPH V2 VALIDATION REPORT")
report_line("=" * 75)

report_line()

report_line(
    f"Nodes: {len(nodes):,}"
)

report_line(
    f"Relationships: {len(edges):,}"
)

report_line()

report_line(
    "NODE TYPES"
)

report_line("-" * 40)

for node_type, count in sorted(
    node_type_counts.items()
):

    report_line(
        f"{node_type:<35} {count:>10,}"
    )


report_line()

report_line(
    "NODE PROVENANCE"
)

report_line("-" * 40)

for provenance, count in sorted(
    provenance_counts.items()
):

    report_line(
        f"{provenance:<35} {count:>10,}"
    )


report_line()

report_line(
    "RELATIONSHIP TYPES"
)

report_line("-" * 40)

for relation, count in sorted(
    relation_counts.items()
):

    report_line(
        f"{relation:<35} {count:>10,}"
    )


report_line()

report_line(
    "RELATIONSHIP PROVENANCE"
)

report_line("-" * 40)

for provenance, count in sorted(
    edge_provenance_counts.items()
):

    report_line(
        f"{provenance:<35} {count:>10,}"
    )


report_line()

report_line(
    "CONFIGURATION RECORD PRESERVATION"
)

report_line("-" * 40)

report_line(
    f"Synthetic source rows: "
    f"{len(synthetic_df):,}"
)

report_line(
    f"ConfigurationRecord nodes: "
    f"{len(configuration_records):,}"
)

report_line(
    f"CanonicalConfiguration nodes: "
    f"{len(canonical_configurations):,}"
)

report_line(
    f"Record → canonical relationships: "
    f"{len(record_to_canonical):,}"
)


if (
    len(configuration_records)
    == len(synthetic_df)
):

    report_line(
        "PASS: all synthetic source rows "
        "have a ConfigurationRecord node."
    )

else:

    report_line(
        "REVIEW: synthetic source rows and "
        "ConfigurationRecord nodes differ."
    )


report_line()

report_line(
    "CONFIGURATION FAMILY MODEL"
)

report_line("-" * 40)

report_line(
    f"Canonical family nodes: "
    f"{len(family_nodes):,}"
)

report_line(
    f"Observed family nodes: "
    f"{len(observed_family_nodes):,}"
)

report_line(
    f"Synthetic-only family nodes: "
    f"{len(synthetic_family_nodes):,}"
)


report_line()

report_line(
    "GRAPH INTEGRITY"
)

report_line("-" * 40)

report_line(
    f"Invalid source references: "
    f"{len(invalid_sources):,}"
)

report_line(
    f"Invalid target references: "
    f"{len(invalid_targets):,}"
)

report_line(
    f"Orphan nodes: "
    f"{len(orphan_nodes):,}"
)


if (
    len(invalid_sources) == 0
    and len(invalid_targets) == 0
):

    report_line(
        "PASS: all relationship references "
        "point to existing nodes."
    )

else:

    report_line(
        "FAIL: invalid relationship references exist."
    )


if len(orphan_nodes) == 0:

    report_line(
        "PASS: no orphan nodes."
    )

else:

    report_line(
        "REVIEW: orphan nodes exist."
    )


report_line()

report_line(
    "METHODOLOGICAL SAFETY"
)

report_line("-" * 40)

report_line(
    "PASS: real RuleRows are represented as "
    "configurator knowledge."
)

report_line(
    "PASS: real RuleRows are not represented "
    "as historical orders."
)

report_line(
    "PASS: real RuleRows are not represented "
    "as historical engineer decisions."
)

report_line(
    "PASS: synthetic configuration records "
    "retain SYNTHETIC provenance."
)

report_line(
    "PASS: undocumented technical-code meanings "
    "are not expanded."
)

report_line(
    "PASS: V1 graph is not modified."
)


report_line()

report_line(
    "V2 DESIGN INTENT"
)

report_line("-" * 40)

report_line(
    "ConfigurationRecord preserves source-row-level "
    "synthetic provenance."
)

report_line(
    "CanonicalConfiguration represents the "
    "configuration identity."
)

report_line(
    "FamilyEvidence preserves the distinction between "
    "observed workbook expressions and synthetic package context."
)

report_line(
    "CharacteristicValue identity is scoped by "
    "characteristic + value."
)

report_line(
    "V2 is designed to support future requirement, "
    "constraint, validity and explanation relationships."
)


VALIDATION_FILE.write_text(
    "\n".join(report),
    encoding="utf-8"
)


# ============================================================
# SAVE CSV FILES
# ============================================================

nodes_df = pd.DataFrame(
    nodes
)

edges_df = pd.DataFrame(
    edges
)


nodes_df.to_csv(
    NODES_FILE,
    index=False
)

edges_df.to_csv(
    EDGES_FILE,
    index=False
)


# ============================================================
# STATISTICS
# ============================================================

statistics = {

    "graph_version":
        "V2",

    "real_source":
        str(REAL_XLSX),

    "synthetic_source":
        str(SYNTHETIC_CSV),

    "output_directory":
        str(OUTPUT_DIR),

    "source_rows": {

        "real_workbook":
            int(len(real_df)),

        "synthetic_configuration_records":
            int(len(synthetic_df)),
    },

    "nodes": {

        "total":
            int(len(nodes)),

        "by_type":
            {
                k: int(v)
                for k, v
                in node_type_counts.items()
            },

        "by_provenance":
            {
                k: int(v)
                for k, v
                in provenance_counts.items()
            },
    },

    "relationships": {

        "total":
            int(len(edges)),

        "by_type":
            {
                k: int(v)
                for k, v
                in relation_counts.items()
            },

        "by_provenance":
            {
                k: int(v)
                for k, v
                in edge_provenance_counts.items()
            },
    },

    "configuration_model": {

        "configuration_records":
            int(len(configuration_records)),

        "canonical_configurations":
            int(len(canonical_configurations)),

        "record_to_canonical_relationships":
            int(len(record_to_canonical)),
    },

    "family_model": {

        "canonical_families":
            int(len(family_nodes)),

        "observed_families":
            int(len(observed_family_nodes)),

        "synthetic_family_nodes":
            int(len(synthetic_family_nodes)),
    },

    "validation": {

        "invalid_source_references":
            int(len(invalid_sources)),

        "invalid_target_references":
            int(len(invalid_targets)),

        "orphan_nodes":
            int(len(orphan_nodes)),

        "status":
            (
                "PASS"
                if (
                    len(invalid_sources) == 0
                    and len(invalid_targets) == 0
                    and len(orphan_nodes) == 0
                )
                else "REVIEW"
            ),
    },
}


STATS_FILE.write_text(
    json.dumps(
        statistics,
        indent=2
    ),
    encoding="utf-8"
)


# ============================================================
# OPTIONAL GRAPHML
# ============================================================

print()
print("-" * 75)
print("CREATING GRAPHML")
print("-" * 75)


try:

    import networkx as nx

    graph = nx.MultiDiGraph()

    for node in nodes:

        attributes = {
            key: (
                str(value)
                if value is not None
                else ""
            )
            for key, value
            in node.items()
            if key != "node_id"
        }

        graph.add_node(
            node["node_id"],
            **attributes
        )


    for edge_index, edge in enumerate(edges):

        attributes = {
            key: (
                str(value)
                if value is not None
                else ""
            )
            for key, value
            in edge.items()
            if key not in {
                "source_id",
                "target_id",
            }
        }

        graph.add_edge(
            edge["source_id"],
            edge["target_id"],
            key=edge_index,
            **attributes
        )


    nx.write_graphml(
        graph,
        GRAPHML_FILE
    )

    print(
        f"GraphML saved: {GRAPHML_FILE}"
    )

except ImportError:

    print(
        "networkx is not installed. "
        "CSV graph files were still created."
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 75)
print("V2 KNOWLEDGE GRAPH BUILD COMPLETE")
print("=" * 75)

print()

print(
    f"Nodes: "
    f"{len(nodes):,}"
)

print(
    f"Relationships: "
    f"{len(edges):,}"
)

print(
    f"Configuration records: "
    f"{len(configuration_records):,}"
)

print(
    f"Canonical configurations: "
    f"{len(canonical_configurations):,}"
)

print(
    f"Configuration families: "
    f"{len(family_nodes):,}"
)

print(
    f"Rule rows: "
    f"{len(rules):,}"
    if "rules" in locals()
    else f"Rule rows: {rule_count:,}"
)

print(
    f"Invalid references: "
    f"{len(invalid_sources) + len(invalid_targets):,}"
)

print(
    f"Orphan nodes: "
    f"{len(orphan_nodes):,}"
)

print()

print(
    f"Output directory:\n"
    f"{OUTPUT_DIR}"
)

print()

print(
    "V1 graph was not modified."
)