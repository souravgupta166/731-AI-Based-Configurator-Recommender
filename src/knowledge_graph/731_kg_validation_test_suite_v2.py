#!/usr/bin/env python3
"""Semantic and structural audit for the 731 V2 knowledge graph.

Run from the repository root:
    python src/knowledge_graph/731_kg_validation_test_suite_v2.py

This script is read-only. It does not alter the graph.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GRAPH_DIR = ROOT / "graphs" / "731_knowledge_graph_v2"
NODES_FILE = GRAPH_DIR / "nodes.csv"
EDGES_FILE = GRAPH_DIR / "relationships.csv"
REPORT_FILE = GRAPH_DIR / "v2_semantic_audit_report.txt"

EXPECTED_COUNTS = {
    "RuleRow": 1056,
    "ConfigurationRecord": 5000,
    "CanonicalConfiguration": 1979,
    "ConfigurationFamily": 9,
    "Characteristic": 28,
}
EXPECTED_TECHNICAL_FIELDS = 19


def main() -> int:
    if not NODES_FILE.exists() or not EDGES_FILE.exists():
        print("ERROR: V2 nodes.csv or relationships.csv is missing.")
        print(f"Expected: {NODES_FILE}")
        print(f"Expected: {EDGES_FILE}")
        return 2

    nodes = pd.read_csv(NODES_FILE, low_memory=False).fillna("")
    edges = pd.read_csv(EDGES_FILE, low_memory=False).fillna("")
    lines: list[str] = []
    failures: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    def check(condition: bool, label: str, detail: str = "") -> None:
        status = "PASS" if condition else "FAIL"
        emit(f"[{status}] {label}" + (f" — {detail}" if detail else ""))
        if not condition:
            failures.append(label)

    emit("=" * 78)
    emit("731 KNOWLEDGE GRAPH V2 — SEMANTIC AUDIT")
    emit("=" * 78)
    emit(f"Graph directory: {GRAPH_DIR}")
    emit(f"Nodes: {len(nodes):,}")
    emit(f"Relationships: {len(edges):,}")
    emit()

    required_node_cols = {
        "node_id", "node_type", "provenance", "source",
        "source_row", "characteristic", "value",
        "package_context", "canonical_configuration_id",
    }
    required_edge_cols = {
        "source_id", "target_id", "relation", "characteristic",
    }
    check(required_node_cols.issubset(nodes.columns), "Required node schema",
          f"missing={sorted(required_node_cols - set(nodes.columns))}")
    check(required_edge_cols.issubset(edges.columns), "Required relationship schema",
          f"missing={sorted(required_edge_cols - set(edges.columns))}")
    if not required_node_cols.issubset(nodes.columns) or not required_edge_cols.issubset(edges.columns):
        REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return 2

    check(nodes["node_id"].astype(str).is_unique, "Unique node IDs")
    node_ids = set(nodes["node_id"].astype(str))
    bad_sources = int((~edges["source_id"].astype(str).isin(node_ids)).sum())
    bad_targets = int((~edges["target_id"].astype(str).isin(node_ids)).sum())
    check(bad_sources == 0, "Relationship source references", f"invalid={bad_sources}")
    check(bad_targets == 0, "Relationship target references", f"invalid={bad_targets}")

    referenced = set(edges["source_id"].astype(str)) | set(edges["target_id"].astype(str))
    orphan_count = len(node_ids - referenced)
    check(orphan_count == 0, "No orphan nodes", f"orphans={orphan_count}")

    type_counts = nodes["node_type"].value_counts().to_dict()
    emit("\nNode type counts:")
    for node_type, expected in EXPECTED_COUNTS.items():
        actual = int(type_counts.get(node_type, 0))
        check(actual == expected, f"{node_type} node count", f"actual={actual}; expected={expected}")

    records = nodes[nodes["node_type"].eq("ConfigurationRecord")].copy()
    canonical = nodes[nodes["node_type"].eq("CanonicalConfiguration")].copy()
    rules = nodes[nodes["node_type"].eq("RuleRow")].copy()
    families = nodes[nodes["node_type"].eq("ConfigurationFamily")].copy()

    check(
        rules["provenance"].eq("OBSERVED_731").all(),
        "RuleRow observed provenance",
        f"non-OBSERVED_731={int((~rules['provenance'].eq('OBSERVED_731')).sum())}",
    )
    check(
        records["provenance"].eq("SYNTHETIC").all(),
        "ConfigurationRecord synthetic provenance",
        f"non-SYNTHETIC={int((~records['provenance'].eq('SYNTHETIC')).sum())}",
    )
    check(
        canonical["provenance"].eq("SYNTHETIC").all(),
        "CanonicalConfiguration synthetic provenance",
        f"non-SYNTHETIC={int((~canonical['provenance'].eq('SYNTHETIC')).sum())}",
    )

    record_ids = set(records["node_id"].astype(str))
    canonical_node_ids = set(canonical["node_id"].astype(str))
    canonical_identity_ids = set(canonical["canonical_configuration_id"].astype(str))
    check(
        records["canonical_configuration_id"].astype(str).ne("").all(),
        "Every record has canonical identity",
    )
    check(
        canonical["canonical_configuration_id"].astype(str).ne("").all(),
        "Every canonical node has canonical identity",
    )
    check(
        canonical["canonical_configuration_id"].astype(str).is_unique,
        "Canonical identities are unique",
    )

    represents = edges[edges["relation"].eq("REPRESENTS_CANONICAL_CONFIGURATION")]
    check(len(represents) == 5000, "Record-to-canonical edge count", f"actual={len(represents)}; expected=5000")
    check(
        represents["source_id"].astype(str).isin(record_ids).all(),
        "REPRESENTS sources are ConfigurationRecord nodes",
    )
    check(
        represents["target_id"].astype(str).isin(canonical_node_ids).all(),
        "REPRESENTS targets are CanonicalConfiguration nodes",
    )
    check(
        represents["source_id"].astype(str).nunique() == len(records),
        "Every record has exactly one representation edge",
        f"represented_records={represents['source_id'].nunique():,}; records={len(records):,}",
    )
    target_count_by_record = represents.groupby("source_id").size()
    check(
        bool((target_count_by_record == 1).all()) and len(target_count_by_record) == len(records),
        "No missing or multiply represented records",
    )

    # Verify each record's stored canonical identity agrees with its edge target.
    canonical_lookup = canonical.set_index("node_id")["canonical_configuration_id"].astype(str).to_dict()
    mismatches = 0
    for row in represents.itertuples(index=False):
        record_match = records.loc[records["node_id"].astype(str).eq(str(row.source_id))]
        if record_match.empty:
            mismatches += 1
            continue
        record_cid = str(record_match.iloc[0]["canonical_configuration_id"])
        target_cid = canonical_lookup.get(str(row.target_id), "")
        if record_cid != target_cid:
            mismatches += 1
    check(mismatches == 0, "Record identity agrees with representation target", f"mismatches={mismatches}")

    # Technical-value semantic checks.
    htv = edges[edges["relation"].eq("HAS_TECHNICAL_VALUE")].copy()
    node_type_lookup = nodes.set_index("node_id")["node_type"].astype(str).to_dict()
    htv_source_types = htv["source_id"].astype(str).map(node_type_lookup)
    check(
        htv_source_types.eq("CanonicalConfiguration").all(),
        "HAS_TECHNICAL_VALUE sources are canonical nodes",
        f"wrong_source_type={int((~htv_source_types.eq('CanonicalConfiguration')).sum())}",
    )
    value_type_lookup = nodes.set_index("node_id")["node_type"].astype(str).to_dict()
    htv_target_types = htv["target_id"].astype(str).map(value_type_lookup)
    check(
        htv_target_types.eq("CharacteristicValue").all(),
        "HAS_TECHNICAL_VALUE targets are characteristic values",
        f"wrong_target_type={int((~htv_target_types.eq('CharacteristicValue')).sum())}",
    )

    duplicate_mask = htv.duplicated(
        subset=["source_id", "target_id", "relation", "characteristic"],
        keep=False,
    )
    duplicate_rows = int(duplicate_mask.sum())
    check(duplicate_rows == 0, "No duplicate technical-value edges", f"duplicate_rows={duplicate_rows}")

    per_canonical = htv.groupby("source_id").size()
    missing_canonical = canonical_node_ids - set(per_canonical.index.astype(str))
    wrong_counts = per_canonical[per_canonical != EXPECTED_TECHNICAL_FIELDS]
    check(
        len(missing_canonical) == 0 and len(per_canonical) == len(canonical),
        "Every canonical configuration has technical-value edges",
        f"missing_canonicals={len(missing_canonical)}",
    )
    check(
        len(wrong_counts) == 0,
        "Exactly 19 technical-value edges per canonical configuration",
        f"canonical_nodes_with_wrong_count={len(wrong_counts)}; total_edges={len(htv)}",
    )
    unique_char_counts = htv.groupby("source_id")["characteristic"].nunique()
    wrong_characteristic_coverage = unique_char_counts[unique_char_counts != EXPECTED_TECHNICAL_FIELDS]
    check(
        len(wrong_characteristic_coverage) == 0,
        "Exactly 19 distinct technical characteristics per canonical configuration",
        f"canonical_nodes_with_wrong_characteristic_coverage={len(wrong_characteristic_coverage)}",
    )
    expected_total = len(canonical) * EXPECTED_TECHNICAL_FIELDS
    check(
        len(htv) == expected_total,
        "Total technical-value edge count",
        f"actual={len(htv):,}; expected={expected_total:,}",
    )

    # Family links and rule semantics.
    belongs = edges[edges["relation"].eq("BELONGS_TO_FAMILY")]
    rule_family = edges[edges["relation"].eq("RULE_APPLIES_TO_FAMILY")]
    rule_specs = edges[edges["relation"].eq("RULE_SPECIFIES")]
    check(len(belongs) == len(canonical), "Canonical-to-family edge count", f"actual={len(belongs)}")
    check(
        belongs["source_id"].astype(str).isin(canonical_node_ids).all()
        and belongs["target_id"].astype(str).isin(set(families["node_id"].astype(str))).all(),
        "Canonical family links use correct node types",
    )
    check(
        len(rule_family) == len(rules)
        and rule_family["source_id"].astype(str).isin(set(rules["node_id"].astype(str))).all()
        and rule_family["target_id"].astype(str).isin(set(families["node_id"].astype(str))).all(),
        "Rule-to-family links are valid",
        f"edges={len(rule_family)}; rules={len(rules)}",
    )
    check(
        rule_specs["source_id"].astype(str).isin(set(rules["node_id"].astype(str))).all()
        and rule_specs["target_id"].astype(str).map(node_type_lookup).eq("CharacteristicValue").all(),
        "Rule-to-value links are valid",
        f"edges={len(rule_specs):,}",
    )

    # Do not mistake internal IDs for commercial model codes.
    check(
        canonical["canonical_configuration_id"].astype(str).str.startswith("CFG731_").all(),
        "Canonical IDs use internal CFG731 namespace",
    )
    emit("\nSummary:")
    emit(f"  Canonical configurations: {len(canonical):,}")
    emit(f"  HAS_TECHNICAL_VALUE edges: {len(htv):,}")
    emit(f"  Expected technical-value edges: {expected_total:,}")
    emit(f"  Duplicate technical-value edge rows: {duplicate_rows:,}")
    emit(f"  Result: {'PASS' if not failures else 'FAIL'}")
    emit("\nNote: ConfigurationRecord/CanonicalConfiguration provenance is SYNTHETIC; RuleRow provenance is OBSERVED_731.")

    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport saved: {REPORT_FILE}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
