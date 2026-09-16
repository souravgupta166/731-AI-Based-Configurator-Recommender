#!/usr/bin/env python3
"""
731 Knowledge Graph Validation Test Suite
=========================================

Purpose
-------
Systematically compares the existing 731 Knowledge Graph candidate-validation
logic with the deterministic mandatory-constraint filter used by the PoC.

This script is VALIDATION-ONLY:
- it does not rebuild or modify the Knowledge Graph;
- it does not train an ML model;
- it does not change the Streamlit application;
- it reports candidate-set agreement and differences.

Run from ~/thesis-configurator:
    python src/knowledge_graph/731_kg_validation_test_suite.py

Outputs:
    data/processed/kg_validation/
        731_kg_validation_test_results.csv
        731_kg_validation_summary.txt

Important research interpretation
---------------------------------
A PASS means the KG candidate-validation adapter and deterministic filter
return the same candidate IDs for that test case.

It does NOT prove that the synthetic catalogue is historical order data or
that the ML recommendation is historically accurate.
"""

from pathlib import Path
from itertools import combinations
import json
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

KG_DIR = ROOT / "graphs" / "731_knowledge_graph"
NODES_PATH = KG_DIR / "nodes.csv"
EDGES_PATH = KG_DIR / "relationships.csv"
STATS_PATH = KG_DIR / "statistics.json"

CONFIG_PATH = (
    ROOT
    / "data"
    / "synthetic"
    / "configurations"
    / "731_synthetic_configurations_v5.csv"
)

OUT_DIR = ROOT / "data" / "processed" / "kg_validation"


TECHNICAL = [
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
    "waveInjector",
    "dev_advMeterVerification",
    "dynamicGasMaster",
    "customUserFluid",
    "steamApplication",
]


def norm(x):
    if pd.isna(x):
        return "__MISSING__"
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return "__MISSING__"
    return s


def fail(message):
    print(f"\nERROR: {message}")
    sys.exit(1)


def load_inputs():
    missing = [
        p for p in [NODES_PATH, EDGES_PATH, CONFIG_PATH]
        if not p.exists()
    ]
    if missing:
        fail(
            "Required file(s) missing:\n"
            + "\n".join(str(p) for p in missing)
        )

    nodes = pd.read_csv(NODES_PATH, low_memory=False)
    edges = pd.read_csv(EDGES_PATH, low_memory=False)
    configs = pd.read_csv(CONFIG_PATH, low_memory=False)

    required_node_cols = {
        "node_id",
        "node_type",
        "label",
        "characteristic",
        "value",
        "package_context",
        "canonical_configuration_id",
    }
    required_edge_cols = {
        "source_id",
        "target_id",
        "relation",
    }
    missing_node = sorted(required_node_cols - set(nodes.columns))
    missing_edge = sorted(required_edge_cols - set(edges.columns))

    if missing_node:
        fail(f"KG nodes.csv is missing columns: {missing_node}")
    if missing_edge:
        fail(f"KG relationships.csv is missing columns: {missing_edge}")
    if "canonical_configuration_id" not in configs.columns:
        fail("Synthetic configuration catalogue lacks canonical_configuration_id.")
    if "package_context" not in configs.columns:
        fail("Synthetic configuration catalogue lacks package_context.")

    configs["canonical_configuration_id"] = (
        configs["canonical_configuration_id"].astype(str)
    )
    nodes["node_id"] = nodes["node_id"].astype(str)
    edges["source_id"] = edges["source_id"].astype(str)
    edges["target_id"] = edges["target_id"].astype(str)

    return nodes, edges, configs


def build_kg_index(nodes, edges):
    node_lookup = nodes.set_index("node_id", drop=False)

    config_nodes = nodes[
        nodes["node_type"].eq("Configuration")
    ].copy()

    if config_nodes["canonical_configuration_id"].isna().any():
        fail("Some Configuration nodes have missing canonical_configuration_id.")

    config_nodes["canonical_configuration_id"] = (
        config_nodes["canonical_configuration_id"].astype(str)
    )

    config_by_id = (
        config_nodes
        .drop_duplicates("canonical_configuration_id")
        .set_index("canonical_configuration_id")["node_id"]
        .to_dict()
    )

    # Configuration -> BELONGS_TO_FAMILY -> ConfigurationFamily
    family_edges = edges[
        edges["relation"].eq("BELONGS_TO_FAMILY")
    ][["source_id", "target_id"]].copy()

    config_family = {}
    for row in family_edges.itertuples(index=False):
        source_id = str(row.source_id)
        target_id = str(row.target_id)
        if source_id not in node_lookup.index or target_id not in node_lookup.index:
            continue
        target = node_lookup.loc[target_id]
        config_family[source_id] = norm(target["label"])

    # Configuration -> HAS_VALUE -> CharacteristicValue
    value_edges = edges[
        edges["relation"].eq("HAS_VALUE")
    ][["source_id", "target_id"]].copy()

    config_values = {}
    for row in value_edges.itertuples(index=False):
        source_id = str(row.source_id)
        target_id = str(row.target_id)

        if source_id not in node_lookup.index or target_id not in node_lookup.index:
            continue

        target = node_lookup.loc[target_id]
        ch = norm(target["characteristic"])
        value = norm(target["value"])

        if ch == "__MISSING__":
            continue

        config_values.setdefault(source_id, {})[ch] = value

    return config_by_id, config_family, config_values


def filter_mandatory(pool, requirements):
    """Same deterministic mandatory filtering principle used by the PoC."""
    filtered = pool.copy()

    for characteristic, value in requirements:
        col = f"config_{characteristic}"
        if col not in filtered.columns:
            return filtered.iloc[0:0]

        filtered = filtered[
            filtered[col].map(norm).eq(norm(value))
        ]

    return filtered


def kg_validate_candidates(
    configs,
    package,
    requirements,
    config_by_id,
    config_family,
    config_values,
):
    """
    Validate each canonical configuration against:
      1. KG Configuration node existence
      2. KG Configuration -> BELONGS_TO_FAMILY
      3. KG Configuration -> HAS_VALUE for every mandatory requirement
    """
    pool = configs[
        configs["package_context"].map(norm).eq(norm(package))
    ].copy()

    pool = pool.drop_duplicates("canonical_configuration_id")

    valid_ids = []

    for cid in pool["canonical_configuration_id"].astype(str):
        node_id = config_by_id.get(cid)
        if node_id is None:
            continue

        if config_family.get(node_id) != norm(package):
            continue

        values = config_values.get(node_id, {})
        ok = True

        for characteristic, value in requirements:
            if values.get(characteristic) != norm(value):
                ok = False
                break

        if ok:
            valid_ids.append(cid)

    return set(valid_ids), pool


def run_test(
    name,
    package,
    requirements,
    configs,
    config_by_id,
    config_family,
    config_values,
    expected_candidates=None,
):
    kg_ids, package_pool = kg_validate_candidates(
        configs,
        package,
        requirements,
        config_by_id,
        config_family,
        config_values,
    )

    deterministic_pool = filter_mandatory(package_pool, requirements)
    deterministic_ids = set(
        deterministic_pool["canonical_configuration_id"].astype(str)
    )

    kg_only = sorted(kg_ids - deterministic_ids)
    deterministic_only = sorted(deterministic_ids - kg_ids)

    agreement_ok = not kg_only and not deterministic_only
    expected_ok = (
        expected_candidates is None
        or (len(kg_ids) == expected_candidates and len(deterministic_ids) == expected_candidates)
    )
    status = "PASS" if agreement_ok and expected_ok else "FAIL"

    return {
        "test": name,
        "package": package,
        "requirements": " AND ".join(
            f"{ch}={value}" for ch, value in requirements
        ) if requirements else "NONE",
        "mandatory_count": len(requirements),
        "package_candidates": len(package_pool),
        "kg_candidates": len(kg_ids),
        "deterministic_candidates": len(deterministic_ids),
        "kg_only_count": len(kg_only),
        "deterministic_only_count": len(deterministic_only),
        "expected_candidates": expected_candidates,
        "expected_count_match": expected_ok,
        "status": status,
        "kg_only_examples": "; ".join(kg_only[:5]),
        "deterministic_only_examples": "; ".join(deterministic_only[:5]),
    }


def choose_existing_value(configs, package, characteristic, preferred=None):
    pool = configs[
        configs["package_context"].map(norm).eq(norm(package))
    ]
    col = f"config_{characteristic}"
    if col not in pool.columns:
        return None

    values = [
        v for v in pool[col].map(norm).unique().tolist()
        if v != "__MISSING__"
    ]

    if preferred is not None and norm(preferred) in values:
        return norm(preferred)

    return sorted(values)[0] if values else None


def choose_valid_pair(configs, package, characteristics):
    """
    Find a pair of requirements that actually has at least one deterministic
    candidate, avoiding an artificial all-zero test where possible.
    """
    pool = configs[
        configs["package_context"].map(norm).eq(norm(package))
    ].drop_duplicates("canonical_configuration_id")

    available = []
    for ch in characteristics:
        col = f"config_{ch}"
        if col not in pool.columns:
            continue
        vals = [
            v for v in pool[col].map(norm).unique().tolist()
            if v != "__MISSING__"
        ]
        if vals:
            available.append((ch, sorted(vals)))

    for (ch1, vals1), (ch2, vals2) in combinations(available, 2):
        for v1 in vals1:
            for v2 in vals2:
                test_pool = filter_mandatory(
                    pool, [(ch1, v1), (ch2, v2)]
                )
                if not test_pool.empty:
                    return [(ch1, v1), (ch2, v2)]

    return None


def main():
    print("=" * 78)
    print("731 KNOWLEDGE GRAPH VALIDATION TEST SUITE")
    print("=" * 78)

    nodes, edges, configs = load_inputs()
    config_by_id, config_family, config_values = build_kg_index(nodes, edges)

    print(f"KG nodes:                 {len(nodes):,}")
    print(f"KG relationships:         {len(edges):,}")
    print(f"Configuration nodes:      {len(config_by_id):,}")
    print(f"Synthetic catalogue rows: {len(configs):,}")
    print(f"Unique catalogue IDs:     {configs.canonical_configuration_id.nunique():,}")

    packages = sorted(
        v for v in configs["package_context"].map(norm).unique()
        if v != "__MISSING__"
    )

    if not packages:
        fail("No package_context values found.")

    results = []

    # ------------------------------------------------------------------
    # Core hand-designed tests
    # ------------------------------------------------------------------

    x731 = "x731" if "x731" in packages else packages[0]

    results.append(
        run_test(
            "T01_no_requirements_x731",
            x731,
            [],
            configs,
            config_by_id,
            config_family,
            config_values,
            expected_candidates=int(
                configs[configs["package_context"].map(norm).eq(norm(x731))]
                ["canonical_configuration_id"].nunique()
            ),
        )
    )

    st_value = choose_existing_value(
        configs, x731, "housing", preferred="ST"
    )
    if st_value:
        results.append(
            run_test(
                "T02_housing_ST_mandatory",
                x731,
                [("housing", st_value)],
                configs,
                config_by_id,
                config_family,
                config_values,
            )
        )

    # Negative boundary test: the V5 catalogue contains only one channel
    # for x731. Keep the impossible value "2" explicit so the test verifies
    # that both validation paths correctly return zero candidates.
    two_channel_value = "2"
    results.append(
        run_test(
            "T03_two_channels_x731_conflict",
            x731,
            [("numberOfChannels", two_channel_value)],
            configs,
            config_by_id,
            config_family,
            config_values,
            expected_candidates=0,
        )
    )

    results.append(
        run_test(
            "T04_ST_plus_two_channels_x731_conflict",
            x731,
            [
                ("housing", st_value),
                ("numberOfChannels", two_channel_value),
            ],
            configs,
            config_by_id,
            config_family,
            config_values,
            expected_candidates=0,
        )
    )

    # F731WD_DualChannel valid combined test, when present.
    wd = "F731WD_DualChannel"
    if wd in packages:
        wd_two = choose_existing_value(
            configs, wd, "numberOfChannels", preferred="2"
        )
        if wd_two:
            results.append(
                run_test(
                    "T05_F731WD_DualChannel_two_channels",
                    wd,
                    [("numberOfChannels", wd_two)],
                    configs,
                    config_by_id,
                    config_family,
                    config_values,
                )
            )

        wd_pair = choose_valid_pair(
            configs,
            wd,
            [
                "housing",
                "numberOfChannels",
                "powerSupply",
                "explosionApproval",
                "protectionArea",
                "certification",
                "dataInterface",
            ],
        )
        if wd_pair:
            results.append(
                run_test(
                    "T06_F731WD_DualChannel_valid_two_constraints",
                    wd,
                    wd_pair,
                    configs,
                    config_by_id,
                    config_family,
                    config_values,
                )
            )

    # ------------------------------------------------------------------
    # One-requirement sweep across all packages / selected technical fields.
    # This gives a broader systematic check without generating an enormous
    # combinatorial test suite.
    # ------------------------------------------------------------------

    sweep_fields = [
        "housing",
        "powerSupply",
        "numberOfChannels",
        "explosionApproval",
        "protectionArea",
        "certification",
        "dataInterface",
    ]

    test_no = 7
    for package in packages:
        for characteristic in sweep_fields:
            value = choose_existing_value(
                configs, package, characteristic
            )
            if value is None:
                continue

            results.append(
                run_test(
                    f"T{test_no:02d}_sweep_{package}_{characteristic}",
                    package,
                    [(characteristic, value)],
                    configs,
                    config_by_id,
                    config_family,
                    config_values,
                )
            )
            test_no += 1

    results_df = pd.DataFrame(results)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "731_kg_validation_test_results.csv"
    summary_path = OUT_DIR / "731_kg_validation_summary.txt"

    results_df.to_csv(results_path, index=False)

    total = len(results_df)
    passed = int(results_df["status"].eq("PASS").sum())
    failed = total - passed

    summary_lines = [
        "731 Knowledge Graph Validation Test Suite",
        "=" * 50,
        f"Total tests: {total}",
        f"Passed:      {passed}",
        f"Failed:      {failed}",
        f"Pass rate:   {(passed / total * 100) if total else 0:.2f}%",
        "",
        "Interpretation:",
        "PASS means KG-valid canonical configuration IDs exactly match",
        "the deterministic mandatory-filter IDs for that test case.",
        "This validates the consistency of the KG adapter against the",
        "deterministic candidate filter; it does not establish historical",
        "engineering accuracy or historical order ground truth.",
        "",
        "Test results:",
        "",
        results_df.to_string(index=False),
    ]

    summary_path.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("\n" + "-" * 78)
    print("RESULTS")
    print("-" * 78)
    print(results_df[
        [
            "test",
            "package",
            "mandatory_count",
            "package_candidates",
            "kg_candidates",
            "deterministic_candidates",
            "kg_only_count",
            "deterministic_only_count",
            "status",
        ]
    ].to_string(index=False))

    print("\n" + "=" * 78)
    print(f"TOTAL TESTS : {total}")
    print(f"PASSED      : {passed}")
    print(f"FAILED      : {failed}")
    print(f"PASS RATE   : {(passed / total * 100) if total else 0:.2f}%")
    print("=" * 78)

    print(f"\nDetailed CSV:     {results_path}")
    print(f"Summary report:   {summary_path}")

    if failed:
        print("\nFAILURES DETECTED.")
        print("Review kg_only_count and deterministic_only_count in the CSV.")
        sys.exit(2)

    print("\nALL TESTS PASSED.")
    sys.exit(0)


if __name__ == "__main__":
    main()
