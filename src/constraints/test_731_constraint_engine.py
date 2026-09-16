#!/usr/bin/env python3

"""
731 CONSTRAINT ENGINE — ROBUST VALIDATION
=========================================

Purpose
-------
Validate the 731 constraint engine using:

1. Real observed configurations
2. Impossible values
3. Package-context violations
4. Independent rule-based expected results
5. Controlled cross-context mutations

Important
---------
A mutation is NOT automatically invalid.

A value change is invalid only when the resulting configuration
is inconsistent with the observed 731 rule structure.
"""


from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd


# =============================================================================
# PROJECT PATH
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "731_constraint_validation_tests.csv"
)

CONSTRAINT_DIR = (
    PROJECT_ROOT
    / "src"
    / "constraints"
)

if str(CONSTRAINT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(CONSTRAINT_DIR),
    )


from build_731_constraint_engine import (  # noqa: E402
    ConstraintEngine,
    NOVALUE,
    build_rule_table,
    load_source_data,
    representative_value,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

TEST_COLUMNS = [
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
    "waveInjector",
    "dev_advMeterVerification",
    "customUserFluid",
]


# =============================================================================
# NORMALIZATION
# =============================================================================

def clean_value(value: Any) -> str:

    if pd.isna(value):
        return NOVALUE

    text = str(value).strip()

    if not text:
        return NOVALUE

    if len(text) >= 2:

        if (
            text[0] == "'"
            and text[-1] == "'"
        ):
            text = text[1:-1]

        elif (
            text[0] == '"'
            and text[-1] == '"'
        ):
            text = text[1:-1]

    return text.strip()


# =============================================================================
# SOURCE ROW → CANDIDATE
# =============================================================================

def row_to_candidate(
    row: pd.Series,
) -> Dict[str, Any]:

    candidate = {}

    for column in row.index:

        candidate[column] = representative_value(
            row[column]
        )

    return candidate


# =============================================================================
# INDEPENDENT VALIDATOR
# =============================================================================

def independent_rule_validation(
    candidate: Dict[str, Any],
    rules: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Independent validation oracle.

    A candidate is valid if there exists at least one source rule row
    whose package context matches and whose allowed values accept every
    characteristic.

    This deliberately does NOT call ConstraintEngine.validate().
    """

    package = clean_value(
        candidate.get(
            "packageFixed_dev",
            NOVALUE,
        )
    )

    candidate_rules = rules[
        rules["package_context"].apply(
            lambda x:
            package == clean_value(x)
            or package in str(x).split("|")
        )
    ]

    matching_rows = []

    for source_row, group in candidate_rules.groupby(
        "source_row"
    ):

        valid = True

        for _, rule in group.iterrows():

            characteristic = (
                rule["characteristic"]
            )

            candidate_value = clean_value(
                candidate.get(
                    characteristic,
                    NOVALUE,
                )
            )

            allowed_values = set(
                value
                for value in str(
                    rule["allowed_values"]
                ).split("|")
                if value
            )

            if not allowed_values:
                continue

            if candidate_value not in allowed_values:

                valid = False
                break

        if valid:
            matching_rows.append(
                int(source_row)
            )

    return {
        "valid": len(matching_rows) > 0,
        "matching_rows": matching_rows,
    }


# =============================================================================
# TEST RESULT
# =============================================================================

def result_record(
    test_id: str,
    test_type: str,
    expected: bool,
    actual: bool,
    source_row: int = -1,
    characteristic: str = "",
    original_value: str = "",
    test_value: str = "",
    details: str = "",
) -> Dict[str, Any]:

    return {
        "test_id": test_id,
        "test_type": test_type,
        "source_row": source_row,
        "characteristic": characteristic,
        "original_value": original_value,
        "test_value": test_value,
        "expected_valid": expected,
        "actual_valid": actual,
        "test_passed": expected == actual,
        "details": details,
    }


# =============================================================================
# TEST 1
# REAL CONFIGURATIONS
# =============================================================================

def test_real_configurations(
    df: pd.DataFrame,
    engine: ConstraintEngine,
) -> List[Dict[str, Any]]:

    print("\n" + "=" * 100)
    print("TEST 1 — REAL OBSERVED CONFIGURATIONS")
    print("=" * 100)

    results = []

    for row_index, row in df.iterrows():

        candidate = row_to_candidate(
            row
        )

        actual = engine.validate(
            candidate
        )["valid"]

        results.append(
            result_record(
                test_id=f"REAL_{row_index:04d}",
                test_type="REAL_CONFIGURATION",
                expected=True,
                actual=actual,
                source_row=int(row_index),
            )
        )

    report = pd.DataFrame(
        results
    )

    print(
        f"\nTests:  {len(report):,}"
    )

    print(
        f"Passed: "
        f"{report['test_passed'].sum():,}"
    )

    print(
        f"Failed: "
        f"{(~report['test_passed']).sum():,}"
    )

    return results


# =============================================================================
# TEST 2
# IMPOSSIBLE VALUES
# =============================================================================

def test_impossible_values(
    df: pd.DataFrame,
    engine: ConstraintEngine,
    rules: pd.DataFrame,
) -> List[Dict[str, Any]]:

    print("\n" + "=" * 100)
    print("TEST 2 — IMPOSSIBLE VALUES")
    print("=" * 100)

    results = []

    impossible_values = {
        "housing": "__INVALID_HOUSING__",
        "powerSupply": "__INVALID_POWER__",
        "numberOfChannels": 999,
        "explosionApproval": "__INVALID_APPROVAL__",
        "protectionArea": "__INVALID_AREA__",
        "certification": "__INVALID_CERTIFICATION__",
        "dataInterface": "__INVALID_INTERFACE__",
        "stromSchaltbar": "__INVALID_OUTPUT__",
        "stromEingaenge": "__INVALID_INPUT__",
        "temperaturEingaenge": "__INVALID_TEMP__",
        "waveInjector": "__INVALID_WAVE__",
    }

    counter = 0

    # Use every 10th row for controlled coverage.
    for row_index, row in df.iloc[::10].iterrows():

        original = row_to_candidate(
            row
        )

        # Ensure original is valid.
        if not engine.validate(
            original
        )["valid"]:
            continue

        for column, invalid_value in (
            impossible_values.items()
        ):

            if column not in df.columns:
                continue

            candidate = copy.deepcopy(
                original
            )

            candidate[column] = (
                invalid_value
            )

            expected = independent_rule_validation(
                candidate,
                rules,
            )["valid"]

            actual = engine.validate(
                candidate
            )["valid"]

            counter += 1

            results.append(
                result_record(
                    test_id=f"IMPOSSIBLE_{counter:05d}",
                    test_type="IMPOSSIBLE_VALUE",
                    expected=expected,
                    actual=actual,
                    source_row=int(row_index),
                    characteristic=column,
                    original_value=clean_value(
                        original[column]
                    ),
                    test_value=clean_value(
                        invalid_value
                    ),
                    details=(
                        "Injected value that does not "
                        "exist in the observed value space."
                    ),
                )
            )

    report = pd.DataFrame(
        results
    )

    print(
        f"\nTests:  {len(report):,}"
    )

    print(
        f"Passed: "
        f"{report['test_passed'].sum():,}"
    )

    print(
        f"Failed: "
        f"{(~report['test_passed']).sum():,}"
    )

    return results


# =============================================================================
# TEST 3
# CROSS-CONTEXT VALUES
# =============================================================================

def test_cross_context_values(
    df: pd.DataFrame,
    engine: ConstraintEngine,
    rules: pd.DataFrame,
) -> List[Dict[str, Any]]:

    print("\n" + "=" * 100)
    print("TEST 3 — CROSS-CONTEXT VALUES")
    print("=" * 100)

    results = []

    counter = 0

    package_column = "packageFixed_dev"

    if package_column not in df.columns:
        return results

    package_groups = (
        df.groupby(
            package_column,
            dropna=False,
        )
    )

    packages = list(
        package_groups.groups.keys()
    )

    for i, package_a in enumerate(packages):

        for package_b in packages[i + 1:]:

            group_a = package_groups.get_group(
                package_a
            )

            group_b = package_groups.get_group(
                package_b
            )

            row_a = group_a.iloc[0]
            row_b = group_b.iloc[0]

            candidate = row_to_candidate(
                row_a
            )

            # Try to import one technical value
            # from another package.
            for column in TEST_COLUMNS:

                if column not in df.columns:
                    continue

                value_a = clean_value(
                    candidate.get(
                        column,
                        NOVALUE,
                    )
                )

                value_b = clean_value(
                    representative_value(
                        row_b[column]
                    )
                )

                if (
                    value_a == value_b
                    or value_b == NOVALUE
                ):
                    continue

                mutated = copy.deepcopy(
                    candidate
                )

                mutated[column] = value_b

                expected = independent_rule_validation(
                    mutated,
                    rules,
                )["valid"]

                actual = engine.validate(
                    mutated
                )["valid"]

                counter += 1

                results.append(
                    result_record(
                        test_id=f"CROSS_{counter:05d}",
                        test_type="CROSS_CONTEXT_MUTATION",
                        expected=expected,
                        actual=actual,
                        source_row=int(
                            row_a.name
                        ),
                        characteristic=column,
                        original_value=value_a,
                        test_value=value_b,
                        details=(
                            f"Base package={package_a}; "
                            f"value taken from package={package_b}"
                        ),
                    )
                )

                # Only one mutation per package pair.
                break

    report = pd.DataFrame(
        results
    )

    print(
        f"\nTests:  {len(report):,}"
    )

    if len(report):

        print(
            f"Passed: "
            f"{report['test_passed'].sum():,}"
        )

        print(
            f"Failed: "
            f"{(~report['test_passed']).sum():,}"
        )

    return results


# =============================================================================
# TEST 4
# RANDOM OBSERVED VALUE MUTATIONS
# =============================================================================

def test_observed_mutations(
    df: pd.DataFrame,
    engine: ConstraintEngine,
    rules: pd.DataFrame,
) -> List[Dict[str, Any]]:

    print("\n" + "=" * 100)
    print("TEST 4 — OBSERVED-VALUE MUTATIONS")
    print("=" * 100)

    results = []

    counter = 0

    for row_index, row in df.iloc[::10].iterrows():

        original = row_to_candidate(
            row
        )

        original_expected = (
            independent_rule_validation(
                original,
                rules,
            )["valid"]
        )

        if not original_expected:
            continue

        for column in TEST_COLUMNS:

            if column not in df.columns:
                continue

            original_value = clean_value(
                original.get(
                    column,
                    NOVALUE,
                )
            )

            observed = set(
                clean_value(v)
                for v in df[column].dropna()
            )

            alternatives = sorted(
                value
                for value in observed
                if value != original_value
            )

            # Test all observed alternatives,
            # not assuming they are invalid.
            for alternative in alternatives[:5]:

                mutated = copy.deepcopy(
                    original
                )

                mutated[column] = (
                    alternative
                )

                expected = (
                    independent_rule_validation(
                        mutated,
                        rules,
                    )["valid"]
                )

                actual = (
                    engine.validate(
                        mutated
                    )["valid"]
                )

                counter += 1

                results.append(
                    result_record(
                        test_id=f"OBSERVED_MUTATION_{counter:06d}",
                        test_type="OBSERVED_VALUE_MUTATION",
                        expected=expected,
                        actual=actual,
                        source_row=int(
                            row_index
                        ),
                        characteristic=column,
                        original_value=original_value,
                        test_value=alternative,
                        details=(
                            "Expected validity is determined "
                            "independently from the normalized "
                            "731 rule table."
                        ),
                    )
                )

    report = pd.DataFrame(
        results
    )

    print(
        f"\nTests:  {len(report):,}"
    )

    print(
        f"Passed: "
        f"{report['test_passed'].sum():,}"
    )

    print(
        f"Failed: "
        f"{(~report['test_passed']).sum():,}"
    )

    return results


# =============================================================================
# SUMMARY
# =============================================================================

def print_summary(
    report: pd.DataFrame,
):

    print("\n" + "=" * 100)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 100)

    total = len(report)

    passed = int(
        report["test_passed"].sum()
    )

    failed = total - passed

    print(
        f"\nTotal tests:     {total:,}"
    )

    print(
        f"Passed:          {passed:,}"
    )

    print(
        f"Failed:          {failed:,}"
    )

    if total:

        print(
            f"Pass rate:       "
            f"{passed / total * 100:.2f}%"
        )

    print(
        "\nBy test type:"
    )

    summary = (
        report.groupby(
            "test_type"
        )["test_passed"]
        .agg(
            tests="count",
            passed="sum",
        )
    )

    summary["failed"] = (
        summary["tests"]
        - summary["passed"]
    )

    summary["pass_rate_%"] = (
        summary["passed"]
        / summary["tests"]
        * 100
    )

    print(
        summary.to_string(
            float_format=lambda x:
            f"{x:.2f}"
        )
    )

    failures = report[
        ~report["test_passed"]
    ]

    if len(failures):

        print(
            "\n" + "-" * 100
        )

        print(
            "FIRST FAILURES"
        )

        print(
            failures[
                [
                    "test_id",
                    "test_type",
                    "source_row",
                    "characteristic",
                    "original_value",
                    "test_value",
                    "expected_valid",
                    "actual_valid",
                    "details",
                ]
            ]
            .head(25)
            .to_string(
                index=False
            )
        )

    else:

        print(
            "\nALL TESTS PASSED."
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 100)
    print("731 CONSTRAINT ENGINE — ROBUST VALIDATION")
    print("=" * 100)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Load real data
    # -------------------------------------------------------------------------

    df = load_source_data()

    # -------------------------------------------------------------------------
    # Build normalized rule table
    # -------------------------------------------------------------------------

    rules = build_rule_table(
        df
    )

    # -------------------------------------------------------------------------
    # Build constraint engine
    # -------------------------------------------------------------------------

    engine = ConstraintEngine(
        rules
    )

    print(
        f"\nConstraint contexts: "
        f"{len(engine.rule_sets):,}"
    )

    # -------------------------------------------------------------------------
    # Run tests
    # -------------------------------------------------------------------------

    results = []

    results.extend(
        test_real_configurations(
            df,
            engine,
        )
    )

    results.extend(
        test_impossible_values(
            df,
            engine,
            rules,
        )
    )

    results.extend(
        test_cross_context_values(
            df,
            engine,
            rules,
        )
    )

    results.extend(
        test_observed_mutations(
            df,
            engine,
            rules,
        )
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    report = pd.DataFrame(
        results
    )

    report.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print_summary(
        report
    )

    print(
        "\n" + "=" * 100
    )

    print(
        "VALIDATION REPORT:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()