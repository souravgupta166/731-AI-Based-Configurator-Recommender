#!/usr/bin/env python3

"""
731 CONSTRAINT ENGINE
=====================

Purpose
-------
Build a constraint/compatibility engine from the real 731 configurator.

Input
-----
data/raw/x731_allgemein_Rev24.xlsx

Output
------
data/processed/731_constraint_rules.csv
data/processed/731_constraint_contexts.csv
data/processed/731_constraint_validation_report.csv

The engine:
1. Loads the original 731 configurator.
2. Parses fixed values and allowed-value expressions.
3. Groups rules by package/configuration context.
4. Validates concrete configurations against observed rules.
5. Tests the validator against configurations derived from the source data.
6. Saves a machine-readable rule representation.

IMPORTANT
---------
This is a rule/constraint engine, NOT an ML model.

The source 731 data is treated as observed product configuration knowledge.
No synthetic engineer preferences are introduced here.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "x731_allgemein_Rev24.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

RULE_OUTPUT = OUTPUT_DIR / "731_constraint_rules.csv"
CONTEXT_OUTPUT = OUTPUT_DIR / "731_constraint_contexts.csv"
VALIDATION_OUTPUT = OUTPUT_DIR / "731_constraint_validation_report.csv"


# =============================================================================
# CONSTANTS
# =============================================================================

NOVALUE = "NOVALUE"

IGNORED_COLUMNS = {
    "Unnamed: 0",
}

# These are metadata / context columns rather than optional technical
# characteristics.
CONTEXT_COLUMNS = {
    "packageFixed_dev",
}

# Columns that are constant across the complete dataset can still be retained
# in the rule table, but are not especially useful for candidate generation.
METADATA_COLUMNS = {
    "marke",
    "installationsart",
    "type",
    "intrinsicSafety",
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ParsedRule:
    source_row: int
    characteristic: str
    raw_value: str
    rule_type: str
    allowed_values: Tuple[str, ...]
    package_context: str


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize_scalar(value: Any) -> str:
    """
    Convert values from Excel/Pandas into a stable string representation.
    """

    if pd.isna(value):
        return NOVALUE

    text = str(value).strip()

    if not text:
        return NOVALUE

    return text


def clean_value(value: Any) -> str:
    """
    Remove surrounding quotes from values such as:
        'GP'
        'ST'
        '1'
    while preserving other content.
    """

    text = normalize_scalar(value)

    if text.upper() == NOVALUE:
        return NOVALUE

    # Remove matching single/double quotes.
    if len(text) >= 2:
        if (text[0] == "'" and text[-1] == "'") or (
            text[0] == '"' and text[-1] == '"'
        ):
            text = text[1:-1]

    return text.strip()


# =============================================================================
# RULE PARSING
# =============================================================================

def parse_allowed_expression(text: str) -> Optional[Set[str]]:
    """
    Parse expressions such as:

        in {'ST', 'AL'}
        in {1, 2}
        in {NOVALUE, 'x731'}
        in {NOVALUE, 1}

    Returns a set of normalized values.

    Returns None if the expression is not an allowed-value expression.
    """

    text = normalize_scalar(text)

    if not text.lower().startswith("in"):
        return None

    match = re.search(r"\{(.*)\}", text)

    if not match:
        return None

    inside = match.group(1).strip()

    if not inside:
        return set()

    # Use Python's AST where possible.
    try:
        parsed = ast.literal_eval("{" + inside + "}")

        if isinstance(parsed, (set, tuple, list)):
            return {
                clean_value(v)
                for v in parsed
            }

    except Exception:
        pass

    # Fallback parser.
    values = []

    for token in inside.split(","):
        token = token.strip()

        if not token:
            continue

        values.append(clean_value(token))

    return set(values)


def parse_rule_value(
    value: Any,
) -> Tuple[str, Tuple[str, ...]]:
    """
    Returns:
        rule_type
        allowed_values
    """

    text = normalize_scalar(value)

    allowed = parse_allowed_expression(text)

    if allowed is not None:
        return (
            "allowed_values",
            tuple(sorted(allowed)),
        )

    cleaned = clean_value(text)

    if cleaned.upper() == NOVALUE:
        return (
            "novalue",
            (NOVALUE,),
        )

    return (
        "fixed_value",
        (cleaned,),
    )


# =============================================================================
# PACKAGE CONTEXT
# =============================================================================

def derive_package_context(
    row: pd.Series,
) -> str:
    """
    Convert packageFixed_dev into a stable package context.

    Examples:
        in {NOVALUE, 'x731'} -> x731
        in {NOVALUE, 'G731ST-LT'} -> G731ST-LT
        'F731PW' -> F731PW
    """

    raw = normalize_scalar(row.get("packageFixed_dev", NOVALUE))

    allowed = parse_allowed_expression(raw)

    if allowed:

        non_novalue = sorted(
            value
            for value in allowed
            if value != NOVALUE
        )

        if non_novalue:
            return "|".join(non_novalue)

        return NOVALUE

    cleaned = clean_value(raw)

    if cleaned == NOVALUE:
        return NOVALUE

    return cleaned


# =============================================================================
# LOAD DATA
# =============================================================================

def load_source_data() -> pd.DataFrame:

    print("=" * 100)
    print("LOADING REAL 731 CONFIGURATOR")
    print("=" * 100)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_excel(INPUT_FILE)

    print(f"\nInput file: {INPUT_FILE}")
    print(f"Rows:       {df.shape[0]:,}")
    print(f"Columns:    {df.shape[1]:,}")

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        print("\nRemoved non-informative column: Unnamed: 0")

    return df


# =============================================================================
# BUILD RULE TABLE
# =============================================================================

def build_rule_table(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print("\n" + "=" * 100)
    print("BUILDING NORMALIZED CONSTRAINT RULES")
    print("=" * 100)

    records: List[Dict[str, Any]] = []

    for row_idx, row in df.iterrows():

        package_context = derive_package_context(row)

        for column in df.columns:

            if column in IGNORED_COLUMNS:
                continue

            raw_value = row[column]

            rule_type, allowed_values = parse_rule_value(raw_value)

            records.append(
                {
                    "source_row": int(row_idx),
                    "package_context": package_context,
                    "characteristic": column,
                    "raw_value": normalize_scalar(raw_value),
                    "rule_type": rule_type,
                    "allowed_values": "|".join(
                        allowed_values
                    ),
                    "allowed_count": len(allowed_values),
                }
            )

    rules = pd.DataFrame(records)

    print(
        f"\nNormalized constraint records: "
        f"{len(rules):,}"
    )

    print("\nRule type distribution:")

    distribution = (
        rules["rule_type"]
        .value_counts()
        .rename_axis("rule_type")
        .reset_index(name="count")
    )

    print(distribution.to_string(index=False))

    return rules


# =============================================================================
# CONTEXT TABLE
# =============================================================================

def build_context_table(
    rules: pd.DataFrame,
) -> pd.DataFrame:

    print("\n" + "=" * 100)
    print("BUILDING PACKAGE-SPECIFIC CONTEXTS")
    print("=" * 100)

    context_rows = []

    grouped = rules.groupby(
        [
            "package_context",
            "characteristic",
        ],
        dropna=False,
    )

    for (
        package_context,
        characteristic,
    ), group in grouped:

        values: Set[str] = set()

        for item in group["allowed_values"]:

            if not item:
                continue

            values.update(
                value
                for value in item.split("|")
                if value
            )

        context_rows.append(
            {
                "package_context": package_context,
                "characteristic": characteristic,
                "observed_values": "|".join(
                    sorted(values)
                ),
                "observed_value_count": len(values),
                "rule_occurrences": len(group),
                "source_rows": group["source_row"].nunique(),
            }
        )

    contexts = pd.DataFrame(context_rows)

    print(
        f"\nPackage contexts found: "
        f"{contexts['package_context'].nunique()}"
    )

    print(
        "\nPackage contexts:"
    )

    print(
        contexts[
            [
                "package_context",
                "source_rows",
            ]
        ]
        .drop_duplicates()
        .sort_values("package_context")
        .to_string(index=False)
    )

    return contexts


# =============================================================================
# ENGINE
# =============================================================================

class ConstraintEngine:
    """
    Constraint engine based exclusively on observed 731 rule rows.

    A concrete candidate configuration is valid when at least one observed
    source-rule row for the relevant package context accepts every populated
    characteristic.
    """

    def __init__(
        self,
        rules: pd.DataFrame,
    ):

        self.rules = rules.copy()

        self.rule_sets: Dict[
            str,
            List[Dict[str, Set[str]]]
        ] = {}

        self._build_rule_sets()

    # -------------------------------------------------------------------------
    # BUILD RULE SETS
    # -------------------------------------------------------------------------

    def _build_rule_sets(self):

        grouped = self.rules.groupby(
            [
                "package_context",
                "source_row",
            ],
            dropna=False,
        )

        for (
            package_context,
            source_row,
        ), group in grouped:

            rule_dict: Dict[str, Set[str]] = {}

            for _, record in group.iterrows():

                characteristic = record["characteristic"]

                values = set(
                    value
                    for value in str(
                        record["allowed_values"]
                    ).split("|")
                    if value
                )

                rule_dict[
                    characteristic
                ] = values

            self.rule_sets.setdefault(
                package_context,
                []
            ).append(
                {
                    "source_row": int(source_row),
                    "rules": rule_dict,
                }
            )

    # -------------------------------------------------------------------------
    # PACKAGE MATCHING
    # -------------------------------------------------------------------------

    def _candidate_package(
        self,
        candidate: Dict[str, Any],
    ) -> str:

        value = candidate.get(
            "packageFixed_dev",
            candidate.get(
                "package_context",
                NOVALUE,
            )
        )

        return clean_value(value)

    # -------------------------------------------------------------------------
    # VALUE MATCH
    # -------------------------------------------------------------------------

    @staticmethod
    def _value_matches(
        candidate_value: Any,
        allowed_values: Set[str],
    ) -> bool:

        candidate = clean_value(candidate_value)

        return candidate in allowed_values

    # -------------------------------------------------------------------------
    # VALIDATE AGAINST ONE RULE ROW
    # -------------------------------------------------------------------------

    def _validate_against_rule(
        self,
        candidate: Dict[str, Any],
        rule: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:

        violations = []

        rule_dict = rule["rules"]

        for characteristic, allowed_values in rule_dict.items():

            # Do not force the candidate to contain metadata fields.
            if characteristic in {
                "marke",
                "installationsart",
                "type",
                "intrinsicSafety",
            }:
                continue

            candidate_value = candidate.get(
                characteristic,
                NOVALUE,
            )

            if not self._value_matches(
                candidate_value,
                allowed_values,
            ):

                violations.append(
                    characteristic
                )

        return (
            len(violations) == 0,
            violations,
        )

    # -------------------------------------------------------------------------
    # VALIDATE CANDIDATE
    # -------------------------------------------------------------------------

    def validate(
        self,
        candidate: Dict[str, Any],
    ) -> Dict[str, Any]:

        package_context = self._candidate_package(
            candidate
        )

        possible_contexts = []

        # Exact package context.
        if package_context in self.rule_sets:
            possible_contexts.append(
                package_context
            )

        # If candidate has no package, evaluate all.
        if package_context == NOVALUE:
            possible_contexts = list(
                self.rule_sets.keys()
            )

        # Some source rules use package contexts containing NOVALUE
        # plus the actual package.
        for context in self.rule_sets.keys():

            if context == NOVALUE:
                continue

            if package_context == context:
                continue

            # Support context strings like:
            # "x731"
            # "G731ST-LT"
            # etc.
            if package_context and package_context in context.split("|"):
                possible_contexts.append(context)

        possible_contexts = list(
            dict.fromkeys(possible_contexts)
        )

        if not possible_contexts:

            return {
                "valid": False,
                "package_context": package_context,
                "matching_rule_rows": [],
                "violations": [
                    "NO_PACKAGE_CONTEXT"
                ],
            }

        matching_rows = []
        all_violations = []

        for context in possible_contexts:

            for rule in self.rule_sets[context]:

                valid, violations = (
                    self._validate_against_rule(
                        candidate,
                        rule,
                    )
                )

                if valid:

                    matching_rows.append(
                        rule["source_row"]
                    )

                else:

                    all_violations.extend(
                        violations
                    )

        if matching_rows:

            return {
                "valid": True,
                "package_context": package_context,
                "matching_rule_rows": sorted(
                    set(matching_rows)
                ),
                "violations": [],
            }

        return {
            "valid": False,
            "package_context": package_context,
            "matching_rule_rows": [],
            "violations": sorted(
                set(all_violations)
            ),
        }


# =============================================================================
# SOURCE ROW → REPRESENTATIVE CANDIDATE
# =============================================================================

def representative_value(
    raw_value: Any,
) -> str:

    rule_type, allowed_values = parse_rule_value(
        raw_value
    )

    if not allowed_values:
        return NOVALUE

    # Deterministic selection:
    # Prefer non-NOVALUE values.
    non_novalue = [
        value
        for value in allowed_values
        if value != NOVALUE
    ]

    if non_novalue:
        return sorted(non_novalue)[0]

    return NOVALUE


def build_representative_candidates(
    df: pd.DataFrame,
) -> List[Dict[str, Any]]:

    candidates = []

    for row_idx, row in df.iterrows():

        candidate = {}

        for column in df.columns:

            if column in IGNORED_COLUMNS:
                continue

            candidate[column] = representative_value(
                row[column]
            )

        candidate["_source_row"] = int(row_idx)

        candidates.append(candidate)

    return candidates


# =============================================================================
# VALIDATION TEST
# =============================================================================

def validate_source_representatives(
    engine: ConstraintEngine,
    candidates: List[Dict[str, Any]],
) -> pd.DataFrame:

    print("\n" + "=" * 100)
    print("VALIDATING REPRESENTATIVE SOURCE CONFIGURATIONS")
    print("=" * 100)

    results = []

    for candidate in candidates:

        result = engine.validate(
            candidate
        )

        results.append(
            {
                "source_row": candidate[
                    "_source_row"
                ],
                "package_context": result[
                    "package_context"
                ],
                "valid": result[
                    "valid"
                ],
                "matching_rule_count": len(
                    result[
                        "matching_rule_rows"
                    ]
                ),
                "matching_rule_rows": "|".join(
                    map(
                        str,
                        result[
                            "matching_rule_rows"
                        ],
                    )
                ),
                "violation_count": len(
                    result[
                        "violations"
                    ]
                ),
                "violations": "|".join(
                    result[
                        "violations"
                    ]
                ),
            }
        )

    report = pd.DataFrame(results)

    valid_count = int(
        report["valid"].sum()
    )

    total = len(report)

    percentage = (
        valid_count / total * 100
        if total
        else 0
    )

    print(
        f"\nValidated source representatives: "
        f"{total:,}"
    )

    print(
        f"Valid: {valid_count:,} "
        f"({percentage:.2f}%)"
    )

    print(
        f"Invalid: {total - valid_count:,}"
    )

    return report


# =============================================================================
# SUMMARY
# =============================================================================

def print_engine_summary(
    rules: pd.DataFrame,
    contexts: pd.DataFrame,
    report: pd.DataFrame,
):

    print("\n" + "=" * 100)
    print("731 CONSTRAINT ENGINE SUMMARY")
    print("=" * 100)

    print(
        f"\nSource rows:                  "
        f"{rules['source_row'].nunique():,}"
    )

    print(
        f"Characteristics:             "
        f"{rules['characteristic'].nunique():,}"
    )

    print(
        f"Normalized rules:             "
        f"{len(rules):,}"
    )

    print(
        f"Package contexts:             "
        f"{contexts['package_context'].nunique():,}"
    )

    print(
        f"Representative candidates:    "
        f"{len(report):,}"
    )

    print(
        f"Valid representatives:        "
        f"{report['valid'].sum():,}"
    )

    print(
        f"Validation success rate:      "
        f"{report['valid'].mean() * 100:.2f}%"
    )

    print("\nMost frequent violations:")

    violations = (
        report[
            report["violations"] != ""
        ]["violations"]
        .str.split("|")
        .explode()
        .value_counts()
        .head(15)
    )

    if len(violations):

        print(
            violations.to_string()
        )

    else:

        print(
            "No violations."
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # 1. LOAD
    # -------------------------------------------------------------------------

    df = load_source_data()

    # -------------------------------------------------------------------------
    # 2. RULES
    # -------------------------------------------------------------------------

    rules = build_rule_table(
        df
    )

    # -------------------------------------------------------------------------
    # 3. CONTEXTS
    # -------------------------------------------------------------------------

    contexts = build_context_table(
        rules
    )

    # -------------------------------------------------------------------------
    # 4. ENGINE
    # -------------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("INITIALIZING CONSTRAINT ENGINE")
    print("=" * 100)

    engine = ConstraintEngine(
        rules
    )

    print(
        f"\nRule contexts loaded: "
        f"{len(engine.rule_sets):,}"
    )

    # -------------------------------------------------------------------------
    # 5. REPRESENTATIVE SOURCE CONFIGURATIONS
    # -------------------------------------------------------------------------

    candidates = build_representative_candidates(
        df
    )

    # -------------------------------------------------------------------------
    # 6. VALIDATE
    # -------------------------------------------------------------------------

    validation_report = (
        validate_source_representatives(
            engine,
            candidates,
        )
    )

    # -------------------------------------------------------------------------
    # 7. SAVE
    # -------------------------------------------------------------------------

    rules.to_csv(
        RULE_OUTPUT,
        index=False,
    )

    contexts.to_csv(
        CONTEXT_OUTPUT,
        index=False,
    )

    validation_report.to_csv(
        VALIDATION_OUTPUT,
        index=False,
    )

    # -------------------------------------------------------------------------
    # 8. SUMMARY
    # -------------------------------------------------------------------------

    print_engine_summary(
        rules,
        contexts,
        validation_report,
    )

    print("\n" + "=" * 100)
    print("FILES SAVED")
    print("=" * 100)

    print(
        f"\nConstraint rules:"
        f"\n{RULE_OUTPUT}"
    )

    print(
        f"\nConstraint contexts:"
        f"\n{CONTEXT_OUTPUT}"
    )

    print(
        f"\nValidation report:"
        f"\n{VALIDATION_OUTPUT}"
    )

    print("\n" + "=" * 100)
    print("CONSTRAINT ENGINE BUILD COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
