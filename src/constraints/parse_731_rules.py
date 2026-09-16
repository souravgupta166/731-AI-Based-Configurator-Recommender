import re
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
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
# RULE PARSER
# ============================================================

def parse_allowed_values(value):
    """
    Parse configurator expressions such as:

        in {'ST', 'AL'}

        in {NOVALUE, 'x731'}

        in {1, 2}

    Returns:
        list of parsed values

    Returns None if the value is not an
    'in {...}' expression.
    """

    if pd.isna(value):
        return None

    text = str(value).strip()

    # Match expressions beginning with "in { ... }"
    match = re.match(
        r"^in\s*\{(.*)\}$",
        text,
        flags=re.IGNORECASE
    )

    if not match:
        return None

    content = match.group(1).strip()

    if not content:
        return []

    # Extract:
    #   'quoted values'
    #   "double quoted values"
    #   unquoted values
    tokens = re.findall(
        r"'([^']*)'|\"([^\"]*)\"|([^,\s]+)",
        content
    )

    values = []

    for single, double, unquoted in tokens:

        if single:
            values.append(single)

        elif double:
            values.append(double)

        else:
            values.append(unquoted)

    return values


# ============================================================
# RULE CLASSIFICATION
# ============================================================

def classify_rule_value(value):
    """
    Classify each configurator cell into a semantic rule type.

    Possible types:

        missing
        novalue
        allowed_values
        fixed_value

    Examples:

        NaN
            -> missing

        NOVALUE
            -> novalue

        in {'ST', 'AL'}
            -> allowed_values

        2
            -> fixed_value

        '731'
            -> fixed_value
    """

    # --------------------------------------------------------
    # Missing value
    # --------------------------------------------------------

    if pd.isna(value):

        return "missing", []


    text = str(value).strip()


    # --------------------------------------------------------
    # Explicit NOVALUE
    # --------------------------------------------------------

    if text.upper() == "NOVALUE":

        return "novalue", ["NOVALUE"]


    # --------------------------------------------------------
    # Allowed-value expression
    # --------------------------------------------------------

    allowed_values = parse_allowed_values(
        value
    )

    if allowed_values is not None:

        return (
            "allowed_values",
            allowed_values
        )


    # --------------------------------------------------------
    # Fixed value
    # --------------------------------------------------------

    return (
        "fixed_value",
        [text]
    )


# ============================================================
# LOAD CONFIGURATOR
# ============================================================

def load_731():

    if not RAW_FILE.exists():

        raise FileNotFoundError(
            f"Could not find:\n{RAW_FILE}"
        )

    return pd.read_excel(
        RAW_FILE
    )


# ============================================================
# EXTRACT ALL RULES
# ============================================================

def extract_rules(df):
    """
    Convert every cell in the original 731 configurator
    into a normalized rule record.

    IMPORTANT:
    Unlike the previous version, this function does NOT
    discard fixed values or NOVALUE entries.

    Every source cell is preserved.
    """

    records = []

    for row_index, row in df.iterrows():

        for column in df.columns:

            raw_value = row[column]

            rule_type, parsed_values = (
                classify_rule_value(
                    raw_value
                )
            )

            records.append({

                # Original Excel row
                "source_row": row_index,

                # Characteristic / column
                "characteristic": column,

                # Original value exactly as represented
                "raw_value": str(
                    raw_value
                ),

                # Semantic classification
                "rule_type": rule_type,

                # Parsed values separated by |
                "parsed_values": "|".join(
                    parsed_values
                ),

                # Number of values represented
                "value_count": len(
                    parsed_values
                )
            })

    return pd.DataFrame(
        records
    )


# ============================================================
# CREATE RULE-ONLY DATASET
# ============================================================

def create_allowed_rules_dataset(
    normalized_rules
):
    """
    Create a second dataset containing only
    'in {...}' rule expressions.

    This preserves the useful output from the
    original parser while the normalized dataset
    contains all cells.
    """

    return normalized_rules[
        normalized_rules["rule_type"]
        == "allowed_values"
    ].copy()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 80)
    print("731 CONFIGURATION RULE EXTRACTION")
    print("=" * 80)


    # --------------------------------------------------------
    # Load source Excel
    # --------------------------------------------------------

    df = load_731()

    print(
        f"\nSource dataset:"
        f" {df.shape[0]:,} rows × "
        f"{df.shape[1]:,} columns"
    )


    # --------------------------------------------------------
    # Extract / normalize all cells
    # --------------------------------------------------------

    normalized_rules = extract_rules(
        df
    )

    print(
        f"\nNormalized records:"
        f" {len(normalized_rules):,}"
    )


    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # OUTPUT 1
    # COMPLETE NORMALIZED RULE MATRIX
    # ========================================================

    normalized_file = (
        OUTPUT_DIR
        / "731_normalized_rule_matrix.csv"
    )

    normalized_rules.to_csv(
        normalized_file,
        index=False
    )

    print(
        "\nComplete normalized rule matrix saved to:"
    )

    print(
        normalized_file
    )


    # ========================================================
    # OUTPUT 2
    # ONLY 'in {...}' RULES
    # ========================================================

    allowed_rules = (
        create_allowed_rules_dataset(
            normalized_rules
        )
    )

    extracted_file = (
        OUTPUT_DIR
        / "731_extracted_rules.csv"
    )

    allowed_rules.to_csv(
        extracted_file,
        index=False
    )

    print(
        "\nAllowed-value rules saved to:"
    )

    print(
        extracted_file
    )


    # ========================================================
    # REPORT 1
    # RULE TYPE DISTRIBUTION
    # ========================================================

    print("\n" + "=" * 80)
    print("RULE TYPE DISTRIBUTION")
    print("=" * 80)

    print(
        normalized_rules[
            "rule_type"
        ]
        .value_counts()
        .to_string()
    )


    # ========================================================
    # REPORT 2
    # CHARACTERISTICS WITH ALLOWED-VALUE RULES
    # ========================================================

    print("\n" + "=" * 80)
    print(
        "CHARACTERISTICS WITH "
        "ALLOWED-VALUE RULES"
    )
    print("=" * 80)

    characteristic_summary = (
        allowed_rules
        .groupby("characteristic")
        .agg(
            rule_occurrences=(
                "source_row",
                "count"
            ),

            distinct_rule_sets=(
                "parsed_values",
                "nunique"
            )
        )
        .sort_values(
            "distinct_rule_sets",
            ascending=False
        )
    )

    print(
        characteristic_summary
        .to_string()
    )


    # ========================================================
    # REPORT 3
    # SAMPLE NORMALIZED RECORDS
    # ========================================================

    print("\n" + "=" * 80)
    print(
        "SAMPLE NORMALIZED RECORDS"
    )
    print("=" * 80)

    print(
        normalized_rules
        .head(30)
        .to_string(index=False)
    )


    # ========================================================
    # REPORT 4
    # SAMPLE ALLOWED-VALUE RULES
    # ========================================================

    print("\n" + "=" * 80)
    print(
        "SAMPLE ALLOWED-VALUE RULES"
    )
    print("=" * 80)

    print(
        allowed_rules
        .head(20)
        .to_string(index=False)
    )


    # ========================================================
    # FINISHED
    # ========================================================

    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)

    print(
        "\nCreated:"
    )

    print(
        f"  1. {normalized_file}"
    )

    print(
        f"  2. {extracted_file}"
    )