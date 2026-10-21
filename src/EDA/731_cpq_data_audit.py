#!/usr/bin/env python3
"""
731 CPQ DATA AUDIT
==================

Purpose
-------
Audit the 731 technical source data before changing the Streamlit CPQ UI.

The audit separates:

1. CPQ-visible technical selection fields
2. Deterministic backend / constraint fields
3. Constant product metadata
4. Legacy / unused fields
5. Fields requiring further investigation
6. Model-code information that still requires authoritative verification

IMPORTANT
---------
This script does NOT invent Emerson model-code meanings.

The internal CFG731_* canonical configuration ID is NOT treated as an
Emerson commercial/model code.

The technical source data remains the source of truth for this audit.

Inputs
------
data/raw/x731_allgemein_Rev24.xlsx

data/processed/731_normalized_rule_matrix.csv

data/processed/731_extracted_rules.csv

data/synthetic/configurations/731_synthetic_configurations_v5.csv

graphs/731_knowledge_graph_v2/nodes.csv

graphs/731_knowledge_graph_v2/relationships.csv


Outputs
-------
data/processed/731_cpq_data_audit.csv

data/processed/731_cpq_data_audit_summary.json

data/processed/731_cpq_data_audit_report.md

data/processed/731_cpq_value_profile_top20.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

SOURCE = ROOT / "data/raw/x731_allgemein_Rev24.xlsx"

NORMALIZED = (
    ROOT / "data/processed/731_normalized_rule_matrix.csv"
)

RULES = (
    ROOT / "data/processed/731_extracted_rules.csv"
)

SYNTHETIC = (
    ROOT
    / "data/synthetic/configurations/731_synthetic_configurations_v5.csv"
)

KG_NODES = (
    ROOT
    / "graphs/731_knowledge_graph_v2/nodes.csv"
)

KG_EDGES = (
    ROOT
    / "graphs/731_knowledge_graph_v2/relationships.csv"
)


# ============================================================
# 2. OUTPUT PATHS
# ============================================================

OUT_DIR = ROOT / "data/processed"

OUT_CSV = (
    OUT_DIR / "731_cpq_data_audit.csv"
)

OUT_JSON = (
    OUT_DIR / "731_cpq_data_audit_summary.json"
)

OUT_MD = (
    OUT_DIR / "731_cpq_data_audit_report.md"
)

VALUE_PROFILE = (
    OUT_DIR / "731_cpq_value_profile_top20.csv"
)


# ============================================================
# 3. CURRENT UI TECHNICAL FIELDS
# ============================================================
#
# These are the 19 technical fields currently represented
# in the configurator backend/UI.
#
# IMPORTANT:
# This list describes the CURRENT implementation.
# It does not claim that these are the only CPQ fields
# that exist in the original Emerson source.
# ============================================================

CURRENT_UI_TECHNICAL_FIELDS = [
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


# ============================================================
# 4. KNOWN CONSTANT PRODUCT FIELDS
# ============================================================
#
# These fields were previously identified as effectively
# constant in the 731 source data.
#
# A constant field should NOT automatically become a customer
# selection in the CPQ UI.
# ============================================================

KNOWN_CONSTANT_FIELDS = {
    "marke",
    "installationsart",
    "type",
    "intrinsicSafety",
    "inclusiveManual",
    "powerCable",
    "carryingCase",
    "measuringTape",
}


# ============================================================
# 5. KNOWN ALL-NOVALUE / LEGACY FIELDS
# ============================================================
#
# These fields were identified in the source audit as fields
# that are effectively unused / NOVALUE in the normalized
# technical data.
#
# They should not be exposed as active CPQ selections unless
# future source evidence proves otherwise.
# ============================================================

KNOWN_ALL_NOVALUE_FIELDS = {
    "analogOutput",
    "analogOutput2",
    "analogInput",
    "analogInput2",
    "digitalOutput",
    "digitalOutput2",
    "digitalInput",
    "digitalInput2",
    "hardwareAP",
}


# ============================================================
# 6. OTHER BACKEND / OPTIONAL METADATA
# ============================================================

KNOWN_BACKEND_METADATA_FIELDS = {
    "tagPlate",
}


# ============================================================
# 7. MODEL-CODE KEYWORDS
# ============================================================
#
# This is ONLY used to flag fields for investigation.
#
# It must NOT be interpreted as proof that the field participates
# in the Emerson model code.
# ============================================================

MODEL_CODE_KEYWORDS = (
    "model",
    "modelcode",
    "order",
    "ordercode",
    "article",
    "part",
    "productcode",
)


# ============================================================
# 8. HELPER FUNCTIONS
# ============================================================

def file_exists(path: Path) -> bool:
    """Return True when a file exists."""
    return path.exists()


def load_csv(path: Path) -> pd.DataFrame | None:
    """
    Load a CSV if it exists.

    Returns None if the file does not exist.
    """
    if not path.exists():
        return None

    return pd.read_csv(
        path,
        low_memory=False,
    )


def safe_unique(series: pd.Series) -> list[str]:
    """
    Return sorted unique non-empty string values.
    """
    values = (
        series
        .dropna()
        .astype(str)
        .str.strip()
    )

    values = [
        value
        for value in values.unique()
        if value
        and value.lower() != "nan"
    ]

    return sorted(values)


def relative_path(path: Path) -> str:
    """
    Return a project-relative path when possible.
    """
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


# ============================================================
# 9. RULE FIELD DETECTION
# ============================================================

def detect_rule_field_column(
    rules: pd.DataFrame | None,
) -> str | None:
    """
    Detect the field-name column used by the extracted rule
    dataset.

    Different pipeline versions can use slightly different
    column names, so we check common possibilities.
    """

    if rules is None or rules.empty:
        return None

    candidates = [
        "field_name",
        "characteristic",
        "field",
        "column",
        "source_column",
    ]

    for candidate in candidates:
        if candidate in rules.columns:
            return candidate

    return None


def get_rule_counts(
    rules: pd.DataFrame | None,
) -> dict[str, int]:
    """
    Count extracted rule occurrences by source field.
    """

    field_column = detect_rule_field_column(rules)

    if field_column is None:
        return {}

    counts = (
        rules[field_column]
        .astype(str)
        .value_counts()
        .to_dict()
    )

    return {
        str(key): int(value)
        for key, value in counts.items()
    }


# ============================================================
# 10. FIELD CLASSIFICATION
# ============================================================

def classify_field(
    field: str,
    source_unique_count: int,
    source_non_null_count: int,
    source_rows: int,
    rule_occurrences: int,
) -> tuple[str, str]:
    """
    Classify a source field.

    The classification is deliberately conservative.

    We do NOT automatically turn every variable source column
    into a CPQ selector.
    """

    # --------------------------------------------------------
    # Current UI technical fields
    # --------------------------------------------------------

    if field in CURRENT_UI_TECHNICAL_FIELDS:
        return (
            "CPQ_VISIBLE_TECHNICAL",
            (
                "Currently represented in the 19-field technical "
                "configuration schema."
            ),
        )

    # --------------------------------------------------------
    # All-NOVALUE / legacy fields
    # --------------------------------------------------------

    if field in KNOWN_ALL_NOVALUE_FIELDS:
        return (
            "LEGACY_OR_UNUSED",
            (
                "Previously identified as effectively NOVALUE "
                "throughout the normalized 731 dataset."
            ),
        )

    # --------------------------------------------------------
    # Known constant fields
    # --------------------------------------------------------

    if field in KNOWN_CONSTANT_FIELDS:
        return (
            "CONSTANT_PRODUCT_METADATA",
            (
                "Field is effectively fixed in the 731 source "
                "and should not automatically become a customer "
                "selection."
            ),
        )

    # --------------------------------------------------------
    # Backend / optional metadata
    # --------------------------------------------------------

    if field in KNOWN_BACKEND_METADATA_FIELDS:
        return (
            "BACKEND_OR_OPTIONAL_METADATA",
            (
                "Field varies in the source but is not part of "
                "the current 19-field technical configuration "
                "schema."
            ),
        )

    # --------------------------------------------------------
    # Single observed value
    # --------------------------------------------------------

    if source_unique_count <= 1:
        return (
            "CONSTANT_PRODUCT_METADATA",
            (
                "Only one observed source value was found. "
                "Investigate whether this represents product "
                "metadata or a fixed CPQ value."
            ),
        )

    # --------------------------------------------------------
    # Model-code investigation
    # --------------------------------------------------------
    #
    # This does NOT mean it IS model-code information.
    # It only flags the field for further investigation.
    # --------------------------------------------------------

    lower_name = field.lower()

    if any(
        keyword in lower_name
        for keyword in MODEL_CODE_KEYWORDS
    ):
        return (
            "MODEL_CODE_INVESTIGATION",
            (
                "Field name suggests possible ordering/model-code "
                "relevance, but no authoritative mapping is "
                "established by this audit."
            ),
        )

    # --------------------------------------------------------
    # Fields participating in extracted rules
    # --------------------------------------------------------

    if rule_occurrences > 0:
        return (
            "BACKEND_RULE_FIELD",
            (
                "Field has extracted deterministic rule occurrences "
                "and therefore participates in source-data logic."
            ),
        )

    # --------------------------------------------------------
    # Partially populated fields
    # --------------------------------------------------------

    if source_non_null_count < source_rows:
        return (
            "INVESTIGATE",
            (
                "Field is variable or partially populated and "
                "requires CPQ/UI mapping evidence before exposure."
            ),
        )

    # --------------------------------------------------------
    # Default
    # --------------------------------------------------------

    return (
        "INVESTIGATE",
        (
            "Field is present in the source but its customer-facing "
            "CPQ role has not yet been established."
        ),
    )


# ============================================================
# 11. BUILD FIELD AUDIT
# ============================================================

def build_source_field_audit(
    source: pd.DataFrame,
    rules: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Build a row-by-row audit of every source column.
    """

    rule_counts = get_rule_counts(rules)

    rows: list[dict] = []

    source_row_count = len(source)

    for field in source.columns:

        series = source[field]

        non_null_count = int(
            series.notna().sum()
        )

        unique_count = len(
            series
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
        )

        rule_occurrences = int(
            rule_counts.get(field, 0)
        )

        classification, rationale = classify_field(
            field=field,
            source_unique_count=unique_count,
            source_non_null_count=non_null_count,
            source_rows=source_row_count,
            rule_occurrences=rule_occurrences,
        )

        is_current_ui = (
            field in CURRENT_UI_TECHNICAL_FIELDS
        )

        if is_current_ui:
            recommended_action = (
                "KEEP_IN_TECHNICAL_SCHEMA"
            )
        else:
            recommended_action = (
                "DO_NOT_EXPOSE_UNTIL_MAPPED"
            )

        rows.append(
            {
                "field_name": field,
                "source_rows": source_row_count,
                "source_non_null": non_null_count,
                "source_null_count": (
                    source_row_count - non_null_count
                ),
                "source_unique_count": unique_count,
                "current_ui_field": is_current_ui,
                "rule_occurrences": rule_occurrences,
                "classification": classification,
                "rationale": rationale,
                "model_code_mapping_status": (
                    "NOT_VERIFIED"
                ),
                "recommended_action": recommended_action,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 12. VALUE PROFILE
# ============================================================

def make_value_profile(
    source: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce the top 20 observed values for every source field.
    """

    rows: list[dict] = []

    total_rows = len(source)

    for field in source.columns:

        values = (
            source[field]
            .dropna()
            .astype(str)
            .str.strip()
        )

        counts = values.value_counts()

        for value, count in counts.head(20).items():

            rows.append(
                {
                    "field_name": field,
                    "value": value,
                    "count": int(count),
                    "share": round(
                        float(count / total_rows),
                        6,
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# 13. MAIN AUDIT
# ============================================================

def main() -> None:

    print()
    print("=" * 80)
    print("731 CPQ DATA AUDIT")
    print("=" * 80)
    print()

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Verify source workbook
    # --------------------------------------------------------

    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Missing source workbook:\n{SOURCE}"
        )

    # --------------------------------------------------------
    # Load source workbook
    # --------------------------------------------------------

    print("Loading source workbook...")

    source = pd.read_excel(
        SOURCE,
        sheet_name="x731_allgemein",
    )

    print(
        f"Source shape: "
        f"{source.shape[0]} rows x "
        f"{source.shape[1]} columns"
    )

    print()

    # --------------------------------------------------------
    # Load supporting artifacts
    # --------------------------------------------------------

    print("Loading supporting artifacts...")

    normalized = load_csv(
        NORMALIZED
    )

    rules = load_csv(
        RULES
    )

    synthetic = load_csv(
        SYNTHETIC
    )

    kg_nodes = load_csv(
        KG_NODES
    )

    kg_edges = load_csv(
        KG_EDGES
    )

    print(
        f"Normalized matrix: "
        f"{'FOUND' if normalized is not None else 'MISSING'}"
    )

    print(
        f"Extracted rules:    "
        f"{'FOUND' if rules is not None else 'MISSING'}"
    )

    print(
        f"Synthetic configs:  "
        f"{'FOUND' if synthetic is not None else 'MISSING'}"
    )

    print(
        f"KG nodes:           "
        f"{'FOUND' if kg_nodes is not None else 'MISSING'}"
    )

    print(
        f"KG relationships:    "
        f"{'FOUND' if kg_edges is not None else 'MISSING'}"
    )

    print()

    # --------------------------------------------------------
    # Build field audit
    # --------------------------------------------------------

    print("Auditing source fields...")

    audit = build_source_field_audit(
        source=source,
        rules=rules,
    )

    # --------------------------------------------------------
    # Save field audit
    # --------------------------------------------------------

    audit.to_csv(
        OUT_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # Determine important field groups
    # --------------------------------------------------------

    variable_fields = audit[
        (
            audit["source_unique_count"] > 1
        )
        &
        (
            audit["classification"]
            != "LEGACY_OR_UNUSED"
        )
    ]["field_name"].tolist()

    constant_fields = audit[
        audit["classification"]
        == "CONSTANT_PRODUCT_METADATA"
    ]["field_name"].tolist()

    legacy_fields = audit[
        audit["classification"]
        == "LEGACY_OR_UNUSED"
    ]["field_name"].tolist()

    backend_fields = audit[
        audit["classification"]
        == "BACKEND_RULE_FIELD"
    ]["field_name"].tolist()

    investigation_fields = audit[
        audit["classification"].isin(
            [
                "INVESTIGATE",
                "MODEL_CODE_INVESTIGATION",
                "BACKEND_OR_OPTIONAL_METADATA",
            ]
        )
    ]["field_name"].tolist()

    ui_missing_from_source = [
        field
        for field in CURRENT_UI_TECHNICAL_FIELDS
        if field not in source.columns
    ]

    variable_source_not_in_ui = [
        field
        for field in variable_fields
        if field not in CURRENT_UI_TECHNICAL_FIELDS
    ]

    # --------------------------------------------------------
    # Build summary
    # --------------------------------------------------------

    summary = {
        "source": {
            "file": relative_path(SOURCE),
            "rows": int(source.shape[0]),
            "columns": int(source.shape[1]),
            "current_ui_technical_field_count": (
                len(CURRENT_UI_TECHNICAL_FIELDS)
            ),
            "ui_fields_missing_from_source": (
                ui_missing_from_source
            ),
        },

        "field_classification": {
            "current_ui_technical_fields": int(
                (
                    audit["classification"]
                    == "CPQ_VISIBLE_TECHNICAL"
                ).sum()
            ),

            "constant_product_metadata": int(
                (
                    audit["classification"]
                    == "CONSTANT_PRODUCT_METADATA"
                ).sum()
            ),

            "legacy_or_unused": int(
                (
                    audit["classification"]
                    == "LEGACY_OR_UNUSED"
                ).sum()
            ),

            "backend_rule_fields": int(
                (
                    audit["classification"]
                    == "BACKEND_RULE_FIELD"
                ).sum()
            ),

            "variable_fields_not_currently_in_ui": int(
                len(variable_source_not_in_ui)
            ),

            "investigate_fields": int(
                len(investigation_fields)
            ),
        },

        "artifacts": {
            "normalized_exists": (
                normalized is not None
            ),

            "normalized_rows": (
                int(len(normalized))
                if normalized is not None
                else None
            ),

            "normalized_columns": (
                int(normalized.shape[1])
                if normalized is not None
                else None
            ),

            "rules_exists": (
                rules is not None
            ),

            "rules_rows": (
                int(len(rules))
                if rules is not None
                else None
            ),

            "rules_columns": (
                int(rules.shape[1])
                if rules is not None
                else None
            ),

            "synthetic_exists": (
                synthetic is not None
            ),

            "synthetic_rows": (
                int(len(synthetic))
                if synthetic is not None
                else None
            ),

            "synthetic_columns": (
                int(synthetic.shape[1])
                if synthetic is not None
                else None
            ),

            "kg_nodes_exists": (
                kg_nodes is not None
            ),

            "kg_nodes": (
                int(len(kg_nodes))
                if kg_nodes is not None
                else None
            ),

            "kg_edges_exists": (
                kg_edges is not None
            ),

            "kg_edges": (
                int(len(kg_edges))
                if kg_edges is not None
                else None
            ),
        },

        "model_code": {
            "status": "NOT_VERIFIED",
            "mapping_inferred": False,
            "canonical_cfg_ids_are_model_codes": False,
            "action": (
                "Obtain authoritative Emerson/CPQ model-code "
                "mapping before implementing model-code generation."
            ),
        },
    }

    # --------------------------------------------------------
    # Save JSON summary
    # --------------------------------------------------------

    OUT_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Create value profile
    # --------------------------------------------------------

    value_profile = make_value_profile(
        source
    )

    value_profile.to_csv(
        VALUE_PROFILE,
        index=False,
    )

    # --------------------------------------------------------
    # Build Markdown report
    # --------------------------------------------------------

    report: list[str] = []

    report.append(
        "# 731 CPQ Data Audit"
    )

    report.append("")

    report.append(
        "## 1. Audit objective"
    )

    report.append("")

    report.append(
        "This audit evaluates whether the 731 source data can "
        "support a CPQ-equivalent technical configuration "
        "workflow without exposing unsupported or invented "
        "configuration fields."
    )

    report.append("")

    report.append(
        "The audit deliberately separates customer-facing "
        "technical selections from deterministic backend rules, "
        "constant product metadata, legacy fields, and fields "
        "requiring further investigation."
    )

    report.append("")

    report.append(
        "## 2. Source inventory"
    )

    report.append("")

    report.append(
        f"- Source workbook: `{relative_path(SOURCE)}`"
    )

    report.append(
        f"- Rows: **{source.shape[0]:,}**"
    )

    report.append(
        f"- Columns: **{source.shape[1]}**"
    )

    report.append(
        f"- Current technical UI fields: "
        f"**{len(CURRENT_UI_TECHNICAL_FIELDS)}**"
    )

    report.append("")

    # --------------------------------------------------------
    # Current UI fields
    # --------------------------------------------------------

    report.append(
        "## 3. Current technical UI schema"
    )

    report.append("")

    report.append(
        "The following fields are currently represented in the "
        "technical configuration schema:"
    )

    report.append("")

    for field in CURRENT_UI_TECHNICAL_FIELDS:
        report.append(
            f"- `{field}`"
        )

    report.append("")

    # --------------------------------------------------------
    # Classification table
    # --------------------------------------------------------

    report.append(
        "## 4. Field classification"
    )

    report.append("")

    report.append(
        "| Classification | Count |"
    )

    report.append(
        "|---|---:|"
    )

    report.append(
        f"| CPQ-visible technical | "
        f"{summary['field_classification']['current_ui_technical_fields']} |"
    )

    report.append(
        f"| Constant product metadata | "
        f"{summary['field_classification']['constant_product_metadata']} |"
    )

    report.append(
        f"| Legacy / unused | "
        f"{summary['field_classification']['legacy_or_unused']} |"
    )

    report.append(
        f"| Backend rule fields | "
        f"{summary['field_classification']['backend_rule_fields']} |"
    )

    report.append(
        f"| Variable fields not currently in UI | "
        f"{summary['field_classification']['variable_fields_not_currently_in_ui']} |"
    )

    report.append(
        f"| Investigate / backend / model-code | "
        f"{summary['field_classification']['investigate_fields']} |"
    )

    report.append("")

    # --------------------------------------------------------
    # Variable fields not in UI
    # --------------------------------------------------------

    report.append(
        "### 4.1 Variable source fields not currently in the UI"
    )

    report.append("")

    if variable_source_not_in_ui:

        for field in variable_source_not_in_ui:
            report.append(
                f"- `{field}`"
            )

    else:

        report.append(
            "No variable source fields were identified "
            "outside the current UI schema."
        )

    report.append("")

    # --------------------------------------------------------
    # Constant fields
    # --------------------------------------------------------

    report.append(
        "### 4.2 Constant product metadata"
    )

    report.append("")

    for field in constant_fields:
        report.append(
            f"- `{field}`"
        )

    report.append("")

    # --------------------------------------------------------
    # Legacy fields
    # --------------------------------------------------------

    report.append(
        "### 4.3 Legacy / unused fields"
    )

    report.append("")

    for field in legacy_fields:
        report.append(
            f"- `{field}`"
        )

    report.append("")

    # --------------------------------------------------------
    # Backend rule fields
    # --------------------------------------------------------

    report.append(
        "### 4.4 Backend rule fields"
    )

    report.append("")

    if backend_fields:

        for field in backend_fields:
            report.append(
                f"- `{field}`"
            )

    else:

        report.append(
            "No additional backend rule fields were detected."
        )

    report.append("")

    # --------------------------------------------------------
    # Investigation fields
    # --------------------------------------------------------

    report.append(
        "### 4.5 Fields requiring investigation"
    )

    report.append("")

    if investigation_fields:

        for field in investigation_fields:
            report.append(
                f"- `{field}`"
            )

    else:

        report.append(
            "No additional investigation fields were detected."
        )

    report.append("")

    # --------------------------------------------------------
    # Model code
    # --------------------------------------------------------

    report.append(
        "## 5. Model-code status"
    )

    report.append("")

    report.append(
        "**NOT VERIFIED.**"
    )

    report.append("")

    report.append(
        "This audit does not infer the meaning of Emerson "
        "model-code positions from configuration IDs, hashes, "
        "column names, observed value combinations, or synthetic "
        "configuration identifiers."
    )

    report.append("")

    report.append(
        "The internal `CFG731_*` canonical configuration "
        "identifier is treated only as an internal deterministic "
        "configuration identity. It is not treated as an Emerson "
        "commercial/model code."
    )

    report.append("")

    # --------------------------------------------------------
    # Backend architecture
    # --------------------------------------------------------

    report.append(
        "## 6. Recommended CPQ architecture"
    )

    report.append("")

    report.append(
        "```text"
    )

    report.append(
        "REAL 731 CPQ DATA"
    )

    report.append(
        "        ↓"
    )

    report.append(
        "Complete CPQ field catalogue"
    )

    report.append(
        "        ↓"
    )

    report.append(
        "Allowed values + fixed values"
    )

    report.append(
        "        ↓"
    )

    report.append(
        "Deterministic compatibility / constraint engine"
    )

    report.append(
        "        ↓"
    )

    report.append(
        "Valid configurations"
    )

    report.append(
        "        ↓"
    )

    report.append(
        "ML ranking"
    )

    report.append(
        "        ↓"
    )

    report.append(
        "Recommended valid configuration"
    )

    report.append(
        "        ↓"
    )

    report.append(
        "Verified model code"
    )

    report.append(
        "```"
    )

    report.append("")

    # --------------------------------------------------------
    # Implementation sequence
    # --------------------------------------------------------

    report.append(
        "## 7. Recommended implementation sequence"
    )

    report.append("")

    implementation_steps = [
        (
            "Freeze the 731 source workbook as the technical "
            "source of truth."
        ),
        (
            "Build a field-by-field CPQ catalogue mapping "
            "customer-facing labels to source columns."
        ),
        (
            "Record the allowed values and fixed values for "
            "each field."
        ),
        (
            "Map each selectable field to deterministic "
            "compatibility / constraint logic."
        ),
        (
            "Separate customer-visible selectors from backend "
            "metadata and constant product fields."
        ),
        (
            "Generate only configurations that pass deterministic "
            "technical validation."
        ),
        (
            "Run ML ranking only over technically valid "
            "configurations."
        ),
        (
            "Add exact model-code generation only after the "
            "authoritative model-code mapping is verified."
        ),
        (
            "Update the Streamlit UI from the backend CPQ "
            "catalogue rather than hard-coding unsupported "
            "values."
        ),
    ]

    for index, step in enumerate(
        implementation_steps,
        start=1,
    ):
        report.append(
            f"{index}. {step}"
        )

    report.append("")

    # --------------------------------------------------------
    # Supporting artifact inventory
    # --------------------------------------------------------

    report.append(
        "## 8. Supporting artifact inventory"
    )

    report.append("")

    report.append(
        "| Artifact | Status | Rows | Columns |"
    )

    report.append(
        "|---|---|---:|---:|"
    )

    if normalized is not None:

        report.append(
            f"| Normalized rule matrix | FOUND | "
            f"{len(normalized):,} | "
            f"{normalized.shape[1]} |"
        )

    else:

        report.append(
            "| Normalized rule matrix | MISSING | - | - |"
        )

    if rules is not None:

        report.append(
            f"| Extracted rules | FOUND | "
            f"{len(rules):,} | "
            f"{rules.shape[1]} |"
        )

    else:

        report.append(
            "| Extracted rules | MISSING | - | - |"
        )

    if synthetic is not None:

        report.append(
            f"| Synthetic configurations | FOUND | "
            f"{len(synthetic):,} | "
            f"{synthetic.shape[1]} |"
        )

    else:

        report.append(
            "| Synthetic configurations | MISSING | - | - |"
        )

    if kg_nodes is not None:

        report.append(
            f"| Knowledge graph nodes | FOUND | "
            f"{len(kg_nodes):,} | "
            f"{kg_nodes.shape[1]} |"
        )

    else:

        report.append(
            "| Knowledge graph nodes | MISSING | - | - |"
        )

    if kg_edges is not None:

        report.append(
            f"| Knowledge graph relationships | FOUND | "
            f"{len(kg_edges):,} | "
            f"{kg_edges.shape[1]} |"
        )

    else:

        report.append(
            "| Knowledge graph relationships | MISSING | - | - |"
        )

    report.append("")

    # --------------------------------------------------------
    # Output files
    # --------------------------------------------------------

    report.append(
        "## 9. Generated outputs"
    )

    report.append("")

    report.append(
        f"- `{relative_path(OUT_CSV)}`"
    )

    report.append(
        f"- `{relative_path(OUT_JSON)}`"
    )

    report.append(
        f"- `{relative_path(OUT_MD)}`"
    )

    report.append(
        f"- `{relative_path(VALUE_PROFILE)}`"
    )

    report.append("")

    # --------------------------------------------------------
    # Final conclusion
    # --------------------------------------------------------

    report.append(
        "## 10. Audit conclusion"
    )

    report.append("")

    report.append(
        "The 731 source data contains substantially more "
        "information than the current 19-field technical UI "
        "schema. However, additional source columns must not "
        "automatically be exposed as customer-facing CPQ "
        "selectors."
    )

    report.append("")

    report.append(
        "The next implementation step is therefore a controlled "
        "field-by-field CPQ mapping rather than a simple "
        "one-to-one exposure of all 60 source columns."
    )

    report.append("")

    report.append(
        "The model-code generation layer remains blocked until "
        "an authoritative mapping between configuration choices "
        "and the actual Emerson/CPQ model code has been verified."
    )

    report.append("")

    # --------------------------------------------------------
    # Save Markdown
    # --------------------------------------------------------

    OUT_MD.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    # ========================================================
    # 14. TERMINAL SUMMARY
    # ========================================================

    print("=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)
    print()

    print("SOURCE")
    print(
        f"  Workbook: {relative_path(SOURCE)}"
    )
    print(
        f"  Rows:     {source.shape[0]:,}"
    )
    print(
        f"  Columns:  {source.shape[1]}"
    )

    print()

    print("FIELD CLASSIFICATION")

    print(
        "  Current UI technical:       "
        f"{summary['field_classification']['current_ui_technical_fields']}"
    )

    print(
        "  Constant metadata:          "
        f"{summary['field_classification']['constant_product_metadata']}"
    )

    print(
        "  Legacy / unused:            "
        f"{summary['field_classification']['legacy_or_unused']}"
    )

    print(
        "  Backend rule fields:        "
        f"{summary['field_classification']['backend_rule_fields']}"
    )

    print(
        "  Variable not in UI:         "
        f"{summary['field_classification']['variable_fields_not_currently_in_ui']}"
    )

    print(
        "  Investigate / backend:      "
        f"{summary['field_classification']['investigate_fields']}"
    )

    print()

    print(
        "VARIABLE SOURCE FIELDS NOT CURRENTLY IN UI"
    )

    if variable_source_not_in_ui:

        for field in variable_source_not_in_ui:
            print(
                f"  - {field}"
            )

    else:

        print(
            "  None"
        )

    print()

    print(
        "MODEL CODE"
    )

    print(
        "  STATUS: NOT_VERIFIED"
    )

    print(
        "  No model-code meanings were inferred."
    )

    print()

    print(
        "GENERATED FILES"
    )

    print(
        f"  {relative_path(OUT_CSV)}"
    )

    print(
        f"  {relative_path(OUT_JSON)}"
    )

    print(
        f"  {relative_path(OUT_MD)}"
    )

    print(
        f"  {relative_path(VALUE_PROFILE)}"
    )

    print()

    print(
        "Next step:"
    )

    print(
        "  Paste this terminal output back into the chat."
    )

    print(
        "  We will then build the field-by-field CPQ mapping."
    )

    print()


# ============================================================
# 15. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()