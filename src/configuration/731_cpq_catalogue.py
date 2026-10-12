#!/usr/bin/env python3
"""
731 CPQ CATALOGUE
=================

Purpose
-------
Create a data-driven CPQ catalogue for the 731 product using the
actual Emerson 731 source workbook.

The catalogue separates:

1. Customer-facing technical configuration fields
2. Backend package/context fields
3. Source/document metadata
4. Observed source values
5. Source rule expressions
6. Package-specific source profiles

IMPORTANT
---------
This module does NOT:

- invent model-code meanings
- generate Emerson commercial/model codes
- replace the deterministic constraint engine
- allow ML to decide technical validity

The source workbook remains the technical source of truth.

Architecture
------------

                    731 SOURCE DATA
                          |
                          v
                 CPQ FIELD CATALOGUE
                          |
             +------------+------------+
             |                         |
             v                         v
       Technical fields          Backend context
             |                         |
             +------------+------------+
                          |
                          v
              Deterministic validation
                          |
                          v
                 Valid configurations
                          |
                          v
                     ML ranking
                          |
                          v
               Recommended configuration


Source
------
data/raw/x731_allgemein_Rev24.xlsx

Main sheet
----------
x731_allgemein

Generated outputs
-----------------
data/processed/731_cpq_catalogue.json
data/processed/731_cpq_catalogue.csv
data/processed/731_cpq_package_profiles.csv
data/processed/731_cpq_catalogue_report.md
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any


import pandas as pd


# ============================================================================
# 1. PROJECT PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]

SOURCE_FILE = (
    ROOT
    / "data"
    / "raw"
    / "x731_allgemein_Rev24.xlsx"
)

SOURCE_SHEET = "x731_allgemein"

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
)

CATALOGUE_JSON = (
    OUTPUT_DIR
    / "731_cpq_catalogue.json"
)

CATALOGUE_CSV = (
    OUTPUT_DIR
    / "731_cpq_catalogue.csv"
)

PACKAGE_PROFILE_CSV = (
    OUTPUT_DIR
    / "731_cpq_package_profiles.csv"
)

CATALOGUE_REPORT = (
    OUTPUT_DIR
    / "731_cpq_catalogue_report.md"
)


# ============================================================================
# 2. CUSTOMER-FACING TECHNICAL FIELDS
# ============================================================================
#
# These are the 19 technical fields currently represented in the application.
#
# IMPORTANT:
# Their allowed/observed values are derived from the real workbook below.
# We do not manually invent the value lists.
# ============================================================================

TECHNICAL_FIELDS = [
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


# ============================================================================
# 3. BACKEND CPQ CONTEXT FIELDS
# ============================================================================
#
# These fields participate in the backend/source configuration structure but
# should NOT automatically become customer-facing dropdowns.
# ============================================================================

BACKEND_FIELDS = {
    "packageFixed_dev": {
        "label": "Package Context",
        "role": "package_context",
        "customer_selectable": False,
        "description": (
            "Backend package/family context used to distinguish "
            "package-specific configuration rules."
        ),
    },
    "OrgUnitAreaOfDocument": {
        "label": "Organizational / Document Area",
        "role": "source_metadata",
        "customer_selectable": False,
        "description": (
            "Source/document metadata. Retained for provenance and "
            "auditability; not treated as a customer technical selector."
        ),
    },
}


# ============================================================================
# 4. SOURCE METADATA FIELDS
# ============================================================================
#
# These are known fixed/product metadata fields identified during the audit.
# They remain in the source but are not customer-facing configuration
# selections.
# ============================================================================

CONSTANT_METADATA_FIELDS = {
    "marke": {
        "role": "product_metadata",
        "customer_selectable": False,
    },
    "installationsart": {
        "role": "product_metadata",
        "customer_selectable": False,
    },
    "type": {
        "role": "product_metadata",
        "customer_selectable": False,
    },
    "intrinsicSafety": {
        "role": "product_metadata",
        "customer_selectable": False,
    },
    "inclusiveManual": {
        "role": "product_metadata",
        "customer_selectable": False,
    },
    "powerCable": {
        "role": "product_metadata",
        "customer_selectable": False,
    },
    "carryingCase": {
        "role": "product_metadata",
        "customer_selectable": False,
    },
    "measuringTape": {
        "role": "product_metadata",
        "customer_selectable": False,
    },
}


# ============================================================================
# 5. LEGACY / UNUSED FIELDS
# ============================================================================

LEGACY_FIELDS = {
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


# ============================================================================
# 6. FIELD LABELS
# ============================================================================
#
# These are UI-friendly labels only.
#
# They do NOT change the underlying source field names.
# ============================================================================

FIELD_LABELS = {
    "devCategory": "Device Category",
    "characteristic": "Characteristic",
    "housing": "Housing",
    "powerSupply": "Power Supply",
    "numberOfChannels": "Number of Channels",
    "explosionApproval": "Explosion Approval",
    "protectionArea": "Protection Area",
    "certification": "Certification",
    "dataInterface": "Data Interface",
    "stromSchaltbar": "Switchable Current",
    "stromEingaenge": "Current Inputs",
    "temperaturEingaenge": "Temperature Inputs",
    "binaerDigitalOpenColl_MN": "Binary Digital Open Collector MN",
    "binaerOpenColl_MP": "Binary Open Collector MP",
    "waveInjector": "Wave Injector",
    "dev_advMeterVerification": "Advanced Meter Verification",
    "dynamicGasMaster": "Dynamic Gas Master",
    "customUserFluid": "Custom User Fluid",
    "steamApplication": "Steam Application",
}


# ============================================================================
# 7. SPECIAL VALUES
# ============================================================================

NOVALUE = "NOVALUE"


# ============================================================================
# 8. NORMALIZATION HELPERS
# ============================================================================

def normalize_scalar(value: Any) -> str | None:
    """
    Convert a raw Excel cell into a clean string.

    NaN / empty values become None.
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    if not text:
        return None

    return text


def strip_outer_quotes(value: str) -> str:
    """
    Remove one layer of matching quotes.
    """

    value = value.strip()

    if len(value) >= 2:

        if (
            value[0] == "'"
            and value[-1] == "'"
        ):
            return value[1:-1]

        if (
            value[0] == '"'
            and value[-1] == '"'
        ):
            return value[1:-1]

    return value


def normalize_rule_token(value: Any) -> str | None:
    """
    Normalize a token from a source rule expression.

    Examples:

        "'ST'"       -> "ST"
        '"ST"'       -> "ST"
        "NOVALUE"    -> "NOVALUE"
        "1"          -> "1"
        "2"          -> "2"
    """

    text = normalize_scalar(value)

    if text is None:
        return None

    text = text.strip()

    text = strip_outer_quotes(text)

    if text.upper() == "NOVALUE":
        return NOVALUE

    return text


# ============================================================================
# 9. RULE EXPRESSION PARSER
# ============================================================================

def extract_rule_values(raw_value: Any) -> list[str]:
    """
    Extract possible values from the source's rule expression.

    The workbook contains expressions such as:

        in {'ST', 'AL'}

        in {1, 2}

        'N'

        in {NOVALUE, 'x731'}

    This function extracts the values conservatively.

    It does not attempt to interpret compatibility semantics.
    """

    text = normalize_scalar(raw_value)

    if text is None:
        return []

    # ------------------------------------------------------------
    # NOVALUE
    # ------------------------------------------------------------

    if text == NOVALUE:
        return [NOVALUE]

    # ------------------------------------------------------------
    # "in {...}" expressions
    # ------------------------------------------------------------

    if text.startswith("in "):

        expression = text[3:].strip()

        try:
            parsed = ast.literal_eval(
                expression.replace(
                    "NOVALUE",
                    repr(NOVALUE),
                )
            )

            if isinstance(
                parsed,
                (set, tuple, list),
            ):
                values = []

                for item in parsed:

                    normalized = normalize_rule_token(
                        item
                    )

                    if normalized is not None:
                        values.append(normalized)

                return sorted(
                    set(values)
                )

        except Exception:
            pass

        # --------------------------------------------------------
        # Fallback parser for malformed expressions
        # --------------------------------------------------------

        inside = expression.strip()

        if (
            inside.startswith("{")
            and inside.endswith("}")
        ):
            inside = inside[1:-1]

        pieces = inside.split(",")

        values = []

        for piece in pieces:

            normalized = normalize_rule_token(
                piece
            )

            if normalized is not None:
                values.append(normalized)

        return sorted(
            set(values)
        )

    # ------------------------------------------------------------
    # Scalar expression
    # ------------------------------------------------------------

    scalar = normalize_rule_token(text)

    if scalar is None:
        return []

    return [scalar]


# ============================================================================
# 10. OBSERVED VALUES FROM A FIELD
# ============================================================================

def extract_observed_values(
    series: pd.Series,
) -> list[str]:
    """
    Extract all distinct normalized values appearing in a source column.

    For rule expressions such as:

        in {'ST', 'AL'}

    both ST and AL are returned.

    This means the catalogue represents the technical value vocabulary
    found in the source rather than simply treating each rule expression
    as a single dropdown option.
    """

    values: set[str] = set()

    for raw_value in series:

        extracted = extract_rule_values(
            raw_value
        )

        for value in extracted:
            values.add(value)

    return sorted(values)


# ============================================================================
# 11. SOURCE EXPRESSION PROFILE
# ============================================================================

def expression_counts(
    series: pd.Series,
) -> list[dict[str, Any]]:
    """
    Count distinct source rule expressions.

    This preserves the source representation and is useful for
    auditing how the technical rules are encoded.
    """

    counts = (
        series
        .dropna()
        .astype(str)
        .str.strip()
        .value_counts()
    )

    result = []

    for expression, count in counts.items():

        result.append(
            {
                "expression": expression,
                "count": int(count),
            }
        )

    return result


# ============================================================================
# 12. FIELD PROFILE
# ============================================================================

def build_field_profile(
    source: pd.DataFrame,
    field: str,
) -> dict[str, Any]:
    """
    Build a complete profile for one technical CPQ field.
    """

    series = source[field]

    observed_values = extract_observed_values(
        series
    )

    raw_expression_profile = expression_counts(
        series
    )

    non_null = int(
        series.notna().sum()
    )

    unique_raw = int(
        series
        .dropna()
        .astype(str)
        .str.strip()
        .nunique()
    )

    return {
        "field_name": field,
        "label": FIELD_LABELS.get(
            field,
            field,
        ),
        "source_column": field,
        "customer_selectable": True,
        "backend_only": False,
        "role": "technical_configuration",
        "source_rows": int(len(source)),
        "source_non_null": non_null,
        "source_null_count": int(
            len(source) - non_null
        ),
        "raw_unique_expression_count": unique_raw,
        "observed_values": observed_values,
        "observed_value_count": len(
            observed_values
        ),
        "source_rule_expressions": (
            raw_expression_profile
        ),
        "model_code_mapping_status": (
            "NOT_VERIFIED"
        ),
    }


# ============================================================================
# 13. PACKAGE VALUE EXTRACTION
# ============================================================================

def extract_package_names(
    source: pd.DataFrame,
) -> list[str]:
    """
    Extract package names from packageFixed_dev.

    The source contains expressions such as:

        in {NOVALUE, 'x731'}

        in {NOVALUE, 'G731ST-LT'}

    The returned list excludes NOVALUE.
    """

    if "packageFixed_dev" not in source.columns:
        return []

    values = extract_observed_values(
        source["packageFixed_dev"]
    )

    return sorted(
        value
        for value in values
        if value != NOVALUE
    )


# ============================================================================
# 14. PACKAGE PROFILE
# ============================================================================

def build_package_profiles(
    source: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Build package-specific technical profiles.

    IMPORTANT:
    The package is used as a backend context.

    It is not marked as a customer-facing selector.
    """

    if "packageFixed_dev" not in source.columns:
        return []

    packages = extract_package_names(
        source
    )

    profiles = []

    for package in packages:

        # --------------------------------------------------------
        # Match rows whose packageFixed_dev expression contains
        # the package name.
        # --------------------------------------------------------

        mask = (
            source["packageFixed_dev"]
            .astype(str)
            .str.contains(
                re.escape(package),
                regex=True,
                na=False,
            )
        )

        package_df = source.loc[
            mask
        ].copy()

        field_values: dict[str, list[str]] = {}

        for field in TECHNICAL_FIELDS:

            if field not in package_df.columns:
                field_values[field] = []
                continue

            field_values[field] = (
                extract_observed_values(
                    package_df[field]
                )
            )

        profiles.append(
            {
                "package": package,
                "row_count": int(
                    len(package_df)
                ),
                "technical_fields": field_values,
            }
        )

    return profiles


# ============================================================================
# 15. BACKEND FIELD CATALOGUE
# ============================================================================

def build_backend_catalogue(
    source: pd.DataFrame,
) -> dict[str, Any]:
    """
    Build metadata for backend-only fields.
    """

    backend = {}

    for field, metadata in BACKEND_FIELDS.items():

        if field not in source.columns:
            backend[field] = {
                **metadata,
                "source_column_exists": False,
                "observed_values": [],
            }
            continue

        observed = extract_observed_values(
            source[field]
        )

        backend[field] = {
            **metadata,
            "source_column_exists": True,
            "observed_values": observed,
            "observed_value_count": len(
                observed
            ),
            "model_code_mapping_status": (
                "NOT_VERIFIED"
            ),
        }

    return backend


# ============================================================================
# 16. CONSTANT METADATA CATALOGUE
# ============================================================================

def build_metadata_catalogue(
    source: pd.DataFrame,
) -> dict[str, Any]:
    """
    Build catalogue information for constant/product metadata fields.
    """

    metadata = {}

    for field, config in CONSTANT_METADATA_FIELDS.items():

        if field not in source.columns:
            metadata[field] = {
                **config,
                "source_column_exists": False,
                "observed_values": [],
            }
            continue

        observed = extract_observed_values(
            source[field]
        )

        metadata[field] = {
            **config,
            "source_column_exists": True,
            "observed_values": observed,
            "observed_value_count": len(
                observed
            ),
        }

    return metadata


# ============================================================================
# 17. LEGACY FIELD CATALOGUE
# ============================================================================

def build_legacy_catalogue(
    source: pd.DataFrame,
) -> dict[str, Any]:
    """
    Build metadata for known legacy / unused fields.
    """

    legacy = {}

    for field in sorted(
        LEGACY_FIELDS
    ):

        if field not in source.columns:
            legacy[field] = {
                "source_column_exists": False,
                "observed_values": [],
            }
            continue

        observed = extract_observed_values(
            source[field]
        )

        legacy[field] = {
            "source_column_exists": True,
            "observed_values": observed,
            "observed_value_count": len(
                observed
            ),
            "customer_selectable": False,
            "role": "legacy_or_unused",
        }

    return legacy


# ============================================================================
# 18. FLAT CSV CATALOGUE
# ============================================================================

def build_flat_catalogue(
    technical_profiles: list[dict[str, Any]],
    backend_catalogue: dict[str, Any],
    metadata_catalogue: dict[str, Any],
    legacy_catalogue: dict[str, Any],
) -> pd.DataFrame:
    """
    Build a compact flat catalogue for inspection and downstream use.
    """

    rows = []

    # ------------------------------------------------------------
    # Technical fields
    # ------------------------------------------------------------

    for profile in technical_profiles:

        rows.append(
            {
                "field_name": profile["field_name"],
                "label": profile["label"],
                "role": profile["role"],
                "customer_selectable": True,
                "backend_only": False,
                "observed_value_count": profile[
                    "observed_value_count"
                ],
                "observed_values": "|".join(
                    profile["observed_values"]
                ),
                "model_code_mapping_status": (
                    "NOT_VERIFIED"
                ),
            }
        )

    # ------------------------------------------------------------
    # Backend fields
    # ------------------------------------------------------------

    for field, config in backend_catalogue.items():

        rows.append(
            {
                "field_name": field,
                "label": config.get(
                    "label",
                    field,
                ),
                "role": config.get(
                    "role",
                    "backend",
                ),
                "customer_selectable": False,
                "backend_only": True,
                "observed_value_count": config.get(
                    "observed_value_count",
                    0,
                ),
                "observed_values": "|".join(
                    config.get(
                        "observed_values",
                        [],
                    )
                ),
                "model_code_mapping_status": (
                    "NOT_VERIFIED"
                ),
            }
        )

    # ------------------------------------------------------------
    # Constant metadata
    # ------------------------------------------------------------

    for field, config in metadata_catalogue.items():

        rows.append(
            {
                "field_name": field,
                "label": field,
                "role": config.get(
                    "role",
                    "product_metadata",
                ),
                "customer_selectable": False,
                "backend_only": True,
                "observed_value_count": config.get(
                    "observed_value_count",
                    0,
                ),
                "observed_values": "|".join(
                    config.get(
                        "observed_values",
                        [],
                    )
                ),
                "model_code_mapping_status": (
                    "NOT_APPLICABLE"
                ),
            }
        )

    # ------------------------------------------------------------
    # Legacy
    # ------------------------------------------------------------

    for field, config in legacy_catalogue.items():

        rows.append(
            {
                "field_name": field,
                "label": field,
                "role": "legacy_or_unused",
                "customer_selectable": False,
                "backend_only": True,
                "observed_value_count": config.get(
                    "observed_value_count",
                    0,
                ),
                "observed_values": "|".join(
                    config.get(
                        "observed_values",
                        [],
                    )
                ),
                "model_code_mapping_status": (
                    "NOT_APPLICABLE"
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# 19. MARKDOWN REPORT
# ============================================================================

def build_report(
    source: pd.DataFrame,
    technical_profiles: list[dict[str, Any]],
    backend_catalogue: dict[str, Any],
    metadata_catalogue: dict[str, Any],
    legacy_catalogue: dict[str, Any],
    package_profiles: list[dict[str, Any]],
) -> str:
    """
    Create a human-readable catalogue report.
    """

    lines: list[str] = []

    lines.append(
        "# 731 CPQ Catalogue"
    )

    lines.append("")

    lines.append(
        "## Source"
    )

    lines.append("")

    lines.append(
        f"- Workbook: `{SOURCE_FILE.relative_to(ROOT)}`"
    )

    lines.append(
        f"- Sheet: `{SOURCE_SHEET}`"
    )

    lines.append(
        f"- Rows: **{len(source):,}**"
    )

    lines.append(
        f"- Columns: **{len(source.columns)}**"
    )

    lines.append("")

    # ------------------------------------------------------------
    # Technical fields
    # ------------------------------------------------------------

    lines.append(
        "## 1. Customer-facing technical fields"
    )

    lines.append("")

    lines.append(
        "These fields form the current technical CPQ catalogue."
    )

    lines.append("")

    lines.append(
        "| Field | Label | Values | Count |"
    )

    lines.append(
        "|---|---|---|---:|"
    )

    for profile in technical_profiles:

        values = ", ".join(
            profile["observed_values"]
        )

        lines.append(
            f"| `{profile['field_name']}` | "
            f"{profile['label']} | "
            f"{values} | "
            f"{profile['observed_value_count']} |"
        )

    lines.append("")

    # ------------------------------------------------------------
    # Backend fields
    # ------------------------------------------------------------

    lines.append(
        "## 2. Backend CPQ context"
    )

    lines.append("")

    for field, config in backend_catalogue.items():

        lines.append(
            f"### `{field}`"
        )

        lines.append("")

        lines.append(
            f"- Role: **{config.get('role', 'backend')}**"
        )

        lines.append(
            "- Customer selectable: **No**"
        )

        values = config.get(
            "observed_values",
            [],
        )

        lines.append(
            f"- Observed values: "
            f"{', '.join(values)}"
        )

        lines.append("")

    # ------------------------------------------------------------
    # Package profiles
    # ------------------------------------------------------------

    lines.append(
        "## 3. Package contexts"
    )

    lines.append("")

    lines.append(
        "Package contexts are backend configuration contexts. "
        "They are not automatically customer-facing selectors."
    )

    lines.append("")

    lines.append(
        "| Package | Source rows |"
    )

    lines.append(
        "|---|---:|"
    )

    for profile in package_profiles:

        lines.append(
            f"| `{profile['package']}` | "
            f"{profile['row_count']:,} |"
        )

    lines.append("")

    # ------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------

    lines.append(
        "## 4. Product/source metadata"
    )

    lines.append("")

    for field, config in metadata_catalogue.items():

        lines.append(
            f"- `{field}` → "
            f"{config.get('observed_values', [])}"
        )

    lines.append("")

    # ------------------------------------------------------------
    # Legacy
    # ------------------------------------------------------------

    lines.append(
        "## 5. Legacy / unused fields"
    )

    lines.append("")

    for field, config in legacy_catalogue.items():

        lines.append(
            f"- `{field}` → "
            f"{config.get('observed_values', [])}"
        )

    lines.append("")

    # ------------------------------------------------------------
    # Model code
    # ------------------------------------------------------------

    lines.append(
        "## 6. Model-code status"
    )

    lines.append("")

    lines.append(
        "**NOT VERIFIED**"
    )

    lines.append("")

    lines.append(
        "This catalogue deliberately does not infer the meaning "
        "of Emerson model-code positions."
    )

    lines.append("")

    lines.append(
        "The internal CFG731_* canonical configuration identifier "
        "must not be treated as an Emerson commercial/model code."
    )

    lines.append("")

    # ------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------

    lines.append(
        "## 7. Configuration architecture"
    )

    lines.append("")

    lines.append(
        "```text"
    )

    lines.append(
        "Customer RFQ"
    )

    lines.append(
        "    ↓"
    )

    lines.append(
        "Structured technical requirements"
    )

    lines.append(
        "    ↓"
    )

    lines.append(
        "731 CPQ Catalogue"
    )

    lines.append(
        "    ↓"
    )

    lines.append(
        "Package/context resolution"
    )

    lines.append(
        "    ↓"
    )

    lines.append(
        "Deterministic constraint validation"
    )

    lines.append(
        "    ↓"
    )

    lines.append(
        "Valid configurations"
    )

    lines.append(
        "    ↓"
    )

    lines.append(
        "ML ranking"
    )

    lines.append(
        "    ↓"
    )

    lines.append(
        "Recommended valid configuration"
    )

    lines.append(
        "    ↓"
    )

    lines.append(
        "Verified model code"
    )

    lines.append(
        "```"
    )

    lines.append("")

    return "\n".join(
        lines
    )


# ============================================================================
# 20. BUILD COMPLETE CATALOGUE
# ============================================================================

def build_catalogue(
    source: pd.DataFrame,
) -> dict[str, Any]:
    """
    Build the complete structured CPQ catalogue.
    """

    # ------------------------------------------------------------
    # Technical profiles
    # ------------------------------------------------------------

    technical_profiles = []

    for field in TECHNICAL_FIELDS:

        if field not in source.columns:

            raise KeyError(
                f"Required technical field is missing "
                f"from source workbook: {field}"
            )

        technical_profiles.append(
            build_field_profile(
                source,
                field,
            )
        )

    # ------------------------------------------------------------
    # Backend catalogue
    # ------------------------------------------------------------

    backend_catalogue = build_backend_catalogue(
        source
    )

    # ------------------------------------------------------------
    # Metadata catalogue
    # ------------------------------------------------------------

    metadata_catalogue = build_metadata_catalogue(
        source
    )

    # ------------------------------------------------------------
    # Legacy catalogue
    # ------------------------------------------------------------

    legacy_catalogue = build_legacy_catalogue(
        source
    )

    # ------------------------------------------------------------
    # Package profiles
    # ------------------------------------------------------------

    package_profiles = build_package_profiles(
        source
    )

    # ------------------------------------------------------------
    # Complete object
    # ------------------------------------------------------------

    catalogue = {
        "catalogue_version": "731_CPQ_CATALOGUE_V1",

        "source": {
            "file": str(
                SOURCE_FILE.relative_to(ROOT)
            ),
            "sheet": SOURCE_SHEET,
            "rows": int(
                source.shape[0]
            ),
            "columns": int(
                source.shape[1]
            ),
        },

        "governance": {
            "technical_source_of_truth": (
                "data/raw/x731_allgemein_Rev24.xlsx"
            ),
            "deterministic_validation_required": True,
            "ml_may_rank_only_valid_candidates": True,
            "model_code_mapping_status": (
                "NOT_VERIFIED"
            ),
            "canonical_cfg_ids_are_model_codes": (
                False
            ),
        },

        "technical_fields": technical_profiles,

        "backend_fields": backend_catalogue,

        "constant_metadata_fields": metadata_catalogue,

        "legacy_fields": legacy_catalogue,

        "package_contexts": package_profiles,
    }

    return catalogue


# ============================================================================
# 21. SAVE CATALOGUE
# ============================================================================

def save_catalogue(
    catalogue: dict[str, Any],
) -> None:
    """
    Save JSON, CSV and Markdown catalogue outputs.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------

    with open(
        CATALOGUE_JSON,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            catalogue,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------

    flat_df = build_flat_catalogue(
        technical_profiles=catalogue[
            "technical_fields"
        ],
        backend_catalogue=catalogue[
            "backend_fields"
        ],
        metadata_catalogue=catalogue[
            "constant_metadata_fields"
        ],
        legacy_catalogue=catalogue[
            "legacy_fields"
        ],
    )

    flat_df.to_csv(
        CATALOGUE_CSV,
        index=False,
    )

    # ------------------------------------------------------------
    # Package CSV
    # ------------------------------------------------------------

    package_rows = []

    for profile in catalogue[
        "package_contexts"
    ]:

        package = profile[
            "package"
        ]

        row_count = profile[
            "row_count"
        ]

        for field, values in profile[
            "technical_fields"
        ].items():

            package_rows.append(
                {
                    "package": package,
                    "source_row_count": row_count,
                    "field_name": field,
                    "observed_values": "|".join(
                        values
                    ),
                    "observed_value_count": len(
                        values
                    ),
                }
            )

    package_df = pd.DataFrame(
        package_rows
    )

    package_df.to_csv(
        PACKAGE_PROFILE_CSV,
        index=False,
    )

    # ------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------

    report = build_report(
        source=pd.read_excel(
            SOURCE_FILE,
            sheet_name=SOURCE_SHEET,
        ),
        technical_profiles=catalogue[
            "technical_fields"
        ],
        backend_catalogue=catalogue[
            "backend_fields"
        ],
        metadata_catalogue=catalogue[
            "constant_metadata_fields"
        ],
        legacy_catalogue=catalogue[
            "legacy_fields"
        ],
        package_profiles=catalogue[
            "package_contexts"
        ],
    )

    CATALOGUE_REPORT.write_text(
        report,
        encoding="utf-8",
    )


# ============================================================================
# 22. VALIDATE CATALOGUE
# ============================================================================

def validate_catalogue(
    catalogue: dict[str, Any],
) -> list[str]:
    """
    Perform internal consistency checks.

    Returns a list of validation errors.
    """

    errors: list[str] = []

    # ------------------------------------------------------------
    # Technical field count
    # ------------------------------------------------------------

    technical_fields = catalogue[
        "technical_fields"
    ]

    if len(technical_fields) != 19:

        errors.append(
            "Expected exactly 19 technical fields."
        )

    # ------------------------------------------------------------
    # Technical field names
    # ------------------------------------------------------------

    actual_names = [
        profile["field_name"]
        for profile in technical_fields
    ]

    if actual_names != TECHNICAL_FIELDS:

        errors.append(
            "Technical field ordering/names do not match "
            "TECHNICAL_FIELDS."
        )

    # ------------------------------------------------------------
    # Customer selectable
    # ------------------------------------------------------------

    for profile in technical_fields:

        if not profile[
            "customer_selectable"
        ]:

            errors.append(
                f"Technical field incorrectly marked "
                f"non-selectable: {profile['field_name']}"
            )

    # ------------------------------------------------------------
    # Backend fields
    # ------------------------------------------------------------

    for field, config in catalogue[
        "backend_fields"
    ].items():

        if config.get(
            "customer_selectable",
            True,
        ):

            errors.append(
                f"Backend field incorrectly marked "
                f"customer selectable: {field}"
            )

    # ------------------------------------------------------------
    # Model code
    # ------------------------------------------------------------

    if catalogue[
        "governance"
    ][
        "model_code_mapping_status"
    ] != "NOT_VERIFIED":

        errors.append(
            "Model-code status must remain NOT_VERIFIED."
        )

    if catalogue[
        "governance"
    ][
        "canonical_cfg_ids_are_model_codes"
    ]:

        errors.append(
            "Canonical CFG731 IDs must not be treated as "
            "model codes."
        )

    return errors


# ============================================================================
# 23. MAIN
# ============================================================================

def main() -> None:

    print()
    print("=" * 80)
    print("731 CPQ CATALOGUE BUILDER")
    print("=" * 80)
    print()

    # ------------------------------------------------------------
    # Check source
    # ------------------------------------------------------------

    if not SOURCE_FILE.exists():

        raise FileNotFoundError(
            "731 source workbook was not found:\n"
            f"{SOURCE_FILE}"
        )

    print(
        f"Source workbook:\n"
        f"  {SOURCE_FILE}"
    )

    print()

    # ------------------------------------------------------------
    # Load source
    # ------------------------------------------------------------

    print(
        "Loading workbook..."
    )

    source = pd.read_excel(
        SOURCE_FILE,
        sheet_name=SOURCE_SHEET,
    )

    print(
        f"Loaded: "
        f"{source.shape[0]:,} rows x "
        f"{source.shape[1]} columns"
    )

    print()

    # ------------------------------------------------------------
    # Verify technical fields exist
    # ------------------------------------------------------------

    missing_fields = [
        field
        for field in TECHNICAL_FIELDS
        if field not in source.columns
    ]

    if missing_fields:

        print(
            "ERROR: Missing technical fields:"
        )

        for field in missing_fields:
            print(
                f"  - {field}"
            )

        raise KeyError(
            "Required technical fields are missing."
        )

    # ------------------------------------------------------------
    # Build catalogue
    # ------------------------------------------------------------

    print(
        "Building technical field profiles..."
    )

    catalogue = build_catalogue(
        source
    )

    # ------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------

    print(
        "Validating catalogue..."
    )

    errors = validate_catalogue(
        catalogue
    )

    if errors:

        print()
        print(
            "CATALOGUE VALIDATION FAILED"
        )
        print()

        for error in errors:
            print(
                f"  ERROR: {error}"
            )

        raise RuntimeError(
            "Catalogue validation failed."
        )

    print(
        "Catalogue validation: PASS"
    )

    print()

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    print(
        "Saving catalogue..."
    )

    save_catalogue(
        catalogue
    )

    # ------------------------------------------------------------
    # Terminal summary
    # ------------------------------------------------------------

    print()
    print("=" * 80)
    print("731 CPQ CATALOGUE COMPLETE")
    print("=" * 80)
    print()

    print(
        "SOURCE"
    )

    print(
        f"  Rows:    {source.shape[0]:,}"
    )

    print(
        f"  Columns: {source.shape[1]}"
    )

    print()

    print(
        "TECHNICAL CPQ FIELDS"
    )

    print(
        f"  Count: {len(catalogue['technical_fields'])}"
    )

    print()

    for profile in catalogue[
        "technical_fields"
    ]:

        print(
            f"  {profile['field_name']:<35}"
            f" {profile['observed_value_count']:>3} values"
        )

    print()

    print(
        "BACKEND FIELDS"
    )

    for field in catalogue[
        "backend_fields"
    ]:

        values = catalogue[
            "backend_fields"
        ][field].get(
            "observed_values",
            [],
        )

        print(
            f"  {field:<35}"
            f" {len(values):>3} observed values"
        )

    print()

    print(
        "PACKAGE CONTEXTS"
    )

    for profile in catalogue[
        "package_contexts"
    ]:

        print(
            f"  {profile['package']:<35}"
            f" {profile['row_count']:>5} rows"
        )

    print()

    print(
        "MODEL CODE"
    )

    print(
        "  Status: NOT_VERIFIED"
    )

    print(
        "  No model-code meanings were inferred."
    )

    print()

    print(
        "OUTPUT FILES"
    )

    print(
        f"  {CATALOGUE_JSON}"
    )

    print(
        f"  {CATALOGUE_CSV}"
    )

    print(
        f"  {PACKAGE_PROFILE_CSV}"
    )

    print(
        f"  {CATALOGUE_REPORT}"
    )

    print()

    print(
        "NEXT STEP"
    )

    print(
        "  Use the generated catalogue as the backend source"
    )

    print(
        "  for the CPQ/Streamlit technical configuration UI."
    )

    print()


# ============================================================================
# 24. SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()