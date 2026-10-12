#!/usr/bin/env python3
"""
731 CPQ CONFIGURATION ENGINE
============================

Deterministic configuration resolver for the Emerson 731 technical
configuration catalogue.

Architecture
------------

    Customer/RFQ requirements
              |
              v
    731 CPQ Catalogue
              |
              v
    Deterministic configuration engine
              |
              v
    Technically valid candidates
              |
              v
    ML ranking
              |
              v
    Recommended valid configuration


IMPORTANT
---------

This module is responsible for TECHNICAL VALIDITY only.

It does NOT:

- use ML to decide technical validity
- generate Emerson commercial/model codes
- infer model-code position meanings
- treat CFG731_* as a commercial model code
- replace the existing canonical_configuration.py
- invent compatibility rules

The source workbook remains the technical rule source.

The engine interprets source expressions such as:

    'ST'

    in {'ST', 'AL'}

    in {1, 2}

    in {NOVALUE, 'FWI'}

A requested value is compatible with a source row when the value
is explicitly allowed by that source row.

Unspecified technical fields are not used to reject a partial RFQ.

Package context is handled as BACKEND context through packageFixed_dev.
It is not intended to be a customer-facing selector.


Inputs
------

data/raw/x731_allgemein_Rev24.xlsx

data/processed/731_cpq_catalogue.json


Outputs
-------

The engine can return:

- valid / invalid result
- matched source rows
- representative technical configurations
- package context
- matched fields
- candidate DataFrames

A test report is written to:

data/processed/731_cpq_configuration_engine_test_report.md
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

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

CATALOGUE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "731_cpq_catalogue.json"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
)

TEST_REPORT = (
    OUTPUT_DIR
    / "731_cpq_configuration_engine_test_report.md"
)


# ============================================================================
# 2. TECHNICAL CPQ FIELDS
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
# 3. BACKEND FIELDS
# ============================================================================

BACKEND_FIELDS = [
    "packageFixed_dev",
    "OrgUnitAreaOfDocument",
]


# ============================================================================
# 4. SPECIAL VALUE
# ============================================================================

NOVALUE = "NOVALUE"


# ============================================================================
# 5. BASIC NORMALIZATION
# ============================================================================

def normalize_scalar(value: Any) -> str | None:
    """
    Convert a source/user value into a clean string.

    None / NaN / empty strings become None.
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


def normalize_value(value: Any) -> str | None:
    """
    Normalize a technical value for deterministic comparison.

    Examples
    --------
    "'ST'" -> "ST"
    '"ST"' -> "ST"
    1     -> "1"
    "1"   -> "1"
    None  -> None
    """

    text = normalize_scalar(value)

    if text is None:
        return None

    if len(text) >= 2:

        if text.startswith("'") and text.endswith("'"):
            text = text[1:-1]

        elif text.startswith('"') and text.endswith('"'):
            text = text[1:-1]

    if text.upper() == NOVALUE:
        return NOVALUE

    return text


def normalize_configuration(
    configuration: dict[str, Any],
) -> dict[str, str]:
    """
    Normalize a partial customer/RFQ configuration.

    None values are removed.

    Example
    -------

    {
        "housing": "ST",
        "powerSupply": 4,
        "numberOfChannels": 1
    }

    becomes

    {
        "housing": "ST",
        "powerSupply": "4",
        "numberOfChannels": "1"
    }
    """

    normalized: dict[str, str] = {}

    for field_name, value in configuration.items():

        normalized_value = normalize_value(
            value
        )

        if normalized_value is not None:
            normalized[field_name] = normalized_value

    return normalized


# ============================================================================
# 6. SOURCE RULE PARSER
# ============================================================================

def parse_rule_expression(
    expression: Any,
) -> set[str]:
    """
    Parse one source rule expression into explicitly allowed values.

    Supported examples:

        'ST'

        in {'ST', 'AL'}

        in {1, 2}

        NOVALUE

        in {NOVALUE, 'FWI'}

    The parser does not infer additional compatibility.
    """

    text = normalize_scalar(
        expression
    )

    if text is None:
        return set()

    # ------------------------------------------------------------
    # Direct NOVALUE
    # ------------------------------------------------------------

    if text == NOVALUE:
        return {NOVALUE}

    # ------------------------------------------------------------
    # Set/list expression
    # ------------------------------------------------------------

    if text.startswith("in "):

        expression_body = text[3:].strip()

        # Python literal_eval cannot understand bare NOVALUE.
        # Replace it with a quoted string before evaluation.
        expression_for_eval = expression_body.replace(
            NOVALUE,
            repr(NOVALUE),
        )

        try:

            parsed = ast.literal_eval(
                expression_for_eval
            )

            if isinstance(
                parsed,
                (set, list, tuple),
            ):

                values: set[str] = set()

                for item in parsed:

                    normalized = normalize_value(
                        item
                    )

                    if normalized is not None:
                        values.add(normalized)

                return values

        except (
            SyntaxError,
            ValueError,
            TypeError,
        ):
            pass

        # --------------------------------------------------------
        # Conservative fallback parser
        # --------------------------------------------------------

        if (
            expression_body.startswith("{")
            and expression_body.endswith("}")
        ):
            expression_body = expression_body[1:-1]

        values = set()

        for token in expression_body.split(","):

            normalized = normalize_value(
                token
            )

            if normalized is not None:
                values.add(normalized)

        return values

    # ------------------------------------------------------------
    # Scalar value
    # ------------------------------------------------------------

    normalized = normalize_value(
        text
    )

    if normalized is None:
        return set()

    return {
        normalized
    }


# ============================================================================
# 7. DATA CLASSES
# ============================================================================

@dataclass
class Candidate:
    """
    One technically compatible source configuration.

    The source row remains available for provenance.

    The engine deliberately does not create a new canonical ID here.
    The existing project's canonical_configuration.py remains responsible
    for canonical configuration identity.
    """

    source_row: int

    configuration: dict[str, Any]

    package_context: str | None

    matched_fields: list[str] = field(
        default_factory=list
    )

    source_package_expression: str | None = None

    @property
    def technical_fields(
        self,
    ) -> dict[str, Any]:

        return {
            field_name: self.configuration.get(
                field_name
            )
            for field_name in TECHNICAL_FIELDS
        }


@dataclass
class ResolutionResult:
    """
    Result of deterministic CPQ resolution.
    """

    valid: bool

    requested_configuration: dict[str, Any]

    normalized_configuration: dict[str, str]

    package_context: str | None

    candidate_count: int

    candidates: list[Candidate]

    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:

        return {
            "valid": self.valid,

            "requested_configuration": (
                self.requested_configuration
            ),

            "normalized_configuration": (
                self.normalized_configuration
            ),

            "package_context": (
                self.package_context
            ),

            "candidate_count": (
                self.candidate_count
            ),

            "reason": self.reason,

            "candidates": [
                {
                    "source_row": (
                        candidate.source_row
                    ),

                    "package_context": (
                        candidate.package_context
                    ),

                    "matched_fields": (
                        candidate.matched_fields
                    ),

                    "source_package_expression": (
                        candidate.source_package_expression
                    ),

                    "configuration": (
                        candidate.configuration
                    ),
                }

                for candidate in self.candidates
            ],
        }


# ============================================================================
# 8. CPQ ENGINE
# ============================================================================

class CPQ731ConfigurationEngine:
    """
    Deterministic 731 configuration resolver.

    The engine evaluates the actual source workbook rows.

    For a partial technical configuration:

        supplied fields
              |
              v
        source rule rows
              |
              v
        rows supporting every supplied value
              |
              v
        valid candidates

    Package context is applied through packageFixed_dev.
    """

    def __init__(
        self,
        source_file: Path | str = SOURCE_FILE,
        source_sheet: str = SOURCE_SHEET,
        catalogue_file: Path | str = CATALOGUE_FILE,
    ) -> None:

        self.source_file = Path(
            source_file
        )

        self.source_sheet = source_sheet

        self.catalogue_file = Path(
            catalogue_file
        )

        # --------------------------------------------------------
        # Load source
        # --------------------------------------------------------

        self.source = pd.read_excel(
            self.source_file,
            sheet_name=self.source_sheet,
        )

        # --------------------------------------------------------
        # Load catalogue
        # --------------------------------------------------------

        self.catalogue = (
            self._load_catalogue()
        )

        # --------------------------------------------------------
        # Validate
        # --------------------------------------------------------

        self._validate_source()

        # --------------------------------------------------------
        # Preparse all rules once
        # --------------------------------------------------------

        self._parsed_rules = (
            self._preparse_rules()
        )

        # --------------------------------------------------------
        # Discover package contexts
        # --------------------------------------------------------

        self.package_contexts = (
            self._discover_packages()
        )

    # ========================================================================
    # CATALOGUE
    # ========================================================================

    def _load_catalogue(
        self,
    ) -> dict[str, Any]:

        if not self.catalogue_file.exists():

            raise FileNotFoundError(
                "731 CPQ catalogue not found:\n"
                f"{self.catalogue_file}"
            )

        with open(
            self.catalogue_file,
            "r",
            encoding="utf-8",
        ) as file:

            return json_load(
                file
            )

    # ========================================================================
    # SOURCE VALIDATION
    # ========================================================================

    def _validate_source(
        self,
    ) -> None:

        missing_technical = [
            field_name
            for field_name in TECHNICAL_FIELDS
            if field_name not in self.source.columns
        ]

        if missing_technical:

            raise KeyError(
                "Missing technical fields:\n"
                + "\n".join(
                    f"  - {field}"
                    for field in missing_technical
                )
            )

        missing_backend = [
            field_name
            for field_name in BACKEND_FIELDS
            if field_name not in self.source.columns
        ]

        if missing_backend:

            raise KeyError(
                "Missing backend fields:\n"
                + "\n".join(
                    f"  - {field}"
                    for field in missing_backend
                )
            )

    # ========================================================================
    # PREPARSE SOURCE RULES
    # ========================================================================

    def _preparse_rules(
        self,
    ) -> dict[int, dict[str, set[str]]]:
        """
        Parse all technical/package expressions once.

        This makes repeated UI resolution much faster than reparsing the
        workbook for every request.
        """

        parsed = {}

        for source_index, row in self.source.iterrows():

            field_rules = {}

            # ----------------------------------------------------
            # Technical fields
            # ----------------------------------------------------

            for field_name in TECHNICAL_FIELDS:

                field_rules[field_name] = (
                    parse_rule_expression(
                        row[field_name]
                    )
                )

            # ----------------------------------------------------
            # Package context
            # ----------------------------------------------------

            field_rules["packageFixed_dev"] = (
                parse_rule_expression(
                    row["packageFixed_dev"]
                )
            )

            # ----------------------------------------------------
            # Document metadata
            # ----------------------------------------------------

            field_rules["OrgUnitAreaOfDocument"] = (
                parse_rule_expression(
                    row["OrgUnitAreaOfDocument"]
                )
            )

            parsed[
                int(source_index)
            ] = field_rules

        return parsed

    # ========================================================================
    # PACKAGE DISCOVERY
    # ========================================================================

    def _discover_packages(
        self,
    ) -> list[str]:

        packages = set()

        for source_index in self._parsed_rules:

            allowed = (
                self._parsed_rules[
                    source_index
                ][
                    "packageFixed_dev"
                ]
            )

            for value in allowed:

                if value != NOVALUE:
                    packages.add(value)

        return sorted(
            packages
        )

    # ========================================================================
    # PACKAGE MATCHING
    # ========================================================================

    def _row_supports_package(
        self,
        source_index: int,
        package_context: str,
    ) -> bool:

        requested_package = normalize_value(
            package_context
        )

        if requested_package is None:
            return True

        allowed_packages = (
            self._parsed_rules[
                source_index
            ][
                "packageFixed_dev"
            ]
        )

        return (
            requested_package
            in allowed_packages
        )

    # ========================================================================
    # TECHNICAL FIELD MATCHING
    # ========================================================================

    def _row_supports_configuration(
        self,
        source_index: int,
        configuration: dict[str, str],
    ) -> tuple[
        bool,
        list[str],
    ]:
        """
        Check whether one source row supports every supplied field.

        Every supplied value must be explicitly contained in the
        source row's allowed values.
        """

        rules = self._parsed_rules[
            source_index
        ]

        matched_fields = []

        for field_name, requested_value in (
            configuration.items()
        ):

            if field_name not in TECHNICAL_FIELDS:

                raise ValueError(
                    f"Unknown technical field: "
                    f"{field_name}"
                )

            allowed_values = rules.get(
                field_name,
                set(),
            )

            if requested_value not in allowed_values:

                return (
                    False,
                    [],
                )

            matched_fields.append(
                field_name
            )

        return (
            True,
            matched_fields,
        )

    # ========================================================================
    # REPRESENTATIVE SOURCE VALUE
    # ========================================================================

    @staticmethod
    def representative_value(
        expression: Any,
    ) -> str | None:
        """
        Return one deterministic representative value from a source rule.

        Preference:
            1. actual technical value
            2. NOVALUE
            3. None

        This is for candidate display/materialization only.

        It does NOT alter the underlying source rule.
        """

        values = parse_rule_expression(
            expression
        )

        if not values:
            return None

        non_novalue = sorted(
            value
            for value in values
            if value != NOVALUE
        )

        if non_novalue:
            return non_novalue[0]

        return NOVALUE

    # ========================================================================
    # SOURCE ROW -> CANDIDATE
    # ========================================================================

    def _candidate_from_row(
        self,
        source_index: int,
        package_context: str | None,
        matched_fields: list[str],
    ) -> Candidate:

        row = self.source.loc[
            source_index
        ]

        configuration = {}

        for field_name in TECHNICAL_FIELDS:

            configuration[field_name] = (
                self.representative_value(
                    row[field_name]
                )
            )

        source_package_expression = (
            normalize_scalar(
                row["packageFixed_dev"]
            )
        )

        return Candidate(
            source_row=int(
                source_index
            ) + 2,

            configuration=configuration,

            package_context=package_context,

            matched_fields=matched_fields,

            source_package_expression=(
                source_package_expression
            ),
        )

    # ========================================================================
    # CANDIDATE DEDUPLICATION
    # ========================================================================

    @staticmethod
    def _configuration_signature(
        candidate: Candidate,
    ) -> tuple:

        return tuple(
            normalize_value(
                candidate.configuration.get(
                    field_name
                )
            )
            for field_name in TECHNICAL_FIELDS
        )

    def _deduplicate_candidates(
        self,
        candidates: Iterable[Candidate],
    ) -> list[Candidate]:

        unique = {}

        for candidate in candidates:

            signature = (
                self._configuration_signature(
                    candidate
                )
            )

            existing = unique.get(
                signature
            )

            if existing is None:

                unique[
                    signature
                ] = candidate

                continue

            # Prefer the candidate that matched more explicit
            # customer fields.
            if len(
                candidate.matched_fields
            ) > len(
                existing.matched_fields
            ):

                unique[
                    signature
                ] = candidate

        return sorted(
            unique.values(),
            key=lambda candidate: (
                candidate.package_context or "",
                candidate.source_row,
            ),
        )

    # ========================================================================
    # MAIN RESOLUTION METHOD
    # ========================================================================

    def resolve(
        self,
        configuration: dict[str, Any],
        package_context: str | None = None,
        limit: int | None = None,
    ) -> ResolutionResult:
        """
        Resolve a partial technical configuration.

        Example:

            result = engine.resolve(
                {
                    "devCategory": "G",
                    "characteristic": "ST",
                    "housing": "ST",
                }
            )

        Package context:

            result = engine.resolve(
                {
                    "characteristic": "ST",
                },
                package_context="G731ST-LT",
            )

        The method returns only configurations supported by the source
        rule rows.
        """

        normalized = (
            normalize_configuration(
                configuration
            )
        )

        # --------------------------------------------------------
        # Check fields
        # --------------------------------------------------------

        unknown_fields = [
            field_name
            for field_name in normalized
            if field_name not in TECHNICAL_FIELDS
        ]

        if unknown_fields:

            raise ValueError(
                "Unknown technical field(s): "
                + ", ".join(
                    unknown_fields
                )
            )

        normalized_package = normalize_value(
            package_context
        )

        # --------------------------------------------------------
        # Validate package if explicitly supplied
        # --------------------------------------------------------

        if normalized_package is not None:

            if normalized_package not in (
                self.package_contexts
            ):

                return ResolutionResult(
                    valid=False,

                    requested_configuration=(
                        configuration
                    ),

                    normalized_configuration=(
                        normalized
                    ),

                    package_context=(
                        normalized_package
                    ),

                    candidate_count=0,

                    candidates=[],

                    reason=(
                        "Unknown package context: "
                        f"{normalized_package}"
                    ),
                )

        # --------------------------------------------------------
        # Search source rules
        # --------------------------------------------------------

        candidates = []

        for source_index in (
            self._parsed_rules
        ):

            # ----------------------------------------------------
            # Package filtering
            # ----------------------------------------------------

            if normalized_package is not None:

                if not self._row_supports_package(
                    source_index,
                    normalized_package,
                ):

                    continue

            # ----------------------------------------------------
            # Technical filtering
            # ----------------------------------------------------

            matches, matched_fields = (
                self._row_supports_configuration(
                    source_index,
                    normalized,
                )
            )

            if not matches:
                continue

            candidate = (
                self._candidate_from_row(
                    source_index=source_index,
                    package_context=normalized_package,
                    matched_fields=matched_fields,
                )
            )

            candidates.append(
                candidate
            )

        # --------------------------------------------------------
        # Deduplicate
        # --------------------------------------------------------

        candidates = (
            self._deduplicate_candidates(
                candidates
            )
        )

        # --------------------------------------------------------
        # Limit
        # --------------------------------------------------------

        if limit is not None:

            if limit < 1:
                raise ValueError(
                    "limit must be >= 1"
                )

            candidates = candidates[
                :limit
            ]

        # --------------------------------------------------------
        # Valid result
        # --------------------------------------------------------

        if candidates:

            return ResolutionResult(
                valid=True,

                requested_configuration=(
                    configuration
                ),

                normalized_configuration=(
                    normalized
                ),

                package_context=(
                    normalized_package
                ),

                candidate_count=len(
                    candidates
                ),

                candidates=candidates,

                reason=None,
            )

        # --------------------------------------------------------
        # Invalid result
        # --------------------------------------------------------

        reason = (
            "No source configuration row satisfies "
            "all supplied technical requirements."
        )

        if normalized_package is not None:

            reason += (
                " Package context: "
                f"{normalized_package}."
            )

        return ResolutionResult(
            valid=False,

            requested_configuration=(
                configuration
            ),

            normalized_configuration=(
                normalized
            ),

            package_context=(
                normalized_package
            ),

            candidate_count=0,

            candidates=[],

            reason=reason,
        )

    # ========================================================================
    # SIMPLE VALIDATION API
    # ========================================================================

    def validate(
        self,
        configuration: dict[str, Any],
        package_context: str | None = None,
    ) -> bool:
        """
        Return True only if at least one source rule row supports
        the supplied technical requirements.
        """

        result = self.resolve(
            configuration=configuration,
            package_context=package_context,
            limit=1,
        )

        return result.valid

    # ========================================================================
    # CANDIDATE DATAFRAME
    # ========================================================================

    def candidates_dataframe(
        self,
        result: ResolutionResult,
    ) -> pd.DataFrame:
        """
        Convert resolution candidates to a DataFrame.

        No model code is generated.
        """

        rows = []

        for candidate in (
            result.candidates
        ):

            row = {
                "source_row": (
                    candidate.source_row
                ),

                "package_context": (
                    candidate.package_context
                ),

                "matched_field_count": len(
                    candidate.matched_fields
                ),

                "matched_fields": "|".join(
                    candidate.matched_fields
                ),

                "source_package_expression": (
                    candidate.source_package_expression
                ),
            }

            row.update(
                candidate.technical_fields
            )

            rows.append(
                row
            )

        return pd.DataFrame(
            rows
        )

    # ========================================================================
    # CATALOGUE API
    # ========================================================================

    def list_technical_fields(
        self,
    ) -> list[str]:

        return list(
            TECHNICAL_FIELDS
        )

    def list_package_contexts(
        self,
    ) -> list[str]:

        return list(
            self.package_contexts
        )

    def get_field_definition(
        self,
        field_name: str,
    ) -> dict[str, Any] | None:

        for definition in (
            self.catalogue.get(
                "technical_fields",
                [],
            )
        ):

            if definition.get(
                "field_name"
            ) == field_name:

                return definition

        return None

    # ========================================================================
    # UI VALUES
    # ========================================================================

    def get_ui_values(
        self,
        field_name: str,
    ) -> list[str]:

        definition = (
            self.get_field_definition(
                field_name
            )
        )

        if definition is None:

            raise KeyError(
                f"Unknown CPQ technical field: "
                f"{field_name}"
            )

        values = definition.get(
            "observed_values",
            [],
        )

        # NOVALUE is an internal source representation.
        # It should not normally be shown as "NOVALUE" in the UI.
        return [
            value
            for value in values
            if value != NOVALUE
        ]


# ============================================================================
# JSON LOADER
# ============================================================================

def json_load(
    file_handle,
) -> dict[str, Any]:
    """
    Small local JSON loader so the engine has no dependency on another
    project module.
    """

    import json

    return json.load(
        file_handle
    )


# ============================================================================
# BASIC TESTS
# ============================================================================

def run_basic_tests(
    engine: CPQ731ConfigurationEngine,
) -> list[str]:
    """
    Run basic deterministic integrity tests.

    These are intentionally simple.

    They verify:

    - the engine can resolve source rows
    - a known compatible configuration is accepted
    - an impossible value is rejected
    - expected package contexts are present
    """

    failures = []

    # ------------------------------------------------------------------------
    # TEST 1
    # Empty requirements
    # ------------------------------------------------------------------------

    result = engine.resolve({})

    if not result.valid:

        failures.append(
            "Empty technical configuration should return candidates."
        )

    if result.candidate_count <= 0:

        failures.append(
            "Empty technical configuration returned zero candidates."
        )

    # ------------------------------------------------------------------------
    # TEST 2
    # Known source-compatible x731 configuration
    # ------------------------------------------------------------------------

    result = engine.resolve(
        {
            "devCategory": "F",
            "characteristic": "GP",
            "housing": "ST",
            "powerSupply": "1",
            "numberOfChannels": "1",
            "explosionApproval": "N",
            "protectionArea": "N",
        },
        package_context="x731",
        limit=5,
    )

    if not result.valid:

        failures.append(
            "Known x731-compatible configuration was rejected."
        )

    # ------------------------------------------------------------------------
    # TEST 3
    # Impossible value
    # ------------------------------------------------------------------------

    result = engine.resolve(
        {
            "housing": "THIS_VALUE_DOES_NOT_EXIST",
        }
    )

    if result.valid:

        failures.append(
            "Invalid housing value was incorrectly accepted."
        )

    # ------------------------------------------------------------------------
    # TEST 4
    # Expected package contexts
    # ------------------------------------------------------------------------

    expected_packages = {
        "x731",
        "G731ST-LT",
        "G731ST-HT",
        "F731TE",
        "G731CA",
        "F731PW",
        "F731WD_DualChannel",
        "F731WD_SingleChannel",
        "G731VG",
    }

    actual_packages = set(
        engine.list_package_contexts()
    )

    missing_packages = (
        expected_packages
        - actual_packages
    )

    if missing_packages:

        failures.append(
            "Missing expected package contexts: "
            + ", ".join(
                sorted(
                    missing_packages
                )
            )
        )

    # ------------------------------------------------------------------------
    # TEST 5
    # Catalogue contains all 19 fields
    # ------------------------------------------------------------------------

    catalogue_fields = set(
        engine.list_technical_fields()
    )

    expected_fields = set(
        TECHNICAL_FIELDS
    )

    missing_fields = (
        expected_fields
        - catalogue_fields
    )

    if missing_fields:

        failures.append(
            "Missing catalogue technical fields: "
            + ", ".join(
                sorted(
                    missing_fields
                )
            )
        )

    return failures


# ============================================================================
# TEST REPORT
# ============================================================================

def build_test_report(
    engine: CPQ731ConfigurationEngine,
    failures: list[str],
) -> str:

    lines = []

    lines.append(
        "# 731 CPQ Configuration Engine Test Report"
    )

    lines.append("")

    lines.append(
        "## Source"
    )

    lines.append("")

    lines.append(
        f"- Workbook: "
        f"`{SOURCE_FILE.relative_to(ROOT)}`"
    )

    lines.append(
        f"- Rows: **{len(engine.source):,}**"
    )

    lines.append(
        f"- Columns: **{len(engine.source.columns)}**"
    )

    lines.append("")

    lines.append(
        "## Technical fields"
    )

    lines.append("")

    for field_name in (
        engine.list_technical_fields()
    ):

        lines.append(
            f"- `{field_name}`"
        )

    lines.append("")

    lines.append(
        "## Package contexts"
    )

    lines.append("")

    for package in (
        engine.list_package_contexts()
    ):

        lines.append(
            f"- `{package}`"
        )

    lines.append("")

    lines.append(
        "## Deterministic validation"
    )

    lines.append("")

    lines.append(
        "Technical validity is determined from the 731 source "
        "rule rows. ML is not used for technical validation."
    )

    lines.append("")

    lines.append(
        "## Test result"
    )

    lines.append("")

    if failures:

        lines.append(
            "**FAIL**"
        )

        lines.append("")

        for failure in failures:

            lines.append(
                f"- {failure}"
            )

    else:

        lines.append(
            "**PASS**"
        )

        lines.append("")

        lines.append(
            "All basic deterministic engine tests passed."
        )

    lines.append("")

    lines.append(
        "## Model-code status"
    )

    lines.append("")

    lines.append(
        "**NOT VERIFIED**"
    )

    lines.append("")

    lines.append(
        "The engine does not generate Emerson commercial/model "
        "codes. Canonical CFG731 configuration identity remains "
        "handled by the existing canonical configuration module."
    )

    return "\n".join(
        lines
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print()

    print(
        "=" * 80
    )

    print(
        "731 CPQ CONFIGURATION ENGINE"
    )

    print(
        "=" * 80
    )

    print()

    # ------------------------------------------------------------------------
    # Load engine
    # ------------------------------------------------------------------------

    print(
        "Loading catalogue and source rules..."
    )

    engine = (
        CPQ731ConfigurationEngine()
    )

    print(
        f"Source: "
        f"{len(engine.source):,} rows x "
        f"{len(engine.source.columns)} columns"
    )

    print()

    # ------------------------------------------------------------------------
    # Technical fields
    # ------------------------------------------------------------------------

    print(
        "TECHNICAL FIELDS"
    )

    for field_name in (
        engine.list_technical_fields()
    ):

        print(
            f"  - {field_name}"
        )

    print()

    # ------------------------------------------------------------------------
    # Package contexts
    # ------------------------------------------------------------------------

    print(
        "PACKAGE CONTEXTS"
    )

    for package in (
        engine.list_package_contexts()
    ):

        print(
            f"  - {package}"
        )

    print()

    # ------------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------------

    print(
        "RUNNING DETERMINISTIC TESTS..."
    )

    failures = run_basic_tests(
        engine
    )

    print()

    if failures:

        print(
            "TEST RESULT: FAIL"
        )

        for failure in failures:

            print(
                f"  ERROR: {failure}"
            )

    else:

        print(
            "TEST RESULT: PASS"
        )

    print()

    # ------------------------------------------------------------------------
    # Example 1
    # ------------------------------------------------------------------------

    print(
        "EXAMPLE 1: PARTIAL TECHNICAL REQUIREMENT"
    )

    example_configuration = {
        "devCategory": "F",
        "characteristic": "GP",
        "housing": "ST",
        "powerSupply": "1",
        "numberOfChannels": "1",
    }

    result = engine.resolve(
        example_configuration,
        package_context="x731",
        limit=5,
    )

    print(
        f"  Valid:            {result.valid}"
    )

    print(
        f"  Package:          {result.package_context}"
    )

    print(
        f"  Candidates:       {result.candidate_count}"
    )

    if result.reason:

        print(
            f"  Reason:           {result.reason}"
        )

    for candidate in result.candidates:

        print()

        print(
            f"  Source row:       {candidate.source_row}"
        )

        print(
            f"  Matched fields:   "
            f"{', '.join(candidate.matched_fields)}"
        )

        print(
            f"  Package rule:     "
            f"{candidate.source_package_expression}"
        )

    print()

    # ------------------------------------------------------------------------
    # Example 2
    # ------------------------------------------------------------------------

    print(
        "EXAMPLE 2: INVALID REQUIREMENT"
    )

    invalid_configuration = {
        "housing": "THIS_VALUE_DOES_NOT_EXIST",
    }

    invalid_result = engine.resolve(
        invalid_configuration
    )

    print(
        f"  Valid:            "
        f"{invalid_result.valid}"
    )

    print(
        f"  Candidates:       "
        f"{invalid_result.candidate_count}"
    )

    print(
        f"  Reason:           "
        f"{invalid_result.reason}"
    )

    print()

    # ------------------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = build_test_report(
        engine,
        failures,
    )

    TEST_REPORT.write_text(
        report,
        encoding="utf-8",
    )

    print(
        "TEST REPORT"
    )

    print(
        f"  {TEST_REPORT}"
    )

    print()

    # ------------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------------

    if failures:

        raise SystemExit(1)

    print(
        "Deterministic CPQ engine is ready."
    )

    print()

    print(
        "Next integration target:"
    )

    print(
        "  Streamlit CPQ UI -> this engine -> valid candidates -> ML ranking"
    )

    print()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()