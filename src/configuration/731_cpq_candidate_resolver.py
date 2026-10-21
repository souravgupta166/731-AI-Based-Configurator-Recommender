#!/usr/bin/env python3
"""
731 CPQ CANDIDATE RESOLVER
==========================

Converts deterministic CPQ source-rule matches into unique technical
configuration candidates.

Architecture
------------

    731 workbook
         |
         v
    CPQ catalogue
         |
         v
    CPQ configuration engine
         |
         v
    compatible source rows
         |
         v
    unique technical configurations
         |
         v
    existing canonical_configuration.py
         |
         v
    candidate set
         |
         v
    ML ranking


IMPORTANT
---------

This module does NOT:

- decide technical validity using ML
- invent compatibility rules
- generate Emerson model codes
- replace the existing canonical configuration implementation
- treat CFG731_* as an Emerson model code

Technical validity remains the responsibility of:

    731_cpq_configuration_engine.py

Canonical configuration identity remains the responsibility of:

    src/synthetic/canonical_configuration.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================================
# 1. PROJECT ROOT
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = ROOT / "src"


# ============================================================================
# 2. MAKE PROJECT ROOT IMPORTABLE
# ============================================================================

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================================
# 3. LOAD NUMERIC-NAMED CPQ ENGINE
# ============================================================================

CPQ_ENGINE_FILE = (
    ROOT
    / "src"
    / "configuration"
    / "731_cpq_configuration_engine.py"
)


def load_cpq_engine_module():
    """
    Load 731_cpq_configuration_engine.py from its file path.

    The filename starts with a number, so normal Python import syntax
    cannot be used.

    The module is registered in sys.modules before execution because
    the engine contains dataclasses.
    """

    if not CPQ_ENGINE_FILE.exists():
        raise FileNotFoundError(
            "CPQ engine file not found:\n"
            f"{CPQ_ENGINE_FILE}"
        )

    module_name = "cpq_731_configuration_engine"

    spec = importlib.util.spec_from_file_location(
        module_name,
        CPQ_ENGINE_FILE,
    )

    if spec is None:
        raise ImportError(
            "Could not create import specification for CPQ engine."
        )

    if spec.loader is None:
        raise ImportError(
            "Could not create loader for CPQ engine."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[module_name] = module

    spec.loader.exec_module(
        module
    )

    return module


engine_module = load_cpq_engine_module()


# ============================================================================
# 4. LOAD EXISTING CANONICAL CONFIGURATION MODULE
# ============================================================================

try:

    from src.synthetic import canonical_configuration as canonical_module

except ImportError as exc:

    raise ImportError(
        "Could not import:\n"
        "  src/synthetic/canonical_configuration.py\n\n"
        f"Original error: {exc}"
    ) from exc


# ============================================================================
# 5. VERIFY EXISTING CANONICAL MODULE API
# ============================================================================

if not hasattr(
    canonical_module,
    "TECHNICAL_COLUMNS",
):

    raise AttributeError(
        "The existing canonical_configuration.py does not expose "
        "TECHNICAL_COLUMNS."
    )


if not hasattr(
    canonical_module,
    "canonical_id",
):

    raise AttributeError(
        "The existing canonical_configuration.py does not expose "
        "canonical_id(). "
        "The candidate resolver will not invent a replacement."
    )


TECHNICAL_COLUMNS = list(
    canonical_module.TECHNICAL_COLUMNS
)

canonical_id = canonical_module.canonical_id


# ============================================================================
# 6. OUTPUT PATHS
# ============================================================================

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
)

CANDIDATE_REPORT = (
    OUTPUT_DIR
    / "731_cpq_candidate_resolver_report.md"
)


# ============================================================================
# 7. TECHNICAL FIELDS
# ============================================================================

TECHNICAL_FIELDS = list(
    engine_module.TECHNICAL_FIELDS
)


# ============================================================================
# 8. CANDIDATE DATA STRUCTURE
# ============================================================================

@dataclass
class CPQCandidate:
    """
    One unique deterministic technical configuration.

    Multiple source rule rows may support the same technical
    configuration. Their source rows are retained as provenance.
    """

    canonical_configuration_id: str

    configuration: dict[str, Any]

    package_context: str | None

    source_rows: list[int] = field(
        default_factory=list
    )

    matched_fields: list[str] = field(
        default_factory=list
    )

    source_package_expressions: list[str] = field(
        default_factory=list
    )

    source_row_count: int = 0

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "canonical_configuration_id": (
                self.canonical_configuration_id
            ),
            "package_context": (
                self.package_context
            ),
            "source_rows": (
                self.source_rows
            ),
            "source_row_count": (
                self.source_row_count
            ),
            "matched_fields": (
                self.matched_fields
            ),
            "source_package_expressions": (
                self.source_package_expressions
            ),
            "configuration": (
                self.configuration
            ),
        }


# ============================================================================
# 9. RESOLUTION RESULT
# ============================================================================

@dataclass
class CandidateResolution:
    """
    Final candidate-resolution result.
    """

    valid: bool

    requested_configuration: dict[str, Any]

    normalized_configuration: dict[str, str]

    package_context: str | None

    source_match_count: int

    unique_candidate_count: int

    candidates: list[CPQCandidate]

    reason: str | None = None

    def to_dict(
        self,
    ) -> dict[str, Any]:

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
            "source_match_count": (
                self.source_match_count
            ),
            "unique_candidate_count": (
                self.unique_candidate_count
            ),
            "reason": self.reason,
            "candidates": [
                candidate.to_dict()
                for candidate in self.candidates
            ],
        }


# ============================================================================
# 10. CANDIDATE RESOLVER
# ============================================================================

class CPQ731CandidateResolver:
    """
    Converts deterministic CPQ matches into unique configurations.

    Technical validity:
        CPQ731ConfigurationEngine

    Canonical configuration identity:
        canonical_configuration.py
    """

    def __init__(
        self,
        cpq_engine=None,
    ) -> None:

        if cpq_engine is None:

            self.engine = (
                engine_module
                .CPQ731ConfigurationEngine()
            )

        else:

            self.engine = cpq_engine

    # ========================================================================
    # CANONICAL CONFIGURATION
    # ========================================================================

    @staticmethod
    def _canonical_configuration(
        configuration: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Keep only the technical columns defined by the existing
        canonical configuration implementation.
        """

        result = {}

        for field_name in TECHNICAL_COLUMNS:

            result[field_name] = (
                configuration.get(
                    field_name
                )
            )

        return result

    @staticmethod
    def _canonical_id(
        configuration: dict[str, Any],
    ) -> str:
        """
        Delegate canonical ID generation to the existing project
        implementation.

        No new hashing algorithm is created here.
        """

        canonical_configuration = (
            CPQ731CandidateResolver
            ._canonical_configuration(
                configuration
            )
        )

        return canonical_id(
            canonical_configuration
        )

    # ========================================================================
    # NORMALIZATION
    # ========================================================================

    @staticmethod
    def _normalize_candidate_configuration(
        configuration: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Normalize technical values consistently.

        NOVALUE remains an internal source representation.
        """

        normalized = {}

        for field_name in TECHNICAL_COLUMNS:

            value = configuration.get(
                field_name
            )

            if value is None:

                normalized[field_name] = None

                continue

            normalized[field_name] = (
                engine_module.normalize_value(
                    value
                )
            )

        return normalized

    # ========================================================================
    # MATERIALIZE CANDIDATE
    # ========================================================================

    def _materialize_candidate(
        self,
        source_candidate,
        existing: CPQCandidate | None = None,
    ) -> CPQCandidate:
        """
        Convert one deterministic-engine candidate into the unique
        CPQ candidate representation.
        """

        configuration = (
            self._normalize_candidate_configuration(
                source_candidate.configuration
            )
        )

        candidate_id = (
            self._canonical_id(
                configuration
            )
        )

        # --------------------------------------------------------------------
        # New candidate
        # --------------------------------------------------------------------

        if existing is None:

            package_expression = (
                source_candidate
                .source_package_expression
            )

            package_expressions = []

            if package_expression:

                package_expressions.append(
                    package_expression
                )

            return CPQCandidate(

                canonical_configuration_id=(
                    candidate_id
                ),

                configuration=(
                    configuration
                ),

                package_context=(
                    source_candidate.package_context
                ),

                source_rows=[
                    source_candidate.source_row
                ],

                matched_fields=sorted(
                    set(
                        source_candidate.matched_fields
                    )
                ),

                source_package_expressions=(
                    package_expressions
                ),

                source_row_count=1,
            )

        # --------------------------------------------------------------------
        # Merge source rows
        # --------------------------------------------------------------------

        if (
            source_candidate.source_row
            not in existing.source_rows
        ):

            existing.source_rows.append(
                source_candidate.source_row
            )

        # --------------------------------------------------------------------
        # Merge matched fields
        # --------------------------------------------------------------------

        for field_name in (
            source_candidate.matched_fields
        ):

            if field_name not in (
                existing.matched_fields
            ):

                existing.matched_fields.append(
                    field_name
                )

        # --------------------------------------------------------------------
        # Merge source package expressions
        # --------------------------------------------------------------------

        expression = (
            source_candidate
            .source_package_expression
        )

        if (
            expression
            and expression
            not in existing.source_package_expressions
        ):

            existing.source_package_expressions.append(
                expression
            )

        # --------------------------------------------------------------------
        # Normalize provenance
        # --------------------------------------------------------------------

        existing.source_rows = sorted(
            set(
                existing.source_rows
            )
        )

        existing.matched_fields = sorted(
            set(
                existing.matched_fields
            )
        )

        existing.source_package_expressions = sorted(
            set(
                existing.source_package_expressions
            )
        )

        existing.source_row_count = (
            len(
                existing.source_rows
            )
        )

        return existing

    # ========================================================================
    # RESOLVE
    # ========================================================================

    def resolve(
        self,
        configuration: dict[str, Any],
        package_context: str | None = None,
        limit: int | None = None,
    ) -> CandidateResolution:
        """
        Resolve a partial CPQ requirement into unique technical
        configuration candidates.

        Technical validity is determined exclusively by the deterministic
        CPQ engine.

        This resolver performs:

        - candidate materialization
        - canonicalization
        - deduplication
        - provenance preservation
        """

        engine_result = self.engine.resolve(
            configuration=configuration,
            package_context=package_context,
            limit=None,
        )

        # --------------------------------------------------------------------
        # No valid source rows
        # --------------------------------------------------------------------

        if not engine_result.valid:

            return CandidateResolution(

                valid=False,

                requested_configuration=(
                    configuration
                ),

                normalized_configuration=(
                    engine_result
                    .normalized_configuration
                ),

                package_context=(
                    engine_result.package_context
                ),

                source_match_count=0,

                unique_candidate_count=0,

                candidates=[],

                reason=(
                    engine_result.reason
                ),
            )

        # --------------------------------------------------------------------
        # Deduplicate by canonical ID
        # --------------------------------------------------------------------

        candidates_by_id: dict[
            str,
            CPQCandidate,
        ] = {}

        for source_candidate in (
            engine_result.candidates
        ):

            normalized_configuration = (
                self._normalize_candidate_configuration(
                    source_candidate.configuration
                )
            )

            candidate_id = (
                self._canonical_id(
                    normalized_configuration
                )
            )

            existing = (
                candidates_by_id.get(
                    candidate_id
                )
            )

            candidates_by_id[
                candidate_id
            ] = self._materialize_candidate(
                source_candidate=source_candidate,
                existing=existing,
            )

        # --------------------------------------------------------------------
        # Deterministic ordering
        # --------------------------------------------------------------------

        candidates = sorted(
            candidates_by_id.values(),
            key=lambda candidate: (
                candidate.canonical_configuration_id
            ),
        )

        # --------------------------------------------------------------------
        # Apply limit AFTER deduplication
        # --------------------------------------------------------------------

        if limit is not None:

            if limit < 1:

                raise ValueError(
                    "limit must be >= 1"
                )

            candidates = candidates[
                :limit
            ]

        # --------------------------------------------------------------------
        # Return
        # --------------------------------------------------------------------

        return CandidateResolution(

            valid=True,

            requested_configuration=(
                configuration
            ),

            normalized_configuration=(
                engine_result
                .normalized_configuration
            ),

            package_context=(
                engine_result.package_context
            ),

            source_match_count=(
                len(
                    engine_result.candidates
                )
            ),

            unique_candidate_count=(
                len(candidates)
            ),

            candidates=candidates,

            reason=None,
        )

    # ========================================================================
    # SIMPLE VALIDATION
    # ========================================================================

    def validate(
        self,
        configuration: dict[str, Any],
        package_context: str | None = None,
    ) -> bool:

        return self.engine.validate(
            configuration=configuration,
            package_context=package_context,
        )

    # ========================================================================
    # DATAFRAME
    # ========================================================================

    def to_dataframe(
        self,
        resolution: CandidateResolution,
    ) -> pd.DataFrame:
        """
        Convert unique candidates into a DataFrame.

        One row = one unique technical configuration.
        """

        rows = []

        for candidate in (
            resolution.candidates
        ):

            row = {
                "canonical_configuration_id": (
                    candidate
                    .canonical_configuration_id
                ),

                "package_context": (
                    candidate.package_context
                ),

                "source_row_count": (
                    candidate.source_row_count
                ),

                "source_rows": "|".join(
                    str(source_row)
                    for source_row
                    in candidate.source_rows
                ),

                "matched_fields": "|".join(
                    candidate.matched_fields
                ),

                "source_package_expressions": (
                    " || ".join(
                        candidate
                        .source_package_expressions
                    )
                ),
            }

            for field_name in (
                TECHNICAL_COLUMNS
            ):

                row[field_name] = (
                    candidate
                    .configuration
                    .get(field_name)
                )

            rows.append(
                row
            )

        return pd.DataFrame(
            rows
        )

    # ========================================================================
    # SAVE JSON
    # ========================================================================

    def save_resolution(
        self,
        resolution: CandidateResolution,
        output_file: Path | str,
    ) -> None:

        output_file = Path(
            output_file
        )

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                resolution.to_dict(),
                file,
                indent=2,
                ensure_ascii=False,
            )


# ============================================================================
# 11. TESTS
# ============================================================================

def run_tests(
    resolver: CPQ731CandidateResolver,
) -> list[str]:
    """
    Basic candidate-resolver integrity tests.
    """

    failures = []

    # ========================================================================
    # TEST 1
    # Known x731 partial configuration
    # ========================================================================

    resolution = resolver.resolve(
        {
            "devCategory": "F",
            "characteristic": "GP",
            "housing": "ST",
            "powerSupply": "1",
            "numberOfChannels": "1",
        },
        package_context="x731",
    )

    if not resolution.valid:

        failures.append(
            "Known x731 partial configuration was rejected."
        )

    if resolution.source_match_count <= 0:

        failures.append(
            "Expected at least one deterministic source match."
        )

    if resolution.unique_candidate_count <= 0:

        failures.append(
            "Expected at least one unique candidate."
        )

    # ========================================================================
    # TEST 2
    # Canonical IDs unique
    # ========================================================================

    ids = [
        candidate.canonical_configuration_id
        for candidate in resolution.candidates
    ]

    if len(ids) != len(set(ids)):

        failures.append(
            "Duplicate canonical configuration IDs were returned."
        )

    # ========================================================================
    # TEST 3
    # All canonical technical columns present
    # ========================================================================

    for candidate in resolution.candidates:

        missing = [
            field_name
            for field_name in TECHNICAL_COLUMNS
            if field_name
            not in candidate.configuration
        ]

        if missing:

            failures.append(
                "Candidate missing technical columns: "
                + ", ".join(
                    missing
                )
            )

            break

    # ========================================================================
    # TEST 4
    # Invalid configuration
    # ========================================================================

    invalid_resolution = resolver.resolve(
        {
            "housing": "NOT_A_REAL_731_VALUE",
        }
    )

    if invalid_resolution.valid:

        failures.append(
            "Invalid technical requirement was accepted."
        )

    if (
        invalid_resolution.unique_candidate_count
        != 0
    ):

        failures.append(
            "Invalid requirement returned candidates."
        )

    # ========================================================================
    # TEST 5
    # Limit
    # ========================================================================

    limited_resolution = resolver.resolve(
        {},
        limit=10,
    )

    if (
        limited_resolution
        .unique_candidate_count
        > 10
    ):

        failures.append(
            "Candidate limit was not respected."
        )

    # ========================================================================
    # TEST 6
    # No model code
    # ========================================================================

    for candidate in (
        limited_resolution.candidates
    ):

        if "model_code" in (
            candidate.configuration
        ):

            failures.append(
                "Candidate unexpectedly contains model_code."
            )

            break

    return failures


# ============================================================================
# 12. REPORT
# ============================================================================

def build_report(
    failures: list[str],
) -> str:

    lines = []

    lines.append(
        "# 731 CPQ Candidate Resolver Test Report"
    )

    lines.append("")

    lines.append(
        "## Purpose"
    )

    lines.append("")

    lines.append(
        "The candidate resolver converts deterministic 731 CPQ "
        "source-rule matches into unique technical configurations."
    )

    lines.append("")

    lines.append(
        "Technical validity is delegated to the deterministic "
        "731 CPQ configuration engine."
    )

    lines.append("")

    lines.append(
        "Canonical configuration identity is delegated to the "
        "existing canonical_configuration.py implementation."
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
            "All candidate resolver tests passed."
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
        "No Emerson model-code mapping is inferred or generated "
        "by this resolver."
    )

    return "\n".join(
        lines
    )


# ============================================================================
# 13. MAIN
# ============================================================================

def main() -> None:

    print()

    print(
        "=" * 80
    )

    print(
        "731 CPQ CANDIDATE RESOLVER"
    )

    print(
        "=" * 80
    )

    print()

    # ========================================================================
    # LOAD
    # ========================================================================

    print(
        "Loading deterministic CPQ engine..."
    )

    resolver = (
        CPQ731CandidateResolver()
    )

    print(
        "Engine loaded successfully."
    )

    print()

    # ========================================================================
    # CANONICAL FIELDS
    # ========================================================================

    print(
        "CANONICAL TECHNICAL FIELDS"
    )

    print()

    for field_name in (
        TECHNICAL_COLUMNS
    ):

        print(
            f"  - {field_name}"
        )

    print()

    # ========================================================================
    # TESTS
    # ========================================================================

    print(
        "RUNNING CANDIDATE RESOLVER TESTS..."
    )

    failures = run_tests(
        resolver
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

    # ========================================================================
    # EXAMPLE
    # ========================================================================

    print(
        "EXAMPLE: UNIQUE CANDIDATE RESOLUTION"
    )

    example_configuration = {
        "devCategory": "F",
        "characteristic": "GP",
        "housing": "ST",
        "powerSupply": "1",
        "numberOfChannels": "1",
    }

    resolution = resolver.resolve(
        example_configuration,
        package_context="x731",
    )

    print(
        f"  Valid:              "
        f"{resolution.valid}"
    )

    print(
        f"  Package:            "
        f"{resolution.package_context}"
    )

    print(
        f"  Source matches:     "
        f"{resolution.source_match_count}"
    )

    print(
        f"  Unique candidates:  "
        f"{resolution.unique_candidate_count}"
    )

    if resolution.reason:

        print(
            f"  Reason:             "
            f"{resolution.reason}"
        )

    # ========================================================================
    # FIRST FIVE CANDIDATES
    # ========================================================================

    for candidate in (
        resolution.candidates[:5]
    ):

        print()

        print(
            "  CANDIDATE"
        )

        print(
            f"    Canonical ID: "
            f"{candidate.canonical_configuration_id}"
        )

        print(
            f"    Source rows: "
            f"{candidate.source_rows}"
        )

        print(
            f"    Source row count: "
            f"{candidate.source_row_count}"
        )

        print(
            f"    Matched fields: "
            f"{', '.join(candidate.matched_fields)}"
        )

        print(
            "    Configuration:"
        )

        for field_name in (
            TECHNICAL_COLUMNS
        ):

            print(
                f"      {field_name}: "
                f"{candidate.configuration.get(field_name)}"
            )

    print()

    # ========================================================================
    # DATAFRAME
    # ========================================================================

    dataframe = resolver.to_dataframe(
        resolution
    )

    print(
        "CANDIDATE DATAFRAME"
    )

    print(
        f"  Shape:    "
        f"{dataframe.shape}"
    )

    print(
        f"  Columns:  "
        f"{len(dataframe.columns)}"
    )

    print()

    # ========================================================================
    # SAVE REPORT
    # ========================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = build_report(
        failures
    )

    CANDIDATE_REPORT.write_text(
        report,
        encoding="utf-8",
    )

    print(
        "TEST REPORT"
    )

    print(
        f"  {CANDIDATE_REPORT}"
    )

    print()

    # ========================================================================
    # EXIT
    # ========================================================================

    if failures:

        raise SystemExit(1)

    print(
        "Candidate resolver is ready."
    )

    print()

    print(
        "Next integration:"
    )

    print(
        "  CPQ UI"
        " -> deterministic engine"
        " -> unique candidates"
        " -> existing ML ranking"
    )

    print()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()