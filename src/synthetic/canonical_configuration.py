#!/usr/bin/env python3

"""
731 CANONICAL CONFIGURATION REPRESENTATION
==========================================

Defines the single canonical representation used throughout
the synthetic-data pipeline.

Purpose
-------
A raw 731 configurator record may contain expressions such as:

    in {NOVALUE, 'G731ST-LT'}

For modelling, this represents the selected technical state:

    G731ST-LT

This module provides:

1. Value normalization
2. Canonical technical configuration
3. Canonical configuration ID
4. Technical distance between configurations
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, Iterable

import pandas as pd


# =============================================================================
# TECHNICAL CHARACTERISTICS
# =============================================================================

TECHNICAL_COLUMNS = [
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


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize_value(value) -> str:
    """
    Convert a raw 731 value to canonical representation.

    Examples
    --------
    in {NOVALUE, 'G731ST-LT'}
        -> G731ST-LT

    'GP'
        -> GP

    NOVALUE
        -> NOVALUE
    """

    if pd.isna(value):
        return "NOVALUE"

    text = str(value).strip()

    if not text:
        return "NOVALUE"

    # ------------------------------------------------------------
    # Configurator set expression
    # ------------------------------------------------------------

    match = re.match(
        r"^in\s*\{(.*)\}$",
        text,
        flags=re.IGNORECASE,
    )

    if match:

        content = match.group(1).strip()

        tokens = re.findall(
            r"'([^']*)'|\"([^\"]*)\"|([^,\s]+)",
            content,
        )

        values = []

        for single, double, unquoted in tokens:

            token = (
                single
                or double
                or unquoted
            )

            if token:
                values.append(
                    token.strip()
                )

        # Prefer a real value over NOVALUE.
        non_novalue = [
            token
            for token in values
            if token.upper() != "NOVALUE"
        ]

        if non_novalue:
            return non_novalue[0]

        return "NOVALUE"

    # ------------------------------------------------------------
    # Remove quotes
    # ------------------------------------------------------------

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
# CANONICAL CONFIGURATION
# =============================================================================

def canonicalize_configuration(
    row_or_dict,
    columns: Iterable[str] = TECHNICAL_COLUMNS,
) -> Dict[str, str]:
    """
    Return a canonical technical configuration dictionary.
    """

    configuration = {}

    for column in columns:

        if isinstance(row_or_dict, dict):
            value = row_or_dict.get(
                column,
                "NOVALUE",
            )
        else:
            value = row_or_dict.get(
                column,
                "NOVALUE",
            )

        configuration[column] = (
            normalize_value(value)
        )

    return configuration


# =============================================================================
# CANONICAL STRING
# =============================================================================

def canonical_string(
    configuration: Dict[str, str],
    columns: Iterable[str] = TECHNICAL_COLUMNS,
) -> str:
    """
    Deterministic representation of a technical configuration.
    """

    return "||".join(
        normalize_value(
            configuration.get(
                column,
                "NOVALUE",
            )
        )
        for column in columns
    )


# =============================================================================
# CANONICAL ID
# =============================================================================

def canonical_id(
    configuration: Dict[str, str],
    columns: Iterable[str] = TECHNICAL_COLUMNS,
) -> str:
    """
    Generate a deterministic SHA-256 based configuration identifier.

    The identifier does not contain customer information and is only
    used to identify the technical configuration.
    """

    representation = canonical_string(
        configuration,
        columns,
    )

    digest = hashlib.sha256(
        representation.encode(
            "utf-8"
        )
    ).hexdigest()

    return (
        "CFG731_"
        + digest[:16]
    )


# =============================================================================
# TECHNICAL DISTANCE
# =============================================================================

def technical_distance(
    configuration_a: Dict[str, str],
    configuration_b: Dict[str, str],
    columns: Iterable[str] = TECHNICAL_COLUMNS,
) -> int:
    """
    Hamming distance across technical characteristics.

    Example
    -------
    1 means exactly one technical characteristic differs.
    """

    return sum(
        normalize_value(
            configuration_a.get(
                column,
                "NOVALUE",
            )
        )
        != normalize_value(
            configuration_b.get(
                column,
                "NOVALUE",
            )
        )
        for column in columns
    )


# =============================================================================
# DATAFRAME HELPER
# =============================================================================

def add_canonical_identity(
    df: pd.DataFrame,
    columns: Iterable[str] = TECHNICAL_COLUMNS,
) -> pd.DataFrame:
    """
    Add canonical values and canonical_configuration_id.

    The original columns are retained.
    """

    result = df.copy()

    for column in columns:

        if column in result.columns:

            result[column] = (
                result[column]
                .apply(
                    normalize_value
                )
            )

    result[
        "canonical_configuration_id"
    ] = result.apply(
        lambda row:
        canonical_id(
            row,
            columns,
        ),
        axis=1,
    )

    return result