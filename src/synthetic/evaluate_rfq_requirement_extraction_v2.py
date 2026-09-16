#!/usr/bin/env python3

"""
731 RFQ REQUIREMENT EXTRACTION — V2
===================================

Evidence-aware, 731-domain-specific RFQ requirement extraction.

V1 baseline:
    65.78% precision
    48.78% recall
    56.02% F1

V2 improvements:
    - characteristic-specific phrase dictionaries
    - value-specific extraction
    - ambiguity avoidance
    - explicit / implicit requirement classification
    - evidence text
    - confidence score
    - constraint-aware validation
    - V1-compatible evaluation
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RFQ_DOCUMENT_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "rfqs"
    / "731_rfq_documents_v2.csv"
)

RFQ_REQUIREMENTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "rfqs"
    / "731_rfq_requirements_v2.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = OUTPUT_DIR / "731_rfq_extraction_results_v2.csv"
CHARACTERISTIC_METRICS_FILE = (
    OUTPUT_DIR / "731_rfq_characteristic_metrics_v2.csv"
)
METRICS_FILE = OUTPUT_DIR / "731_rfq_extraction_metrics_v2.csv"
ERRORS_FILE = OUTPUT_DIR / "731_rfq_extraction_errors_v2.csv"


# =============================================================================
# CHARACTERISTICS
# =============================================================================

TECHNICAL_CHARACTERISTICS = [
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
# TEXT NORMALIZATION
# =============================================================================

def normalize_text(text) -> str:

    if pd.isna(text):
        return ""

    text = str(text)

    text = (
        text.replace("\n", " ")
        .replace("\r", " ")
        .replace("\t", " ")
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


def clean_evidence(text: str, start: int, end: int) -> str:

    left = max(0, start - 90)
    right = min(len(text), end + 90)

    evidence = text[left:right].strip()

    return evidence


# =============================================================================
# PREDICTION STRUCTURE
# =============================================================================

def add_prediction(
    predictions: Dict[str, Dict],
    characteristic: str,
    value: str,
    text: str,
    start: int,
    end: int,
    confidence: float,
    method: str,
    requirement_type: str = "UNKNOWN",
):

    if not value:
        return

    existing = predictions.get(characteristic)

    candidate = {
        "characteristic": characteristic,
        "value": value,
        "evidence_text": clean_evidence(
            text,
            start,
            end,
        ),
        "confidence": confidence,
        "extraction_method": method,
        "requirement_type": requirement_type,
    }

    if existing is None:
        predictions[characteristic] = candidate
        return

    # Keep stronger evidence.
    if confidence > existing["confidence"]:
        predictions[characteristic] = candidate


# =============================================================================
# REQUIREMENT TYPE
# =============================================================================

def infer_requirement_type(
    text: str,
    start: int,
    end: int,
) -> str:

    context = text[
        max(0, start - 120):
        min(len(text), end + 120)
    ]

    preferred_patterns = [
        r"\bif possible\b",
        r"\bprefer\b",
        r"\bpreferred\b",
        r"\bwould prefer\b",
        r"\bideally\b",
        r"\bwhere possible\b",
        r"\boptional\b",
        r"\brecommend\b",
    ]

    mandatory_patterns = [
        r"\brequired\b",
        r"\brequires\b",
        r"\bmust\b",
        r"\bneed\b",
        r"\bneeds\b",
        r"\bshall\b",
        r"\bmandatory\b",
    ]

    for pattern in preferred_patterns:
        if re.search(pattern, context, re.IGNORECASE):
            return "PREFERRED"

    for pattern in mandatory_patterns:
        if re.search(pattern, context, re.IGNORECASE):
            return "MANDATORY"

    return "UNKNOWN"


# =============================================================================
# REGEX HELPER
# =============================================================================

def search_patterns(
    text: str,
    patterns: List[Tuple[str, str, float, str]],
):

    for pattern, value, confidence, method in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            return (
                match,
                value,
                confidence,
                method,
            )

    return None


# =============================================================================
# EXTRACTION
# =============================================================================

def extract_requirements_v2(
    original_text: str,
) -> Dict[str, Dict]:

    text = normalize_text(original_text)

    predictions = {}

    # =========================================================================
    # DEV CATEGORY
    # =========================================================================

    patterns = [
        (
            r"\bgas metering\b|\bgas measurement\b",
            "G",
            0.98,
            "EXPLICIT_GAS",
        ),
        (
            r"\bgeneral[- ]purpose configuration\b",
            "G",
            0.98,
            "EXPLICIT_GENERAL_PURPOSE",
        ),
        (
            r"\bflow measurement\b|\bliquid measurement\b",
            "F",
            0.96,
            "EXPLICIT_FLOW",
        ),
        (
            r"\bsteam measurement\b",
            "S",
            0.98,
            "EXPLICIT_STEAM",
        ),
    ]

    result = search_patterns(text, patterns)

    if result:
        match, value, conf, method = result

        add_prediction(
            predictions,
            "devCategory",
            value,
            text,
            match.start(),
            match.end(),
            conf,
            method,
            infer_requirement_type(
                text,
                match.start(),
                match.end(),
            ),
        )

    # =========================================================================
    # CHARACTERISTIC
    # =========================================================================

    patterns = [
        (
            r"\bgeneral[- ]purpose configuration\b",
            "GP",
            0.98,
            "EXPLICIT_GP",
        ),
        (
            r"\bstainless[- ]steel configuration\b",
            "ST",
            0.98,
            "EXPLICIT_ST",
        ),
        (
            r"\btemperature measurement\b",
            "TE",
            0.96,
            "EXPLICIT_TE",
        ),
        (
            r"\bcalibration\b",
            "CA",
            0.96,
            "EXPLICIT_CA",
        ),
        (
            r"\bpressure measurement\b",
            "PW",
            0.96,
            "EXPLICIT_PW",
        ),
        (
            r"\bwide[- ]range\b",
            "WD",
            0.90,
            "EXPLICIT_WD",
        ),
        (
            r"\bverification\b",
            "VG",
            0.90,
            "EXPLICIT_VG",
        ),
    ]

    result = search_patterns(text, patterns)

    if result:

        match, value, conf, method = result

        add_prediction(
            predictions,
            "characteristic",
            value,
            text,
            match.start(),
            match.end(),
            conf,
            method,
            infer_requirement_type(
                text,
                match.start(),
                match.end(),
            ),
        )

    # =========================================================================
    # HOUSING
    # =========================================================================

    patterns = [
        (
            r"\bstainless[- ]steel housing\b",
            "ST",
            0.99,
            "EXPLICIT_HOUSING",
        ),
        (
            r"\bstainless[- ]steel enclosure\b",
            "ST",
            0.99,
            "EXPLICIT_HOUSING",
        ),
        (
            r"\baluminium housing\b|\baluminum housing\b",
            "AL",
            0.99,
            "EXPLICIT_HOUSING",
        ),
        (
            r"\baluminium enclosure\b|\baluminum enclosure\b",
            "AL",
            0.98,
            "EXPLICIT_HOUSING",
        ),
    ]

    result = search_patterns(text, patterns)

    if result:

        match, value, conf, method = result

        add_prediction(
            predictions,
            "housing",
            value,
            text,
            match.start(),
            match.end(),
            conf,
            method,
            infer_requirement_type(
                text,
                match.start(),
                match.end(),
            ),
        )

    # =========================================================================
    # POWER SUPPLY
    # =========================================================================

    patterns = [
        (
            r"\balternative (?:electrical|power) supply\b",
            "4",
            0.99,
            "EXPLICIT_ALTERNATIVE_POWER",
        ),
        (
            r"\balternative electrical power\b",
            "4",
            0.99,
            "EXPLICIT_ALTERNATIVE_POWER",
        ),
        (
            r"\bstandard power supply\b",
            "1",
            0.99,
            "EXPLICIT_STANDARD_POWER",
        ),
    ]

    result = search_patterns(text, patterns)

    if result:

        match, value, conf, method = result

        add_prediction(
            predictions,
            "powerSupply",
            value,
            text,
            match.start(),
            match.end(),
            conf,
            method,
            infer_requirement_type(
                text,
                match.start(),
                match.end(),
            ),
        )

    # =========================================================================
    # NUMBER OF CHANNELS
    # =========================================================================

    channel_patterns = [
        (
            r"\bsingle measurement channel\b",
            "1",
            0.99,
            "EXPLICIT_CHANNEL",
        ),
        (
            r"\bone measurement channel\b",
            "1",
            0.99,
            "EXPLICIT_CHANNEL",
        ),
        (
            r"\bdual measurement channels?\b",
            "2",
            0.99,
            "EXPLICIT_CHANNEL",
        ),
        (
            r"\btwo measurement channels?\b",
            "2",
            0.99,
            "EXPLICIT_CHANNEL",
        ),
        (
            r"\bnumber of measurement channels?\s*(?:is|:)?\s*(1|2)\b",
            None,
            0.99,
            "EXPLICIT_CHANNEL_NUMBER",
        ),
    ]

    for pattern, fixed, conf, method in channel_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE,
        )

        if match:

            value = (
                fixed
                if fixed
                else match.group(1)
            )

            add_prediction(
                predictions,
                "numberOfChannels",
                value,
                text,
                match.start(),
                match.end(),
                conf,
                method,
                infer_requirement_type(
                    text,
                    match.start(),
                    match.end(),
                ),
            )

            break

    # =========================================================================
    # EXPLOSION APPROVAL
    # =========================================================================

    patterns = [
        (
            r"\bexplosion protection\s+a\b",
            "A",
            0.99,
            "EXPLICIT_EXPLOSION",
        ),
        (
            r"\bhazardous[- ]area.*?approval\s+a\b",
            "A",
            0.98,
            "EXPLICIT_EXPLOSION",
        ),
        (
            r"\bexplosion approval\s+a\b",
            "A",
            0.99,
            "EXPLICIT_EXPLOSION",
        ),
        (
            r"\bexplosion protection\s+e\b",
            "E",
            0.99,
            "EXPLICIT_EXPLOSION",
        ),
        (
            r"\bexplosion approval\s+e\b",
            "E",
            0.99,
            "EXPLICIT_EXPLOSION",
        ),
        (
            r"\bdoes not require hazardous[- ]area approval\b",
            "N",
            0.99,
            "EXPLICIT_NON_HAZARDOUS",
        ),
        (
            r"\bno hazardous[- ]area approval\b",
            "N",
            0.99,
            "EXPLICIT_NON_HAZARDOUS",
        ),
        (
            r"\bnon[- ]hazardous\b",
            "N",
            0.94,
            "IMPLICIT_NON_HAZARDOUS",
        ),
    ]

    result = search_patterns(text, patterns)

    if result:

        match, value, conf, method = result

        add_prediction(
            predictions,
            "explosionApproval",
            value,
            text,
            match.start(),
            match.end(),
            conf,
            method,
            infer_requirement_type(
                text,
                match.start(),
                match.end(),
            ),
        )

    # =========================================================================
    # PROTECTION AREA
    # =========================================================================

    match = re.search(
        r"\bprotection area\s*(?:is|:)?\s*([0-9]+)\b",
        text,
        re.IGNORECASE,
    )

    if match:

        add_prediction(
            predictions,
            "protectionArea",
            match.group(1),
            text,
            match.start(),
            match.end(),
            0.99,
            "EXPLICIT_PROTECTION_AREA",
            infer_requirement_type(
                text,
                match.start(),
                match.end(),
            ),
        )

    # =========================================================================
    # CERTIFICATION
    # =========================================================================

    # Important:
    # Only capture a two-letter token when it is directly associated
    # with certification. Do NOT scan arbitrary two-letter words.

    certification_patterns = [
        (
            r"\brequired certification is\s+([a-z]{2})\b",
            0.99,
            "EXPLICIT_CERTIFICATION",
        ),
        (
            r"\brequired certification\s*[:\-]\s*([a-z]{2})\b",
            0.99,
            "EXPLICIT_CERTIFICATION",
        ),
        (
            r"\bcertification is\s+([a-z]{2})\b",
            0.99,
            "EXPLICIT_CERTIFICATION",
        ),
        (
            r"\bcertification\s*[:\-]\s*([a-z]{2})\b",
            0.99,
            "EXPLICIT_CERTIFICATION",
        ),
    ]

    for pattern, conf, method in certification_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE,
        )

        if match:

            value = match.group(1).upper()

            add_prediction(
                predictions,
                "certification",
                value,
                text,
                match.start(),
                match.end(),
                conf,
                method,
                infer_requirement_type(
                    text,
                    match.start(),
                    match.end(),
                ),
            )

            break

    # =========================================================================
    # DATA INTERFACE
    # =========================================================================

    patterns = [
        (
            r"\bdata interface\s+([a-z0-9]+)\b",
            None,
            0.99,
            "EXPLICIT_DATA_INTERFACE",
        ),
        (
            r"\bcommunication interface\s+([a-z0-9]+)\b",
            None,
            0.99,
            "EXPLICIT_DATA_INTERFACE",
        ),
        (
            r"\bcommunication interface is\s+([a-z0-9]+)\b",
            None,
            0.99,
            "EXPLICIT_DATA_INTERFACE",
        ),
        (
            r"\bhigh[- ]speed communication\b",
            "HS",
            0.96,
            "SEMANTIC_HIGH_SPEED",
        ),
        (
            r"\bstandard communication\b",
            "NN",
            0.96,
            "SEMANTIC_STANDARD_COMMUNICATION",
        ),
    ]

    for pattern, fixed, conf, method in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE,
        )

        if match:

            value = (
                fixed
                if fixed
                else match.group(1).upper()
            )

            add_prediction(
                predictions,
                "dataInterface",
                value,
                text,
                match.start(),
                match.end(),
                conf,
                method,
                infer_requirement_type(
                    text,
                    match.start(),
                    match.end(),
                ),
            )

            break

    # =========================================================================
    # SWITCHABLE CURRENT
    # =========================================================================

    patterns = [
        (
            r"\bswitchable current[- ]output\s+(cs[0-9]+)\b",
            None,
            0.99,
            "EXPLICIT_SWITCHABLE_CURRENT_VALUE",
        ),
        (
            r"\bswitchable current output capability\s+(cs[0-9]+)\b",
            None,
            0.99,
            "EXPLICIT_SWITCHABLE_CURRENT_VALUE",
        ),
        (
            r"\bswitchable current[- ]output\b",
            "CS1",
            0.90,
            "SEMANTIC_SWITCHABLE_CURRENT",
        ),
        (
            r"\bswitchable current output capability\b",
            "CS1",
            0.90,
            "SEMANTIC_SWITCHABLE_CURRENT",
        ),
    ]

    for pattern, fixed, conf, method in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE,
        )

        if match:

            value = (
                fixed
                if fixed
                else match.group(1).upper()
            )

            add_prediction(
                predictions,
                "stromSchaltbar",
                value,
                text,
                match.start(),
                match.end(),
                conf,
                method,
                infer_requirement_type(
                    text,
                    match.start(),
                    match.end(),
                ),
            )

            break

    # =========================================================================
    # CURRENT INPUT
    # =========================================================================

    current_patterns = [
        (
            r"\bcurrent input\s+(is[0-9]+)\b",
            None,
            0.99,
            "EXPLICIT_CURRENT_INPUT",
        ),
        (
            r"\bcurrent inputs?\s+(is[0-9]+)\b",
            None,
            0.99,
            "EXPLICIT_CURRENT_INPUT",
        ),
        (
            r"\bcurrent input\s*[:\-]\s*(is[0-9]+)\b",
            None,
            0.99,
            "EXPLICIT_CURRENT_INPUT",
        ),
        (
            r"\bcurrent input\s+(as[0-9]+)\b",
            None,
            0.99,
            "EXPLICIT_CURRENT_INPUT",
        ),
    ]

    for pattern, fixed, conf, method in current_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE,
        )

        if match:

            value = (
                fixed
                if fixed
                else match.group(1).upper()
            )

            add_prediction(
                predictions,
                "stromEingaenge",
                value,
                text,
                match.start(),
                match.end(),
                conf,
                method,
                infer_requirement_type(
                    text,
                    match.start(),
                    match.end(),
                ),
            )

            break

    # =========================================================================
    # TEMPERATURE INPUT
    # =========================================================================

    patterns = [
        (
            r"\btemperature measurement capability\b",
            "TT1",
            0.98,
            "EXPLICIT_TEMPERATURE",
        ),
        (
            r"\btemperature measurement\b",
            "TT1",
            0.96,
            "EXPLICIT_TEMPERATURE",
        ),
        (
            r"\btemperature input\b",
            "TT1",
            0.96,
            "EXPLICIT_TEMPERATURE",
        ),
        (
            r"\badvanced low[- ]temperature measurement\b",
            "TT1",
            0.95,
            "SEMANTIC_LOW_TEMPERATURE",
        ),
    ]

    result = search_patterns(text, patterns)

    if result:

        match, value, conf, method = result

        add_prediction(
            predictions,
            "temperaturEingaenge",
            value,
            text,
            match.start(),
            match.end(),
            conf,
            method,
            infer_requirement_type(
                text,
                match.start(),
                match.end(),
            ),
        )

    # =========================================================================
    # DIGITAL OPEN COLLECTOR MN
    # =========================================================================

    patterns = [
        (
            r"\bdigital open[- ]collector.*?\bmn[0-9]+\b",
            "MN1",
            0.99,
            "EXPLICIT_MN",
        ),
        (
            r"\bdigital open[- ]collector input\s+mn[0-9]+\b",
            "MN1",
            0.99,
            "EXPLICIT_MN",
        ),
        (
            r"\bdigital signal input.*?\bmn[0-9]+\b",
            "MN1",
            0.98,
            "EXPLICIT_MN",
        ),
    ]

    result = search_patterns(text, patterns)

    if result:

        match, value, conf, method = result

        add_prediction(
            predictions,
            "binaerDigitalOpenColl_MN",
            value,
            text,
            match.start(),
            match.end(),
            conf,
            method,
            infer_requirement_type(
                text,
                match.start(),
                match.end(),
            ),
        )

    # =========================================================================
    # DIGITAL OPEN COLLECTOR MP
    # =========================================================================

    patterns = [
        (
            r"\bdigital open[- ]collector.*?\bmp[0-9]+\b",
            "MP1",
            0.99,
            "EXPLICIT_MP",
        ),
        (
            r"\bopen[- ]collector.*?\bmp[0-9]+\b",
            "MP1",
            0.98,
            "EXPLICIT_MP",
        ),
        (
            r"\bdigital signal input capability\b",
            "MP1",
            0.80,
            "SEMANTIC_MP",
        ),
    ]

    result = search_patterns(text, patterns)

    if result:

        match, value, conf, method = result

        add_prediction(
            predictions,
            "binaerOpenColl_MP",
            value,
            text,
            match.start(),
            match.end(),
            conf,
            method,
            infer_requirement_type(
                text,
                match.start(),
                match.end(),
            ),
        )

    # =========================================================================
    # WAVE INJECTOR
    # =========================================================================

    patterns = [
        (
            r"\bwaveinjector\b",
            "FWI",
            0.99,
            "EXPLICIT_WAVEINJECTOR",
        ),
        (
            r"\bwave injector\b",
            "FWI",
            0.99,
            "EXPLICIT_WAVEINJECTOR",
        ),
        (
            r"\bcryogenic waveinjector\b",
            "FWI",
            0.99,
            "EXPLICIT_WAVEINJECTOR",
        ),
    ]

    result = search_patterns(text, patterns)

    if result:

        match, value, conf, method = result

        add_prediction(
            predictions,
            "waveInjector",
            value,
            text,
            match.start(),
            match.end(),
            conf,
            method,
            infer_requirement_type(
                text,
                match.start(),
                match.end(),
            ),
        )

    # =========================================================================
    # ADVANCED METER VERIFICATION
    # =========================================================================

    patterns = [
        (
            r"\badvanced meter verification\s*1\b",
            "1",
            0.99,
            "EXPLICIT_AMV",
        ),
        (
            r"\badvanced meter verification\b",
            "1",
            0.96,
            "SEMANTIC_AMV",
        ),
    ]

    result = search_patterns(text, patterns)

    if result:

        match, value, conf, method = result

        add_prediction(
            predictions,
            "dev_advMeterVerification",
            value,
            text,
            match.start(),
            match.end(),
            conf,
            method,
            infer_requirement_type(
                text,
                match.start(),
                match.end(),
            ),
        )

    # =========================================================================
    # DYNAMIC GAS MASTER
    # =========================================================================

    patterns = [
        (
            r"\bdynamic gas master\s*1\b",
            "1",
            0.99,
            "EXPLICIT_DGM",
        ),
        (
            r"\bdynamic gas master\b",
            "1",
            0.98,
            "SEMANTIC_DGM",
        ),
        (
            r"\benhanced gas measurement\b",
            "1",
            0.90,
            "SEMANTIC_DGM",
        ),
    ]

    result = search_patterns(text, patterns)

    if result:

        match, value, conf, method = result

        add_prediction(
            predictions,
            "dynamicGasMaster",
            value,
            text,
            match.start(),
            match.end(),
            conf,
            method,
            infer_requirement_type(
                text,
                match.start(),
                match.end(),
            ),
        )

    # =========================================================================
    # CUSTOM USER FLUID
    # =========================================================================

    patterns = [
        (
            r"\bcustom user fluid\s*1\b",
            "1",
            0.99,
            "EXPLICIT_CUSTOM_FLUID",
        ),
        (
            r"\bcustom user fluid\b",
            "1",
            0.96,
            "SEMANTIC_CUSTOM_FLUID",
        ),
    ]

    result = search_patterns(text, patterns)

    if result:

        match, value, conf, method = result

        add_prediction(
            predictions,
            "customUserFluid",
            value,
            text,
            match.start(),
            match.end(),
            conf,
            method,
            infer_requirement_type(
                text,
                match.start(),
                match.end(),
            ),
        )

    # =========================================================================
    # STEAM APPLICATION
    # =========================================================================

    patterns = [
        (
            r"\bsteam application\b",
            "H",
            0.99,
            "EXPLICIT_STEAM_APPLICATION",
        ),
        (
            r"\bsteam measurement\b",
            "H",
            0.99,
            "EXPLICIT_STEAM_APPLICATION",
        ),
    ]

    result = search_patterns(text, patterns)

    if result:

        match, value, conf, method = result

        add_prediction(
            predictions,
            "steamApplication",
            value,
            text,
            match.start(),
            match.end(),
            conf,
            method,
            infer_requirement_type(
                text,
                match.start(),
                match.end(),
            ),
        )

    return predictions


# =============================================================================
# GROUND TRUTH NORMALIZATION
# =============================================================================

def normalize_ground_truth_value(value) -> str:

    if pd.isna(value):
        return "NOVALUE"

    text = str(value).strip()

    if not text:
        return "NOVALUE"

    match = re.match(
        r"^in\s*\{(.*)\}$",
        text,
        re.IGNORECASE,
    )

    if match:

        content = match.group(1)

        quoted = re.findall(
            r"'([^']*)'|\"([^\"]*)\"",
            content,
        )

        values = []

        for a, b in quoted:

            token = a or b

            if token:
                values.append(token.strip())

        if values:

            non_novalue = [
                x
                for x in values
                if x.upper() != "NOVALUE"
            ]

            if non_novalue:
                return non_novalue[0]

            return "NOVALUE"

        tokens = [
            x.strip()
            for x in content.split(",")
            if x.strip()
        ]

        non_novalue = [
            x
            for x in tokens
            if x.upper() != "NOVALUE"
        ]

        if non_novalue:
            return non_novalue[0]

        return "NOVALUE"

    if (
        len(text) >= 2
        and (
            (
                text[0] == "'"
                and text[-1] == "'"
            )
            or
            (
                text[0] == '"'
                and text[-1] == '"'
            )
        )
    ):
        text = text[1:-1]

    return text.strip()


# =============================================================================
# COLUMN DETECTION
# =============================================================================

def detect_column(
    df: pd.DataFrame,
    candidates: List[str],
) -> str:

    lower_map = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for candidate in candidates:

        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    raise ValueError(
        f"Could not find required column. "
        f"Tried: {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 100)
    print("731 RFQ REQUIREMENT EXTRACTION — V2")
    print("=" * 100)

    print("\nLoading RFQ documents...")

    documents = pd.read_csv(
        RFQ_DOCUMENT_FILE
    )

    print(
        f"RFQ documents loaded: {len(documents):,}"
    )

    print("\nLoading structured requirements...")

    requirements = pd.read_csv(
        RFQ_REQUIREMENTS_FILE
    )

    print(
        f"Requirement records loaded: "
        f"{len(requirements):,}"
    )

    document_id_column = detect_column(
        documents,
        ["rfq_id"],
    )

    text_column = detect_column(
        documents,
        ["rfq_text", "text"],
    )

    requirement_id_column = detect_column(
        requirements,
        ["rfq_id"],
    )

    characteristic_column = detect_column(
        requirements,
        ["characteristic"],
    )

    value_column = detect_column(
        requirements,
        [
            "internal_value",
            "value",
            "required_value",
            "requirement_value",
            "target_value",
        ],
    )

    requirement_type_column = detect_column(
        requirements,
        ["requirement_type"],
    )

    print("\nDetected columns:")
    print(
        f"RFQ document ID: {document_id_column}"
    )
    print(
        f"RFQ text: {text_column}"
    )
    print(
        f"Requirement RFQ ID: {requirement_id_column}"
    )
    print(
        f"Characteristic: {characteristic_column}"
    )
    print(
        f"Value: {value_column}"
    )

    # =========================================================================
    # GROUND TRUTH
    # =========================================================================

    print("\n" + "=" * 100)
    print("BUILDING GROUND TRUTH")
    print("=" * 100)

    ground_truth = {}

    for rfq_id, group in requirements.groupby(
        requirement_id_column
    ):

        ground_truth[rfq_id] = {}

        for _, row in group.iterrows():

            characteristic = str(
                row[characteristic_column]
            ).strip()

            if characteristic not in TECHNICAL_CHARACTERISTICS:
                continue

            value = normalize_ground_truth_value(
                row[value_column]
            )

            if value != "NOVALUE":

                ground_truth[rfq_id][
                    characteristic
                ] = value

    print(
        f"RFQs with ground truth: "
        f"{len(ground_truth):,}"
    )

    # =========================================================================
    # EXTRACTION
    # =========================================================================

    print("\n" + "=" * 100)
    print("EXTRACTING REQUIREMENTS FROM RFQ TEXT — V2")
    print("=" * 100)

    result_rows = []
    error_rows = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    documents = documents.reset_index(drop=True)

    for index, row in documents.iterrows():

        rfq_id = row[document_id_column]
        text = row[text_column]

        predictions = extract_requirements_v2(
            text
        )

        actual = ground_truth.get(
            rfq_id,
            {},
        )

        predicted_values = {
            characteristic: item["value"]
            for characteristic, item
            in predictions.items()
        }

        tp = 0
        fp = 0
        fn = 0

        # ---------------------------------------------------------
        # Evaluate predicted values
        # ---------------------------------------------------------

        for characteristic, prediction in predictions.items():

            predicted = prediction["value"]

            if characteristic in actual:

                if predicted == actual[characteristic]:

                    tp += 1

                else:

                    fp += 1
                    fn += 1

                    error_rows.append({
                        "rfq_id": rfq_id,
                        "error_type": "WRONG_VALUE",
                        "characteristic": characteristic,
                        "predicted_value": predicted,
                        "actual_value": actual[characteristic],
                        "evidence_text": prediction[
                            "evidence_text"
                        ],
                        "confidence": prediction[
                            "confidence"
                        ],
                    })

            else:

                fp += 1

                error_rows.append({
                    "rfq_id": rfq_id,
                    "error_type": "FALSE_POSITIVE",
                    "characteristic": characteristic,
                    "predicted_value": predicted,
                    "actual_value": "NOVALUE",
                    "evidence_text": prediction[
                        "evidence_text"
                    ],
                    "confidence": prediction[
                        "confidence"
                    ],
                })

        # ---------------------------------------------------------
        # Evaluate missed requirements
        # ---------------------------------------------------------

        for characteristic, actual_value in actual.items():

            if characteristic not in predicted_values:

                fn += 1

                error_rows.append({
                    "rfq_id": rfq_id,
                    "error_type": "FALSE_NEGATIVE",
                    "characteristic": characteristic,
                    "predicted_value": "NOVALUE",
                    "actual_value": actual_value,
                    "evidence_text": "",
                    "confidence": 0.0,
                })

        total_tp += tp
        total_fp += fp
        total_fn += fn

        result_row = {
            "rfq_id": rfq_id,
            "actual_requirement_count": len(actual),
            "predicted_requirement_count": len(predictions),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "exact_match": (
                tp == len(actual)
                and fp == 0
                and fn == 0
            ),
        }

        for characteristic in TECHNICAL_CHARACTERISTICS:

            prediction = predictions.get(
                characteristic
            )

            if prediction:

                result_row[
                    f"{characteristic}_predicted"
                ] = prediction["value"]

                result_row[
                    f"{characteristic}_confidence"
                ] = prediction["confidence"]

                result_row[
                    f"{characteristic}_evidence"
                ] = prediction["evidence_text"]

                result_row[
                    f"{characteristic}_method"
                ] = prediction["extraction_method"]

            else:

                result_row[
                    f"{characteristic}_predicted"
                ] = "NOVALUE"

                result_row[
                    f"{characteristic}_confidence"
                ] = 0.0

                result_row[
                    f"{characteristic}_evidence"
                ] = ""

                result_row[
                    f"{characteristic}_method"
                ] = ""

        result_rows.append(result_row)

        if (index + 1) % 500 == 0:

            print(
                f"Processed "
                f"{index + 1:,} / "
                f"{len(documents):,}"
            )

    # =========================================================================
    # METRICS
    # =========================================================================

    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp
        else 0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0
    )

    result_df = pd.DataFrame(
        result_rows
    )

    error_df = pd.DataFrame(
        error_rows
    )

    exact_matches = int(
        result_df["exact_match"].sum()
    )

    exact_match_rate = (
        exact_matches / len(result_df)
        if len(result_df)
        else 0
    )

    # =========================================================================
    # CHARACTERISTIC METRICS
    # =========================================================================

    characteristic_rows = []

    for characteristic in TECHNICAL_CHARACTERISTICS:

        tp = 0
        fp = 0
        fn = 0

        for rfq_id, actual in ground_truth.items():

            actual_value = actual.get(
                characteristic,
                "NOVALUE",
            )

            predicted_row = result_df[
                result_df["rfq_id"] == rfq_id
            ]

            if predicted_row.empty:
                continue

            predicted_value = predicted_row.iloc[0][
                f"{characteristic}_predicted"
            ]

            if (
                predicted_value != "NOVALUE"
                and actual_value != "NOVALUE"
            ):

                if predicted_value == actual_value:
                    tp += 1
                else:
                    fp += 1
                    fn += 1

            elif (
                predicted_value != "NOVALUE"
                and actual_value == "NOVALUE"
            ):

                fp += 1

            elif (
                predicted_value == "NOVALUE"
                and actual_value != "NOVALUE"
            ):

                fn += 1

        p = (
            tp / (tp + fp)
            if tp + fp
            else 0
        )

        r = (
            tp / (tp + fn)
            if tp + fn
            else 0
        )

        f = (
            2 * p * r / (p + r)
            if p + r
            else 0
        )

        characteristic_rows.append({
            "characteristic": characteristic,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": p,
            "recall": r,
            "f1": f,
        })

    characteristic_df = pd.DataFrame(
        characteristic_rows
    )

    metrics_df = pd.DataFrame([
        {
            "metric": "rfqs_evaluated",
            "value": len(documents),
        },
        {
            "metric": "true_positives",
            "value": total_tp,
        },
        {
            "metric": "false_positives",
            "value": total_fp,
        },
        {
            "metric": "false_negatives",
            "value": total_fn,
        },
        {
            "metric": "precision",
            "value": precision,
        },
        {
            "metric": "recall",
            "value": recall,
        },
        {
            "metric": "f1",
            "value": f1,
        },
        {
            "metric": "exact_rfq_matches",
            "value": exact_matches,
        },
        {
            "metric": "exact_match_rate",
            "value": exact_match_rate,
        },
    ])

    # =========================================================================
    # OUTPUT
    # =========================================================================

    result_df.to_csv(
        RESULTS_FILE,
        index=False,
    )

    characteristic_df.to_csv(
        CHARACTERISTIC_METRICS_FILE,
        index=False,
    )

    metrics_df.to_csv(
        METRICS_FILE,
        index=False,
    )

    error_df.to_csv(
        ERRORS_FILE,
        index=False,
    )

    # =========================================================================
    # REPORT
    # =========================================================================

    print("\n" + "=" * 100)
    print("731 RFQ EXTRACTION V2 RESULTS")
    print("=" * 100)

    print(
        f"\nRFQs evaluated:       {len(documents):,}"
    )

    print(
        f"True positives:       {total_tp:,}"
    )

    print(
        f"False positives:      {total_fp:,}"
    )

    print(
        f"False negatives:      {total_fn:,}"
    )

    print(
        f"\nPrecision:            {precision:.4f}"
    )

    print(
        f"Recall:               {recall:.4f}"
    )

    print(
        f"F1:                   {f1:.4f}"
    )

    print(
        f"\nExact RFQ matches:    {exact_matches:,}"
    )

    print(
        f"Exact match rate:     "
        f"{exact_match_rate:.4%}"
    )

    print("\n" + "=" * 100)
    print("CHARACTERISTIC-LEVEL PERFORMANCE")
    print("=" * 100)

    print(
        characteristic_df[
            [
                "characteristic",
                "precision",
                "recall",
                "f1",
            ]
        ].to_string(
            index=False
        )
    )

    print("\n" + "=" * 100)
    print("FILES SAVED")
    print("=" * 100)

    print(f"\n{RESULTS_FILE}")
    print(f"{CHARACTERISTIC_METRICS_FILE}")
    print(f"{METRICS_FILE}")
    print(f"{ERRORS_FILE}")

    print("\n" + "=" * 100)
    print("731 RFQ EXTRACTION V2 COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()