#!/usr/bin/env python3

"""
731 RFQ REQUIREMENT EXTRACTION — V1 BASELINE
=============================================

Purpose
-------
Evaluate a transparent rule-based NLP baseline for extracting
technical 731 requirements from customer-facing RFQ text.

Input
-----
data/synthetic/rfqs/731_rfq_documents_v2.csv
data/synthetic/rfqs/731_rfq_requirements_v2.csv

Output
------
data/processed/731_rfq_extraction_results_v1.csv
data/processed/731_rfq_characteristic_metrics_v1.csv
data/processed/731_rfq_extraction_metrics_v1.csv
data/processed/731_rfq_extraction_errors_v1.csv

IMPORTANT
---------
The extractor does NOT use the hidden RFQ ground truth while
performing extraction.

Ground truth is used only AFTER extraction for evaluation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_FILE = (
    OUTPUT_DIR
    / "731_rfq_extraction_results_v1.csv"
)

CHARACTERISTIC_METRICS_FILE = (
    OUTPUT_DIR
    / "731_rfq_characteristic_metrics_v1.csv"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "731_rfq_extraction_metrics_v1.csv"
)

ERRORS_FILE = (
    OUTPUT_DIR
    / "731_rfq_extraction_errors_v1.csv"
)


# =============================================================================
# TECHNICAL CHARACTERISTICS
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

def normalize_text(text: str) -> str:

    if pd.isna(text):
        return ""

    text = str(text)

    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip().lower()


# =============================================================================
# VALUE EXTRACTION HELPERS
# =============================================================================

def add_prediction(
    predictions: Dict[str, str],
    characteristic: str,
    value: str,
):
    """
    Add a prediction without overwriting an existing value.
    """

    if (
        characteristic not in predictions
        or predictions[characteristic] == "NOVALUE"
    ):
        predictions[characteristic] = value


def find_value(
    text: str,
    patterns: List[Tuple[str, str]],
):
    """
    Return first matching value.
    """

    for pattern, value in patterns:

        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):
            return value

    return None


# =============================================================================
# REQUIREMENT EXTRACTION
# =============================================================================

def extract_requirements(
    text: str,
) -> Dict[str, str]:

    text = normalize_text(text)

    predictions = {}

    # =========================================================================
    # DEV CATEGORY
    # =========================================================================

    value = find_value(
        text,
        [
            (
                r"general[- ]purpose configuration",
                "G",
            ),
            (
                r"gas measurement",
                "G",
            ),
            (
                r"flow measurement",
                "F",
            ),
            (
                r"liquid measurement",
                "F",
            ),
            (
                r"steam measurement",
                "S",
            ),
        ],
    )

    if value:
        add_prediction(
            predictions,
            "devCategory",
            value,
        )

    # =========================================================================
    # CHARACTERISTIC
    # =========================================================================

    value = find_value(
        text,
        [
            (
                r"general[- ]purpose configuration",
                "GP",
            ),
            (
                r"stainless[- ]steel configuration",
                "ST",
            ),
            (
                r"temperature measurement",
                "TE",
            ),
            (
                r"calibration",
                "CA",
            ),
            (
                r"pressure measurement",
                "PW",
            ),
            (
                r"wide[- ]range",
                "WD",
            ),
            (
                r"verification",
                "VG",
            ),
        ],
    )

    if value:
        add_prediction(
            predictions,
            "characteristic",
            value,
        )

    # =========================================================================
    # HOUSING
    # =========================================================================

    value = find_value(
        text,
        [
            (
                r"stainless[- ]steel housing",
                "ST",
            ),
            (
                r"stainless[- ]steel enclosure",
                "ST",
            ),
            (
                r"industrial enclosure",
                "ST",
            ),
            (
                r"robust industrial enclosure",
                "ST",
            ),
            (
                r"aluminium housing",
                "AL",
            ),
            (
                r"aluminum housing",
                "AL",
            ),
        ],
    )

    if value:
        add_prediction(
            predictions,
            "housing",
            value,
        )

    # =========================================================================
    # POWER SUPPLY
    # =========================================================================

    value = find_value(
        text,
        [
            (
                r"alternative electrical supply",
                "4",
            ),
            (
                r"alternative power supply",
                "4",
            ),
            (
                r"alternative electrical power",
                "4",
            ),
            (
                r"standard power supply",
                "1",
            ),
        ],
    )

    if value:
        add_prediction(
            predictions,
            "powerSupply",
            value,
        )

    # =========================================================================
    # NUMBER OF CHANNELS
    # =========================================================================

    match = re.search(
        r"(?:single|one)\s+measurement\s+channel",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        add_prediction(
            predictions,
            "numberOfChannels",
            "1",
        )

    match = re.search(
        r"(?:dual|two|double)\s+measurement\s+channels?",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        add_prediction(
            predictions,
            "numberOfChannels",
            "2",
        )

    match = re.search(
        r"number of measurement channels?\s*(?:is|:)?\s*(1|2)",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        add_prediction(
            predictions,
            "numberOfChannels",
            match.group(1),
        )

    # =========================================================================
    # EXPLOSION APPROVAL
    # =========================================================================

    value = find_value(
        text,
        [
            (
                r"explosion protection\s+a",
                "A",
            ),
            (
                r"hazardous[- ]area.*approval\s+a",
                "A",
            ),
            (
                r"explosion approval\s+a",
                "A",
            ),
            (
                r"explosion protection\s+e",
                "E",
            ),
            (
                r"explosion approval\s+e",
                "E",
            ),
            (
                r"does not require hazardous[- ]area approval",
                "N",
            ),
            (
                r"no hazardous[- ]area approval",
                "N",
            ),
            (
                r"non[- ]hazardous",
                "N",
            ),
        ],
    )

    if value:
        add_prediction(
            predictions,
            "explosionApproval",
            value,
        )

    # =========================================================================
    # PROTECTION AREA
    # =========================================================================

    value = find_value(
        text,
        [
            (
                r"protection area\s+([0-9]+)",
                None,
            ),
        ],
    )

    match = re.search(
        r"protection area\s+([0-9]+)",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        add_prediction(
            predictions,
            "protectionArea",
            match.group(1),
        )

    # =========================================================================
    # CERTIFICATION
    # =========================================================================

    certification_patterns = [
        (r"certification.*\bmn\b", "MN"),
        (r"certification.*\bnn\b", "NN"),
        (r"certification.*\bex\b", "EX"),
        (r"certification.*\b[a-z]{2}\b", None),
    ]

    for pattern, value in certification_patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:

            if value:
                add_prediction(
                    predictions,
                    "certification",
                    value,
                )

                break

            # Extract last two-letter certification token.
            tokens = re.findall(
                r"\b[A-Za-z]{2}\b",
                match.group(0),
            )

            if tokens:

                candidate = tokens[-1].upper()

                if candidate not in {
                    "IS",
                    "OF",
                    "IF",
                    "TO",
                    "AS",
                }:
                    add_prediction(
                        predictions,
                        "certification",
                        candidate,
                    )

                    break

    # =========================================================================
    # DATA INTERFACE
    # =========================================================================

    data_interface_patterns = [
        (r"data interface\s+([A-Za-z0-9]+)", None),
        (r"communication interface\s+([A-Za-z0-9]+)", None),
        (r"high[- ]speed communication", "HS"),
        (r"standard communication", "NN"),
        (r"communication interface is\s+([A-Za-z0-9]+)", None),
    ]

    for pattern, fixed_value in data_interface_patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:

            value = (
                fixed_value
                if fixed_value
                else match.group(1).upper()
            )

            add_prediction(
                predictions,
                "dataInterface",
                value,
            )

            break

    # =========================================================================
    # SWITCHABLE CURRENT
    # =========================================================================

    match = re.search(
        r"(?:switchable current[- ]output|switchable current output capability)"
        r"(?:\s+([A-Za-z0-9]+))?",
        text,
        flags=re.IGNORECASE,
    )

    if match:

        value = (
            match.group(1).upper()
            if match.group(1)
            else "CS1"
        )

        add_prediction(
            predictions,
            "stromSchaltbar",
            value,
        )

    # =========================================================================
    # CURRENT INPUT
    # =========================================================================

    match = re.search(
        r"(?:current input|current inputs?)\s+([A-Za-z0-9]+)",
        text,
        flags=re.IGNORECASE,
    )

    if match:

        add_prediction(
            predictions,
            "stromEingaenge",
            match.group(1).upper(),
        )

    # =========================================================================
    # TEMPERATURE INPUT
    # =========================================================================

    value = find_value(
        text,
        [
            (
                r"temperature measurement capability",
                "TT1",
            ),
            (
                r"temperature measurement",
                "TT1",
            ),
            (
                r"temperature input",
                "TT1",
            ),
            (
                r"advanced low[- ]temperature measurement",
                "TT1",
            ),
        ],
    )

    if value:
        add_prediction(
            predictions,
            "temperaturEingaenge",
            value,
        )

    # =========================================================================
    # DIGITAL OPEN COLLECTOR
    # =========================================================================

    value = find_value(
        text,
        [
            (
                r"digital open[- ]collector input",
                "MP1",
            ),
            (
                r"digital signal input capability",
                "MP1",
            ),
            (
                r"open[- ]collector",
                "MP1",
            ),
        ],
    )

    if value:
        add_prediction(
            predictions,
            "binaerOpenColl_MP",
            value,
        )

    # =========================================================================
    # DIGITAL OPEN COLLECTOR MN
    # =========================================================================

    value = find_value(
        text,
        [
            (
                r"digital signal input capability",
                "MN1",
            ),
            (
                r"digital open[- ]collector.*mn1",
                "MN1",
            ),
        ],
    )

    if value:
        add_prediction(
            predictions,
            "binaerDigitalOpenColl_MN",
            value,
        )

    # =========================================================================
    # WAVE INJECTOR
    # =========================================================================

    value = find_value(
        text,
        [
            (
                r"waveinjector cryogenic",
                "FWI",
            ),
            (
                r"waveinjector",
                "FWI",
            ),
            (
                r"wave injector",
                "FWI",
            ),
        ],
    )

    if value:
        add_prediction(
            predictions,
            "waveInjector",
            value,
        )

    # =========================================================================
    # ADVANCED METER VERIFICATION
    # =========================================================================

    value = find_value(
        text,
        [
            (
                r"advanced meter verification\s*1",
                "1",
            ),
            (
                r"advanced meter verification",
                "1",
            ),
        ],
    )

    if value:
        add_prediction(
            predictions,
            "dev_advMeterVerification",
            value,
        )

    # =========================================================================
    # DYNAMIC GAS MASTER
    # =========================================================================

    value = find_value(
        text,
        [
            (
                r"dynamic gas master\s*1",
                "1",
            ),
            (
                r"enhanced gas measurement",
                "1",
            ),
            (
                r"dynamic gas master",
                "1",
            ),
        ],
    )

    if value:
        add_prediction(
            predictions,
            "dynamicGasMaster",
            value,
        )

    # =========================================================================
    # CUSTOM USER FLUID
    # =========================================================================

    value = find_value(
        text,
        [
            (
                r"custom user fluid\s*1",
                "1",
            ),
            (
                r"custom user fluid",
                "1",
            ),
        ],
    )

    if value:
        add_prediction(
            predictions,
            "customUserFluid",
            value,
        )

    # =========================================================================
    # STEAM APPLICATION
    # =========================================================================

    value = find_value(
        text,
        [
            (
                r"steam application",
                "H",
            ),
            (
                r"steam measurement",
                "H",
            ),
        ],
    )

    if value:
        add_prediction(
            predictions,
            "steamApplication",
            value,
        )

    return predictions


# =============================================================================
# COLUMN DETECTION
# =============================================================================

def detect_column(
    df: pd.DataFrame,
    candidates: List[str],
    required: bool = True,
) -> str | None:

    lower_map = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    if required:
        raise ValueError(
            f"Could not find required column. "
            f"Tried: {candidates}. "
            f"Available columns: {list(df.columns)}"
        )

    return None


# =============================================================================
# GROUND-TRUTH NORMALIZATION
# =============================================================================

def normalize_ground_truth_value(value) -> str:

    if pd.isna(value):
        return "NOVALUE"

    text = str(value).strip()

    if not text:
        return "NOVALUE"

    # Handle set expressions.
    match = re.match(
        r"^in\s*\{(.*)\}$",
        text,
        flags=re.IGNORECASE,
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
                values.append(
                    token.strip()
                )

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
# MAIN
# =============================================================================

def main():

    print("=" * 100)
    print("731 RFQ REQUIREMENT EXTRACTION — V1 BASELINE")
    print("=" * 100)

    # =========================================================================
    # LOAD
    # =========================================================================

    print()
    print("Loading RFQ documents...")

    documents = pd.read_csv(
        RFQ_DOCUMENT_FILE
    )

    print(
        f"RFQ documents loaded: {len(documents):,}"
    )

    print()
    print("Document columns:")
    print(
        list(documents.columns)
    )

    print()
    print("Loading structured requirements...")

    requirements = pd.read_csv(
        RFQ_REQUIREMENTS_FILE
    )

    print(
        f"Requirement records loaded: "
        f"{len(requirements):,}"
    )

    print()
    print("Requirement columns:")
    print(
        list(requirements.columns)
    )

    # =========================================================================
    # IDENTIFY COLUMNS
    # =========================================================================

    rfq_id_document_column = detect_column(
        documents,
        [
            "rfq_id",
            "RFQ_ID",
            "RFQ Reference",
            "rfq_reference",
        ],
    )

    text_column = detect_column(
        documents,
        [
            "rfq_text",
            "document_text",
            "text",
            "rfq_document",
            "document",
            "content",
        ],
    )

    rfq_id_requirement_column = detect_column(
        requirements,
        [
            "rfq_id",
            "RFQ_ID",
            "RFQ Reference",
            "rfq_reference",
        ],
    )

    characteristic_column = detect_column(
        requirements,
        [
            "characteristic",
            "technical_characteristic",
            "characteristic_name",
        ],
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
        [
            "requirement_type",
            "requirementType",
            "type",
        ],
        required=False,
    )

    implicit_column = detect_column(
        requirements,
        [
            "implicit_requirement",
            "implicitRequirement",
            "is_implicit",
        ],
        required=False,
    )

    print()
    print("Detected columns:")
    print(
        "RFQ document ID:",
        rfq_id_document_column,
    )
    print(
        "RFQ text:",
        text_column,
    )
    print(
        "Requirement RFQ ID:",
        rfq_id_requirement_column,
    )
    print(
        "Characteristic:",
        characteristic_column,
    )
    print(
        "Value:",
        value_column,
    )

    # =========================================================================
    # BUILD GROUND TRUTH
    # =========================================================================

    print()
    print("=" * 100)
    print("BUILDING GROUND TRUTH")
    print("=" * 100)

    ground_truth = {}

    for rfq_id, group in requirements.groupby(
        rfq_id_requirement_column
    ):

        gt = {}

        for _, row in group.iterrows():

            characteristic = str(
                row[characteristic_column]
            ).strip()

            if characteristic not in TECHNICAL_CHARACTERISTICS:
                continue

            value = normalize_ground_truth_value(
                row[value_column]
            )

            # Ignore NOVALUE as an explicit customer requirement.
            if value == "NOVALUE":
                continue

            gt[characteristic] = value

        ground_truth[rfq_id] = gt

    print(
        f"RFQs with ground truth: "
        f"{len(ground_truth):,}"
    )

    # =========================================================================
    # EXTRACTION
    # =========================================================================

    print()
    print("=" * 100)
    print("EXTRACTING REQUIREMENTS FROM RFQ TEXT")
    print("=" * 100)

    result_records = []
    error_records = []

    total = len(documents)

    for index, row in documents.iterrows():

        rfq_id = row[
            rfq_id_document_column
        ]

        text = row[
            text_column
        ]

        predictions = extract_requirements(
            text
        )

        truth = ground_truth.get(
            rfq_id,
            {},
        )

        all_characteristics = set(
            TECHNICAL_CHARACTERISTICS
        )

        all_characteristics.update(
            predictions.keys()
        )

        all_characteristics.update(
            truth.keys()
        )

        tp = 0
        fp = 0
        fn = 0

        for characteristic in sorted(
            all_characteristics
        ):

            predicted = predictions.get(
                characteristic,
                "NOVALUE",
            )

            actual = truth.get(
                characteristic,
                "NOVALUE",
            )

            predicted_present = (
                predicted != "NOVALUE"
            )

            actual_present = (
                actual != "NOVALUE"
            )

            if (
                predicted_present
                and actual_present
                and predicted == actual
            ):
                tp += 1

            elif (
                predicted_present
                and (
                    not actual_present
                    or predicted != actual
                )
            ):
                fp += 1

                error_records.append(
                    {
                        "rfq_id": rfq_id,
                        "characteristic": characteristic,
                        "error_type": "FALSE_POSITIVE_OR_WRONG_VALUE",
                        "predicted_value": predicted,
                        "actual_value": actual,
                    }
                )

            elif (
                not predicted_present
                and actual_present
            ):
                fn += 1

                error_records.append(
                    {
                        "rfq_id": rfq_id,
                        "characteristic": characteristic,
                        "error_type": "MISSED_REQUIREMENT",
                        "predicted_value": "NOVALUE",
                        "actual_value": actual,
                    }
                )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        result_records.append(
            {
                "rfq_id": rfq_id,
                "predicted_requirement_count": len(
                    predictions
                ),
                "actual_requirement_count": len(
                    truth
                ),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "predicted_requirements": str(
                    predictions
                ),
                "actual_requirements": str(
                    truth
                ),
            }
        )

        if (
            (index + 1) % 500 == 0
            or index + 1 == total
        ):
            print(
                f"Processed "
                f"{index + 1:,} / {total:,}"
            )

    results = pd.DataFrame(
        result_records
    )

    errors = pd.DataFrame(
        error_records
    )

    # =========================================================================
    # GLOBAL METRICS
    # =========================================================================

    total_tp = results[
        "true_positives"
    ].sum()

    total_fp = results[
        "false_positives"
    ].sum()

    total_fn = results[
        "false_negatives"
    ].sum()

    precision = (
        total_tp
        / (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0.0
    )

    recall = (
        total_tp
        / (total_tp + total_fn)
        if total_tp + total_fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    exact_rfqs = (
        (
            results["false_positives"] == 0
        )
        &
        (
            results["false_negatives"] == 0
        )
    ).sum()

    exact_match_rate = (
        exact_rfqs
        / len(results)
        if len(results)
        else 0.0
    )

    # =========================================================================
    # CHARACTERISTIC METRICS
    # =========================================================================

    characteristic_records = []

    for characteristic in TECHNICAL_CHARACTERISTICS:

        tp = 0
        fp = 0
        fn = 0

        for rfq_id, truth in ground_truth.items():

            actual = truth.get(
                characteristic,
                "NOVALUE",
            )

            document_rows = documents[
                documents[
                    rfq_id_document_column
                ] == rfq_id
            ]

            if document_rows.empty:
                continue

            text = document_rows.iloc[0][
                text_column
            ]

            prediction = extract_requirements(
                text
            )

            predicted = prediction.get(
                characteristic,
                "NOVALUE",
            )

            actual_present = (
                actual != "NOVALUE"
            )

            predicted_present = (
                predicted != "NOVALUE"
            )

            if (
                actual_present
                and predicted_present
                and actual == predicted
            ):
                tp += 1

            elif predicted_present:
                fp += 1

            elif actual_present:
                fn += 1

        p = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0.0
        )

        r = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0.0
        )

        f = (
            2 * p * r / (p + r)
            if p + r > 0
            else 0.0
        )

        characteristic_records.append(
            {
                "characteristic": characteristic,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": p,
                "recall": r,
                "f1": f,
            }
        )

    characteristic_metrics = pd.DataFrame(
        characteristic_records
    )

    # =========================================================================
    # SAVE
    # =========================================================================

    results.to_csv(
        RESULTS_FILE,
        index=False,
    )

    characteristic_metrics.to_csv(
        CHARACTERISTIC_METRICS_FILE,
        index=False,
    )

    errors.to_csv(
        ERRORS_FILE,
        index=False,
    )

    metrics = pd.DataFrame(
        [
            {
                "metric": "RFQ_count",
                "value": len(results),
            },
            {
                "metric": "True_positives",
                "value": total_tp,
            },
            {
                "metric": "False_positives",
                "value": total_fp,
            },
            {
                "metric": "False_negatives",
                "value": total_fn,
            },
            {
                "metric": "Precision",
                "value": precision,
            },
            {
                "metric": "Recall",
                "value": recall,
            },
            {
                "metric": "F1",
                "value": f1,
            },
            {
                "metric": "Exact_RFQ_match_count",
                "value": exact_rfqs,
            },
            {
                "metric": "Exact_RFQ_match_rate",
                "value": exact_match_rate,
            },
        ]
    )

    metrics.to_csv(
        METRICS_FILE,
        index=False,
    )

    # =========================================================================
    # REPORT
    # =========================================================================

    print()
    print("=" * 100)
    print("731 RFQ EXTRACTION BASELINE RESULTS")
    print("=" * 100)

    print()
    print(
        f"RFQs evaluated:       {len(results):,}"
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

    print()
    print(
        f"Precision:            {precision:.4f}"
    )

    print(
        f"Recall:               {recall:.4f}"
    )

    print(
        f"F1:                   {f1:.4f}"
    )

    print()
    print(
        f"Exact RFQ matches:    {exact_rfqs:,}"
    )

    print(
        f"Exact match rate:     {exact_match_rate:.4%}"
    )

    print()
    print("=" * 100)
    print("CHARACTERISTIC-LEVEL PERFORMANCE")
    print("=" * 100)

    print(
        characteristic_metrics[
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

    print()
    print("=" * 100)
    print("FILES SAVED")
    print("=" * 100)

    print()
    print(
        RESULTS_FILE
    )

    print(
        CHARACTERISTIC_METRICS_FILE
    )

    print(
        METRICS_FILE
    )

    print(
        ERRORS_FILE
    )

    print()
    print("=" * 100)
    print("RFQ EXTRACTION BASELINE COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()