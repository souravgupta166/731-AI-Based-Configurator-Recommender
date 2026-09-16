#!/usr/bin/env python3

"""
731 RFQ REQUIREMENT EXTRACTION — V4
===================================

V4 improvements over V3
------------------------
1. Characteristic-specific confidence thresholds.
2. Sentence-level evidence filtering.
3. Value-specific phrase validation.
4. Suppression of weak contextual/metadata matches.
5. Deduplication of predictions.
6. Evaluation against the structured V2 ground truth.
7. Saves prediction, metrics and error-analysis files.

The V4 extractor uses the V3 extraction output as its candidate
evidence layer and applies a stricter decision layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Set, Tuple

import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "731_rfq_extraction_results_v3.csv"
)

GROUND_TRUTH_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "rfqs"
    / "731_rfq_requirements_v2.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

RESULT_FILE = OUTPUT_DIR / "731_rfq_extraction_results_v4.csv"
METRICS_FILE = OUTPUT_DIR / "731_rfq_extraction_metrics_v4.csv"
CHAR_METRICS_FILE = (
    OUTPUT_DIR / "731_rfq_characteristic_metrics_v4.csv"
)
ERROR_FILE = OUTPUT_DIR / "731_rfq_extraction_errors_v4.csv"


# =============================================================================
# CHARACTERISTIC-SPECIFIC THRESHOLDS
# =============================================================================

CHARACTERISTIC_THRESHOLDS: Dict[str, float] = {

    "housing": 0.20,

    "protectionArea": 1.30,

    "customUserFluid": 1.30,

    "stromSchaltbar": 1.20,

    "numberOfChannels": 1.10,

    "powerSupply": 1.50,

    "stromEingaenge": 1.50,

    "dev_advMeterVerification": 1.50,

    "characteristic": 1.30,

    "binaerDigitalOpenColl_MN": 1.40,

    "waveInjector": 1.20,

    "dynamicGasMaster": 1.10,

    "temperaturEingaenge": 1.20,

    "binaerOpenColl_MP": 1.20,

    "devCategory": 0.90,

    "explosionApproval": 1.50,

    "dataInterface": 1.30,

    "certification": 0.30,

    "steamApplication": 0.20,
}


# =============================================================================
# CHARACTERISTICS REQUIRING STRONGER CONTEXT
# =============================================================================

CONTEXT_SENSITIVE = {
    "devCategory",
    "characteristic",
    "certification",
    "explosionApproval",
    "dataInterface",
    "protectionArea",
    "steamApplication",
    "waveInjector",
}


# =============================================================================
# GENERIC / NON-TECHNICAL PHRASES
# =============================================================================

GENERIC_PHRASES = {
    "configuration",
    "requirements",
    "technical requirements",
    "technical and application requirements",
    "following technical requirements",
    "application requirements",
    "suitable measurement solution",
    "measurement solution",
    "quotation",
    "quotation based on",
    "following",
    "standard",
    "appropriate",
    "required",
    "requirement",
    "system",
    "equipment",
    "unit",
    "device",
}


# =============================================================================
# HELPERS
# =============================================================================

def normalize_text(value) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def normalize_value(value) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def threshold_for(characteristic: str) -> float:
    """
    Return characteristic-specific threshold.
    """

    return CHARACTERISTIC_THRESHOLDS.get(
        characteristic,
        1.25,
    )


# =============================================================================
# CONTEXT FILTER
# =============================================================================

def passes_context_filter(row: pd.Series) -> bool:
    """
    Reject evidence that is likely to be generic contextual language.

    This is deliberately conservative. V4 should primarily remove
    obvious false positives without sacrificing strong technical evidence.
    """

    characteristic = str(
        row.get("characteristic", "")
    )

    characteristic_phrase = normalize_text(
        row.get("characteristic_phrase", "")
    )

    value_phrase = normalize_text(
        row.get("value_phrase", "")
    )

    sentence = normalize_text(
        row.get("source_sentence", "")
    )

    # ------------------------------------------------------------
    # Completely generic characteristic phrases
    # ------------------------------------------------------------

    if (
        characteristic_phrase in GENERIC_PHRASES
        and characteristic in CONTEXT_SENSITIVE
    ):
        return False

    # ------------------------------------------------------------
    # Generic value phrases
    # ------------------------------------------------------------

    if value_phrase in {
        "",
        "required",
        "appropriate",
        "suitable",
        "standard",
        "option",
        "capability",
        "support",
    }:
        return False

    # ------------------------------------------------------------
    # Avoid generic quotation / introduction sentences
    # ------------------------------------------------------------

    introduction_patterns = [
        "please provide a quotation",
        "please provide your best",
        "quotation based on the following",
        "following technical and application requirements",
        "receive a quotation for a suitable",
        "please provide a suitable",
    ]

    if any(
        pattern in sentence
        for pattern in introduction_patterns
    ):
        if characteristic in CONTEXT_SENSITIVE:
            return False

    return True


# =============================================================================
# VALUE-SPECIFIC FILTER
# =============================================================================

def passes_value_filter(row: pd.Series) -> bool:
    """
    Check whether the candidate has meaningful value-level evidence.
    """

    characteristic = str(
        row.get("characteristic", "")
    )

    value = normalize_value(
        row.get("internal_value", "")
    )

    value_phrase = normalize_text(
        row.get("value_phrase", "")
    )

    characteristic_phrase = normalize_text(
        row.get("characteristic_phrase", "")
    )

    # No value evidence.
    if not value_phrase:
        return False

    # ------------------------------------------------------------
    # Generic characteristics
    # ------------------------------------------------------------

    if characteristic in {
        "devCategory",
        "characteristic",
    }:

        if value in {"g", "gp", "f"}:

            technical_terms = {
                "gas",
                "liquid",
                "flow",
                "measurement",
                "general-purpose",
                "general purpose",
                "configuration",
            }

            if not any(
                term in value_phrase
                for term in technical_terms
            ):
                return False

    # ------------------------------------------------------------
    # Certification
    # ------------------------------------------------------------

    if characteristic == "certification":

        if value in {"nn", "mn", "mp"}:

            meaningful = {
                "certification",
                "certified",
                "standard",
                "approval",
            }

            if not any(
                term in value_phrase
                for term in meaningful
            ):
                return False

    # ------------------------------------------------------------
    # Data interface
    # ------------------------------------------------------------

    if characteristic == "dataInterface":

        meaningful = {
            "communication",
            "interface",
            "communication interface",
            "communication protocol",
            "high-speed communication",
            "high speed communication",
            "data interface",
        }

        if not any(
            term in value_phrase
            for term in meaningful
        ):
            return False

    # ------------------------------------------------------------
    # Explosion approval
    # ------------------------------------------------------------

    if characteristic == "explosionApproval":

        meaningful = {
            "explosion",
            "hazardous",
            "hazardous-area",
            "hazardous area",
            "explosive",
            "approval",
        }

        if not any(
            term in value_phrase
            for term in meaningful
        ):
            return False

    # ------------------------------------------------------------
    # Protection area
    # ------------------------------------------------------------

    if characteristic == "protectionArea":

        meaningful = {
            "area",
            "hazardous area",
            "hazardous",
            "protection area",
        }

        if not any(
            term in value_phrase
            for term in meaningful
        ):
            return False

    # ------------------------------------------------------------
    # Steam application
    # ------------------------------------------------------------

    if characteristic == "steamApplication":

        meaningful = {
            "steam",
            "steam application",
            "steam measurement",
            "steam service",
        }

        if not any(
            term in value_phrase
            for term in meaningful
        ):
            return False

    return True


# =============================================================================
# LOAD DATA
# =============================================================================

def load_data():

    print("=" * 100)
    print("731 RFQ REQUIREMENT EXTRACTION — V4")
    print("=" * 100)

    print()
    print("Loading V3 extraction candidates...")

    predictions = pd.read_csv(
        PREDICTION_FILE
    )

    print(
        f"V3 candidate rows: {len(predictions):,}"
    )

    print()
    print("Loading structured ground truth...")

    ground_truth = pd.read_csv(
        GROUND_TRUTH_FILE
    )

    print(
        f"Ground-truth requirement rows: "
        f"{len(ground_truth):,}"
    )

    return predictions, ground_truth


# =============================================================================
# BUILD GROUND TRUTH
# =============================================================================

def build_ground_truth(
    ground_truth: pd.DataFrame,
) -> Set[Tuple[str, str, str]]:

    required = [
        "rfq_id",
        "characteristic",
        "internal_value",
    ]

    missing = [
        c for c in required
        if c not in ground_truth.columns
    ]

    if missing:
        raise ValueError(
            f"Ground truth missing columns: {missing}"
        )

    return set(
        zip(
            ground_truth["rfq_id"],
            ground_truth["characteristic"],
            ground_truth["internal_value"],
        )
    )


# =============================================================================
# APPLY V4 FILTER
# =============================================================================

def extract_v4(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("=" * 100)
    print("APPLYING V4 DECISION LAYER")
    print("=" * 100)

    accepted = []

    counters = {
        "below_threshold": 0,
        "context_rejected": 0,
        "value_rejected": 0,
        "accepted": 0,
    }

    for _, row in predictions.iterrows():

        characteristic = str(
            row["characteristic"]
        )

        confidence = float(
            row["confidence"]
        )

        threshold = threshold_for(
            characteristic
        )

        # --------------------------------------------------------
        # Threshold
        # --------------------------------------------------------

        if confidence < threshold:

            counters["below_threshold"] += 1

            continue

        # --------------------------------------------------------
        # Context
        # --------------------------------------------------------

        if not passes_context_filter(row):

            counters["context_rejected"] += 1

            continue

        # --------------------------------------------------------
        # Value
        # --------------------------------------------------------

        if not passes_value_filter(row):

            counters["value_rejected"] += 1

            continue

        record = row.to_dict()

        record["v4_threshold"] = threshold
        record["v4_decision"] = "ACCEPT"

        accepted.append(record)

        counters["accepted"] += 1

    result = pd.DataFrame(
        accepted
    )

    # ------------------------------------------------------------
    # Deduplicate
    # ------------------------------------------------------------

    if len(result):

        result = (
            result
            .sort_values(
                [
                    "rfq_id",
                    "characteristic",
                    "internal_value",
                    "confidence",
                ],
                ascending=[
                    True,
                    True,
                    True,
                    False,
                ],
            )
            .drop_duplicates(
                subset=[
                    "rfq_id",
                    "characteristic",
                    "internal_value",
                ],
                keep="first",
            )
            .reset_index(drop=True)
        )

    print()
    print("V4 filtering statistics:")

    for key, value in counters.items():

        print(
            f"{key:25s}: {value:,}"
        )

    print()
    print(
        f"Final unique predictions: "
        f"{len(result):,}"
    )

    return result


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(
    predictions: pd.DataFrame,
    truth: Set[Tuple[str, str, str]],
):

    predicted = set(
        zip(
            predictions["rfq_id"],
            predictions["characteristic"],
            predictions["internal_value"],
        )
    )

    tp = len(
        predicted & truth
    )

    fp = len(
        predicted - truth
    )

    fn = len(
        truth - predicted
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0
    )

    print()
    print("=" * 100)
    print("731 RFQ EXTRACTION V4 RESULTS")
    print("=" * 100)

    print()
    print(
        f"Predicted requirements: {len(predicted):,}"
    )

    print(
        f"Ground-truth requirements: {len(truth):,}"
    )

    print()
    print(
        f"True positives:  {tp:,}"
    )

    print(
        f"False positives: {fp:,}"
    )

    print(
        f"False negatives: {fn:,}"
    )

    print()
    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1:        {f1:.4f}"
    )

    return {
        "predictions": len(predicted),
        "ground_truth": len(truth),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
    }


# =============================================================================
# CHARACTERISTIC METRICS
# =============================================================================

def characteristic_metrics(
    predictions: pd.DataFrame,
    truth: Set[Tuple[str, str, str]],
) -> pd.DataFrame:

    characteristics = sorted(
        set(
            list(predictions["characteristic"].dropna().unique())
            + [
                x[1]
                for x in truth
            ]
        )
    )

    rows = []

    for characteristic in characteristics:

        predicted = {
            x
            for x in zip(
                predictions["rfq_id"],
                predictions["characteristic"],
                predictions["internal_value"],
            )
            if x[1] == characteristic
        }

        actual = {
            x
            for x in truth
            if x[1] == characteristic
        }

        tp = len(
            predicted & actual
        )

        fp = len(
            predicted - actual
        )

        fn = len(
            actual - predicted
        )

        precision = (
            tp / (tp + fp)
            if tp + fp
            else 0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn
            else 0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if precision + recall
            else 0
        )

        rows.append({
            "characteristic": characteristic,
            "threshold": threshold_for(
                characteristic
            ),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            "f1",
            ascending=False,
        )
    )


# =============================================================================
# ERROR ANALYSIS
# =============================================================================

def build_error_analysis(
    predictions: pd.DataFrame,
    truth: Set[Tuple[str, str, str]],
) -> pd.DataFrame:

    predicted = set(
        zip(
            predictions["rfq_id"],
            predictions["characteristic"],
            predictions["internal_value"],
        )
    )

    rows = []

    # ------------------------------------------------------------
    # True positives
    # ------------------------------------------------------------

    for item in predicted & truth:

        rfq_id, characteristic, value = item

        rows.append({
            "rfq_id": rfq_id,
            "characteristic": characteristic,
            "internal_value": value,
            "error_type": "TRUE_POSITIVE",
        })

    # ------------------------------------------------------------
    # False positives
    # ------------------------------------------------------------

    for item in predicted - truth:

        rfq_id, characteristic, value = item

        rows.append({
            "rfq_id": rfq_id,
            "characteristic": characteristic,
            "internal_value": value,
            "error_type": "FALSE_POSITIVE",
        })

    # ------------------------------------------------------------
    # False negatives
    # ------------------------------------------------------------

    for item in truth - predicted:

        rfq_id, characteristic, value = item

        rows.append({
            "rfq_id": rfq_id,
            "characteristic": characteristic,
            "internal_value": value,
            "error_type": "FALSE_NEGATIVE",
        })

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions, ground_truth = load_data()

    truth = build_ground_truth(
        ground_truth
    )

    print()
    print("=" * 100)
    print("GROUND TRUTH")
    print("=" * 100)

    print(
        f"Unique ground-truth requirements: "
        f"{len(truth):,}"
    )

    # ------------------------------------------------------------
    # V4 extraction
    # ------------------------------------------------------------

    v4 = extract_v4(
        predictions
    )

    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------

    metrics = evaluate(
        v4,
        truth,
    )

    # ------------------------------------------------------------
    # Characteristic metrics
    # ------------------------------------------------------------

    char_metrics = characteristic_metrics(
        v4,
        truth,
    )

    print()
    print("=" * 100)
    print("CHARACTERISTIC-LEVEL PERFORMANCE")
    print("=" * 100)

    print(
        char_metrics[
            [
                "characteristic",
                "threshold",
                "precision",
                "recall",
                "f1",
            ]
        ].to_string(
            index=False
        )
    )

    # ------------------------------------------------------------
    # Error analysis
    # ------------------------------------------------------------

    errors = build_error_analysis(
        v4,
        truth,
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    v4.to_csv(
        RESULT_FILE,
        index=False,
    )

    char_metrics.to_csv(
        CHAR_METRICS_FILE,
        index=False,
    )

    errors.to_csv(
        ERROR_FILE,
        index=False,
    )

    metrics_df = pd.DataFrame([
        metrics
    ])

    metrics_df.to_csv(
        METRICS_FILE,
        index=False,
    )

    print()
    print("=" * 100)
    print("FILES SAVED")
    print("=" * 100)

    print()
    print(RESULT_FILE)

    print(CHAR_METRICS_FILE)

    print(METRICS_FILE)

    print(ERROR_FILE)

    print()
    print("=" * 100)
    print("731 RFQ EXTRACTION V4 COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()