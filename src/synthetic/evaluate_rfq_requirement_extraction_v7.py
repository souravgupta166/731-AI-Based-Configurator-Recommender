#!/usr/bin/env python3

"""
731 RFQ REQUIREMENT EXTRACTION — V7

V7 extends V6 with deterministic phrase extraction
for steamApplication.

Evidence from the synthetic RFQ corpus shows that the
requirement is explicitly expressed using:

    "steamApplication configuration"

Therefore V7 intentionally avoids broad semantic matching.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

V6_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "731_rfq_extraction_results_v6.csv"
)

RFQ_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "rfqs"
    / "731_rfq_documents_v2.csv"
)

GROUND_TRUTH_FILE = (
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

OUTPUT_FILE = (
    OUTPUT_DIR
    / "731_rfq_extraction_results_v7.csv"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "731_rfq_extraction_metrics_v7.csv"
)

CHAR_METRICS_FILE = (
    OUTPUT_DIR
    / "731_rfq_characteristic_metrics_v7.csv"
)

ERROR_FILE = (
    OUTPUT_DIR
    / "731_rfq_extraction_errors_v7.csv"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

TARGET_CHARACTERISTIC = "steamApplication"

TARGET_VALUE = "1"


STEAM_PATTERNS = [
    r"\bsteamapplication\s+configuration\b",
]


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for deterministic phrase matching."""

    if pd.isna(text):
        return ""

    text = str(text)

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# =============================================================================
# PHRASE DETECTION
# =============================================================================

def detect_steam(sentence: str) -> bool:
    """
    Detect explicit steamApplication configuration language.

    Intentionally conservative:
    only explicit 'steamApplication configuration'
    wording is accepted.
    """

    text = normalize_text(sentence)

    for pattern in STEAM_PATTERNS:

        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):
            return True

    return False


# =============================================================================
# SENTENCE SPLITTING
# =============================================================================

def split_sentences(text: str):

    if pd.isna(text):
        return []

    text = str(text)

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 100)
    print("731 RFQ REQUIREMENT EXTRACTION — V7")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # Load V6
    # -------------------------------------------------------------------------

    print()
    print("Loading V6 predictions...")

    v6 = pd.read_csv(
        V6_FILE
    )

    print(
        f"V6 predictions: {len(v6):,}"
    )

    # -------------------------------------------------------------------------
    # Load RFQs
    # -------------------------------------------------------------------------

    print()
    print("Loading RFQ documents...")

    rfqs = pd.read_csv(
        RFQ_FILE
    )

    print(
        f"RFQ documents: {len(rfqs):,}"
    )

    # -------------------------------------------------------------------------
    # Load ground truth
    # -------------------------------------------------------------------------

    print()
    print("Loading structured requirements...")

    gt = pd.read_csv(
        GROUND_TRUTH_FILE
    )

    print(
        f"Ground-truth requirement rows: {len(gt):,}"
    )

    # -------------------------------------------------------------------------
    # Remove existing steam predictions
    # -------------------------------------------------------------------------

    baseline = v6[
        v6["characteristic"]
        != TARGET_CHARACTERISTIC
    ].copy()

    removed = (
        len(v6)
        - len(baseline)
    )

    print()
    print(
        "Removed existing steamApplication predictions:",
        removed,
    )

    # -------------------------------------------------------------------------
    # Build deterministic steam predictions
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("V7 STEAM APPLICATION EXTRACTION")
    print("=" * 100)

    new_predictions = []

    processed = 0

    for _, row in rfqs.iterrows():

        rfq_id = row["rfq_id"]

        text = row["rfq_text"]

        sentences = split_sentences(
            text
        )

        for sentence_index, sentence in enumerate(
            sentences
        ):

            if not detect_steam(
                sentence
            ):
                continue

            new_predictions.append(
                {
                    "rfq_id": rfq_id,
                    "sentence_index": sentence_index,
                    "characteristic": TARGET_CHARACTERISTIC,
                    "internal_value": TARGET_VALUE,
                    "confidence": 3.0,
                    "characteristic_phrase": (
                        "steamApplication"
                    ),
                    "value_phrase": (
                        "steamApplication configuration"
                    ),
                    "source_sentence": sentence,
                    "v6_decision": "NEW_V7",
                    "v7_source": (
                        "EXPLICIT_STEAM_APPLICATION_PHRASE"
                    ),
                }
            )

        processed += 1

        if processed % 500 == 0:

            print(
                f"Processed {processed:,} / "
                f"{len(rfqs):,}"
            )

    new_df = pd.DataFrame(
        new_predictions
    )

    print()
    print(
        "New steamApplication predictions:",
        len(new_df),
    )

    # -------------------------------------------------------------------------
    # Deduplicate
    # -------------------------------------------------------------------------

    combined = pd.concat(
        [
            baseline,
            new_df,
        ],
        ignore_index=True,
    )

    combined = combined.drop_duplicates(
        subset=[
            "rfq_id",
            "characteristic",
            "internal_value",
        ]
    ).reset_index(
        drop=True
    )

    print()
    print(
        "Final V7 unique predictions:",
        len(combined),
    )

    # -------------------------------------------------------------------------
    # Ground truth
    # -------------------------------------------------------------------------

    truth = gt[
        [
            "rfq_id",
            "characteristic",
            "internal_value",
        ]
    ].drop_duplicates()

    predicted_set = set(
        zip(
            combined["rfq_id"],
            combined["characteristic"],
            combined["internal_value"],
        )
    )

    truth_set = set(
        zip(
            truth["rfq_id"],
            truth["characteristic"],
            truth["internal_value"],
        )
    )

    tp = len(
        predicted_set
        & truth_set
    )

    fp = len(
        predicted_set
        - truth_set
    )

    fn = len(
        truth_set
        - predicted_set
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

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("731 RFQ EXTRACTION V7 RESULTS")
    print("=" * 100)

    print()
    print(
        f"Predicted requirements:    {len(predicted_set):,}"
    )

    print(
        f"Ground-truth requirements: {len(truth_set):,}"
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

    # -------------------------------------------------------------------------
    # Characteristic metrics
    # -------------------------------------------------------------------------

    rows = []

    characteristics = sorted(
        set(
            truth["characteristic"]
        )
    )

    for characteristic in characteristics:

        truth_c = {
            x
            for x in truth_set
            if x[1] == characteristic
        }

        pred_c = {
            x
            for x in predicted_set
            if x[1] == characteristic
        }

        tp_c = len(
            truth_c
            & pred_c
        )

        fp_c = len(
            pred_c
            - truth_c
        )

        fn_c = len(
            truth_c
            - pred_c
        )

        p = (
            tp_c / (tp_c + fp_c)
            if tp_c + fp_c
            else 0
        )

        r = (
            tp_c / (tp_c + fn_c)
            if tp_c + fn_c
            else 0
        )

        f = (
            2 * p * r / (p + r)
            if p + r
            else 0
        )

        rows.append(
            {
                "characteristic": characteristic,
                "true_positive": tp_c,
                "false_positive": fp_c,
                "false_negative": fn_c,
                "precision": p,
                "recall": r,
                "f1": f,
            }
        )

    metrics_df = pd.DataFrame(
        rows
    )

    print()
    print("=" * 100)
    print("CHARACTERISTIC-LEVEL PERFORMANCE")
    print("=" * 100)

    print(
        metrics_df[
            [
                "characteristic",
                "precision",
                "recall",
                "f1",
                "true_positive",
                "false_positive",
                "false_negative",
            ]
        ].to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Error analysis
    # -------------------------------------------------------------------------

    errors = []

    for item in predicted_set:

        if item not in truth_set:

            errors.append(
                {
                    "rfq_id": item[0],
                    "characteristic": item[1],
                    "internal_value": item[2],
                    "error_type": "FALSE_POSITIVE",
                }
            )

    for item in truth_set:

        if item not in predicted_set:

            errors.append(
                {
                    "rfq_id": item[0],
                    "characteristic": item[1],
                    "internal_value": item[2],
                    "error_type": "FALSE_NEGATIVE",
                }
            )

    errors_df = pd.DataFrame(
        errors
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    combined.to_csv(
        OUTPUT_FILE,
        index=False
    )

    metrics_summary = pd.DataFrame(
        [
            {
                "predicted_requirements": len(predicted_set),
                "ground_truth_requirements": len(truth_set),
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        ]
    )

    metrics_summary.to_csv(
        METRICS_FILE,
        index=False
    )

    metrics_df.to_csv(
        CHAR_METRICS_FILE,
        index=False
    )

    errors_df.to_csv(
        ERROR_FILE,
        index=False
    )

    print()
    print("=" * 100)
    print("V7 FILES SAVED")
    print("=" * 100)

    print()
    print(OUTPUT_FILE)

    print(METRICS_FILE)

    print(CHAR_METRICS_FILE)

    print(ERROR_FILE)

    print()
    print("=" * 100)
    print("731 RFQ EXTRACTION V7 COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
