#!/usr/bin/env python3

"""
731 RFQ REQUIREMENT EXTRACTION — V8
===================================

V8 focuses on ambiguity-aware extraction.

Key improvement over V7
------------------------
steamApplication is explicitly treated as an ambiguous characteristic.

The RFQ text contains evidence that steamApplication is requested,
but the customer-facing language does not uniquely identify the
internal values H versus N.

Therefore:

    characteristic detected
            +
    value not identifiable
            =
    ABSTAIN / UNRESOLVED

V8 reports:

1. Strict value-level extraction
2. Characteristic-level detection
3. Steam-specific characteristic detection
4. Steam value abstention statistics

V6 is used as the baseline because V7 incorrectly added
1,153 steamApplication predictions with zero true positives.
"""

from __future__ import annotations

from pathlib import Path
import re

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

DOC_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "rfqs"
    / "731_rfq_documents_v2.csv"
)

REQ_FILE = (
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

RESULT_FILE = (
    OUTPUT_DIR
    / "731_rfq_extraction_results_v8.csv"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "731_rfq_extraction_metrics_v8.csv"
)

CHAR_METRICS_FILE = (
    OUTPUT_DIR
    / "731_rfq_characteristic_metrics_v8.csv"
)

ERROR_FILE = (
    OUTPUT_DIR
    / "731_rfq_extraction_errors_v8.csv"
)

ABSTENTION_FILE = (
    OUTPUT_DIR
    / "731_rfq_extraction_abstentions_v8.csv"
)


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    text = str(value).lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# =============================================================================
# STEAM DETECTION
# =============================================================================

STEAM_PATTERNS = [
    "steamapplication",
    "steam application",
]


def contains_steam_requirement(text: str) -> bool:

    text = normalize_text(text)

    return any(
        pattern in text
        for pattern in STEAM_PATTERNS
    )


def steam_value_is_identifiable(
    text: str,
) -> bool:
    """
    Determine whether the RFQ explicitly identifies the
    internal steamApplication value.

    The synthetic V2 RFQs intentionally use the same
    customer-facing phrase for internal values H and N.

    Therefore generic phrases such as:

        "required steamApplication configuration"

    are NOT sufficient to identify H or N.

    Only explicit H/N notation is accepted.
    """

    text = normalize_text(text)

    explicit_h_patterns = [
        r"\bsteamapplication\s*[:=]\s*h\b",
        r"\bsteam application\s*[:=]\s*h\b",
    ]

    explicit_n_patterns = [
        r"\bsteamapplication\s*[:=]\s*n\b",
        r"\bsteam application\s*[:=]\s*n\b",
    ]

    for pattern in (
        explicit_h_patterns
        + explicit_n_patterns
    ):

        if re.search(pattern, text):
            return True

    return False


def extract_explicit_steam_value(
    text: str,
):
    """
    Return H or N if explicitly present.
    Otherwise return None.
    """

    text = normalize_text(text)

    if re.search(
        r"\bsteamapplication\s*[:=]\s*h\b",
        text,
    ):
        return "H"

    if re.search(
        r"\bsteam application\s*[:=]\s*h\b",
        text,
    ):
        return "H"

    if re.search(
        r"\bsteamapplication\s*[:=]\s*n\b",
        text,
    ):
        return "N"

    if re.search(
        r"\bsteam application\s*[:=]\s*n\b",
        text,
    ):
        return "N"

    return None


# =============================================================================
# BUILD GROUND TRUTH
# =============================================================================

def build_ground_truth(
    requirements: pd.DataFrame,
):

    required_columns = [
        "rfq_id",
        "characteristic",
        "internal_value",
    ]

    missing = [
        column
        for column in required_columns
        if column not in requirements.columns
    ]

    if missing:
        raise ValueError(
            "Missing ground-truth columns: "
            + ", ".join(missing)
        )

    gt = requirements[
        required_columns
    ].copy()

    gt["rfq_id"] = (
        gt["rfq_id"]
        .astype(str)
        .str.strip()
    )

    gt["characteristic"] = (
        gt["characteristic"]
        .astype(str)
        .str.strip()
    )

    gt["internal_value"] = (
        gt["internal_value"]
        .astype(str)
        .str.strip()
    )

    gt = gt.drop_duplicates()

    return gt


# =============================================================================
# METRIC HELPERS
# =============================================================================

def calculate_metrics(
    predicted_set,
    truth_set,
):

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
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    return (
        tp,
        fp,
        fn,
        precision,
        recall,
        f1,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 100)
    print("731 RFQ REQUIREMENT EXTRACTION — V8")
    print("=" * 100)

    # -----------------------------------------------------------------
    # LOAD V6
    # -----------------------------------------------------------------

    print()
    print("Loading V6 predictions...")

    v6 = pd.read_csv(
        V6_FILE,
        low_memory=False,
    )

    print(
        f"V6 predictions: {len(v6):,}"
    )

    # -----------------------------------------------------------------
    # LOAD DOCUMENTS
    # -----------------------------------------------------------------

    print()
    print("Loading RFQ documents...")

    documents = pd.read_csv(
        DOC_FILE,
        low_memory=False,
    )

    print(
        f"RFQ documents: {len(documents):,}"
    )

    # -----------------------------------------------------------------
    # LOAD REQUIREMENTS
    # -----------------------------------------------------------------

    print()
    print("Loading structured requirements...")

    requirements = pd.read_csv(
        REQ_FILE,
        low_memory=False,
    )

    print(
        f"Ground-truth requirement rows: "
        f"{len(requirements):,}"
    )

    # -----------------------------------------------------------------
    # GROUND TRUTH
    # -----------------------------------------------------------------

    print()
    print("=" * 100)
    print("BUILDING GROUND TRUTH")
    print("=" * 100)

    ground_truth = build_ground_truth(
        requirements
    )

    print(
        f"Unique ground-truth requirements: "
        f"{len(ground_truth):,}"
    )

    truth_set = set(
        zip(
            ground_truth["rfq_id"],
            ground_truth["characteristic"],
            ground_truth["internal_value"],
        )
    )

    # -----------------------------------------------------------------
    # NORMALIZE V6
    # -----------------------------------------------------------------

    required_v6_columns = [
        "rfq_id",
        "characteristic",
        "internal_value",
    ]

    missing_v6 = [
        column
        for column in required_v6_columns
        if column not in v6.columns
    ]

    if missing_v6:
        raise ValueError(
            "Missing V6 columns: "
            + ", ".join(missing_v6)
        )

    v6["rfq_id"] = (
        v6["rfq_id"]
        .astype(str)
        .str.strip()
    )

    v6["characteristic"] = (
        v6["characteristic"]
        .astype(str)
        .str.strip()
    )

    v6["internal_value"] = (
        v6["internal_value"]
        .astype(str)
        .str.strip()
    )

    # -----------------------------------------------------------------
    # REMOVE OLD STEAM PREDICTIONS
    # -----------------------------------------------------------------

    baseline = v6[
        v6["characteristic"]
        != "steamApplication"
    ].copy()

    old_steam = v6[
        v6["characteristic"]
        == "steamApplication"
    ].copy()

    print()
    print(
        "Existing V6 steamApplication predictions:",
        len(old_steam),
    )

    print(
        "Baseline predictions:",
        len(baseline),
    )

    # -----------------------------------------------------------------
    # STEAM AMBIGUITY-AWARE EXTRACTION
    # -----------------------------------------------------------------

    print()
    print("=" * 100)
    print("V8 STEAM AMBIGUITY-AWARE EXTRACTION")
    print("=" * 100)

    steam_records = []
    abstentions = []

    for index, row in documents.iterrows():

        rfq_id = str(
            row["rfq_id"]
        ).strip()

        text = normalize_text(
            row["rfq_text"]
        )

        # -------------------------------------------------------------
        # Detect steam characteristic
        # -------------------------------------------------------------

        if not contains_steam_requirement(
            text
        ):
            continue

        # -------------------------------------------------------------
        # Determine whether value is identifiable
        # -------------------------------------------------------------

        identifiable = (
            steam_value_is_identifiable(
                text
            )
        )

        # -------------------------------------------------------------
        # Characteristic detected but value unresolved
        # -------------------------------------------------------------

        if not identifiable:

            abstentions.append(
                {
                    "rfq_id": rfq_id,
                    "characteristic":
                        "steamApplication",
                    "value_status":
                        "UNRESOLVED",
                    "reason":
                        "VALUE_NOT_IDENTIFIABLE",
                    "confidence": 1.0,
                    "source_sentence":
                        text[:500],
                }
            )

            if (
                (index + 1) % 500 == 0
            ):

                print(
                    f"Processed "
                    f"{index + 1:,} / "
                    f"{len(documents):,}"
                )

            continue

        # -------------------------------------------------------------
        # Explicit H/N evidence
        # -------------------------------------------------------------

        value = extract_explicit_steam_value(
            text
        )

        if value is not None:

            steam_records.append(
                {
                    "rfq_id": rfq_id,
                    "sentence_index": -1,
                    "characteristic":
                        "steamApplication",
                    "internal_value":
                        value,
                    "confidence": 1.0,
                    "characteristic_phrase":
                        "steamApplication",
                    "value_phrase":
                        value,
                    "source_sentence":
                        text[:500],
                    "v8_source":
                        "EXPLICIT_VALUE",
                }
            )

        if (
            (index + 1) % 500 == 0
        ):

            print(
                f"Processed "
                f"{index + 1:,} / "
                f"{len(documents):,}"
            )

    # -----------------------------------------------------------------
    # DATAFRAMES
    # -----------------------------------------------------------------

    steam_df = pd.DataFrame(
        steam_records
    )

    abstention_df = pd.DataFrame(
        abstentions
    )

    print()
    print(
        "Explicit steam predictions:",
        len(steam_df),
    )

    print(
        "Steam characteristic detections:",
        len(abstention_df) + len(steam_df),
    )

    print(
        "Steam value abstentions:",
        len(abstention_df),
    )

    # -----------------------------------------------------------------
    # COMBINE STRICT VALUE PREDICTIONS
    # -----------------------------------------------------------------

    print()
    print("=" * 100)
    print("COMBINING V6 + V8")
    print("=" * 100)

    baseline["v8_source"] = "V6"

    if len(steam_df):

        final = pd.concat(
            [
                baseline,
                steam_df,
            ],
            ignore_index=True,
        )

    else:

        final = baseline.copy()

    # -----------------------------------------------------------------
    # UNIQUE STRICT PREDICTIONS
    # -----------------------------------------------------------------

    final = final.drop_duplicates(
        subset=[
            "rfq_id",
            "characteristic",
            "internal_value",
        ]
    )

    print(
        "Final V8 strict predictions:",
        len(final),
    )

    # -----------------------------------------------------------------
    # STRICT VALUE EVALUATION
    # -----------------------------------------------------------------

    predicted_set = set(
        zip(
            final["rfq_id"],
            final["characteristic"],
            final["internal_value"],
        )
    )

    (
        tp,
        fp,
        fn,
        precision,
        recall,
        f1,
    ) = calculate_metrics(
        predicted_set,
        truth_set,
    )

    # -----------------------------------------------------------------
    # CHARACTERISTIC-LEVEL GROUND TRUTH
    # -----------------------------------------------------------------

    truth_characteristics = set(
        zip(
            ground_truth["rfq_id"],
            ground_truth["characteristic"],
        )
    )

    # Strict predictions converted to characteristic pairs
    prediction_characteristics = set(
        zip(
            final["rfq_id"],
            final["characteristic"],
        )
    )

    # IMPORTANT:
    # Add unresolved steam characteristic detections.
    #
    # These are NOT added to strict value predictions.
    # They only count toward characteristic detection.

    if len(abstention_df):

        steam_characteristic_pairs = set(
            zip(
                abstention_df["rfq_id"],
                abstention_df["characteristic"],
            )
        )

        prediction_characteristics.update(
            steam_characteristic_pairs
        )

    (
        char_tp,
        char_fp,
        char_fn,
        char_precision,
        char_recall,
        char_f1,
    ) = calculate_metrics(
        prediction_characteristics,
        truth_characteristics,
    )

    # -----------------------------------------------------------------
    # STEAM CHARACTERISTIC DETECTION
    # -----------------------------------------------------------------

    steam_truth = set(
        ground_truth[
            ground_truth["characteristic"]
            == "steamApplication"
        ]["rfq_id"]
    )

    steam_detected = set()

    # Explicit predictions
    if len(steam_df):

        steam_detected.update(
            steam_df["rfq_id"]
            .astype(str)
        )

    # Ambiguous but detected requirements
    if len(abstention_df):

        steam_detected.update(
            abstention_df["rfq_id"]
            .astype(str)
        )

    (
        steam_char_tp,
        steam_char_fp,
        steam_char_fn,
        steam_char_precision,
        steam_char_recall,
        steam_char_f1,
    ) = calculate_metrics(
        steam_detected,
        steam_truth,
    )

    # -----------------------------------------------------------------
    # STEAM VALUE-LEVEL EVALUATION
    # -----------------------------------------------------------------

    steam_truth_values = set(
        zip(
            ground_truth[
                ground_truth["characteristic"]
                == "steamApplication"
            ]["rfq_id"],
            ground_truth[
                ground_truth["characteristic"]
                == "steamApplication"
            ]["characteristic"],
            ground_truth[
                ground_truth["characteristic"]
                == "steamApplication"
            ]["internal_value"],
        )
    )

    steam_predicted_values = set(
        item
        for item in predicted_set
        if item[1] == "steamApplication"
    )

    (
        steam_value_tp,
        steam_value_fp,
        steam_value_fn,
        steam_value_precision,
        steam_value_recall,
        steam_value_f1,
    ) = calculate_metrics(
        steam_predicted_values,
        steam_truth_values,
    )

    # -----------------------------------------------------------------
    # RESULTS
    # -----------------------------------------------------------------

    print()
    print("=" * 100)
    print("731 RFQ EXTRACTION V8 RESULTS")
    print("=" * 100)

    print()
    print(
        f"Predicted requirements:    "
        f"{len(predicted_set):,}"
    )

    print(
        f"Ground-truth requirements: "
        f"{len(truth_set):,}"
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

    # -----------------------------------------------------------------
    # CHARACTERISTIC RESULTS
    # -----------------------------------------------------------------

    print()
    print("=" * 100)
    print("CHARACTERISTIC-LEVEL DETECTION")
    print("=" * 100)

    print(
        f"Detected characteristic pairs: "
        f"{len(prediction_characteristics):,}"
    )

    print(
        f"Ground-truth characteristic pairs: "
        f"{len(truth_characteristics):,}"
    )

    print(
        f"True positives:  {char_tp:,}"
    )

    print(
        f"False positives: {char_fp:,}"
    )

    print(
        f"False negatives: {char_fn:,}"
    )

    print()

    print(
        f"Precision: {char_precision:.4f}"
    )

    print(
        f"Recall:    {char_recall:.4f}"
    )

    print(
        f"F1:        {char_f1:.4f}"
    )

    # -----------------------------------------------------------------
    # STEAM RESULTS
    # -----------------------------------------------------------------

    print()
    print("=" * 100)
    print("STEAM APPLICATION")
    print("=" * 100)

    print()
    print(
        f"Ground-truth steam RFQs: "
        f"{len(steam_truth):,}"
    )

    print(
        f"Steam characteristic detected: "
        f"{len(steam_detected):,}"
    )

    print(
        f"Steam value abstentions: "
        f"{len(abstention_df):,}"
    )

    print()

    print(
        f"Characteristic precision: "
        f"{steam_char_precision:.4f}"
    )

    print(
        f"Characteristic recall: "
        f"{steam_char_recall:.4f}"
    )

    print(
        f"Characteristic F1: "
        f"{steam_char_f1:.4f}"
    )

    print()

    print(
        f"Steam value precision: "
        f"{steam_value_precision:.4f}"
    )

    print(
        f"Steam value recall: "
        f"{steam_value_recall:.4f}"
    )

    print(
        f"Steam value F1: "
        f"{steam_value_f1:.4f}"
    )

    # -----------------------------------------------------------------
    # SAVE METRICS
    # -----------------------------------------------------------------

    metrics = pd.DataFrame(
        [
            {
                "metric":
                    "strict_precision",
                "value":
                    precision,
            },
            {
                "metric":
                    "strict_recall",
                "value":
                    recall,
            },
            {
                "metric":
                    "strict_f1",
                "value":
                    f1,
            },
            {
                "metric":
                    "characteristic_precision",
                "value":
                    char_precision,
            },
            {
                "metric":
                    "characteristic_recall",
                "value":
                    char_recall,
            },
            {
                "metric":
                    "characteristic_f1",
                "value":
                    char_f1,
            },
            {
                "metric":
                    "steam_characteristic_precision",
                "value":
                    steam_char_precision,
            },
            {
                "metric":
                    "steam_characteristic_recall",
                "value":
                    steam_char_recall,
            },
            {
                "metric":
                    "steam_characteristic_f1",
                "value":
                    steam_char_f1,
            },
            {
                "metric":
                    "steam_value_precision",
                "value":
                    steam_value_precision,
            },
            {
                "metric":
                    "steam_value_recall",
                "value":
                    steam_value_recall,
            },
            {
                "metric":
                    "steam_value_f1",
                "value":
                    steam_value_f1,
            },
            {
                "metric":
                    "steam_value_abstentions",
                "value":
                    len(abstention_df),
            },
            {
                "metric":
                    "steam_characteristic_detections",
                "value":
                    len(steam_detected),
            },
        ]
    )

    # -----------------------------------------------------------------
    # ERROR TABLE
    # -----------------------------------------------------------------

    errors = []

    # Strict false positives
    for item in predicted_set:

        if item not in truth_set:

            errors.append(
                {
                    "rfq_id":
                        item[0],
                    "characteristic":
                        item[1],
                    "internal_value":
                        item[2],
                    "error_type":
                        "FALSE_POSITIVE",
                }
            )

    # Strict false negatives
    for item in truth_set:

        if item not in predicted_set:

            errors.append(
                {
                    "rfq_id":
                        item[0],
                    "characteristic":
                        item[1],
                    "internal_value":
                        item[2],
                    "error_type":
                        "FALSE_NEGATIVE",
                }
            )

    # Steam ambiguity is deliberately NOT treated as
    # a false positive or false negative at the
    # characteristic level.
    #
    # It is recorded separately as VALUE_UNRESOLVED.

    for _, row in abstention_df.iterrows():

        errors.append(
            {
                "rfq_id":
                    row["rfq_id"],
                "characteristic":
                    "steamApplication",
                "internal_value":
                    "",
                "error_type":
                    "VALUE_UNRESOLVED",
            }
        )

    errors_df = pd.DataFrame(
        errors
    )

    # -----------------------------------------------------------------
    # CHARACTERISTIC METRICS
    # -----------------------------------------------------------------

    char_rows = []

    all_characteristics = sorted(
        set(
            ground_truth[
                "characteristic"
            ]
        )
    )

    for characteristic in all_characteristics:

        truth_c = set(
            zip(
                ground_truth[
                    ground_truth[
                        "characteristic"
                    ]
                    == characteristic
                ]["rfq_id"],
                ground_truth[
                    ground_truth[
                        "characteristic"
                    ]
                    == characteristic
                ]["characteristic"],
            )
        )

        pred_c = set(
            item
            for item in prediction_characteristics
            if item[1] == characteristic
        )

        (
            ctp,
            cfp,
            cfn,
            cp,
            cr,
            cf1,
        ) = calculate_metrics(
            pred_c,
            truth_c,
        )

        char_rows.append(
            {
                "characteristic":
                    characteristic,
                "precision":
                    cp,
                "recall":
                    cr,
                "f1":
                    cf1,
                "true_positive":
                    ctp,
                "false_positive":
                    cfp,
                "false_negative":
                    cfn,
            }
        )

    char_metrics_df = pd.DataFrame(
        char_rows
    )

    # -----------------------------------------------------------------
    # OUTPUT DIRECTORY
    # -----------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # SAVE STRICT RESULTS
    # -----------------------------------------------------------------

    final.to_csv(
        RESULT_FILE,
        index=False,
    )

    # -----------------------------------------------------------------
    # SAVE METRICS
    # -----------------------------------------------------------------

    metrics.to_csv(
        METRICS_FILE,
        index=False,
    )

    # -----------------------------------------------------------------
    # SAVE CHARACTERISTIC METRICS
    # -----------------------------------------------------------------

    char_metrics_df.to_csv(
        CHAR_METRICS_FILE,
        index=False,
    )

    # -----------------------------------------------------------------
    # SAVE ERRORS
    # -----------------------------------------------------------------

    errors_df.to_csv(
        ERROR_FILE,
        index=False,
    )

    # -----------------------------------------------------------------
    # SAVE ABSTENTIONS
    # -----------------------------------------------------------------

    abstention_df.to_csv(
        ABSTENTION_FILE,
        index=False,
    )

    # -----------------------------------------------------------------
    # FINAL FILE REPORT
    # -----------------------------------------------------------------

    print()
    print("=" * 100)
    print("V8 FILES SAVED")
    print("=" * 100)

    print()
    print(
        "Strict extraction results:"
    )
    print(
        RESULT_FILE
    )

    print()
    print(
        "Metrics:"
    )
    print(
        METRICS_FILE
    )

    print()
    print(
        "Characteristic metrics:"
    )
    print(
        CHAR_METRICS_FILE
    )

    print()
    print(
        "Errors:"
    )
    print(
        ERROR_FILE
    )

    print()
    print(
        "Steam abstentions:"
    )
    print(
        ABSTENTION_FILE
    )

    print()
    print("=" * 100)
    print("731 RFQ EXTRACTION V8 COMPLETE")
    print("=" * 100)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()