#!/usr/bin/env python3

"""
731 RFQ REQUIREMENT EXTRACTION — V3
===================================

Evidence-driven RFQ requirement extraction.

V3 improves over V1/V2 by using the linguistic knowledge base
generated from the existing RFQ corpus.

Inputs
------
data/synthetic/rfqs/731_rfq_documents_v2.csv
data/synthetic/rfqs/731_rfq_requirements_v2.csv
data/synthetic/rfqs/731_rfq_ground_truth_v2.csv

data/processed/731_rfq_extraction_phrase_knowledge.csv
data/processed/731_rfq_characteristic_lexicon.csv
data/processed/731_rfq_value_language_map.csv

Outputs
-------
data/processed/731_rfq_extraction_results_v3.csv
data/processed/731_rfq_characteristic_metrics_v3.csv
data/processed/731_rfq_extraction_metrics_v3.csv
data/processed/731_rfq_extraction_errors_v3.csv

IMPORTANT
---------
This is an interpretable rule/evidence-based baseline, not an ML model.

The objective is to establish a strong, transparent benchmark before
introducing a machine-learning / NLP model.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOCUMENT_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "rfqs"
    / "731_rfq_documents_v2.csv"
)

REQUIREMENT_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "rfqs"
    / "731_rfq_requirements_v2.csv"
)

GROUND_TRUTH_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "rfqs"
    / "731_rfq_ground_truth_v2.csv"
)

PHRASE_KNOWLEDGE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "731_rfq_extraction_phrase_knowledge.csv"
)

CHARACTERISTIC_LEXICON_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "731_rfq_characteristic_lexicon.csv"
)

VALUE_LANGUAGE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "731_rfq_value_language_map.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

RESULTS_FILE = (
    OUTPUT_DIR
    / "731_rfq_extraction_results_v3.csv"
)

CHARACTERISTIC_METRICS_FILE = (
    OUTPUT_DIR
    / "731_rfq_characteristic_metrics_v3.csv"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "731_rfq_extraction_metrics_v3.csv"
)

ERROR_FILE = (
    OUTPUT_DIR
    / "731_rfq_extraction_errors_v3.csv"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

MIN_EVIDENCE_SCORE = 0.18
MIN_VALUE_SCORE = 0.12

MAX_PHRASE_WORDS = 6

MAX_CANDIDATES_PER_CHARACTERISTIC = 5

GENERIC_PHRASES = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "for",
    "with",
    "without",
    "is",
    "are",
    "be",
    "required",
    "require",
    "requires",
    "equipment",
    "system",
    "device",
    "option",
    "options",
    "please",
    "provide",
    "include",
    "including",
    "support",
    "suitable",
    "appropriate",
    "capability",
    "capabilities",
    "measurement",
    "solution",
    "quotation",
    "quote",
    "offer",
    "following",
    "technical",
    "application",
    "requirements",
    "based",
    "customer",
    "team",
    "project",
}


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize_text(value) -> str:

    if pd.isna(value):
        return ""

    text = str(value).lower()

    text = text.replace(
        "\n",
        " ",
    )

    text = text.replace(
        "\r",
        " ",
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def normalize_value(value) -> str:

    if pd.isna(value):
        return "NOVALUE"

    text = str(value).strip()

    if not text:
        return "NOVALUE"

    match = re.match(
        r"^in\s*\{(.*)\}$",
        text,
        flags=re.IGNORECASE,
    )

    if match:

        content = match.group(1)

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
                token = token.strip()

                if token.upper() != "NOVALUE":
                    values.append(token)

        if values:
            return values[0]

        return "NOVALUE"

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


def normalize_phrase(value) -> str:

    text = normalize_text(value)

    text = re.sub(
        r"[^a-z0-9\s\-]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# =============================================================================
# PHRASE QUALITY
# =============================================================================

def phrase_quality(
    phrase: str,
) -> float:
    """
    Estimate how discriminative a phrase is.

    Longer technical phrases receive more weight.

    Generic phrases are heavily penalized.
    """

    phrase = normalize_phrase(
        phrase
    )

    if not phrase:
        return 0.0

    words = phrase.split()

    if not words:
        return 0.0

    if len(words) > MAX_PHRASE_WORDS:
        return 0.0

    generic_count = sum(
        word in GENERIC_PHRASES
        for word in words
    )

    generic_ratio = (
        generic_count
        / len(words)
    )

    # Reject phrases that are almost entirely generic.
    if generic_ratio >= 0.75:
        return 0.0

    score = 1.0

    # Longer phrases generally carry more information.
    score += min(
        0.6,
        0.15 * (len(words) - 1),
    )

    # Generic words reduce specificity.
    score *= (
        1.0
        - 0.65 * generic_ratio
    )

    # Single generic words are weak.
    if (
        len(words) == 1
        and words[0] in GENERIC_PHRASES
    ):
        return 0.0

    return max(
        0.0,
        min(
            1.5,
            score,
        ),
    )


# =============================================================================
# SENTENCE PROCESSING
# =============================================================================

def split_sentences(
    text: str,
) -> List[str]:

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


def sentence_phrases(
    sentence: str,
) -> List[str]:
    """
    Generate contiguous phrases up to six words.
    """

    sentence = normalize_phrase(
        sentence
    )

    words = sentence.split()

    phrases = set()

    for n in range(
        1,
        min(
            MAX_PHRASE_WORDS,
            len(words),
        ) + 1,
    ):

        for start in range(
            len(words) - n + 1
        ):

            phrase = " ".join(
                words[
                    start:start + n
                ]
            )

            if phrase_quality(
                phrase
            ) > 0:

                phrases.add(
                    phrase
                )

    return list(phrases)


# =============================================================================
# KNOWLEDGE BASE
# =============================================================================

def load_knowledge_base():

    print()
    print("=" * 100)
    print(
        "LOADING V3 EXTRACTION KNOWLEDGE BASE"
    )
    print("=" * 100)

    phrase_df = pd.read_csv(
        PHRASE_KNOWLEDGE_FILE
    )

    lexicon_df = pd.read_csv(
        CHARACTERISTIC_LEXICON_FILE
    )

    value_df = pd.read_csv(
        VALUE_LANGUAGE_FILE
    )

    print(
        f"Phrase records: "
        f"{len(phrase_df):,}"
    )

    print(
        f"Characteristic lexicon: "
        f"{len(lexicon_df):,}"
    )

    print(
        f"Value-language mappings: "
        f"{len(value_df):,}"
    )

    return (
        phrase_df,
        lexicon_df,
        value_df,
    )


def build_evidence_maps(
    phrase_df: pd.DataFrame,
):

    characteristic_map = defaultdict(list)

    value_map = defaultdict(list)

    # -------------------------------------------------------------------------
    # Calculate global phrase frequency.
    #
    # A phrase occurring with many different characteristics is less
    # discriminative than one strongly associated with one characteristic.
    # -------------------------------------------------------------------------

    phrase_characteristics = (
        phrase_df
        .groupby("phrase")[
            "characteristic"
        ]
        .nunique()
        .to_dict()
    )

    for _, row in phrase_df.iterrows():

        characteristic = str(
            row["characteristic"]
        )

        value = normalize_value(
            row["internal_value"]
        )

        phrase = normalize_phrase(
            row["phrase"]
        )

        if not phrase:
            continue

        quality = phrase_quality(
            phrase
        )

        if quality <= 0:
            continue

        unique_rfqs = float(
            row.get(
                "unique_rfqs",
                0,
            )
        )

        evidence_score = float(
            row.get(
                "evidence_score",
                0,
            )
        )

        # ---------------------------------------------------------------------
        # Discriminative phrase factor.
        #
        # If a phrase occurs for many characteristics, its value as a
        # characteristic signal is reduced.
        # ---------------------------------------------------------------------

        characteristic_count = (
            phrase_characteristics.get(
                phrase,
                1,
            )
        )

        discriminative_factor = (
            1.0
            / (
                1.0
                + 0.35
                * max(
                    0,
                    characteristic_count - 1,
                )
            )
        )

        score = (
            quality
            * discriminative_factor
            * (
                1.0
                + min(
                    1.5,
                    evidence_score
                    / 100.0,
                )
            )
        )

        record = {
            "phrase": phrase,
            "value": value,
            "score": score,
            "unique_rfqs": unique_rfqs,
        }

        characteristic_map[
            characteristic
        ].append(
            record
        )

        value_map[
            (
                characteristic,
                value,
            )
        ].append(
            record
        )

    # -------------------------------------------------------------------------
    # Remove duplicate phrases and retain strongest evidence.
    # -------------------------------------------------------------------------

    for characteristic in characteristic_map:

        best = {}

        for record in characteristic_map[
            characteristic
        ]:

            phrase = record[
                "phrase"
            ]

            existing = best.get(
                phrase
            )

            if (
                existing is None
                or record["score"]
                > existing["score"]
            ):
                best[phrase] = record

        characteristic_map[
            characteristic
        ] = sorted(
            best.values(),
            key=lambda x: x["score"],
            reverse=True,
        )

    for key in value_map:

        best = {}

        for record in value_map[key]:

            phrase = record[
                "phrase"
            ]

            existing = best.get(
                phrase
            )

            if (
                existing is None
                or record["score"]
                > existing["score"]
            ):
                best[phrase] = record

        value_map[key] = sorted(
            best.values(),
            key=lambda x: x["score"],
            reverse=True,
        )

    return (
        characteristic_map,
        value_map,
    )


# =============================================================================
# VALUE EXTRACTION
# =============================================================================

def extract_value_candidates(
    sentence: str,
    characteristic: str,
    value_map,
) -> List[Tuple[str, float, str]]:
    """
    Return possible values for one characteristic.
    """

    normalized_sentence = (
        normalize_phrase(
            sentence
        )
    )

    if not normalized_sentence:
        return []

    candidates = []

    possible_keys = [
        key
        for key in value_map
        if key[0] == characteristic
    ]

    for (
        key_characteristic,
        value,
    ) in possible_keys:

        records = value_map[
            (
                key_characteristic,
                value,
            )
        ]

        best_score = 0.0
        best_phrase = ""

        for record in records:

            phrase = record[
                "phrase"
            ]

            if phrase not in normalized_sentence:
                continue

            score = (
                record["score"]
            )

            if score > best_score:

                best_score = score
                best_phrase = phrase

        if (
            best_score
            >= MIN_VALUE_SCORE
        ):

            candidates.append(
                (
                    value,
                    best_score,
                    best_phrase,
                )
            )

    candidates.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    return candidates[
        :MAX_CANDIDATES_PER_CHARACTERISTIC
    ]


# =============================================================================
# CHARACTERISTIC EXTRACTION
# =============================================================================

def extract_characteristics(
    sentence: str,
    characteristic_map,
) -> List[
    Tuple[str, float, str]
]:
    """
    Detect technical characteristics from a sentence.
    """

    normalized_sentence = (
        normalize_phrase(
            sentence
        )
    )

    if not normalized_sentence:
        return []

    candidates = []

    for characteristic, records in (
        characteristic_map.items()
    ):

        best_score = 0.0
        best_phrase = ""

        for record in records:

            phrase = record[
                "phrase"
            ]

            if phrase not in normalized_sentence:
                continue

            score = record[
                "score"
            ]

            if score > best_score:

                best_score = score
                best_phrase = phrase

        if (
            best_score
            >= MIN_EVIDENCE_SCORE
        ):

            candidates.append(
                (
                    characteristic,
                    best_score,
                    best_phrase,
                )
            )

    candidates.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    return candidates


# =============================================================================
# EXTRACTION
# =============================================================================

def extract_rfq(
    rfq_id: str,
    text: str,
    characteristic_map,
    value_map,
) -> List[Dict]:

    results = []

    sentences = split_sentences(
        text
    )

    for sentence_index, sentence in (
        enumerate(sentences)
    ):

        characteristics = (
            extract_characteristics(
                sentence,
                characteristic_map,
            )
        )

        if not characteristics:
            continue

        for (
            characteristic,
            characteristic_score,
            characteristic_phrase,
        ) in characteristics:

            values = (
                extract_value_candidates(
                    sentence,
                    characteristic,
                    value_map,
                )
            )

            # -----------------------------------------------------------------
            # If a value can be identified, produce the strongest candidate.
            # -----------------------------------------------------------------

            if values:

                value, value_score, value_phrase = (
                    values[0]
                )

                combined_score = (
                    0.55
                    * characteristic_score
                    + 0.45
                    * value_score
                )

                results.append(
                    {
                        "rfq_id": rfq_id,
                        "sentence_index":
                            sentence_index,
                        "characteristic":
                            characteristic,
                        "internal_value":
                            value,
                        "confidence":
                            combined_score,
                        "characteristic_phrase":
                            characteristic_phrase,
                        "value_phrase":
                            value_phrase,
                        "source_sentence":
                            sentence,
                    }
                )

    return results


# =============================================================================
# DEDUPLICATION
# =============================================================================

def deduplicate_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    if predictions.empty:
        return predictions

    predictions = (
        predictions
        .sort_values(
            "confidence",
            ascending=False,
        )
    )

    predictions = (
        predictions
        .drop_duplicates(
            subset=[
                "rfq_id",
                "characteristic",
                "internal_value",
            ],
            keep="first",
        )
    )

    return predictions.reset_index(
        drop=True
    )


# =============================================================================
# GROUND TRUTH
# =============================================================================

def build_ground_truth(
    requirements: pd.DataFrame,
) -> pd.DataFrame:

    truth = requirements.copy()

    truth[
        "internal_value"
    ] = (
        truth[
            "internal_value"
        ]
        .apply(
            normalize_value
        )
    )

    truth = (
        truth[
            [
                "rfq_id",
                "requirement_id",
                "characteristic",
                "internal_value",
                "requirement_type",
                "implicit_requirement",
            ]
        ]
        .drop_duplicates()
    )

    return truth


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_predictions(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
):

    prediction_keys = set()

    for _, row in predictions.iterrows():

        prediction_keys.add(
            (
                str(row["rfq_id"]),
                str(row["characteristic"]),
                normalize_value(
                    row["internal_value"]
                ),
            )
        )

    truth_keys = set()

    for _, row in ground_truth.iterrows():

        truth_keys.add(
            (
                str(row["rfq_id"]),
                str(row["characteristic"]),
                normalize_value(
                    row["internal_value"]
                ),
            )
        )

    true_positives = (
        len(
            prediction_keys
            & truth_keys
        )
    )

    false_positives = (
        len(
            prediction_keys
            - truth_keys
        )
    )

    false_negatives = (
        len(
            truth_keys
            - prediction_keys
        )
    )

    precision = (
        true_positives
        / (
            true_positives
            + false_positives
        )
        if (
            true_positives
            + false_positives
        )
        else 0.0
    )

    recall = (
        true_positives
        / (
            true_positives
            + false_negatives
        )
        if (
            true_positives
            + false_negatives
        )
        else 0.0
    )

    f1 = (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
        if (
            precision
            + recall
        )
        else 0.0
    )

    return {
        "true_positives":
            true_positives,
        "false_positives":
            false_positives,
        "false_negatives":
            false_negatives,
        "precision":
            precision,
        "recall":
            recall,
        "f1":
            f1,
    }


# =============================================================================
# CHARACTERISTIC METRICS
# =============================================================================

def calculate_characteristic_metrics(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:

    characteristics = sorted(
        set(
            ground_truth[
                "characteristic"
            ].astype(str)
        )
    )

    rows = []

    for characteristic in characteristics:

        pred_subset = predictions[
            predictions[
                "characteristic"
            ].astype(str)
            == characteristic
        ]

        truth_subset = ground_truth[
            ground_truth[
                "characteristic"
            ].astype(str)
            == characteristic
        ]

        pred_keys = {
            (
                str(row["rfq_id"]),
                normalize_value(
                    row["internal_value"]
                ),
            )
            for _, row in
            pred_subset.iterrows()
        }

        truth_keys = {
            (
                str(row["rfq_id"]),
                normalize_value(
                    row["internal_value"]
                ),
            )
            for _, row in
            truth_subset.iterrows()
        }

        tp = len(
            pred_keys & truth_keys
        )

        fp = len(
            pred_keys - truth_keys
        )

        fn = len(
            truth_keys - pred_keys
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
            2
            * precision
            * recall
            / (precision + recall)
            if precision + recall
            else 0.0
        )

        rows.append(
            {
                "characteristic":
                    characteristic,
                "true_positives":
                    tp,
                "false_positives":
                    fp,
                "false_negatives":
                    fn,
                "precision":
                    precision,
                "recall":
                    recall,
                "f1":
                    f1,
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# EXACT RFQ MATCHES
# =============================================================================

def calculate_exact_matches(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> Tuple[int, int]:

    pred_map = defaultdict(set)
    truth_map = defaultdict(set)

    for _, row in predictions.iterrows():

        pred_map[
            str(row["rfq_id"])
        ].add(
            (
                str(row["characteristic"]),
                normalize_value(
                    row["internal_value"]
                ),
            )
        )

    for _, row in ground_truth.iterrows():

        truth_map[
            str(row["rfq_id"])
        ].add(
            (
                str(row["characteristic"]),
                normalize_value(
                    row["internal_value"]
                ),
            )
        )

    all_rfqs = set(
        truth_map.keys()
    )

    exact_matches = 0

    for rfq_id in all_rfqs:

        if (
            pred_map.get(
                rfq_id,
                set(),
            )
            == truth_map.get(
                rfq_id,
                set(),
            )
        ):
            exact_matches += 1

    return (
        exact_matches,
        len(all_rfqs),
    )


# =============================================================================
# ERROR ANALYSIS
# =============================================================================

def build_error_table(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:

    prediction_keys = {}

    for _, row in predictions.iterrows():

        key = (
            str(row["rfq_id"]),
            str(row["characteristic"]),
            normalize_value(
                row["internal_value"]
            ),
        )

        prediction_keys[
            key
        ] = row

    truth_keys = {}

    for _, row in ground_truth.iterrows():

        key = (
            str(row["rfq_id"]),
            str(row["characteristic"]),
            normalize_value(
                row["internal_value"]
            ),
        )

        truth_keys[
            key
        ] = row

    rows = []

    for key in (
        set(prediction_keys)
        | set(truth_keys)
    ):

        in_prediction = (
            key in prediction_keys
        )

        in_truth = (
            key in truth_keys
        )

        if (
            in_prediction
            and in_truth
        ):
            error_type = "TRUE_POSITIVE"

        elif in_prediction:
            error_type = "FALSE_POSITIVE"

        else:
            error_type = "FALSE_NEGATIVE"

        rfq_id, characteristic, value = (
            key
        )

        row = {
            "rfq_id": rfq_id,
            "characteristic":
                characteristic,
            "internal_value":
                value,
            "error_type":
                error_type,
        }

        if in_prediction:

            prediction_row = (
                prediction_keys[key]
            )

            row[
                "confidence"
            ] = prediction_row[
                "confidence"
            ]

            row[
                "characteristic_phrase"
            ] = prediction_row[
                "characteristic_phrase"
            ]

            row[
                "value_phrase"
            ] = prediction_row[
                "value_phrase"
            ]

            row[
                "source_sentence"
            ] = prediction_row[
                "source_sentence"
            ]

        else:

            row[
                "confidence"
            ] = None

            row[
                "characteristic_phrase"
            ] = ""

            row[
                "value_phrase"
            ] = ""

            row[
                "source_sentence"
            ] = ""

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print(
        "731 RFQ REQUIREMENT EXTRACTION — V3"
    )
    print("=" * 100)

    # -------------------------------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------------------------------

    print()
    print("Loading RFQ documents...")

    documents = pd.read_csv(
        DOCUMENT_FILE
    )

    print(
        f"RFQ documents loaded: "
        f"{len(documents):,}"
    )

    print()
    print("Loading structured requirements...")

    requirements = pd.read_csv(
        REQUIREMENT_FILE
    )

    print(
        f"Requirement records loaded: "
        f"{len(requirements):,}"
    )

    print()
    print("Loading ground truth...")

    ground_truth_source = (
        pd.read_csv(
            GROUND_TRUTH_FILE
        )
    )

    print(
        f"Ground truth records loaded: "
        f"{len(ground_truth_source):,}"
    )

    (
        phrase_df,
        lexicon_df,
        value_df,
    ) = load_knowledge_base()

    # -------------------------------------------------------------------------
    # BUILD EVIDENCE MAPS
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "BUILDING DISCRIMINATIVE EVIDENCE MAPS"
    )
    print("=" * 100)

    (
        characteristic_map,
        value_map,
    ) = build_evidence_maps(
        phrase_df
    )

    print(
        f"Characteristics in evidence map: "
        f"{len(characteristic_map):,}"
    )

    print(
        f"Characteristic/value mappings: "
        f"{len(value_map):,}"
    )

    # -------------------------------------------------------------------------
    # GROUND TRUTH
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "BUILDING GROUND TRUTH"
    )
    print("=" * 100)

    ground_truth = build_ground_truth(
        requirements
    )

    print(
        f"RFQs with ground truth: "
        f"{ground_truth['rfq_id'].nunique():,}"
    )

    # -------------------------------------------------------------------------
    # EXTRACTION
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "EXTRACTING REQUIREMENTS FROM RFQ TEXT — V3"
    )
    print("=" * 100)

    all_predictions = []

    total = len(documents)

    for index, row in (
        documents.iterrows()
    ):

        rfq_id = str(
            row["rfq_id"]
        )

        text = str(
            row["rfq_text"]
        )

        predictions = extract_rfq(
            rfq_id,
            text,
            characteristic_map,
            value_map,
        )

        all_predictions.extend(
            predictions
        )

        if (
            (index + 1) % 500 == 0
            or index + 1 == total
        ):

            print(
                f"Processed "
                f"{index + 1:,} / "
                f"{total:,}"
            )

    predictions = pd.DataFrame(
        all_predictions
    )

    if predictions.empty:

        predictions = pd.DataFrame(
            columns=[
                "rfq_id",
                "sentence_index",
                "characteristic",
                "internal_value",
                "confidence",
                "characteristic_phrase",
                "value_phrase",
                "source_sentence",
            ]
        )

    predictions = (
        deduplicate_predictions(
            predictions
        )
    )

    # -------------------------------------------------------------------------
    # EVALUATION
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "731 RFQ EXTRACTION V3 RESULTS"
    )
    print("=" * 100)

    metrics = evaluate_predictions(
        predictions,
        ground_truth,
    )

    exact_matches, total_rfqs = (
        calculate_exact_matches(
            predictions,
            ground_truth,
        )
    )

    exact_match_rate = (
        exact_matches
        / total_rfqs
        if total_rfqs
        else 0.0
    )

    print()
    print(
        f"RFQs evaluated:       "
        f"{total_rfqs:,}"
    )

    print(
        f"Predicted requirements: "
        f"{len(predictions):,}"
    )

    print(
        f"Ground-truth requirements: "
        f"{len(ground_truth):,}"
    )

    print()
    print(
        f"True positives:       "
        f"{metrics['true_positives']:,}"
    )

    print(
        f"False positives:      "
        f"{metrics['false_positives']:,}"
    )

    print(
        f"False negatives:      "
        f"{metrics['false_negatives']:,}"
    )

    print()
    print(
        f"Precision:            "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall:               "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1:                   "
        f"{metrics['f1']:.4f}"
    )

    print()
    print(
        f"Exact RFQ matches:    "
        f"{exact_matches:,}"
    )

    print(
        f"Exact match rate:     "
        f"{exact_match_rate:.4%}"
    )

    # -------------------------------------------------------------------------
    # CHARACTERISTIC METRICS
    # -------------------------------------------------------------------------

    characteristic_metrics = (
        calculate_characteristic_metrics(
            predictions,
            ground_truth,
        )
    )

    print()
    print("=" * 100)
    print(
        "CHARACTERISTIC-LEVEL PERFORMANCE"
    )
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

    # -------------------------------------------------------------------------
    # CONFIDENCE DISTRIBUTION
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "CONFIDENCE DISTRIBUTION"
    )
    print("=" * 100)

    if not predictions.empty:

        print(
            predictions[
                "confidence"
            ]
            .describe()
            .to_string()
        )

        print()
        print(
            "Confidence buckets:"
        )

        buckets = pd.cut(
            predictions[
                "confidence"
            ],
            bins=[
                -float("inf"),
                0.20,
                0.30,
                0.40,
                0.50,
                0.60,
                0.70,
                0.80,
                0.90,
                float("inf"),
            ],
        )

        print(
            buckets
            .value_counts()
            .sort_index()
            .to_string()
        )

    # -------------------------------------------------------------------------
    # ERROR ANALYSIS
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "ERROR ANALYSIS"
    )
    print("=" * 100)

    errors = build_error_table(
        predictions,
        ground_truth,
    )

    print(
        errors[
            "error_type"
        ]
        .value_counts()
        .to_string()
    )

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------

    predictions.to_csv(
        RESULTS_FILE,
        index=False,
    )

    characteristic_metrics.to_csv(
        CHARACTERISTIC_METRICS_FILE,
        index=False,
    )

    metrics_table = pd.DataFrame(
        [
            {
                "metric":
                    "rfqs_evaluated",
                "value":
                    total_rfqs,
            },
            {
                "metric":
                    "predicted_requirements",
                "value":
                    len(predictions),
            },
            {
                "metric":
                    "ground_truth_requirements",
                "value":
                    len(ground_truth),
            },
            {
                "metric":
                    "true_positives",
                "value":
                    metrics[
                        "true_positives"
                    ],
            },
            {
                "metric":
                    "false_positives",
                "value":
                    metrics[
                        "false_positives"
                    ],
            },
            {
                "metric":
                    "false_negatives",
                "value":
                    metrics[
                        "false_negatives"
                    ],
            },
            {
                "metric":
                    "precision",
                "value":
                    metrics[
                        "precision"
                    ],
            },
            {
                "metric":
                    "recall",
                "value":
                    metrics[
                        "recall"
                    ],
            },
            {
                "metric":
                    "f1",
                "value":
                    metrics[
                        "f1"
                    ],
            },
            {
                "metric":
                    "exact_rfq_matches",
                "value":
                    exact_matches,
            },
            {
                "metric":
                    "exact_match_rate",
                "value":
                    exact_match_rate,
            },
        ]
    )

    metrics_table.to_csv(
        METRICS_FILE,
        index=False,
    )

    errors.to_csv(
        ERROR_FILE,
        index=False,
    )

    # -------------------------------------------------------------------------
    # FILE REPORT
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "FILES SAVED"
    )
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
        ERROR_FILE
    )

    print()
    print("=" * 100)
    print(
        "731 RFQ EXTRACTION V3 COMPLETE"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()