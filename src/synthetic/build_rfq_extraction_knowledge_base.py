#!/usr/bin/env python3

"""
731 RFQ EXTRACTION KNOWLEDGE BASE BUILDER — V3
==============================================

Purpose
-------
Build an evidence-driven linguistic knowledge base from the existing
synthetic RFQ V2 corpus.

The knowledge base connects:

    Customer language
            ↓
    Technical characteristic
            ↓
    Internal 731 value
            ↓
    Requirement type
            ↓
    Explicit / implicit requirement

This module does NOT generate new RFQs.

Input
-----
data/synthetic/rfqs/731_rfq_documents_v2.csv
data/synthetic/rfqs/731_rfq_requirements_v2.csv
data/synthetic/rfqs/731_rfq_ground_truth_v2.csv

Output
------
data/processed/731_rfq_extraction_phrase_knowledge.csv
data/processed/731_rfq_characteristic_lexicon.csv
data/processed/731_rfq_value_language_map.csv
data/processed/731_rfq_extraction_knowledge_summary.csv

Design principle
----------------
The V3 extractor should be based on observed linguistic evidence
from the synthetic RFQ corpus rather than an arbitrary manually
created keyword list.

The resulting files are intended to be consumed by the next-stage
V3 extraction engine.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

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

RFQ_REQUIREMENT_FILE = (
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

PHRASE_OUTPUT = (
    OUTPUT_DIR
    / "731_rfq_extraction_phrase_knowledge.csv"
)

CHARACTERISTIC_OUTPUT = (
    OUTPUT_DIR
    / "731_rfq_characteristic_lexicon.csv"
)

VALUE_LANGUAGE_OUTPUT = (
    OUTPUT_DIR
    / "731_rfq_value_language_map.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "731_rfq_extraction_knowledge_summary.csv"
)


# =============================================================================
# CONSTANTS
# =============================================================================

MIN_PHRASE_LENGTH = 3
MAX_PHRASE_WORDS = 8

STOPWORDS = {
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
}

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
# GENERAL HELPERS
# =============================================================================

def normalize_text(value) -> str:
    """
    Normalize arbitrary text into a stable lower-case representation.
    """

    if pd.isna(value):
        return ""

    text = str(value)

    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


def normalize_value(value) -> str:
    """
    Normalize an internal 731 value.

    Handles:
        NOVALUE
        'GP'
        in {'GP', 'ST'}
    """

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


def tokenize(text: str) -> List[str]:
    """
    Tokenize natural-language text.
    """

    text = normalize_text(text)

    return re.findall(
        r"[a-z0-9]+(?:[-'][a-z0-9]+)*",
        text,
    )


def clean_phrase(text: str) -> str:
    """
    Normalize a candidate phrase.
    """

    text = normalize_text(text)

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def phrase_is_useful(
    phrase: str,
) -> bool:
    """
    Remove obviously uninformative phrases.
    """

    phrase = clean_phrase(phrase)

    if len(phrase) < MIN_PHRASE_LENGTH:
        return False

    words = phrase.split()

    if not words:
        return False

    if len(words) > MAX_PHRASE_WORDS:
        return False

    # Ignore phrases consisting entirely of stopwords.
    if all(
        word in STOPWORDS
        for word in words
    ):
        return False

    return True


# =============================================================================
# PHRASE EXTRACTION
# =============================================================================

def generate_ngrams(
    text: str,
    max_words: int = MAX_PHRASE_WORDS,
) -> Set[str]:
    """
    Generate contiguous n-grams from a requirement sentence.

    We intentionally retain relatively short phrases because the next
    extractor can use them as evidence rather than treating the entire
    sentence as a keyword.
    """

    tokens = tokenize(text)

    if not tokens:
        return set()

    phrases = set()

    for n in range(
        1,
        min(max_words, len(tokens)) + 1,
    ):

        for start in range(
            0,
            len(tokens) - n + 1,
        ):

            phrase_tokens = tokens[
                start:start + n
            ]

            phrase = " ".join(
                phrase_tokens
            )

            if phrase_is_useful(phrase):
                phrases.add(phrase)

    return phrases


def generate_compact_phrases(
    text: str,
) -> Set[str]:
    """
    Generate stronger phrase candidates.

    These are shorter linguistic units that are generally more useful
    for matching than complete RFQ sentences.
    """

    phrases = set()

    sentences = re.split(
        r"[.!?;]+",
        normalize_text(text),
    )

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        tokens = tokenize(sentence)

        if not tokens:
            continue

        # 1- to 5-word phrases.
        for n in range(
            1,
            min(5, len(tokens)) + 1,
        ):

            for start in range(
                len(tokens) - n + 1
            ):

                phrase = " ".join(
                    tokens[
                        start:start + n
                    ]
                )

                if phrase_is_useful(
                    phrase
                ):
                    phrases.add(phrase)

    return phrases


# =============================================================================
# REQUIREMENT SENTENCE ALIGNMENT
# =============================================================================

def find_requirement_sentences(
    rfq_text: str,
    customer_language: str,
) -> List[str]:
    """
    Find sentences in the RFQ that contain the structured
    customer-language requirement.

    If an exact customer-language phrase cannot be found,
    return the closest sentence candidates containing
    characteristic-related words.
    """

    text = str(rfq_text)

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    customer_text = normalize_text(
        customer_language
    )

    if customer_text:

        customer_tokens = set(
            tokenize(customer_text)
        )

        exact_matches = []

        for sentence in sentences:

            normalized = normalize_text(
                sentence
            )

            if (
                customer_text in normalized
            ):
                exact_matches.append(
                    sentence.strip()
                )

        if exact_matches:
            return exact_matches

        # Token-overlap fallback.
        scored = []

        for sentence in sentences:

            sentence_tokens = set(
                tokenize(sentence)
            )

            overlap = len(
                customer_tokens
                & sentence_tokens
            )

            if overlap > 0:
                scored.append(
                    (
                        overlap,
                        sentence.strip(),
                    )
                )

        scored.sort(
            reverse=True,
            key=lambda x: x[0],
        )

        if scored:
            return [
                sentence
                for _, sentence
                in scored[:3]
            ]

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """
    Load the existing RFQ V2 corpus.
    """

    print("=" * 100)
    print(
        "731 RFQ EXTRACTION KNOWLEDGE BASE — V3"
    )
    print("=" * 100)

    print()
    print("Loading RFQ documents...")

    documents = pd.read_csv(
        RFQ_DOCUMENT_FILE
    )

    print(
        f"RFQ documents: {len(documents):,}"
    )

    print()
    print("Loading structured requirements...")

    requirements = pd.read_csv(
        RFQ_REQUIREMENT_FILE
    )

    print(
        f"Requirement records: "
        f"{len(requirements):,}"
    )

    print()
    print("Loading ground truth...")

    ground_truth = pd.read_csv(
        GROUND_TRUTH_FILE
    )

    print(
        f"Ground truth records: "
        f"{len(ground_truth):,}"
    )

    return (
        documents,
        requirements,
        ground_truth,
    )


# =============================================================================
# COLUMN VALIDATION
# =============================================================================

def validate_columns(
    documents: pd.DataFrame,
    requirements: pd.DataFrame,
    ground_truth: pd.DataFrame,
):
    """
    Verify the expected V2 schema.
    """

    required_document_columns = {
        "rfq_id",
        "rfq_text",
    }

    required_requirement_columns = {
        "rfq_id",
        "requirement_id",
        "characteristic",
        "internal_value",
        "requirement_type",
        "customer_language",
        "implicit_requirement",
    }

    missing_documents = (
        required_document_columns
        - set(documents.columns)
    )

    missing_requirements = (
        required_requirement_columns
        - set(requirements.columns)
    )

    if missing_documents:
        raise ValueError(
            "Missing RFQ document columns: "
            + str(sorted(missing_documents))
        )

    if missing_requirements:
        raise ValueError(
            "Missing RFQ requirement columns: "
            + str(sorted(missing_requirements))
        )

    if "rfq_id" not in ground_truth.columns:
        raise ValueError(
            "Ground truth must contain rfq_id."
        )


# =============================================================================
# BUILD PHRASE KNOWLEDGE
# =============================================================================

def build_phrase_knowledge(
    documents: pd.DataFrame,
    requirements: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build phrase-level evidence.

    Each row represents a linguistic phrase observed in RFQ text
    associated with a structured requirement.
    """

    print()
    print("=" * 100)
    print(
        "BUILDING PHRASE-LEVEL EVIDENCE"
    )
    print("=" * 100)

    document_map = dict(
        zip(
            documents["rfq_id"].astype(str),
            documents["rfq_text"].astype(str),
        )
    )

    records = []

    total = len(requirements)

    for index, row in requirements.iterrows():

        if (
            index > 0
            and index % 10_000 == 0
        ):
            print(
                f"Processed requirements: "
                f"{index:,} / {total:,}"
            )

        rfq_id = str(
            row["rfq_id"]
        )

        characteristic = str(
            row["characteristic"]
        ).strip()

        internal_value = normalize_value(
            row["internal_value"]
        )

        requirement_type = str(
            row["requirement_type"]
        ).strip()

        customer_language = str(
            row["customer_language"]
        ).strip()

        implicit = bool(
            row["implicit_requirement"]
        )

        rfq_text = document_map.get(
            rfq_id,
            "",
        )

        if not rfq_text:
            continue

        sentences = find_requirement_sentences(
            rfq_text,
            customer_language,
        )

        sentence_text = " ".join(
            sentences
        )

        phrases = (
            generate_compact_phrases(
                sentence_text
            )
        )

        # Also include the complete customer-language expression.
        customer_phrase = clean_phrase(
            customer_language
        )

        if phrase_is_useful(
            customer_phrase
        ):
            phrases.add(
                customer_phrase
            )

        for phrase in phrases:

            records.append(
                {
                    "rfq_id": rfq_id,
                    "requirement_id": str(
                        row["requirement_id"]
                    ),
                    "characteristic":
                        characteristic,
                    "internal_value":
                        internal_value,
                    "requirement_type":
                        requirement_type,
                    "implicit_requirement":
                        implicit,
                    "customer_language":
                        customer_language,
                    "phrase":
                        phrase,
                }
            )

    phrase_df = pd.DataFrame(
        records
    )

    if phrase_df.empty:
        raise ValueError(
            "No phrase evidence was generated."
        )

    # Aggregate identical observations.
    grouped = (
        phrase_df
        .groupby(
            [
                "characteristic",
                "internal_value",
                "phrase",
            ],
            dropna=False,
        )
        .agg(
            occurrence_count=(
                "rfq_id",
                "count",
            ),
            unique_rfqs=(
                "rfq_id",
                "nunique",
            ),
            mandatory_count=(
                "requirement_type",
                lambda x: (
                    x.astype(str)
                    .str.upper()
                    .eq("MANDATORY")
                    .sum()
                ),
            ),
            preferred_count=(
                "requirement_type",
                lambda x: (
                    x.astype(str)
                    .str.upper()
                    .eq("PREFERRED")
                    .sum()
                ),
            ),
            implicit_count=(
                "implicit_requirement",
                "sum",
            ),
        )
        .reset_index()
    )

    grouped["mandatory_rate"] = (
        grouped["mandatory_count"]
        / grouped["occurrence_count"]
    )

    grouped["preferred_rate"] = (
        grouped["preferred_count"]
        / grouped["occurrence_count"]
    )

    grouped["implicit_rate"] = (
        grouped["implicit_count"]
        / grouped["occurrence_count"]
    )

    # Evidence score.
    #
    # Frequent phrases across multiple RFQs receive stronger evidence.
    # Very generic phrases are down-weighted by requiring multiple
    # unique RFQs for high confidence.
    grouped["evidence_score"] = (
        grouped["unique_rfqs"]
        * (
            1.0
            + grouped["mandatory_rate"]
            + 0.5 * grouped["preferred_rate"]
        )
    )

    grouped["evidence_level"] = (
        grouped["unique_rfqs"]
        .apply(
            classify_evidence_level
        )
    )

    grouped = grouped.sort_values(
        [
            "characteristic",
            "internal_value",
            "evidence_score",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )

    return grouped


def classify_evidence_level(
    unique_rfqs: int,
) -> str:
    """
    Classify linguistic evidence strength.
    """

    if unique_rfqs >= 100:
        return "VERY_STRONG"

    if unique_rfqs >= 50:
        return "STRONG"

    if unique_rfqs >= 20:
        return "MODERATE"

    if unique_rfqs >= 5:
        return "WEAK"

    return "RARE"


# =============================================================================
# CHARACTERISTIC LEXICON
# =============================================================================

def build_characteristic_lexicon(
    phrase_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build characteristic-level linguistic lexicon.

    This identifies phrases associated with each characteristic,
    regardless of the specific internal value.
    """

    grouped = (
        phrase_df
        .groupby(
            [
                "characteristic",
                "phrase",
            ],
            dropna=False,
        )
        .agg(
            occurrence_count=(
                "occurrence_count",
                "sum",
            ),
            unique_rfqs=(
                "unique_rfqs",
                "sum",
            ),
            internal_value_count=(
                "internal_value",
                "nunique",
            ),
            value_variants=(
                "internal_value",
                lambda x: "|".join(
                    sorted(
                        set(
                            map(
                                str,
                                x,
                            )
                        )
                    )
                ),
            ),
        )
        .reset_index()
    )

    grouped["characteristic_evidence_score"] = (
        grouped["unique_rfqs"]
        / (
            1
            + grouped[
                "internal_value_count"
            ]
        )
    )

    grouped["evidence_level"] = (
        grouped["unique_rfqs"]
        .apply(
            classify_evidence_level
        )
    )

    grouped = grouped.sort_values(
        [
            "characteristic",
            "characteristic_evidence_score",
        ],
        ascending=[
            True,
            False,
        ],
    )

    return grouped


# =============================================================================
# VALUE LANGUAGE MAP
# =============================================================================

def build_value_language_map(
    phrase_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build characteristic + value → language mappings.

    This is the most important table for V3 extraction.
    """

    grouped = (
        phrase_df
        .groupby(
            [
                "characteristic",
                "internal_value",
                "phrase",
            ],
            dropna=False,
        )
        .agg(
            unique_rfqs=(
                "unique_rfqs",
                "sum",
            ),
            occurrence_count=(
                "occurrence_count",
                "sum",
            ),
            mandatory_count=(
                "mandatory_count",
                "sum",
            ),
            preferred_count=(
                "preferred_count",
                "sum",
            ),
            implicit_count=(
                "implicit_count",
                "sum",
            ),
        )
        .reset_index()
    )

    grouped["value_precision_proxy"] = (
        grouped["unique_rfqs"]
        / grouped.groupby(
            [
                "characteristic",
                "phrase",
            ]
        )["unique_rfqs"]
        .transform("sum")
    )

    grouped["value_evidence_score"] = (
        grouped["unique_rfqs"]
        * grouped["value_precision_proxy"]
    )

    grouped["evidence_level"] = (
        grouped["unique_rfqs"]
        .apply(
            classify_evidence_level
        )
    )

    grouped = grouped.sort_values(
        [
            "characteristic",
            "internal_value",
            "value_evidence_score",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )

    return grouped


# =============================================================================
# CHARACTERISTIC SUMMARY
# =============================================================================

def build_characteristic_summary(
    phrase_df: pd.DataFrame,
    requirements: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a high-level summary for every technical characteristic.
    """

    requirement_summary = (
        requirements
        .assign(
            normalized_value=
            requirements[
                "internal_value"
            ].apply(
                normalize_value
            )
        )
        .groupby(
            "characteristic"
        )
        .agg(
            requirement_records=(
                "rfq_id",
                "count",
            ),
            unique_rfqs=(
                "rfq_id",
                "nunique",
            ),
            unique_values=(
                "normalized_value",
                "nunique",
            ),
            mandatory_records=(
                "requirement_type",
                lambda x: (
                    x.astype(str)
                    .str.upper()
                    .eq("MANDATORY")
                    .sum()
                ),
            ),
            preferred_records=(
                "requirement_type",
                lambda x: (
                    x.astype(str)
                    .str.upper()
                    .eq("PREFERRED")
                    .sum()
                ),
            ),
            implicit_records=(
                "implicit_requirement",
                "sum",
            ),
        )
        .reset_index()
    )

    phrase_summary = (
        phrase_df
        .groupby(
            "characteristic"
        )
        .agg(
            linguistic_phrases=(
                "phrase",
                "nunique",
            ),
            phrase_observations=(
                "occurrence_count",
                "sum",
            ),
        )
        .reset_index()
    )

    summary = (
        pd.DataFrame(
            {
                "characteristic":
                TECHNICAL_CHARACTERISTICS
            }
        )
        .merge(
            requirement_summary,
            on="characteristic",
            how="left",
        )
        .merge(
            phrase_summary,
            on="characteristic",
            how="left",
        )
    )

    numeric_columns = [
        "requirement_records",
        "unique_rfqs",
        "unique_values",
        "mandatory_records",
        "preferred_records",
        "implicit_records",
        "linguistic_phrases",
        "phrase_observations",
    ]

    for column in numeric_columns:
        if column in summary.columns:
            summary[column] = (
                summary[column]
                .fillna(0)
                .astype(int)
            )

    summary["mandatory_rate"] = (
        summary["mandatory_records"]
        / summary["requirement_records"]
        .replace(0, pd.NA)
    )

    summary["preferred_rate"] = (
        summary["preferred_records"]
        / summary["requirement_records"]
        .replace(0, pd.NA)
    )

    summary["implicit_rate"] = (
        summary["implicit_records"]
        / summary["requirement_records"]
        .replace(0, pd.NA)
    )

    summary["coverage_status"] = (
        summary["linguistic_phrases"]
        .apply(
            lambda x:
            "GOOD"
            if x >= 20
            else (
                "LIMITED"
                if x > 0
                else "MISSING"
            )
        )
    )

    return summary


# =============================================================================
# VALUE SUMMARY
# =============================================================================

def build_value_summary(
    requirements: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce a compact characteristic/value distribution table.
    """

    temp = requirements.copy()

    temp["internal_value"] = (
        temp["internal_value"]
        .apply(normalize_value)
    )

    summary = (
        temp
        .groupby(
            [
                "characteristic",
                "internal_value",
            ]
        )
        .agg(
            requirement_records=(
                "rfq_id",
                "count",
            ),
            unique_rfqs=(
                "rfq_id",
                "nunique",
            ),
            mandatory_records=(
                "requirement_type",
                lambda x: (
                    x.astype(str)
                    .str.upper()
                    .eq("MANDATORY")
                    .sum()
                ),
            ),
            preferred_records=(
                "requirement_type",
                lambda x: (
                    x.astype(str)
                    .str.upper()
                    .eq("PREFERRED")
                    .sum()
                ),
            ),
            implicit_records=(
                "implicit_requirement",
                "sum",
            ),
        )
        .reset_index()
    )

    return summary.sort_values(
        [
            "characteristic",
            "requirement_records",
        ],
        ascending=[
            True,
            False,
        ],
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        documents,
        requirements,
        ground_truth,
    ) = load_data()

    validate_columns(
        documents,
        requirements,
        ground_truth,
    )

    print()
    print("=" * 100)
    print("INPUT VALIDATION")
    print("=" * 100)

    print()
    print(
        f"RFQs: "
        f"{documents['rfq_id'].nunique():,}"
    )

    print(
        f"Requirements: "
        f"{len(requirements):,}"
    )

    print(
        f"Ground truth: "
        f"{ground_truth['rfq_id'].nunique():,} RFQs"
    )

    print()
    print(
        "Characteristics present:"
    )

    present_characteristics = sorted(
        requirements[
            "characteristic"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    for characteristic in (
        present_characteristics
    ):
        print(
            f"  {characteristic}"
        )

    # -------------------------------------------------------------------------
    # PHRASE KNOWLEDGE
    # -------------------------------------------------------------------------

    phrase_knowledge = (
        build_phrase_knowledge(
            documents,
            requirements,
        )
    )

    # -------------------------------------------------------------------------
    # CHARACTERISTIC LEXICON
    # -------------------------------------------------------------------------

    characteristic_lexicon = (
        build_characteristic_lexicon(
            phrase_knowledge
        )
    )

    # -------------------------------------------------------------------------
    # VALUE LANGUAGE MAP
    # -------------------------------------------------------------------------

    value_language_map = (
        build_value_language_map(
            phrase_knowledge
        )
    )

    # -------------------------------------------------------------------------
    # CHARACTERISTIC SUMMARY
    # -------------------------------------------------------------------------

    characteristic_summary = (
        build_characteristic_summary(
            phrase_knowledge,
            requirements,
        )
    )

    value_summary = (
        build_value_summary(
            requirements
        )
    )

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------

    phrase_knowledge.to_csv(
        PHRASE_OUTPUT,
        index=False,
    )

    characteristic_lexicon.to_csv(
        CHARACTERISTIC_OUTPUT,
        index=False,
    )

    value_language_map.to_csv(
        VALUE_LANGUAGE_OUTPUT,
        index=False,
    )

    characteristic_summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    # -------------------------------------------------------------------------
    # REPORT
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("KNOWLEDGE BASE SUMMARY")
    print("=" * 100)

    print()
    print(
        f"Phrase evidence records: "
        f"{len(phrase_knowledge):,}"
    )

    print(
        f"Characteristic lexicon records: "
        f"{len(characteristic_lexicon):,}"
    )

    print(
        f"Value-language mappings: "
        f"{len(value_language_map):,}"
    )

    print()
    print(
        "Characteristic coverage:"
    )

    display_columns = [
        "characteristic",
        "requirement_records",
        "unique_rfqs",
        "unique_values",
        "linguistic_phrases",
        "coverage_status",
    ]

    print(
        characteristic_summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Evidence level distribution:"
    )

    print(
        phrase_knowledge[
            "evidence_level"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Top linguistic phrases:"
    )

    top_phrases = (
        phrase_knowledge
        .sort_values(
            "evidence_score",
            ascending=False,
        )
        .head(30)
    )

    print(
        top_phrases[
            [
                "characteristic",
                "internal_value",
                "phrase",
                "unique_rfqs",
                "evidence_level",
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
        f"Phrase knowledge:\n"
        f"{PHRASE_OUTPUT}"
    )

    print()
    print(
        f"Characteristic lexicon:\n"
        f"{CHARACTERISTIC_OUTPUT}"
    )

    print()
    print(
        f"Value-language map:\n"
        f"{VALUE_LANGUAGE_OUTPUT}"
    )

    print()
    print(
        f"Knowledge summary:\n"
        f"{SUMMARY_OUTPUT}"
    )

    print()
    print("=" * 100)
    print(
        "731 RFQ EXTRACTION KNOWLEDGE BASE COMPLETE"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()