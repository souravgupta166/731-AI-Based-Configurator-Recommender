from pathlib import Path
import re
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

PROCESSED = ROOT / "data" / "processed"
RFQ_DIR = ROOT / "data" / "synthetic" / "rfqs"

V5_RESULTS = PROCESSED / "731_rfq_extraction_results_v5.csv"
REQUIREMENTS_FILE = RFQ_DIR / "731_rfq_requirements_v2.csv"
DOCUMENTS_FILE = RFQ_DIR / "731_rfq_documents_v2.csv"

OUT_RESULTS = PROCESSED / "731_rfq_extraction_results_v6.csv"
OUT_METRICS = PROCESSED / "731_rfq_extraction_metrics_v6.csv"
OUT_CHAR_METRICS = PROCESSED / "731_rfq_characteristic_metrics_v6.csv"
OUT_ERRORS = PROCESSED / "731_rfq_extraction_errors_v6.csv"


def banner(text):
    print()
    print("=" * 100)
    print(text)
    print("=" * 100)


# =============================================================================
# TEXT HELPERS
# =============================================================================

def normalise_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = (
        text.replace("–", "-")
            .replace("—", "-")
            .replace("’", "'")
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_sentences(text):

    if pd.isna(text):
        return []

    sentences = []

    for line in str(text).splitlines():

        line = line.strip()

        if not line:
            continue

        lower = line.lower()

        # Administrative/header material
        ignored = (
            lower.startswith("customer:"),
            lower.startswith("rfq reference:"),
            lower.startswith("customer reference:"),
            lower.startswith("country:"),
            lower.startswith("dear "),
            lower.startswith("best regards"),
        )

        if any(ignored):
            continue

        parts = re.split(
            r"(?<=[.!?])\s+",
            line
        )

        for part in parts:

            part = part.strip()

            if part:
                sentences.append(part)

    return sentences


# =============================================================================
# GENERIC SEMANTIC PREDICTION
# =============================================================================

def semantic_predictions(
    rfq_id,
    text,
    characteristic,
    rules,
    blockers=None,
    confidence=3.0,
    source_name="V6_SEMANTIC_RULE"
):

    predictions = []

    blockers = blockers or []

    sentences = split_sentences(text)

    for sentence_index, sentence in enumerate(sentences):

        s = normalise_text(sentence)

        # ---------------------------------------------------------
        # Block generic/internal placeholder language
        # ---------------------------------------------------------

        if any(
            re.search(pattern, s)
            for pattern in blockers
        ):
            continue

        # ---------------------------------------------------------
        # Search value-specific semantic rules
        # ---------------------------------------------------------

        for value, patterns in rules.items():

            for pattern in patterns:

                match = re.search(
                    pattern,
                    s
                )

                if not match:
                    continue

                predictions.append({
                    "rfq_id": rfq_id,
                    "sentence_index": sentence_index,
                    "characteristic": characteristic,
                    "internal_value": value,
                    "confidence": confidence,
                    "characteristic_phrase": source_name,
                    "value_phrase": match.group(0),
                    "source_sentence": sentence,
                    "v6_source": source_name,
                })

                # One value per characteristic per sentence
                break

    return predictions


# =============================================================================
# DATA INTERFACE RULES
# =============================================================================
#
# These are deliberately conservative.
#
# We do not use generic words such as:
#
#     interface
#     communication
#     capability
#
# by themselves because they do not identify the actual internal value.
#
# =============================================================================

DATA_INTERFACE_RULES = {

    "NN": [
        r"\bstandard communication interface\b",
        r"\bstandard communication capability\b",
        r"\bstandard communication\b",
        r"\bcommunication interface nn\b",
        r"\bdata interface nn\b",
    ],

    "HS": [
        r"\bhigh[- ]speed communication\b",
        r"\bhigh[- ]speed communication interface\b",
        r"\bhigh[- ]speed interface\b",
        r"\bhigh speed data interface\b",
        r"\bdata interface hs\b",
    ],

    "PA": [
        r"\bprocess automation communication\b",
        r"\bprocess automation interface\b",
        r"\bprocess automation communication interface\b",
        r"\bdata interface pa\b",
    ],

    "MR": [
        r"\bmeasurement reporting interface\b",
        r"\bmeasurement reporting communication\b",
        r"\bmeasurement reporting capability\b",
        r"\bdata interface mr\b",
    ],
}


DATA_INTERFACE_BLOCKERS = [
    r"\brequired datainterface configuration\b",
    r"\bdatainterface configuration\b",
    r"\bdata interface configuration\b",
]


# =============================================================================
# STEAM APPLICATION RULES
# =============================================================================

STEAM_APPLICATION_RULES = {

    "1": [
        r"\bsteam application\b",
        r"\bsteam measurement\b",
        r"\bsteam measurement application\b",
        r"\bsteam service\b",
        r"\bsteam measurement capability\b",
    ],

    "2": [
        r"\badvanced steam application\b",
        r"\bsteam application type 2\b",
        r"\bsteam measurement type 2\b",
    ],
}


STEAM_APPLICATION_BLOCKERS = [
    r"\brequired steamapplication configuration\b",
    r"\bsteamapplication configuration\b",
    r"\bsteam application configuration\b",
]


# =============================================================================
# LOAD V5
# =============================================================================

banner("731 RFQ REQUIREMENT EXTRACTION — V6")

print("Loading V5 predictions...")

v5 = pd.read_csv(V5_RESULTS)

print(
    f"V5 predictions: {len(v5):,}"
)

print()
print("V5 columns:")
print(v5.columns.tolist())


# =============================================================================
# LOAD DOCUMENTS
# =============================================================================

print()
print("Loading RFQ documents...")

docs = pd.read_csv(
    DOCUMENTS_FILE
)

print(
    f"RFQ documents: {len(docs):,}"
)


# =============================================================================
# LOAD GROUND TRUTH
# =============================================================================

print()
print("Loading structured requirements...")

gt = pd.read_csv(
    REQUIREMENTS_FILE
)

print(
    f"Ground-truth requirement rows: {len(gt):,}"
)


# =============================================================================
# REMOVE OLD V5 VERSIONS OF TARGET CHARACTERISTICS
# =============================================================================

banner("PREPARING V5 BASELINE")

TARGET_CHARACTERISTICS = {
    "dataInterface",
    "steamApplication",
}

baseline = v5[
    ~v5["characteristic"].isin(
        TARGET_CHARACTERISTICS
    )
].copy()

removed = v5[
    v5["characteristic"].isin(
        TARGET_CHARACTERISTICS
    )
]

print(
    "V5 predictions:",
    f"{len(v5):,}"
)

print(
    "Removed target-characteristic predictions:",
    f"{len(removed):,}"
)

print(
    "Remaining baseline predictions:",
    f"{len(baseline):,}"
)


# =============================================================================
# BUILD V6 SEMANTIC PREDICTIONS
# =============================================================================

banner("V6 SEMANTIC EXTRACTION")

new_predictions = []

for i, row in docs.iterrows():

    rfq_id = row["rfq_id"]
    text = row["rfq_text"]

    # ---------------------------------------------------------
    # DATA INTERFACE
    # ---------------------------------------------------------

    data_preds = semantic_predictions(
        rfq_id=rfq_id,
        text=text,
        characteristic="dataInterface",
        rules=DATA_INTERFACE_RULES,
        blockers=DATA_INTERFACE_BLOCKERS,
        confidence=3.0,
        source_name="V6_DATA_INTERFACE_SEMANTIC"
    )

    new_predictions.extend(
        data_preds
    )

    # ---------------------------------------------------------
    # STEAM APPLICATION
    # ---------------------------------------------------------

    steam_preds = semantic_predictions(
        rfq_id=rfq_id,
        text=text,
        characteristic="steamApplication",
        rules=STEAM_APPLICATION_RULES,
        blockers=STEAM_APPLICATION_BLOCKERS,
        confidence=3.0,
        source_name="V6_STEAM_APPLICATION_SEMANTIC"
    )

    new_predictions.extend(
        steam_preds
    )

    if (i + 1) % 500 == 0:

        print(
            f"Processed {i + 1:,} / {len(docs):,}"
        )


new_df = pd.DataFrame(
    new_predictions
)

print()
print(
    "New V6 predictions:",
    f"{len(new_df):,}"
)


if len(new_df):

    print()
    print(
        "New predictions by characteristic:"
    )

    print(
        new_df["characteristic"]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "New predictions by characteristic/value:"
    )

    print(
        new_df[
            [
                "characteristic",
                "internal_value"
            ]
        ]
        .value_counts()
        .to_string()
    )


# =============================================================================
# COMBINE
# =============================================================================

banner("COMBINING V5 + V6")

all_columns = sorted(
    set(baseline.columns)
    |
    set(new_df.columns)
)

for column in all_columns:

    if column not in baseline.columns:
        baseline[column] = pd.NA

    if column not in new_df.columns:
        new_df[column] = pd.NA


combined = pd.concat(
    [
        baseline[all_columns],
        new_df[all_columns]
    ],
    ignore_index=True
)


# =============================================================================
# DEDUPLICATION
# =============================================================================

combined["confidence"] = pd.to_numeric(
    combined["confidence"],
    errors="coerce"
)

combined = combined.sort_values(
    "confidence",
    ascending=False
)

combined = combined.drop_duplicates(
    subset=[
        "rfq_id",
        "characteristic",
        "internal_value"
    ],
    keep="first"
).reset_index(drop=True)


print(
    "Final V6 predictions:",
    f"{len(combined):,}"
)


# =============================================================================
# GROUND TRUTH SET
# =============================================================================

truth_df = gt[
    [
        "rfq_id",
        "characteristic",
        "internal_value"
    ]
].drop_duplicates()

pred_df = combined[
    [
        "rfq_id",
        "characteristic",
        "internal_value"
    ]
].drop_duplicates()


truth = set(
    map(
        tuple,
        truth_df.itertuples(
            index=False,
            name=None
        )
    )
)

predicted = set(
    map(
        tuple,
        pred_df.itertuples(
            index=False,
            name=None
        )
    )
)


# =============================================================================
# OVERALL METRICS
# =============================================================================

tp_set = predicted & truth
fp_set = predicted - truth
fn_set = truth - predicted

tp = len(tp_set)
fp = len(fp_set)
fn = len(fn_set)

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
    2 * precision * recall /
    (precision + recall)
    if precision + recall
    else 0
)


banner("731 RFQ EXTRACTION V6 RESULTS")

print()
print(
    f"Predicted requirements:    {len(predicted):,}"
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


# =============================================================================
# CHARACTERISTIC METRICS
# =============================================================================

banner("CHARACTERISTIC-LEVEL PERFORMANCE")

characteristics = sorted(
    set(truth_df["characteristic"])
    |
    set(pred_df["characteristic"])
)

metric_rows = []

for characteristic in characteristics:

    truth_c = {
        x for x in truth
        if x[1] == characteristic
    }

    pred_c = {
        x for x in predicted
        if x[1] == characteristic
    }

    ctp = len(
        truth_c & pred_c
    )

    cfp = len(
        pred_c - truth_c
    )

    cfn = len(
        truth_c - pred_c
    )

    cp = (
        ctp / (ctp + cfp)
        if ctp + cfp
        else 0
    )

    cr = (
        ctp / (ctp + cfn)
        if ctp + cfn
        else 0
    )

    cf1 = (
        2 * cp * cr /
        (cp + cr)
        if cp + cr
        else 0
    )

    metric_rows.append({
        "characteristic": characteristic,
        "true_positive": ctp,
        "false_positive": cfp,
        "false_negative": cfn,
        "precision": cp,
        "recall": cr,
        "f1": cf1
    })


char_metrics = pd.DataFrame(
    metric_rows
).sort_values(
    "f1",
    ascending=False
)


print(
    char_metrics[
        [
            "characteristic",
            "precision",
            "recall",
            "f1",
            "true_positive",
            "false_positive",
            "false_negative"
        ]
    ].to_string(index=False)
)


# =============================================================================
# V5 → V6 COMPARISON
# =============================================================================

banner("V5 → V6 TARGET CHARACTERISTIC COMPARISON")

for characteristic in sorted(
    TARGET_CHARACTERISTICS
):

    row = char_metrics[
        char_metrics["characteristic"]
        == characteristic
    ]

    if len(row):

        print()
        print(
            row.to_string(index=False)
        )


# =============================================================================
# ERRORS
# =============================================================================

error_rows = []

for rfq_id, characteristic, value in tp_set:

    error_rows.append({
        "rfq_id": rfq_id,
        "characteristic": characteristic,
        "internal_value": value,
        "error_type": "TRUE_POSITIVE"
    })


for rfq_id, characteristic, value in fp_set:

    error_rows.append({
        "rfq_id": rfq_id,
        "characteristic": characteristic,
        "internal_value": value,
        "error_type": "FALSE_POSITIVE"
    })


for rfq_id, characteristic, value in fn_set:

    error_rows.append({
        "rfq_id": rfq_id,
        "characteristic": characteristic,
        "internal_value": value,
        "error_type": "FALSE_NEGATIVE"
    })


errors = pd.DataFrame(
    error_rows
)


# =============================================================================
# SAVE
# =============================================================================

overall_metrics = pd.DataFrame([
    {
        "version": "V6",
        "predicted_requirements": len(predicted),
        "ground_truth_requirements": len(truth),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
])


combined.to_csv(
    OUT_RESULTS,
    index=False
)

overall_metrics.to_csv(
    OUT_METRICS,
    index=False
)

char_metrics.to_csv(
    OUT_CHAR_METRICS,
    index=False
)

errors.to_csv(
    OUT_ERRORS,
    index=False
)


# =============================================================================
# FINAL
# =============================================================================

banner("V6 FILES SAVED")

print()
print(OUT_RESULTS)
print(OUT_METRICS)
print(OUT_CHAR_METRICS)
print(OUT_ERRORS)

banner("731 RFQ EXTRACTION V6 COMPLETE")