from pathlib import Path
import re
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

PROCESSED = ROOT / "data" / "processed"
RFQ_DIR = ROOT / "data" / "synthetic" / "rfqs"

V4_RESULTS = PROCESSED / "731_rfq_extraction_results_v4.csv"
REQUIREMENTS_FILE = RFQ_DIR / "731_rfq_requirements_v2.csv"
DOCUMENTS_FILE = RFQ_DIR / "731_rfq_documents_v2.csv"

OUT_RESULTS = PROCESSED / "731_rfq_extraction_results_v5.csv"
OUT_METRICS = PROCESSED / "731_rfq_extraction_metrics_v5.csv"
OUT_CHAR_METRICS = PROCESSED / "731_rfq_characteristic_metrics_v5.csv"
OUT_ERRORS = PROCESSED / "731_rfq_extraction_errors_v5.csv"


# =============================================================================
# PRINTING
# =============================================================================

def banner(text):
    print()
    print("=" * 100)
    print(text)
    print("=" * 100)


# =============================================================================
# NORMALISATION
# =============================================================================

def normalise_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = (
        text.replace("–", "-")
            .replace("—", "-")
            .replace("-", "-")
            .replace("’", "'")
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =============================================================================
# SENTENCE SPLITTING
# =============================================================================

def split_sentences(text):

    if pd.isna(text):
        return []

    lines = []

    for line in str(text).splitlines():

        line = line.strip()

        if not line:
            continue

        # Ignore administrative/header/footer material
        lower = line.lower()

        if lower.startswith("customer:"):
            continue

        if lower.startswith("rfq reference:"):
            continue

        if lower.startswith("customer reference:"):
            continue

        if lower.startswith("country:"):
            continue

        if lower.startswith("dear "):
            continue

        if lower.startswith("best regards"):
            continue

        if lower == "procurement team":
            continue

        parts = re.split(r"(?<=[.!?])\s+", line)

        for part in parts:
            part = part.strip()

            if part:
                lines.append(part)

    return lines


# =============================================================================
# DEV CATEGORY RULES
# =============================================================================
#
# IMPORTANT:
#
# These rules only use customer-visible semantic evidence.
#
# We DO NOT infer S/H from:
#
#     "required devCategory configuration"
#
# because that sentence contains no semantic value.
#
# =============================================================================


DEV_CATEGORY_RULES = {

    "F": [

        # Strong liquid-flow wording
        r"\bliquid flow measurement\b",
        r"\bflow measurement for liquid service\b",
        r"\bflow measurement for liquid\b",

        # Other explicit liquid-service wording
        r"\bliquid measurement\b",
        r"\bliquid metering\b",
        r"\bliquid metering application\b",
        r"\bliquid service\b",

        # Water/process-liquid formulations seen in the RFQs
        r"\bwater measurement service\b",
        r"\bprocess water measurement\b",
        r"\bwater-process measurement\b",
    ],

    "G": [

        # Strong gas wording
        r"\bgas measurement\b",
        r"\bgas metering\b",
        r"\bgas metering application\b",
        r"\bgas measurement application\b",

        # More specific gas-service expressions
        r"\bgas flow measurement\b",
        r"\bgas service\b",
    ],

    # S and H intentionally remain conservative.
    #
    # Do NOT create rules from generic phrases such as:
    #
    # "required devCategory configuration"
    #
    # unless the RFQ generator actually provides customer-visible
    # semantic language for these values.
}


# =============================================================================
# FALSE POSITIVE BLOCKERS
# =============================================================================

DEV_CATEGORY_BLOCKERS = [

    # Internal synthetic placeholders
    r"\brequired devcategory configuration\b",
    r"\bdevcategory configuration\b",

    # Generic sentences which do not identify device category
    r"\btechnical and application requirements\b",
    r"\bsuitable measurement solution\b",
    r"\bmeasurement system\b",
]


# =============================================================================
# DEV CATEGORY EXTRACTION
# =============================================================================

def extract_dev_category(rfq_id, text):

    predictions = []

    sentences = split_sentences(text)

    for sentence_index, sentence in enumerate(sentences):

        s = normalise_text(sentence)

        # ---------------------------------------------------------
        # Explicitly ignore internal placeholder wording
        # ---------------------------------------------------------

        if any(re.search(pattern, s) for pattern in DEV_CATEGORY_BLOCKERS):
            continue

        # ---------------------------------------------------------
        # Search semantic category rules
        # ---------------------------------------------------------

        for value, patterns in DEV_CATEGORY_RULES.items():

            for pattern in patterns:

                match = re.search(pattern, s)

                if not match:
                    continue

                predictions.append({
                    "rfq_id": rfq_id,
                    "sentence_index": sentence_index,
                    "characteristic": "devCategory",
                    "internal_value": value,

                    # Deterministic semantic evidence.
                    # Keep > normal V4 threshold.
                    "confidence": 3.0,

                    "characteristic_phrase": "semantic_devCategory_rule",
                    "value_phrase": match.group(0),
                    "source_sentence": sentence,

                    "v5_source": "DEV_CATEGORY_SEMANTIC_RULE"
                })

                # One value per sentence is sufficient
                break

    return predictions


# =============================================================================
# LOAD V4
# =============================================================================

banner("731 RFQ REQUIREMENT EXTRACTION — V5")

print("Loading V4 predictions...")

v4 = pd.read_csv(V4_RESULTS)

print(f"V4 predictions: {len(v4):,}")

print()
print("V4 columns:")
print(v4.columns.tolist())


# =============================================================================
# LOAD DOCUMENTS
# =============================================================================

print()
print("Loading RFQ documents...")

docs = pd.read_csv(DOCUMENTS_FILE)

print(f"RFQ documents: {len(docs):,}")

required_doc_cols = {"rfq_id", "rfq_text"}

missing = required_doc_cols - set(docs.columns)

if missing:
    raise ValueError(
        f"Missing RFQ document columns: {sorted(missing)}"
    )


# =============================================================================
# LOAD GROUND TRUTH
# =============================================================================

print()
print("Loading structured requirements...")

gt = pd.read_csv(REQUIREMENTS_FILE)

print(f"Ground-truth requirement rows: {len(gt):,}")

required_gt_cols = {
    "rfq_id",
    "characteristic",
    "internal_value"
}

missing = required_gt_cols - set(gt.columns)

if missing:
    raise ValueError(
        f"Missing ground-truth columns: {sorted(missing)}"
    )


# =============================================================================
# PREPARE V4
# =============================================================================

banner("PREPARING V4 BASELINE")

# Remove V4 devCategory predictions.
#
# We are replacing them with the new deterministic semantic extractor.

v4_without_dev = v4[
    v4["characteristic"] != "devCategory"
].copy()

print(
    "V4 predictions excluding devCategory:",
    f"{len(v4_without_dev):,}"
)

old_dev = v4[
    v4["characteristic"] == "devCategory"
].copy()

print(
    "Removed V4 devCategory predictions:",
    f"{len(old_dev):,}"
)


# Ensure metadata column exists

if "v5_source" not in v4_without_dev.columns:
    v4_without_dev["v5_source"] = "V4_DECISION_LAYER"


# =============================================================================
# EXTRACT DEV CATEGORY
# =============================================================================

banner("V5 DEV CATEGORY SEMANTIC EXTRACTION")

dev_predictions = []

for i, row in docs.iterrows():

    rfq_id = row["rfq_id"]
    text = row["rfq_text"]

    preds = extract_dev_category(
        rfq_id,
        text
    )

    dev_predictions.extend(preds)

    if (i + 1) % 500 == 0:
        print(
            f"Processed {i + 1:,} / {len(docs):,}"
        )


dev_df = pd.DataFrame(dev_predictions)

print()
print(
    "New devCategory predictions:",
    f"{len(dev_df):,}"
)


if len(dev_df):

    print()
    print("Predicted devCategory distribution:")

    print(
        dev_df["internal_value"]
        .value_counts()
        .to_string()
    )

    print()
    print("Matched semantic phrases:")

    print(
        dev_df[
            ["internal_value", "value_phrase"]
        ]
        .value_counts()
        .head(30)
        .to_string()
    )


# =============================================================================
# ALIGN COLUMNS
# =============================================================================

banner("COMBINING V4 + V5")

# Add missing columns on either side so concat is safe

all_columns = sorted(
    set(v4_without_dev.columns)
    |
    set(dev_df.columns)
)

for col in all_columns:

    if col not in v4_without_dev.columns:
        v4_without_dev[col] = pd.NA

    if col not in dev_df.columns:
        dev_df[col] = pd.NA


combined = pd.concat(
    [
        v4_without_dev[all_columns],
        dev_df[all_columns]
    ],
    ignore_index=True
)


# =============================================================================
# DEDUPLICATE
# =============================================================================

# Sort so strongest prediction survives

if "confidence" in combined.columns:

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
    "Final V5 unique predictions:",
    f"{len(combined):,}"
)


# =============================================================================
# BUILD SETS
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
    if (tp + fp)
    else 0
)

recall = (
    tp / (tp + fn)
    if (tp + fn)
    else 0
)

f1 = (
    2 * precision * recall /
    (precision + recall)
    if (precision + recall)
    else 0
)


banner("731 RFQ EXTRACTION V5 RESULTS")

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

    t = {
        x
        for x in truth
        if x[1] == characteristic
    }

    p = {
        x
        for x in predicted
        if x[1] == characteristic
    }

    ctp = len(t & p)
    cfp = len(p - t)
    cfn = len(t - p)

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


char_metrics = pd.DataFrame(metric_rows)

char_metrics = char_metrics.sort_values(
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
# DEV CATEGORY DETAILS
# =============================================================================

banner("DEV CATEGORY V5 PERFORMANCE")

dev_metric = char_metrics[
    char_metrics["characteristic"] == "devCategory"
]

print(
    dev_metric.to_string(index=False)
)


# =============================================================================
# ERROR FILE
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


errors = pd.DataFrame(error_rows)


# =============================================================================
# SAVE METRICS
# =============================================================================

overall_metrics = pd.DataFrame([
    {
        "version": "V5",
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

banner("FILES SAVED")

print()
print(OUT_RESULTS)
print(OUT_CHAR_METRICS)
print(OUT_METRICS)
print(OUT_ERRORS)

banner("731 RFQ EXTRACTION V5 COMPLETE")