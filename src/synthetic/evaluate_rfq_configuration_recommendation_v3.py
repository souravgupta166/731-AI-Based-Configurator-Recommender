import os
import math
import numpy as np
import pandas as pd
from collections import defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

BASE = "/home/e1546562/thesis-configurator"

V8_PREDICTIONS = os.path.join(
    BASE,
    "data/processed/731_rfq_extraction_results_v8.csv"
)

V5_CONFIGURATIONS = os.path.join(
    BASE,
    "data/synthetic/configurations/731_synthetic_configurations_v5.csv"
)

RFQ_GROUND_TRUTH = os.path.join(
    BASE,
    "data/synthetic/rfqs/731_rfq_ground_truth_v2.csv"
)

STRUCTURED_REQUIREMENTS = os.path.join(
    BASE,
    "data/synthetic/rfqs/731_rfq_requirements_v2.csv"
)

OUTPUT_DIR = os.path.join(
    BASE,
    "data/processed"
)


# ============================================================
# CHARACTERISTICS
# ============================================================

CHARACTERISTICS = [
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


# ============================================================
# OUTPUT FILES
# ============================================================

RECOMMENDATIONS_FILE = os.path.join(
    OUTPUT_DIR,
    "731_rfq_configuration_recommendations_v3.csv"
)

METRICS_FILE = os.path.join(
    OUTPUT_DIR,
    "731_rfq_configuration_recommendation_metrics_v3.csv"
)

TOPK_FILE = os.path.join(
    OUTPUT_DIR,
    "731_rfq_configuration_topk_v3.csv"
)

ERRORS_FILE = os.path.join(
    OUTPUT_DIR,
    "731_rfq_configuration_recommendation_errors_v3.csv"
)

PACKAGE_METRICS_FILE = os.path.join(
    OUTPUT_DIR,
    "731_rfq_configuration_package_metrics_v3.csv"
)

SCORING_STATS_FILE = os.path.join(
    OUTPUT_DIR,
    "731_rfq_configuration_scoring_statistics_v3.csv"
)

CHARACTERISTIC_METRICS_FILE = os.path.join(
    OUTPUT_DIR,
    "731_rfq_configuration_characteristic_metrics_v3.csv"
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def normalize_value(value):
    if pd.isna(value):
        return "NOVALUE"

    value = str(value).strip()

    if value == "":
        return "NOVALUE"

    return value


def safe_float(value, default=0.0):
    try:
        value = float(value)

        if math.isnan(value):
            return default

        return value

    except Exception:
        return default


def canonical_requirement_key(rfq_id, characteristic, value):
    return (
        str(rfq_id),
        str(characteristic),
        normalize_value(value)
    )


# ============================================================
# BUILD VALUE EVIDENCE WEIGHTS
# ============================================================

def build_value_weights(requirements):

    print()
    print("Building value evidence weights...")

    frequency = (
        requirements[
            ["characteristic", "internal_value"]
        ]
        .drop_duplicates()
        .groupby(
            ["characteristic", "internal_value"]
        )
        .size()
    )

    # Actual RFQ-level frequency
    rfq_frequency = (
        requirements[
            ["rfq_id", "characteristic", "internal_value"]
        ]
        .drop_duplicates()
        .groupby(
            ["characteristic", "internal_value"]
        )
        .size()
    )

    total_rfqs = requirements["rfq_id"].nunique()

    weights = {}

    for key, count in rfq_frequency.items():

        freq = safe_float(count)

        # Smoothed inverse-frequency weight.
        #
        # Rare values are more discriminative.
        # Common values still receive a non-zero weight.
        weight = math.log(
            1.0 + total_rfqs / (1.0 + freq)
        )

        weight = max(0.25, min(weight, 5.0))

        weights[key] = weight

    return weights


# ============================================================
# BUILD CHARACTERISTIC WEIGHTS
# ============================================================

def build_characteristic_weights(requirements):

    print("Building characteristic weights...")

    rfq_counts = (
        requirements[
            ["rfq_id", "characteristic"]
        ]
        .drop_duplicates()
        .groupby("characteristic")
        .size()
    )

    total_rfqs = requirements["rfq_id"].nunique()

    weights = {}

    for characteristic in CHARACTERISTICS:

        count = safe_float(
            rfq_counts.get(characteristic, 0)
        )

        if count <= 0:
            weights[characteristic] = 1.0
            continue

        weight = math.log(
            1.0 + total_rfqs / (1.0 + count)
        )

        weights[characteristic] = max(
            0.5,
            min(weight, 3.0)
        )

    return weights


# ============================================================
# BUILD REQUIREMENT LOOKUP
# ============================================================

def build_requirement_lookup(requirements):

    lookup = defaultdict(list)

    for row in requirements.itertuples(index=False):

        rfq_id = str(row.rfq_id)
        characteristic = str(row.characteristic)
        value = normalize_value(row.internal_value)

        requirement_type = str(
            getattr(row, "requirement_type", "MANDATORY")
        ).upper()

        if requirement_type not in {
            "MANDATORY",
            "PREFERRED"
        }:
            requirement_type = "MANDATORY"

        lookup[rfq_id].append({
            "characteristic": characteristic,
            "value": value,
            "requirement_type": requirement_type
        })

    return lookup


# ============================================================
# BUILD CONFIGURATION LOOKUP
# ============================================================

def build_configuration_lookup(configurations):

    config_records = []

    for row in configurations.itertuples(index=False):

        record = {
            "canonical_configuration_id": str(
                row.canonical_configuration_id
            ),
            "package_context": str(
                row.package_context
            ),
        }

        for characteristic in CHARACTERISTICS:

            if hasattr(row, characteristic):

                record[characteristic] = normalize_value(
                    getattr(row, characteristic)
                )

            else:

                record[characteristic] = "NOVALUE"

        config_records.append(record)

    return config_records


# ============================================================
# REQUIREMENT MATCH CLASSIFICATION
# ============================================================

def classify_match(
    required_value,
    candidate_value
):

    required_value = normalize_value(required_value)
    candidate_value = normalize_value(candidate_value)

    # Exact match
    if required_value == candidate_value:

        return "EXACT"

    # Candidate NOVALUE means the candidate does not actively
    # contradict the requirement, but does not satisfy it.
    if candidate_value == "NOVALUE":

        return "NOVALUE"

    # Required NOVALUE is treated as absence/no explicit demand.
    if required_value == "NOVALUE":

        return "NOVALUE"

    # Otherwise it is an explicit value mismatch.
    return "CONTRADICTION"


# ============================================================
# SCORE ONE CONFIGURATION
# ============================================================

def score_configuration(
    requirements,
    configuration,
    value_weights,
    characteristic_weights
):

    mandatory_total = 0.0
    mandatory_matched = 0.0

    preferred_total = 0.0
    preferred_matched = 0.0

    exact_score = 0.0
    contradiction_penalty = 0.0
    novalue_penalty = 0.0

    exact_count = 0
    contradiction_count = 0
    novalue_count = 0

    matched_characteristics = set()

    mandatory_requirements = 0
    preferred_requirements = 0

    for requirement in requirements:

        characteristic = requirement["characteristic"]
        required_value = requirement["value"]
        requirement_type = requirement["requirement_type"]

        candidate_value = configuration.get(
            characteristic,
            "NOVALUE"
        )

        match_type = classify_match(
            required_value,
            candidate_value
        )

        characteristic_weight = characteristic_weights.get(
            characteristic,
            1.0
        )

        value_weight = value_weights.get(
            (
                characteristic,
                required_value
            ),
            1.0
        )

        evidence_weight = (
            characteristic_weight *
            value_weight
        )

        # ----------------------------------------------------
        # MANDATORY
        # ----------------------------------------------------

        if requirement_type == "MANDATORY":

            mandatory_requirements += 1

            mandatory_total += evidence_weight

            if match_type == "EXACT":

                mandatory_matched += evidence_weight

                exact_score += (
                    2.0 * evidence_weight
                )

                exact_count += 1

                matched_characteristics.add(
                    characteristic
                )

            elif match_type == "CONTRADICTION":

                contradiction_penalty += (
                    4.0 * evidence_weight
                )

                contradiction_count += 1

            elif match_type == "NOVALUE":

                novalue_penalty += (
                    0.75 * evidence_weight
                )

                novalue_count += 1

        # ----------------------------------------------------
        # PREFERRED
        # ----------------------------------------------------

        else:

            preferred_requirements += 1

            preferred_total += evidence_weight

            if match_type == "EXACT":

                preferred_matched += evidence_weight

                exact_score += (
                    0.75 * evidence_weight
                )

                exact_count += 1

                matched_characteristics.add(
                    characteristic
                )

            elif match_type == "CONTRADICTION":

                contradiction_penalty += (
                    1.5 * evidence_weight
                )

                contradiction_count += 1

            elif match_type == "NOVALUE":

                novalue_penalty += (
                    0.25 * evidence_weight
                )

                novalue_count += 1

    # --------------------------------------------------------
    # COVERAGE
    # --------------------------------------------------------

    if mandatory_total > 0:

        mandatory_coverage = (
            mandatory_matched /
            mandatory_total
        )

    else:

        mandatory_coverage = 1.0

    if preferred_total > 0:

        preferred_coverage = (
            preferred_matched /
            preferred_total
        )

    else:

        preferred_coverage = 1.0

    total_requirements = (
        mandatory_requirements +
        preferred_requirements
    )

    total_matched = exact_count

    if total_requirements > 0:

        overall_coverage = (
            total_matched /
            total_requirements
        )

    else:

        overall_coverage = 0.0

    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    score = (

        # Strong emphasis on mandatory satisfaction
        8.0 * mandatory_coverage

        +

        # Preferred requirements still matter
        2.5 * preferred_coverage

        +

        # Exact evidence
        exact_score

        +

        # Overall coverage
        3.0 * overall_coverage

        -

        # Strong contradiction penalty
        contradiction_penalty

        -

        # Smaller penalty for NOVALUE
        novalue_penalty
    )

    return {
        "score": score,
        "mandatory_coverage": mandatory_coverage,
        "preferred_coverage": preferred_coverage,
        "overall_coverage": overall_coverage,
        "exact_count": exact_count,
        "contradiction_count": contradiction_count,
        "novalue_count": novalue_count,
        "matched_characteristics": len(
            matched_characteristics
        ),
    }


# ============================================================
# RANK CONFIGURATIONS
# ============================================================

def rank_configurations(
    rfq_requirements,
    configurations,
    value_weights,
    characteristic_weights,
    package_aware=False,
    target_package=None
):

    scored = []

    for configuration in configurations:

        if (
            package_aware
            and target_package is not None
            and configuration["package_context"]
            != target_package
        ):

            continue

        metrics = score_configuration(
            rfq_requirements,
            configuration,
            value_weights,
            characteristic_weights
        )

        scored.append({
            **configuration,
            **metrics
        })

    if not scored:

        return []

    scored.sort(
        key=lambda x: (
            x["score"],
            x["mandatory_coverage"],
            x["preferred_coverage"],
            x["overall_coverage"],
            x["exact_count"],
            -x["contradiction_count"]
        ),
        reverse=True
    )

    return scored


# ============================================================
# EVALUATE ONE MODE
# ============================================================

def evaluate_mode(
    mode_name,
    requirements_by_rfq,
    configurations,
    ground_truth,
    value_weights,
    characteristic_weights,
    rfq_package_lookup,
    package_aware=False
):

    print()
    print("=" * 80)
    print(
        f"MODE: {mode_name} | "
        f"package_aware={package_aware}"
    )
    print("=" * 80)

    results = []
    topk_rows = []
    error_rows = []
    scoring_rows = []

    rfq_ids = sorted(
        ground_truth["rfq_id"].astype(str).unique()
    )

    for idx, rfq_id in enumerate(rfq_ids, start=1):

        requirements = requirements_by_rfq.get(
            rfq_id,
            []
        )

        if not requirements:

            continue

        target_package = rfq_package_lookup.get(
            rfq_id
        )

        ranked = rank_configurations(
            requirements,
            configurations,
            value_weights,
            characteristic_weights,
            package_aware=package_aware,
            target_package=target_package
        )

        if not ranked:

            continue

        truth_rows = ground_truth[
            ground_truth["rfq_id"].astype(str)
            == rfq_id
        ]

        true_configuration_ids = set(
            truth_rows[
                "canonical_configuration_id"
            ]
            .astype(str)
            .tolist()
        )

        true_rank = None

        for rank, candidate in enumerate(
            ranked,
            start=1
        ):

            if (
                candidate[
                    "canonical_configuration_id"
                ]
                in true_configuration_ids
            ):

                true_rank = rank
                break

        top1 = int(
            true_rank is not None
            and true_rank <= 1
        )

        top3 = int(
            true_rank is not None
            and true_rank <= 3
        )

        top5 = int(
            true_rank is not None
            and true_rank <= 5
        )

        top10 = int(
            true_rank is not None
            and true_rank <= 10
        )

        reciprocal_rank = (
            1.0 / true_rank
            if true_rank is not None
            else 0.0
        )

        top_candidate = ranked[0]

        results.append({
            "rfq_id": rfq_id,
            "target_package": target_package,
            "true_rank": true_rank,
            "top1": top1,
            "top3": top3,
            "top5": top5,
            "top10": top10,
            "reciprocal_rank": reciprocal_rank,
            "recommended_configuration_id":
                top_candidate[
                    "canonical_configuration_id"
                ],
            "recommended_package":
                top_candidate[
                    "package_context"
                ],
            "recommended_score":
                top_candidate["score"],
            "true_configuration_ids":
                "|".join(
                    sorted(true_configuration_ids)
                ),
            "mode": mode_name,
            "package_aware": package_aware,
        })

        # Top-K
        for rank, candidate in enumerate(
            ranked[:10],
            start=1
        ):

            is_true = int(
                candidate[
                    "canonical_configuration_id"
                ]
                in true_configuration_ids
            )

            topk_rows.append({
                "rfq_id": rfq_id,
                "rank": rank,
                "configuration_id":
                    candidate[
                        "canonical_configuration_id"
                    ],
                "package_context":
                    candidate["package_context"],
                "score": candidate["score"],
                "mandatory_coverage":
                    candidate["mandatory_coverage"],
                "preferred_coverage":
                    candidate["preferred_coverage"],
                "overall_coverage":
                    candidate["overall_coverage"],
                "exact_count":
                    candidate["exact_count"],
                "contradiction_count":
                    candidate["contradiction_count"],
                "novalue_count":
                    candidate["novalue_count"],
                "matched_characteristics":
                    candidate[
                        "matched_characteristics"
                    ],
                "is_ground_truth": is_true,
                "mode": mode_name,
                "package_aware": package_aware,
            })

        scoring_rows.append({
            "rfq_id": rfq_id,
            "candidate_count": len(ranked),
            "top_score": ranked[0]["score"],
            "top_mandatory_coverage":
                ranked[0]["mandatory_coverage"],
            "top_preferred_coverage":
                ranked[0]["preferred_coverage"],
            "top_overall_coverage":
                ranked[0]["overall_coverage"],
            "top_exact_count":
                ranked[0]["exact_count"],
            "top_contradiction_count":
                ranked[0]["contradiction_count"],
            "top_novalue_count":
                ranked[0]["novalue_count"],
            "true_rank": true_rank,
            "mode": mode_name,
            "package_aware": package_aware,
        })

        if true_rank is None or true_rank > 10:

            error_rows.append({
                "rfq_id": rfq_id,
                "true_rank": true_rank,
                "target_package": target_package,
                "recommended_configuration_id":
                    top_candidate[
                        "canonical_configuration_id"
                    ],
                "recommended_package":
                    top_candidate[
                        "package_context"
                    ],
                "recommended_score":
                    top_candidate["score"],
                "mode": mode_name,
                "package_aware": package_aware,
            })

        if idx % 500 == 0:

            print(
                f"Processed {idx:,} / "
                f"{len(rfq_ids):,}"
            )

    result_df = pd.DataFrame(results)

    if len(result_df) == 0:

        return {
            "results": result_df,
            "topk": pd.DataFrame(topk_rows),
            "errors": pd.DataFrame(error_rows),
            "scoring": pd.DataFrame(scoring_rows),
            "metrics": {}
        }

    metrics = {
        "mode": mode_name,
        "package_aware": package_aware,
        "rfqs_evaluated": len(result_df),
        "top1_accuracy":
            result_df["top1"].mean(),
        "top3_accuracy":
            result_df["top3"].mean(),
        "top5_accuracy":
            result_df["top5"].mean(),
        "top10_accuracy":
            result_df["top10"].mean(),
        "MRR":
            result_df["reciprocal_rank"].mean(),
        "mean_true_rank":
            result_df["true_rank"].dropna().mean(),
        "median_true_rank":
            result_df["true_rank"].dropna().median(),
        "unranked":
            result_df["true_rank"].isna().sum(),
    }

    return {
        "results": result_df,
        "topk": pd.DataFrame(topk_rows),
        "errors": pd.DataFrame(error_rows),
        "scoring": pd.DataFrame(scoring_rows),
        "metrics": metrics
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("731 RFQ CONFIGURATION RECOMMENDATION — V3")
    print("=" * 100)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD V8
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("LOADING V8 EXTRACTED REQUIREMENTS")
    print("=" * 80)

    v8 = pd.read_csv(
        V8_PREDICTIONS,
        low_memory=False
    )

    print(
        "V8 predictions:",
        len(v8)
    )

    # --------------------------------------------------------
    # LOAD CONFIGURATIONS
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("LOADING V5 TECHNICAL CONFIGURATIONS")
    print("=" * 80)

    configurations_df = pd.read_csv(
        V5_CONFIGURATIONS,
        low_memory=False
    )

    print(
        "V5 configurations:",
        len(configurations_df)
    )

    configurations = build_configuration_lookup(
        configurations_df
    )

    print(
        "Unique configurations:",
        len(configurations)
    )

    # --------------------------------------------------------
    # LOAD RFQ GROUND TRUTH
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("LOADING RFQ GROUND TRUTH")
    print("=" * 80)

    ground_truth = pd.read_csv(
        RFQ_GROUND_TRUTH,
        low_memory=False
    )

    ground_truth["rfq_id"] = (
        ground_truth["rfq_id"]
        .astype(str)
    )

    ground_truth[
        "canonical_configuration_id"
    ] = (
        ground_truth[
            "canonical_configuration_id"
        ]
        .astype(str)
    )

    print(
        "Ground-truth RFQs:",
        ground_truth["rfq_id"].nunique()
    )

    # --------------------------------------------------------
    # LOAD STRUCTURED REQUIREMENTS
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("LOADING STRUCTURED REQUIREMENTS")
    print("=" * 80)

    requirements = pd.read_csv(
        STRUCTURED_REQUIREMENTS,
        low_memory=False
    )

    print(
        "Ground-truth requirements:",
        len(requirements)
    )

    requirements["rfq_id"] = (
        requirements["rfq_id"]
        .astype(str)
    )

    requirements["characteristic"] = (
        requirements["characteristic"]
        .astype(str)
        .str.strip()
    )

    requirements["internal_value"] = (
        requirements["internal_value"]
        .apply(normalize_value)
    )

    requirements["requirement_type"] = (
        requirements["requirement_type"]
        .astype(str)
        .str.upper()
    )

    # --------------------------------------------------------
    # BUILD WEIGHTS
    # --------------------------------------------------------

    print()
    print(
        "Building evidence-weighted "
        "V3 scoring model..."
    )

    value_weights = build_value_weights(
        requirements
    )

    characteristic_weights = (
        build_characteristic_weights(
            requirements
        )
    )

    print(
        "Value evidence weights:",
        len(value_weights)
    )

    print(
        "Characteristic weights:",
        len(characteristic_weights)
    )

    # --------------------------------------------------------
    # BUILD ORACLE REQUIREMENT LOOKUP
    # --------------------------------------------------------

    oracle_lookup = build_requirement_lookup(
        requirements
    )

    # --------------------------------------------------------
    # BUILD V8 REQUIREMENT LOOKUP
    # --------------------------------------------------------

    extracted = v8.copy()

    extracted["rfq_id"] = (
        extracted["rfq_id"]
        .astype(str)
    )

    extracted["characteristic"] = (
        extracted["characteristic"]
        .astype(str)
        .str.strip()
    )

    extracted["internal_value"] = (
        extracted["internal_value"]
        .apply(normalize_value)
    )

    # V8 does not necessarily carry requirement_type.
    #
    # Therefore extracted requirements are treated as
    # MANDATORY by default for the main benchmark.
    extracted["requirement_type"] = "MANDATORY"

    extracted = extracted[
        [
            "rfq_id",
            "characteristic",
            "internal_value",
            "requirement_type"
        ]
    ].drop_duplicates()

    extracted_lookup = build_requirement_lookup(
        extracted
    )

    print(
        "RFQs with extracted requirements:",
        len(extracted_lookup)
    )

    print(
        "RFQs with oracle requirements:",
        len(oracle_lookup)
    )

    # --------------------------------------------------------
    # PACKAGE LOOKUP
    # --------------------------------------------------------

    rfq_package_lookup = (
        ground_truth[
            ["rfq_id", "target_package"]
        ]
        .drop_duplicates(
            subset=["rfq_id"]
        )
        .set_index("rfq_id")[
            "target_package"
        ]
        .astype(str)
        .to_dict()
    )

    # --------------------------------------------------------
    # RUN EXPERIMENTS
    # --------------------------------------------------------

    all_metrics = []
    all_results = []
    all_topk = []
    all_errors = []
    all_scoring = []

    experiments = [
        (
            "V8_EXTRACTED",
            extracted_lookup
        ),
        (
            "ORACLE",
            oracle_lookup
        ),
    ]

    for mode_name, lookup in experiments:

        for package_aware in [False, True]:

            output = evaluate_mode(
                mode_name,
                lookup,
                configurations,
                ground_truth,
                value_weights,
                characteristic_weights,
                rfq_package_lookup,
                package_aware=package_aware
            )

            metrics = output["metrics"]

            all_metrics.append(metrics)

            all_results.append(
                output["results"]
            )

            all_topk.append(
                output["topk"]
            )

            all_errors.append(
                output["errors"]
            )

            all_scoring.append(
                output["scoring"]
            )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(
        all_metrics
    )

    results_df = pd.concat(
        all_results,
        ignore_index=True
    )

    topk_df = pd.concat(
        all_topk,
        ignore_index=True
    )

    errors_df = pd.concat(
        all_errors,
        ignore_index=True
    )

    scoring_df = pd.concat(
        all_scoring,
        ignore_index=True
    )

    metrics_df.to_csv(
        METRICS_FILE,
        index=False
    )

    results_df.to_csv(
        RECOMMENDATIONS_FILE,
        index=False
    )

    topk_df.to_csv(
        TOPK_FILE,
        index=False
    )

    errors_df.to_csv(
        ERRORS_FILE,
        index=False
    )

    scoring_df.to_csv(
        SCORING_STATS_FILE,
        index=False
    )

    # --------------------------------------------------------
    # PACKAGE METRICS
    # --------------------------------------------------------

    package_rows = []

    for (
        mode,
        package_aware
    ), group in results_df.groupby(
        ["mode", "package_aware"]
    ):

        for package, package_group in group.groupby(
            "target_package"
        ):

            package_rows.append({
                "mode": mode,
                "package_aware": package_aware,
                "target_package": package,
                "rfqs": len(package_group),
                "top1":
                    package_group["top1"].mean(),
                "top3":
                    package_group["top3"].mean(),
                "top5":
                    package_group["top5"].mean(),
                "top10":
                    package_group["top10"].mean(),
                "MRR":
                    package_group[
                        "reciprocal_rank"
                    ].mean(),
                "mean_rank":
                    package_group[
                        "true_rank"
                    ].dropna().mean(),
                "median_rank":
                    package_group[
                        "true_rank"
                    ].dropna().median()
            })

    package_metrics_df = pd.DataFrame(
        package_rows
    )

    package_metrics_df.to_csv(
        PACKAGE_METRICS_FILE,
        index=False
    )

    # --------------------------------------------------------
    # CHARACTERISTIC-LEVEL ORACLE ANALYSIS
    # --------------------------------------------------------

    characteristic_rows = []

    # Compare ground truth configuration values
    # against the top-ranked configuration.

    for (
        mode,
        package_aware
    ), group in results_df.groupby(
        ["mode", "package_aware"]
    ):

        for characteristic in CHARACTERISTICS:

            tp = 0
            fp = 0
            fn = 0

            for row in group.itertuples():

                rfq_id = row.rfq_id

                predicted_id = (
                    row.recommended_configuration_id
                )

                predicted_config = next(
                    (
                        c for c in configurations
                        if c[
                            "canonical_configuration_id"
                        ] == predicted_id
                    ),
                    None
                )

                if predicted_config is None:
                    continue

                gt_rows = requirements[
                    requirements["rfq_id"]
                    == rfq_id
                ]

                gt_values = set(
                    gt_rows[
                        gt_rows[
                            "characteristic"
                        ] == characteristic
                    ]["internal_value"]
                    .astype(str)
                    .tolist()
                )

                predicted_value = (
                    predicted_config.get(
                        characteristic,
                        "NOVALUE"
                    )
                )

                if predicted_value in gt_values:

                    tp += 1

                elif gt_values:

                    fp += 1
                    fn += 1

            precision = (
                tp / (tp + fp)
                if (tp + fp) else 0.0
            )

            recall = (
                tp / (tp + fn)
                if (tp + fn) else 0.0
            )

            f1 = (
                2 * precision * recall /
                (precision + recall)
                if precision + recall
                else 0.0
            )

            characteristic_rows.append({
                "mode": mode,
                "package_aware": package_aware,
                "characteristic": characteristic,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1
            })

    characteristic_df = pd.DataFrame(
        characteristic_rows
    )

    characteristic_df.to_csv(
        CHARACTERISTIC_METRICS_FILE,
        index=False
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("731 RFQ CONFIGURATION RECOMMENDATION V3 RESULTS")
    print("=" * 100)

    for row in all_metrics:

        print()
        print(
            f"{row['mode']} | "
            f"package_aware="
            f"{row['package_aware']}"
        )

        print(
            f"RFQs evaluated:       "
            f"{row['rfqs_evaluated']:,}"
        )

        print(
            f"Top-1 accuracy:       "
            f"{row['top1_accuracy']:.4f}"
        )

        print(
            f"Top-3 accuracy:       "
            f"{row['top3_accuracy']:.4f}"
        )

        print(
            f"Top-5 accuracy:       "
            f"{row['top5_accuracy']:.4f}"
        )

        print(
            f"Top-10 accuracy:      "
            f"{row['top10_accuracy']:.4f}"
        )

        print(
            f"MRR:                  "
            f"{row['MRR']:.4f}"
        )

        print(
            f"Mean true rank:       "
            f"{row['mean_true_rank']:.2f}"
        )

        print(
            f"Median true rank:     "
            f"{row['median_true_rank']:.2f}"
        )

        print(
            f"Unranked:             "
            f"{row['unranked']:,}"
        )

    # --------------------------------------------------------
    # FILE SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("V3 FILES SAVED")
    print("=" * 100)

    print()
    print("Recommendations:")
    print(RECOMMENDATIONS_FILE)

    print()
    print("Metrics:")
    print(METRICS_FILE)

    print()
    print("Top-K rankings:")
    print(TOPK_FILE)

    print()
    print("Recommendation errors:")
    print(ERRORS_FILE)

    print()
    print("Package metrics:")
    print(PACKAGE_METRICS_FILE)

    print()
    print("Scoring statistics:")
    print(SCORING_STATS_FILE)

    print()
    print("Characteristic metrics:")
    print(CHARACTERISTIC_METRICS_FILE)

    print()
    print("=" * 100)
    print(
        "731 RFQ CONFIGURATION "
        "RECOMMENDATION V3 COMPLETE"    
    )
    print("=" * 100)


if __name__ == "__main__":
    main()