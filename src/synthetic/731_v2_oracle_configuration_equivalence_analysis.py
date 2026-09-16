import os
import numpy as np
import pandas as pd

# ============================================================
# 731 V2 ORACLE CONFIGURATION EQUIVALENCE ANALYSIS
# ============================================================
#
# Purpose
# -------
# The previous diagnostic showed:
#   - 5,000 configuration rows
#   - 1,979 canonical IDs
#   - 1,097 IDs with duplicate rows
#   - all duplicate IDs classified as exact duplicates under the
#     19 technical characteristics used for comparison
#   - Oracle ID-level Top-1 accuracy = 50.58%
#
# This script tests whether some apparent ID-level failures are
# actually technically equivalent configurations.
#
# It reports:
#   1. Exact canonical-ID Top-1 accuracy
#   2. Technical-equivalence Top-1 accuracy
#   3. Package + technical-equivalence Top-1 accuracy
#   4. Top-3 / Top-5 / Top-10 technical-equivalence recall
#   5. MRR using technical equivalence
#   6. Exact technical-signature equivalence
#   7. Extra-optional-feature cases
#   8. True technical differences
#   9. Mandatory/preferred requirement mismatches
#
# IMPORTANT:
# The 731 configurator data is product/rule knowledge.
# It is NOT historical engineer-decision data.
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE = "/home/e1546562/thesis-configurator"

TOPK_FILE = os.path.join(
    BASE,
    "data/processed/731_rfq_configuration_topk_v2.csv"
)

RECOMMENDATIONS_FILE = os.path.join(
    BASE,
    "data/processed/731_rfq_configuration_recommendations_v2.csv"
)

CONFIGURATIONS_FILE = os.path.join(
    BASE,
    "data/synthetic/configurations/731_synthetic_configurations_v5.csv"
)

REQUIREMENTS_FILE = os.path.join(
    BASE,
    "data/synthetic/rfqs/731_rfq_requirements_v2.csv"
)

OUTPUT_DIR = os.path.join(
    BASE,
    "data/processed"
)

PAIR_FILE = os.path.join(
    OUTPUT_DIR,
    "731_v2_oracle_equivalence_pair_analysis.csv"
)

RANK_FILE = os.path.join(
    OUTPUT_DIR,
    "731_v2_oracle_technical_equivalence_rank_analysis.csv"
)

DIFF_FILE = os.path.join(
    OUTPUT_DIR,
    "731_v2_oracle_equivalence_characteristic_differences.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "731_v2_oracle_equivalence_summary.csv"
)

CATEGORY_FILE = os.path.join(
    OUTPUT_DIR,
    "731_v2_oracle_equivalence_case_categories.csv"
)

SIGNATURE_FILE = os.path.join(
    OUTPUT_DIR,
    "731_v2_configuration_signature_analysis.csv"
)


# ============================================================
# TECHNICAL CHARACTERISTICS
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
# HELPERS
# ============================================================

def norm(value):
    """Normalize configuration values for reliable comparison."""
    if pd.isna(value):
        return "NOVALUE"

    value = str(value).strip()

    if value == "":
        return "NOVALUE"

    return value


def require_file(path):
    """Raise a clear error if an input file does not exist."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}\n"
        )


def normalize_id_series(series):
    return (
        series.astype(str)
        .str.strip()
    )


def load_configurations():
    """Load configurations while preserving duplicate rows."""
    require_file(CONFIGURATIONS_FILE)

    df = pd.read_csv(
        CONFIGURATIONS_FILE,
        low_memory=False
    )

    required = {"canonical_configuration_id"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Configuration file is missing columns: "
            + str(sorted(missing))
        )

    df["canonical_configuration_id"] = normalize_id_series(
        df["canonical_configuration_id"]
    )

    for characteristic in CHARACTERISTICS:
        if characteristic in df.columns:
            df[characteristic] = df[characteristic].apply(norm)

    if "package_context" in df.columns:
        df["package_context"] = (
            df["package_context"]
            .astype(str)
            .str.strip()
        )

    return df


def build_configuration_lookup(configurations):
    """
    Build one representative row per canonical ID.

    Duplicate IDs are not discarded from the analysis:
    they are separately measured by the signature analysis.
    """
    representative = (
        configurations
        .drop_duplicates(
            subset=["canonical_configuration_id"],
            keep="first"
        )
    )

    return {
        str(row["canonical_configuration_id"]).strip(): row
        for _, row in representative.iterrows()
    }


def get_value(configuration, characteristic):
    if configuration is None:
        return "NOVALUE"

    if characteristic not in configuration.index:
        return "NOVALUE"

    return norm(configuration[characteristic])


def signature(configuration):
    """
    Technical signature across the 19 comparison characteristics.
    """
    if configuration is None:
        return tuple("NOVALUE" for _ in CHARACTERISTICS)

    return tuple(
        get_value(configuration, characteristic)
        for characteristic in CHARACTERISTICS
    )


def package_value(configuration):
    if configuration is None:
        return "NOVALUE"

    if "package_context" not in configuration.index:
        return "NOVALUE"

    return norm(configuration["package_context"])


def safe_float(value):
    return pd.to_numeric(value, errors="coerce")


def requirement_context(requirements, rfq_id, characteristic):
    """
    Return the strongest requirement context for one RFQ/characteristic.

    Priority:
        MANDATORY > PREFERRED > any other requirement
    """
    if requirements is None:
        return "NOT_MENTIONED", "NOVALUE"

    matches = requirements[
        (requirements["rfq_id"] == rfq_id)
        & (requirements["characteristic"] == characteristic)
    ]

    if len(matches) == 0:
        return "NOT_MENTIONED", "NOVALUE"

    if "requirement_type" in matches.columns:
        types = (
            matches["requirement_type"]
            .astype(str)
            .str.upper()
            .tolist()
        )

        if "MANDATORY" in types:
            row = matches[
                matches["requirement_type"].astype(str).str.upper()
                == "MANDATORY"
            ].iloc[0]
            req_type = "MANDATORY"

        elif "PREFERRED" in types:
            row = matches[
                matches["requirement_type"].astype(str).str.upper()
                == "PREFERRED"
            ].iloc[0]
            req_type = "PREFERRED"

        else:
            row = matches.iloc[0]
            req_type = str(
                row.get("requirement_type", "UNKNOWN")
            ).upper()

    else:
        row = matches.iloc[0]
        req_type = "UNKNOWN"

    if "internal_value" in row.index:
        required_value = norm(row["internal_value"])
    else:
        required_value = "NOVALUE"

    return req_type, required_value


def changed_characteristics(
    true_configuration,
    predicted_configuration
):
    differences = []

    for characteristic in CHARACTERISTICS:
        true_value = get_value(
            true_configuration,
            characteristic
        )

        predicted_value = get_value(
            predicted_configuration,
            characteristic
        )

        if true_value == predicted_value:
            continue

        if (
            true_value == "NOVALUE"
            and predicted_value != "NOVALUE"
        ):
            relation = "PREDICTED_ADDITIONAL_VALUE"

        elif (
            predicted_value == "NOVALUE"
            and true_value != "NOVALUE"
        ):
            relation = "PREDICTED_MISSING_VALUE"

        else:
            relation = "VALUE_CONFLICT"

        differences.append(
            {
                "characteristic": characteristic,
                "true_value": true_value,
                "predicted_value": predicted_value,
                "relation": relation,
            }
        )

    return differences


def classify_pair(
    true_configuration,
    predicted_configuration,
    requirements,
    rfq_id,
    true_package,
    predicted_package,
    is_exact_id
):
    """
    Classify the predicted configuration relative to the true one.

    Categories are mutually exclusive, in this order:

    EXACT_ID_MATCH
    TECHNICALLY_EQUIVALENT
    EXTRA_OPTIONAL_FEATURES
    TRUE_TECHNICAL_DIFFERENCE
    """
    diffs = changed_characteristics(
        true_configuration,
        predicted_configuration
    )

    same_signature = len(diffs) == 0
    same_package = (
        norm(true_package) == norm(predicted_package)
    )

    if is_exact_id:
        category = "EXACT_ID_MATCH"

    elif same_signature:
        category = "TECHNICALLY_EQUIVALENT"

    else:
        extra_optional_only = True

        for diff in diffs:
            req_type, required_value = requirement_context(
                requirements,
                rfq_id,
                diff["characteristic"]
            )

            # If the changed characteristic is mandatory or preferred,
            # it is not merely an optional extra.
            if req_type in {"MANDATORY", "PREFERRED"}:
                extra_optional_only = False
                break

            # A value conflict is a genuine technical difference,
            # not an additional optional capability.
            if diff["relation"] != "PREDICTED_ADDITIONAL_VALUE":
                extra_optional_only = False
                break

        if extra_optional_only:
            category = "EXTRA_OPTIONAL_FEATURES"
        else:
            category = "TRUE_TECHNICAL_DIFFERENCE"

    if category in {
        "EXACT_ID_MATCH",
        "TECHNICALLY_EQUIVALENT"
    }:
        technical_equivalent = 1
    else:
        technical_equivalent = 0

    package_technical_equivalent = int(
        technical_equivalent == 1
        and same_package
    )

    return {
        "category": category,
        "technical_equivalent": technical_equivalent,
        "same_technical_signature": int(same_signature),
        "same_package": int(same_package),
        "package_technical_equivalent": package_technical_equivalent,
        "differences": diffs,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("731 V2 ORACLE CONFIGURATION EQUIVALENCE ANALYSIS")
    print("=" * 100)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # ========================================================
    # 1. LOAD TOP-K
    # ========================================================

    print("\n[1/6] Loading Oracle Top-K results...")

    require_file(TOPK_FILE)

    topk = pd.read_csv(
        TOPK_FILE,
        low_memory=False
    )

    required_topk = {
        "rfq_id",
        "mode",
        "rank",
        "canonical_configuration_id",
        "package_context",
        "score",
    }

    missing = required_topk - set(topk.columns)

    if missing:
        raise ValueError(
            "Top-K file is missing columns: "
            + str(sorted(missing))
        )

    topk["rfq_id"] = normalize_id_series(
        topk["rfq_id"]
    )

    topk["mode"] = (
        topk["mode"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    topk["canonical_configuration_id"] = normalize_id_series(
        topk["canonical_configuration_id"]
    )

    topk["rank"] = pd.to_numeric(
        topk["rank"],
        errors="coerce"
    )

    topk["score"] = pd.to_numeric(
        topk["score"],
        errors="coerce"
    )

    topk["package_context"] = (
        topk["package_context"]
        .apply(norm)
    )

    oracle = (
        topk[
            topk["mode"] == "ORACLE"
        ]
        .copy()
    )

    print(f"Total Top-K rows: {len(topk):,}")
    print(f"Oracle rows: {len(oracle):,}")
    print(f"Oracle RFQs: {oracle['rfq_id'].nunique():,}")

    # ========================================================
    # 2. LOAD RECOMMENDATION METADATA
    # ========================================================

    print("\n[2/6] Loading recommendation metadata...")

    require_file(RECOMMENDATIONS_FILE)

    recommendations = pd.read_csv(
        RECOMMENDATIONS_FILE,
        low_memory=False
    )

    required_recommendation = {
        "rfq_id",
        "mode",
        "true_configuration_id",
    }

    missing = (
        required_recommendation
        - set(recommendations.columns)
    )

    if missing:
        raise ValueError(
            "Recommendation file is missing columns: "
            + str(sorted(missing))
        )

    recommendations["rfq_id"] = normalize_id_series(
        recommendations["rfq_id"]
    )

    recommendations["mode"] = (
        recommendations["mode"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    for column in [
        "true_configuration_id",
        "recommended_configuration_id",
        "target_package",
        "recommended_package",
    ]:
        if column in recommendations.columns:
            recommendations[column] = normalize_id_series(
                recommendations[column]
            )

    oracle_recommendations = (
        recommendations[
            recommendations["mode"] == "ORACLE"
        ]
        .copy()
    )

    print(
        "Oracle recommendation rows: "
        f"{len(oracle_recommendations):,}"
    )

    # One true-config metadata record per RFQ.
    if "package_aware" in oracle_recommendations.columns:

        package_aware_string = (
            oracle_recommendations["package_aware"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        package_aware = oracle_recommendations[
            package_aware_string == "true"
        ].copy()

        if len(package_aware) > 0:
            true_records = (
                package_aware
                .sort_values("rfq_id")
                .drop_duplicates(
                    subset=["rfq_id"],
                    keep="first"
                )
            )
        else:
            true_records = (
                oracle_recommendations
                .sort_values("rfq_id")
                .drop_duplicates(
                    subset=["rfq_id"],
                    keep="first"
                )
            )

    else:
        true_records = (
            oracle_recommendations
            .sort_values("rfq_id")
            .drop_duplicates(
                subset=["rfq_id"],
                keep="first"
            )
        )

    true_map = (
        true_records
        .set_index("rfq_id")
        .to_dict("index")
    )

    print(
        "RFQs with true configuration metadata: "
        f"{len(true_map):,}"
    )

    # ========================================================
    # 3. LOAD CONFIGURATIONS
    # ========================================================

    print("\n[3/6] Loading V5 configurations...")

    configurations = load_configurations()

    print(
        f"Configuration rows: {len(configurations):,}"
    )

    print(
        "Unique canonical IDs: "
        f"{configurations['canonical_configuration_id'].nunique():,}"
    )

    configuration_lookup = build_configuration_lookup(
        configurations
    )

    # ========================================================
    # 4. LOAD REQUIREMENTS
    # ========================================================

    print("\n[4/6] Loading structured RFQ requirements...")

    requirements = None

    if os.path.exists(REQUIREMENTS_FILE):

        requirements = pd.read_csv(
            REQUIREMENTS_FILE,
            low_memory=False
        )

        required_requirement = {
            "rfq_id",
            "characteristic",
        }

        missing = (
            required_requirement
            - set(requirements.columns)
        )

        if missing:
            raise ValueError(
                "Requirements file is missing columns: "
                + str(sorted(missing))
            )

        requirements["rfq_id"] = normalize_id_series(
            requirements["rfq_id"]
        )

        requirements["characteristic"] = normalize_id_series(
            requirements["characteristic"]
        )

        if "requirement_type" in requirements.columns:
            requirements["requirement_type"] = (
                requirements["requirement_type"]
                .astype(str)
                .str.strip()
                .str.upper()
            )

        if "internal_value" in requirements.columns:
            requirements["internal_value"] = (
                requirements["internal_value"]
                .apply(norm)
            )

        print(
            f"Requirement rows: {len(requirements):,}"
        )

    else:
        print(
            "WARNING: Requirements file not found. "
            "Requirement-context categories will be limited."
        )

    # ========================================================
    # 5. PAIR + RANK ANALYSIS
    # ========================================================

    print("\n[5/6] Computing technical equivalence...")

    top1 = (
        oracle[
            oracle["rank"] == 1
        ]
        .sort_values(
            ["rfq_id", "score"],
            ascending=[True, False]
        )
        .drop_duplicates(
            subset=["rfq_id"],
            keep="first"
        )
        .copy()
    )

    print(
        f"Oracle Top-1 RFQs: "
        f"{top1['rfq_id'].nunique():,}"
    )

    pair_rows = []
    difference_rows = []
    rank_rows = []
    category_counts = {}

    # Pre-index Oracle rows by RFQ for speed.
    oracle_by_rfq = {
        rfq_id: group.sort_values("rank")
        for rfq_id, group in oracle.groupby("rfq_id")
    }

    for _, top1_row in top1.iterrows():

        rfq_id = str(
            top1_row["rfq_id"]
        ).strip()

        if rfq_id not in true_map:
            continue

        truth = true_map[rfq_id]

        true_id = str(
            truth.get(
                "true_configuration_id",
                ""
            )
        ).strip()

        predicted_id = str(
            top1_row["canonical_configuration_id"]
        ).strip()

        true_configuration = configuration_lookup.get(
            true_id
        )

        predicted_configuration = configuration_lookup.get(
            predicted_id
        )

        true_package = truth.get(
            "target_package",
            package_value(true_configuration)
        )

        predicted_package = top1_row.get(
            "package_context",
            package_value(predicted_configuration)
        )

        is_exact_id = int(
            predicted_id == true_id
        )

        classification = classify_pair(
            true_configuration=true_configuration,
            predicted_configuration=predicted_configuration,
            requirements=requirements,
            rfq_id=rfq_id,
            true_package=true_package,
            predicted_package=predicted_package,
            is_exact_id=is_exact_id
        )

        diffs = classification["differences"]

        true_score_rows = oracle[
            (oracle["rfq_id"] == rfq_id)
            & (
                oracle["canonical_configuration_id"]
                == true_id
            )
        ]

        if len(true_score_rows) > 0:
            true_score = true_score_rows["score"].max()
        else:
            true_score = np.nan

        predicted_score = safe_float(
            top1_row["score"]
        )

        score_difference = (
            predicted_score - true_score
            if pd.notna(true_score)
            and pd.notna(predicted_score)
            else np.nan
        )

        rank_group = oracle_by_rfq.get(
            rfq_id,
            pd.DataFrame()
        )

        # Determine first rank at which a technically equivalent
        # configuration appears.
        technical_rank = np.nan
        package_technical_rank = np.nan
        exact_id_rank = np.nan

        if len(rank_group) > 0:

            for _, candidate in rank_group.iterrows():

                candidate_id = str(
                    candidate[
                        "canonical_configuration_id"
                    ]
                ).strip()

                candidate_configuration = (
                    configuration_lookup.get(candidate_id)
                )

                candidate_same_signature = int(
                    signature(candidate_configuration)
                    == signature(true_configuration)
                )

                candidate_same_package = int(
                    norm(candidate["package_context"])
                    == norm(true_package)
                )

                candidate_technical_equivalent = (
                    candidate_same_signature
                    or candidate_id == true_id
                )

                if (
                    pd.isna(technical_rank)
                    and candidate_technical_equivalent
                ):
                    technical_rank = candidate["rank"]

                if (
                    pd.isna(package_technical_rank)
                    and candidate_technical_equivalent
                    and candidate_same_package
                ):
                    package_technical_rank = candidate["rank"]

                if (
                    pd.isna(exact_id_rank)
                    and candidate_id == true_id
                ):
                    exact_id_rank = candidate["rank"]

        pair_rows.append(
            {
                "rfq_id": rfq_id,
                "target_package": true_package,
                "predicted_package": predicted_package,
                "true_configuration_id": true_id,
                "predicted_configuration_id": predicted_id,
                "true_score": true_score,
                "predicted_score": predicted_score,
                "score_difference_predicted_minus_true": score_difference,
                "exact_id_match": is_exact_id,
                "same_technical_signature": classification[
                    "same_technical_signature"
                ],
                "same_package": classification[
                    "same_package"
                ],
                "technical_equivalent": classification[
                    "technical_equivalent"
                ],
                "package_technical_equivalent": classification[
                    "package_technical_equivalent"
                ],
                "top1_case_category": classification[
                    "category"
                ],
                "num_changed_characteristics": len(diffs),
                "changed_characteristics": "|".join(
                    d["characteristic"] for d in diffs
                ),
                "technical_equivalent_rank": technical_rank,
                "package_technical_equivalent_rank": (
                    package_technical_rank
                ),
                "exact_id_rank": exact_id_rank,
            }
        )

        category = classification["category"]
        category_counts[category] = (
            category_counts.get(category, 0) + 1
        )

        for diff in diffs:

            req_type, required_value = requirement_context(
                requirements,
                rfq_id,
                diff["characteristic"]
            )

            extra_not_required = int(
                diff["relation"]
                == "PREDICTED_ADDITIONAL_VALUE"
                and req_type == "NOT_MENTIONED"
            )

            difference_rows.append(
                {
                    "rfq_id": rfq_id,
                    "target_package": true_package,
                    "true_configuration_id": true_id,
                    "predicted_configuration_id": predicted_id,
                    "top1_case_category": category,
                    "characteristic": diff[
                        "characteristic"
                    ],
                    "true_value": diff["true_value"],
                    "predicted_value": diff[
                        "predicted_value"
                    ],
                    "relation": diff["relation"],
                    "requirement_presence": req_type,
                    "required_value": required_value,
                    "predicted_extra_not_required": (
                        extra_not_required
                    ),
                }
            )

        rank_rows.append(
            {
                "rfq_id": rfq_id,
                "true_configuration_id": true_id,
                "exact_id_rank": exact_id_rank,
                "technical_equivalent_rank": (
                    technical_rank
                ),
                "package_technical_equivalent_rank": (
                    package_technical_rank
                ),
                "top1_exact_id": int(
                    not pd.isna(exact_id_rank)
                    and exact_id_rank == 1
                ),
                "top3_exact_id": int(
                    not pd.isna(exact_id_rank)
                    and exact_id_rank <= 3
                ),
                "top5_exact_id": int(
                    not pd.isna(exact_id_rank)
                    and exact_id_rank <= 5
                ),
                "top10_exact_id": int(
                    not pd.isna(exact_id_rank)
                    and exact_id_rank <= 10
                ),
                "top1_technical_equivalent": int(
                    not pd.isna(technical_rank)
                    and technical_rank == 1
                ),
                "top3_technical_equivalent": int(
                    not pd.isna(technical_rank)
                    and technical_rank <= 3
                ),
                "top5_technical_equivalent": int(
                    not pd.isna(technical_rank)
                    and technical_rank <= 5
                ),
                "top10_technical_equivalent": int(
                    not pd.isna(technical_rank)
                    and technical_rank <= 10
                ),
                "top1_package_technical_equivalent": int(
                    not pd.isna(package_technical_rank)
                    and package_technical_rank == 1
                ),
                "top3_package_technical_equivalent": int(
                    not pd.isna(package_technical_rank)
                    and package_technical_rank <= 3
                ),
                "top5_package_technical_equivalent": int(
                    not pd.isna(package_technical_rank)
                    and package_technical_rank <= 5
                ),
                "top10_package_technical_equivalent": int(
                    not pd.isna(package_technical_rank)
                    and package_technical_rank <= 10
                ),
            }
        )

    pairs = pd.DataFrame(pair_rows)
    differences = pd.DataFrame(difference_rows)
    ranks = pd.DataFrame(rank_rows)

    # ========================================================
    # 6. CONFIGURATION SIGNATURE ANALYSIS
    # ========================================================

    print("\n[6/6] Analyzing configuration signatures...")

    signature_rows = []

    grouped = configurations.groupby(
        "canonical_configuration_id"
    )

    for configuration_id, group in grouped:

        signatures = set()

        for _, row in group.iterrows():
            signatures.add(
                tuple(
                    norm(row.get(c, "NOVALUE"))
                    for c in CHARACTERISTICS
                )
            )

        package_contexts = set()

        if "package_context" in group.columns:
            package_contexts = set(
                group["package_context"]
                .apply(norm)
                .tolist()
            )

        if len(group) == 1:
            status = "UNIQUE_ROW"

        elif len(signatures) == 1 and len(package_contexts) <= 1:
            status = "EXACT_DUPLICATE_ROWS"

        elif len(signatures) == 1 and len(package_contexts) > 1:
            status = "SAME_TECHNICAL_SIGNATURE_MULTIPLE_PACKAGES"

        else:
            status = "MULTIPLE_TECHNICAL_SIGNATURES"

        signature_rows.append(
            {
                "canonical_configuration_id": (
                    configuration_id
                ),
                "row_count": len(group),
                "unique_technical_signatures": len(signatures),
                "unique_package_contexts": len(
                    package_contexts
                ),
                "identity_status": status,
            }
        )

    signature_analysis = pd.DataFrame(
        signature_rows
    )

    # ========================================================
    # METRICS
    # ========================================================

    total_rfqs = len(ranks)

    if total_rfqs > 0:

        def mean_col(column):
            return ranks[column].mean()

        exact_top1 = mean_col("top1_exact_id")
        exact_top3 = mean_col("top3_exact_id")
        exact_top5 = mean_col("top5_exact_id")
        exact_top10 = mean_col("top10_exact_id")

        technical_top1 = mean_col(
            "top1_technical_equivalent"
        )
        technical_top3 = mean_col(
            "top3_technical_equivalent"
        )
        technical_top5 = mean_col(
            "top5_technical_equivalent"
        )
        technical_top10 = mean_col(
            "top10_technical_equivalent"
        )

        package_top1 = mean_col(
            "top1_package_technical_equivalent"
        )
        package_top3 = mean_col(
            "top3_package_technical_equivalent"
        )
        package_top5 = mean_col(
            "top5_package_technical_equivalent"
        )
        package_top10 = mean_col(
            "top10_package_technical_equivalent"
        )

        exact_rr = (
            1.0 / ranks["exact_id_rank"]
        ).where(
            ranks["exact_id_rank"].notna()
        )

        technical_rr = (
            1.0 / ranks["technical_equivalent_rank"]
        ).where(
            ranks["technical_equivalent_rank"].notna()
        )

        package_rr = (
            1.0
            / ranks["package_technical_equivalent_rank"]
        ).where(
            ranks[
                "package_technical_equivalent_rank"
            ].notna()
        )

        exact_mrr = exact_rr.mean()
        technical_mrr = technical_rr.mean()
        package_mrr = package_rr.mean()

        mean_technical_rank = ranks[
            "technical_equivalent_rank"
        ].mean()

        median_technical_rank = ranks[
            "technical_equivalent_rank"
        ].median()

        mean_exact_rank = ranks[
            "exact_id_rank"
        ].mean()

        median_exact_rank = ranks[
            "exact_id_rank"
        ].median()

    else:

        exact_top1 = np.nan
        exact_top3 = np.nan
        exact_top5 = np.nan
        exact_top10 = np.nan

        technical_top1 = np.nan
        technical_top3 = np.nan
        technical_top5 = np.nan
        technical_top10 = np.nan

        package_top1 = np.nan
        package_top3 = np.nan
        package_top5 = np.nan
        package_top10 = np.nan

        exact_mrr = np.nan
        technical_mrr = np.nan
        package_mrr = np.nan

        mean_technical_rank = np.nan
        median_technical_rank = np.nan

        mean_exact_rank = np.nan
        median_exact_rank = np.nan

    # Case counts
    category_series = (
        pairs["top1_case_category"]
        if len(pairs) > 0
        else pd.Series(dtype=str)
    )

    exact_cases = int(
        (category_series == "EXACT_ID_MATCH").sum()
    )

    equivalent_cases = int(
        (category_series == "TECHNICALLY_EQUIVALENT").sum()
    )

    extra_optional_cases = int(
        (category_series == "EXTRA_OPTIONAL_FEATURES").sum()
    )

    true_difference_cases = int(
        (category_series == "TRUE_TECHNICAL_DIFFERENCE").sum()
    )

    apparent_failures = int(
        total_rfqs - exact_cases
    )

    equivalent_or_better_top1 = (
        exact_cases + equivalent_cases
    )

    # Requirement-level differences
    if len(differences) > 0:

        mandatory_diff_rows = int(
            (
                differences["requirement_presence"]
                == "MANDATORY"
            ).sum()
        )

        preferred_diff_rows = int(
            (
                differences["requirement_presence"]
                == "PREFERRED"
            ).sum()
        )

        not_mentioned_diff_rows = int(
            (
                differences["requirement_presence"]
                == "NOT_MENTIONED"
            ).sum()
        )

        extra_not_required_rows = int(
            differences[
                "predicted_extra_not_required"
            ].sum()
        )

    else:

        mandatory_diff_rows = 0
        preferred_diff_rows = 0
        not_mentioned_diff_rows = 0
        extra_not_required_rows = 0

    # Signature duplicate statistics
    exact_duplicate_ids = int(
        (
            signature_analysis["identity_status"]
            == "EXACT_DUPLICATE_ROWS"
        ).sum()
    )

    same_signature_multiple_package_ids = int(
        (
            signature_analysis["identity_status"]
            == "SAME_TECHNICAL_SIGNATURE_MULTIPLE_PACKAGES"
        ).sum()
    )

    multiple_signature_ids = int(
        (
            signature_analysis["identity_status"]
            == "MULTIPLE_TECHNICAL_SIGNATURES"
        ).sum()
    )

    summary_rows = [
        ("Oracle RFQs analyzed", total_rfqs),

        ("Exact ID Top-1 accuracy", exact_top1),
        ("Exact ID Top-3 accuracy", exact_top3),
        ("Exact ID Top-5 accuracy", exact_top5),
        ("Exact ID Top-10 accuracy", exact_top10),
        ("Exact ID MRR", exact_mrr),
        ("Mean exact ID rank", mean_exact_rank),
        ("Median exact ID rank", median_exact_rank),

        (
            "Technical-equivalence Top-1 accuracy",
            technical_top1
        ),
        (
            "Technical-equivalence Top-3 accuracy",
            technical_top3
        ),
        (
            "Technical-equivalence Top-5 accuracy",
            technical_top5
        ),
        (
            "Technical-equivalence Top-10 accuracy",
            technical_top10
        ),
        (
            "Technical-equivalence MRR",
            technical_mrr
        ),
        (
            "Mean technical-equivalence rank",
            mean_technical_rank
        ),
        (
            "Median technical-equivalence rank",
            median_technical_rank
        ),

        (
            "Package + technical-equivalence Top-1",
            package_top1
        ),
        (
            "Package + technical-equivalence Top-3",
            package_top3
        ),
        (
            "Package + technical-equivalence Top-5",
            package_top5
        ),
        (
            "Package + technical-equivalence Top-10",
            package_top10
        ),
        (
            "Package + technical-equivalence MRR",
            package_mrr
        ),

        ("Exact ID Top-1 cases", exact_cases),
        (
            "Technically equivalent Top-1 cases",
            equivalent_cases
        ),
        (
            "Extra optional feature Top-1 cases",
            extra_optional_cases
        ),
        (
            "True technical difference Top-1 cases",
            true_difference_cases
        ),
        ("Apparent ID-level failures", apparent_failures),
        (
            "Top-1 exact-or-equivalent cases",
            equivalent_or_better_top1
        ),
        (
            "Top-1 exact-or-equivalent rate",
            (
                equivalent_or_better_top1 / total_rfqs
                if total_rfqs > 0
                else np.nan
            )
        ),

        (
            "Characteristic difference rows",
            len(differences)
        ),
        (
            "Mandatory difference rows",
            mandatory_diff_rows
        ),
        (
            "Preferred difference rows",
            preferred_diff_rows
        ),
        (
            "Not-mentioned difference rows",
            not_mentioned_diff_rows
        ),
        (
            "Predicted extra values not required",
            extra_not_required_rows
        ),

        (
            "Configuration rows",
            len(configurations)
        ),
        (
            "Unique canonical IDs",
            configurations[
                "canonical_configuration_id"
            ].nunique()
        ),
        (
            "Exact duplicate canonical IDs",
            exact_duplicate_ids
        ),
        (
            "Same technical signature / multiple packages",
            same_signature_multiple_package_ids
        ),
        (
            "IDs with multiple technical signatures",
            multiple_signature_ids
        ),
    ]

    summary = pd.DataFrame(
        summary_rows,
        columns=["metric", "value"]
    )

    # ========================================================
    # SAVE
    # ========================================================

    print("\nSaving output files...")

    pairs.to_csv(
        PAIR_FILE,
        index=False
    )

    ranks.to_csv(
        RANK_FILE,
        index=False
    )

    differences.to_csv(
        DIFF_FILE,
        index=False
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False
    )

    category_summary = (
        pairs["top1_case_category"]
        .value_counts()
        .rename_axis("category")
        .reset_index(name="count")
    )

    if len(category_summary) > 0:
        category_summary["percentage"] = (
            category_summary["count"]
            / total_rfqs
            if total_rfqs > 0
            else np.nan
        )

    category_summary.to_csv(
        CATEGORY_FILE,
        index=False
    )

    signature_analysis.to_csv(
        SIGNATURE_FILE,
        index=False
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n")
    print("=" * 100)
    print("KEY EQUIVALENCE FINDINGS")
    print("=" * 100)

    print(
        f"\nOracle RFQs analyzed: {total_rfqs:,}"
    )

    print(
        f"Exact ID Top-1 accuracy: "
        f"{exact_top1:.4f}"
        if pd.notna(exact_top1)
        else "Exact ID Top-1 accuracy: N/A"
    )

    print(
        f"Technical-equivalence Top-1 accuracy: "
        f"{technical_top1:.4f}"
        if pd.notna(technical_top1)
        else "Technical-equivalence Top-1 accuracy: N/A"
    )

    print(
        f"Package + technical-equivalence Top-1: "
        f"{package_top1:.4f}"
        if pd.notna(package_top1)
        else "Package + technical-equivalence Top-1: N/A"
    )

    print(
        f"Exact ID MRR: "
        f"{exact_mrr:.4f}"
        if pd.notna(exact_mrr)
        else "Exact ID MRR: N/A"
    )

    print(
        f"Technical-equivalence MRR: "
        f"{technical_mrr:.4f}"
        if pd.notna(technical_mrr)
        else "Technical-equivalence MRR: N/A"
    )

    print("\nTop-1 case categories:")

    if len(category_summary) > 0:
        print(
            category_summary.to_string(
                index=False
            )
        )
    else:
        print("No pair results available.")

    print("\n")
    print("-" * 100)
    print("TECHNICAL-EQUIVALENCE RANK METRICS")
    print("-" * 100)

    print(
        f"Top-1:  {technical_top1:.4f}"
        if pd.notna(technical_top1)
        else "Top-1:  N/A"
    )

    print(
        f"Top-3:  {technical_top3:.4f}"
        if pd.notna(technical_top3)
        else "Top-3:  N/A"
    )

    print(
        f"Top-5:  {technical_top5:.4f}"
        if pd.notna(technical_top5)
        else "Top-5:  N/A"
    )

    print(
        f"Top-10: {technical_top10:.4f}"
        if pd.notna(technical_top10)
        else "Top-10: N/A"
    )

    print(
        f"MRR:    {technical_mrr:.4f}"
        if pd.notna(technical_mrr)
        else "MRR:    N/A"
    )

    print("\n")
    print("-" * 100)
    print("CONFIGURATION SIGNATURE ANALYSIS")
    print("-" * 100)

    print(
        f"Configuration rows: "
        f"{len(configurations):,}"
    )

    print(
        "Unique canonical IDs: "
        f"{configurations['canonical_configuration_id'].nunique():,}"
    )

    print(
        f"Exact duplicate canonical IDs: "
        f"{exact_duplicate_ids:,}"
    )

    print(
        f"Same technical signature / multiple packages: "
        f"{same_signature_multiple_package_ids:,}"
    )

    print(
        f"IDs with multiple technical signatures: "
        f"{multiple_signature_ids:,}"
    )

    print("\n")
    print("=" * 100)
    print("OUTPUT FILES")
    print("=" * 100)

    print(f"\n1. {PAIR_FILE}")
    print(f"2. {RANK_FILE}")
    print(f"3. {DIFF_FILE}")
    print(f"4. {SUMMARY_FILE}")
    print(f"5. {CATEGORY_FILE}")
    print(f"6. {SIGNATURE_FILE}")

    print("\n")
    print("=" * 100)
    print(
        "731 V2 ORACLE CONFIGURATION EQUIVALENCE ANALYSIS COMPLETE"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
