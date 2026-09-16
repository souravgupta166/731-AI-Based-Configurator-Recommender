import os
import numpy as np
import pandas as pd

# ============================================================
# 731 V2 ORACLE RECOMMENDATION ERROR ANALYSIS - CORRECTED
# ============================================================
# Purpose:
#   Analyse the ORACLE rankings from the V2 recommendation run.
#
# Important V2 schema correction:
#   The V2 TOP-K file does NOT contain `package_aware`.
#   Its configuration identifier is `canonical_configuration_id`.
#   Therefore this script never references topk["package_aware"] or
#   topk["configuration_id"].
#
#   Package-aware information is taken from the V2 recommendations
#   file, where `package_aware`, `target_package`, and
#   `true_configuration_id` are available.
# ============================================================

BASE = "/home/e1546562/thesis-configurator"

TOPK_FILE = os.path.join(
    BASE,
    "data/processed/731_rfq_configuration_topk_v2.csv",
)

RECOMMENDATIONS_FILE = os.path.join(
    BASE,
    "data/processed/731_rfq_configuration_recommendations_v2.csv",
)

CONFIGURATIONS_FILE = os.path.join(
    BASE,
    "data/synthetic/configurations/731_synthetic_configurations_v5.csv",
)

GROUND_TRUTH_FILE = os.path.join(
    BASE,
    "data/synthetic/rfqs/731_rfq_ground_truth_v2.csv",
)

REQUIREMENTS_FILE = os.path.join(
    BASE,
    "data/synthetic/rfqs/731_rfq_requirements_v2.csv",
)

OUTPUT_DIR = os.path.join(BASE, "data/processed")

ERROR_FILE = os.path.join(
    OUTPUT_DIR,
    "731_rfq_configuration_v2_oracle_error_analysis.csv",
)

CHARACTERISTIC_FILE = os.path.join(
    OUTPUT_DIR,
    "731_rfq_configuration_v2_oracle_characteristic_errors.csv",
)

RANK_FILE = os.path.join(
    OUTPUT_DIR,
    "731_rfq_configuration_v2_oracle_rank_analysis.csv",
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "731_rfq_configuration_v2_oracle_error_summary.csv",
)

COMPETITOR_FILE = os.path.join(
    OUTPUT_DIR,
    "731_rfq_configuration_v2_oracle_top1_competitor_analysis.csv",
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
    """Normalise configuration/requirement values."""
    if pd.isna(value):
        return "NOVALUE"

    value = str(value).strip()

    if value == "":
        return "NOVALUE"

    return value


def clean_id(value):
    """Normalise identifiers without converting missing values to 'nan'."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def load_configurations():
    """Load and normalise V5 configuration records."""
    df = pd.read_csv(CONFIGURATIONS_FILE, low_memory=False)

    if "canonical_configuration_id" not in df.columns:
        raise ValueError(
            "Configuration file does not contain "
            "'canonical_configuration_id'."
        )

    df["canonical_configuration_id"] = (
        df["canonical_configuration_id"].apply(clean_id)
    )

    if "package_context" in df.columns:
        df["package_context"] = df["package_context"].apply(clean_id)

    for characteristic in CHARACTERISTICS:
        if characteristic in df.columns:
            df[characteristic] = df[characteristic].apply(norm)

    return df


def get_config_row(configurations, configuration_id):
    """Return the first configuration row for a canonical ID."""
    configuration_id = clean_id(configuration_id)

    if not configuration_id:
        return None

    rows = configurations[
        configurations["canonical_configuration_id"] == configuration_id
    ]

    if rows.empty:
        return None

    return rows.iloc[0]


def compare_configurations(true_config, predicted_config):
    """Return technical characteristics whose values differ."""
    changed = []

    if true_config is None or predicted_config is None:
        return changed

    for characteristic in CHARACTERISTICS:
        true_value = norm(
            true_config.get(characteristic, "NOVALUE")
        )
        predicted_value = norm(
            predicted_config.get(characteristic, "NOVALUE")
        )

        if true_value != predicted_value:
            changed.append(characteristic)

    return changed


def get_rank_band(rank):
    if pd.isna(rank):
        return "UNRANKED"

    rank = int(rank)

    if rank == 1:
        return "1"
    if rank <= 3:
        return "2-3"
    if rank <= 5:
        return "4-5"
    if rank <= 10:
        return "6-10"
    if rank <= 20:
        return "11-20"
    if rank <= 50:
        return "21-50"
    return "51+"


def first_existing_column(df, candidates):
    """Return the first available column from candidates."""
    for column in candidates:
        if column in df.columns:
            return column
    return None


def build_recommendation_lookup(recommendations):
    """
    Build one recommendation record per (rfq_id, package_aware) when possible.

    V2 recommendations are the authoritative source for the selected true
    configuration and package-awareness metadata because these fields are
    explicitly present there.
    """
    required = [
        "rfq_id",
        "true_configuration_id",
        "target_package",
    ]

    missing = [c for c in required if c not in recommendations.columns]
    if missing:
        raise ValueError(
            "V2 recommendations file is missing required columns: "
            f"{missing}"
        )

    rec = recommendations.copy()
    rec["rfq_id"] = rec["rfq_id"].apply(clean_id)
    rec["true_configuration_id"] = rec["true_configuration_id"].apply(clean_id)
    rec["target_package"] = rec["target_package"].apply(clean_id)

    if "mode" in rec.columns:
        rec["mode"] = rec["mode"].astype(str).str.strip().str.upper()

    if "package_aware" in rec.columns:
        rec["package_aware"] = rec["package_aware"].astype(str).str.strip().str.lower()

    # Keep the ORACLE recommendation rows if mode is available.
    if "mode" in rec.columns:
        oracle = rec[rec["mode"] == "ORACLE"].copy()
        if not oracle.empty:
            rec = oracle

    # Prefer package-aware rows if both True/False versions exist. The
    # resulting lookup is only used to obtain the target configuration and
    # package metadata, not to recreate the ranking itself.
    sort_columns = [c for c in ["rfq_id", "package_aware"] if c in rec.columns]
    if sort_columns:
        rec = rec.sort_values(sort_columns)

    rec = rec.drop_duplicates(subset=["rfq_id"], keep="first")

    return rec.set_index("rfq_id").to_dict("index")


def calculate_requirement_satisfaction(config, rfq_requirements):
    """Calculate mandatory/preferred requirement matches for a configuration."""
    mandatory_matches = 0
    mandatory_mismatches = 0
    preferred_matches = 0
    preferred_mismatches = 0

    if config is None:
        for _, req in rfq_requirements.iterrows():
            if req["requirement_type"] == "MANDATORY":
                mandatory_mismatches += 1
            else:
                preferred_mismatches += 1

        return (
            mandatory_matches,
            mandatory_mismatches,
            preferred_matches,
            preferred_mismatches,
        )

    for _, req in rfq_requirements.iterrows():
        characteristic = req["characteristic"]
        required_value = norm(req["internal_value"])
        predicted_value = norm(config.get(characteristic, "NOVALUE"))
        match = required_value == predicted_value

        if req["requirement_type"] == "MANDATORY":
            if match:
                mandatory_matches += 1
            else:
                mandatory_mismatches += 1
        else:
            if match:
                preferred_matches += 1
            else:
                preferred_mismatches += 1

    return (
        mandatory_matches,
        mandatory_mismatches,
        preferred_matches,
        preferred_mismatches,
    )


def calculate_true_mandatory_satisfaction(true_config, mandatory):
    """Calculate mandatory requirement satisfaction of the true config."""
    matches = 0
    mismatches = 0

    if true_config is None:
        return matches, len(mandatory)

    for _, req in mandatory.iterrows():
        characteristic = req["characteristic"]
        required_value = norm(req["internal_value"])
        true_value = norm(true_config.get(characteristic, "NOVALUE"))

        if required_value == true_value:
            matches += 1
        else:
            mismatches += 1

    return matches, mismatches


def get_true_rank(oracle_topk_for_rfq, true_ids):
    """Return the minimum rank at which a true configuration appears."""
    matching = oracle_topk_for_rfq[
        oracle_topk_for_rfq["canonical_configuration_id"].isin(true_ids)
    ]

    if matching.empty:
        return np.nan

    return int(matching["rank"].min())


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 100)
    print("731 V2 ORACLE RECOMMENDATION ERROR ANALYSIS - CORRECTED")
    print("=" * 100)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ========================================================
    # LOAD V2 TOP-K
    # ========================================================
    print("\n" + "=" * 100)
    print("LOADING V2 TOP-K RANKINGS")
    print("=" * 100)

    topk = pd.read_csv(TOPK_FILE, low_memory=False)

    required_topk_columns = [
        "rfq_id",
        "mode",
        "rank",
        "canonical_configuration_id",
        "package_context",
        "score",
        "positive_score",
        "negative_score",
        "matched_requirements",
    ]

    missing_topk = [c for c in required_topk_columns if c not in topk.columns]
    if missing_topk:
        raise ValueError(
            "V2 Top-K file is missing required columns: "
            f"{missing_topk}"
        )

    topk["rfq_id"] = topk["rfq_id"].apply(clean_id)
    topk["mode"] = topk["mode"].astype(str).str.strip().str.upper()
    topk["canonical_configuration_id"] = (
        topk["canonical_configuration_id"].apply(clean_id)
    )
    topk["package_context"] = topk["package_context"].apply(clean_id)
    topk["rank"] = pd.to_numeric(topk["rank"], errors="coerce")

    print(f"Top-K rows: {len(topk):,}")
    print(f"Top-K RFQs: {topk['rfq_id'].nunique():,}")
    print("Top-K columns:")
    print(topk.columns.tolist())
    print("Modes:")
    print(topk["mode"].value_counts(dropna=False).to_string())

    # ========================================================
    # LOAD V2 RECOMMENDATIONS
    # ========================================================
    print("\n" + "=" * 100)
    print("LOADING V2 RECOMMENDATIONS")
    print("=" * 100)

    recommendations = pd.read_csv(RECOMMENDATIONS_FILE, low_memory=False)
    print(f"Recommendation rows: {len(recommendations):,}")
    print("Recommendation columns:")
    print(recommendations.columns.tolist())

    recommendation_lookup = build_recommendation_lookup(recommendations)
    print(f"Recommendation lookup RFQs: {len(recommendation_lookup):,}")

    # ========================================================
    # LOAD CONFIGURATIONS
    # ========================================================
    print("\n" + "=" * 100)
    print("LOADING V5 CONFIGURATIONS")
    print("=" * 100)

    configurations = load_configurations()
    print(f"Configuration rows: {len(configurations):,}")
    print(
        "Unique canonical configuration IDs: "
        f"{configurations['canonical_configuration_id'].nunique():,}"
    )

    # ========================================================
    # LOAD GROUND TRUTH
    # ========================================================
    print("\n" + "=" * 100)
    print("LOADING GROUND TRUTH")
    print("=" * 100)

    ground_truth = pd.read_csv(GROUND_TRUTH_FILE, low_memory=False)

    required_gt_columns = [
        "rfq_id",
        "canonical_configuration_id",
        "target_package",
    ]

    missing_gt = [c for c in required_gt_columns if c not in ground_truth.columns]
    if missing_gt:
        raise ValueError(
            "Ground-truth file is missing columns: " f"{missing_gt}"
        )

    ground_truth["rfq_id"] = ground_truth["rfq_id"].apply(clean_id)
    ground_truth["canonical_configuration_id"] = (
        ground_truth["canonical_configuration_id"].apply(clean_id)
    )
    ground_truth["target_package"] = ground_truth["target_package"].apply(clean_id)

    print(f"Ground-truth RFQs: {ground_truth['rfq_id'].nunique():,}")

    # Ground truth can contain more than one row per RFQ. Preserve all true
    # configuration IDs, while retaining one target package per RFQ.
    gt_lookup = {}
    for rfq_id, group in ground_truth.groupby("rfq_id"):
        gt_lookup[rfq_id] = {
            "configuration_ids": set(
                group["canonical_configuration_id"].dropna().astype(str)
            ),
            "target_package": (
                group["target_package"].iloc[0] if len(group) else ""
            ),
        }

    # ========================================================
    # LOAD STRUCTURED REQUIREMENTS
    # ========================================================
    print("\n" + "=" * 100)
    print("LOADING STRUCTURED REQUIREMENTS")
    print("=" * 100)

    requirements = pd.read_csv(REQUIREMENTS_FILE, low_memory=False)

    required_req_columns = [
        "rfq_id",
        "characteristic",
        "internal_value",
        "requirement_type",
    ]

    missing_req = [c for c in required_req_columns if c not in requirements.columns]
    if missing_req:
        raise ValueError(
            "Requirements file is missing columns: " f"{missing_req}"
        )

    requirements["rfq_id"] = requirements["rfq_id"].apply(clean_id)
    requirements["characteristic"] = requirements["characteristic"].astype(str).str.strip()
    requirements["internal_value"] = requirements["internal_value"].apply(norm)
    requirements["requirement_type"] = (
        requirements["requirement_type"].astype(str).str.strip().str.upper()
    )

    print(f"Requirement rows: {len(requirements):,}")

    # Pre-group requirements for performance.
    requirements_by_rfq = {
        rfq_id: group.copy()
        for rfq_id, group in requirements.groupby("rfq_id")
    }

    # ========================================================
    # IDENTIFY ORACLE TOP-K
    # ========================================================
    print("\n" + "=" * 100)
    print("IDENTIFYING ORACLE TOP-K")
    print("=" * 100)

    # CRITICAL FIX:
    # The V2 TOP-K schema has no package_aware field. Filter only by mode.
    oracle_topk = topk[topk["mode"] == "ORACLE"].copy()

    if oracle_topk.empty:
        raise ValueError("No ORACLE rows found in V2 Top-K file.")

    oracle_topk = oracle_topk.sort_values(
        ["rfq_id", "rank"],
        kind="stable",
    )

    print(f"Oracle Top-K rows: {len(oracle_topk):,}")
    print(f"Oracle RFQs: {oracle_topk['rfq_id'].nunique():,}")

    # One Top-1 row per RFQ.
    top1 = (
        oracle_topk[oracle_topk["rank"] == 1]
        .drop_duplicates(subset=["rfq_id"], keep="first")
        .copy()
    )

    print(f"Oracle Top-1 records: {len(top1):,}")

    # ========================================================
    # ANALYSIS CONTAINERS
    # ========================================================
    error_rows = []
    rank_rows = []
    characteristic_rows = []
    competitor_rows = []

    # ========================================================
    # PROCESS RFQs
    # ========================================================
    print("\n" + "=" * 100)
    print("ANALYZING ORACLE RANKING ERRORS")
    print("=" * 100)

    total = len(top1)

    for counter, (_, row) in enumerate(top1.iterrows(), start=1):
        if counter % 500 == 0:
            print(f"Processed {counter:,} / {total:,}")

        rfq_id = clean_id(row["rfq_id"])

        # ----------------------------------------------------
        # TRUE CONFIGURATION
        # ----------------------------------------------------
        # Prefer V2 recommendations when available because it explicitly
        # stores the true configuration used by that recommendation run.
        rec_info = recommendation_lookup.get(rfq_id, {})

        rec_true_id = clean_id(rec_info.get("true_configuration_id", ""))
        rec_target_package = clean_id(rec_info.get("target_package", ""))

        if rfq_id in gt_lookup:
            true_ids = set(gt_lookup[rfq_id]["configuration_ids"])
            target_package = clean_id(gt_lookup[rfq_id]["target_package"])
        else:
            true_ids = set()
            target_package = ""

        if rec_true_id:
            true_ids.add(rec_true_id)

        if not target_package and rec_target_package:
            target_package = rec_target_package

        # ----------------------------------------------------
        # PREDICTED CONFIGURATION
        # ----------------------------------------------------
        predicted_id = clean_id(row["canonical_configuration_id"])
        predicted_config = get_config_row(configurations, predicted_id)

        # ----------------------------------------------------
        # TRUE RANK
        # ----------------------------------------------------
        rfq_oracle = oracle_topk[oracle_topk["rfq_id"] == rfq_id]
        true_rank = get_true_rank(rfq_oracle, true_ids)

        is_top1 = predicted_id in true_ids

        # ----------------------------------------------------
        # SELECT A TRUE CONFIGURATION FOR DETAILED COMPARISON
        # ----------------------------------------------------
        # Use the recommendation's true ID first. If unavailable, use the
        # first ground-truth ID that exists in the configuration table.
        true_config_id = rec_true_id if rec_true_id else None

        if not true_config_id:
            for candidate_id in sorted(true_ids):
                if get_config_row(configurations, candidate_id) is not None:
                    true_config_id = candidate_id
                    break

        true_config = get_config_row(configurations, true_config_id)

        # ----------------------------------------------------
        # CHARACTERISTIC DIFFERENCES
        # ----------------------------------------------------
        changed = compare_configurations(true_config, predicted_config)

        # ----------------------------------------------------
        # RFQ REQUIREMENTS
        # ----------------------------------------------------
        rfq_requirements = requirements_by_rfq.get(
            rfq_id,
            pd.DataFrame(columns=requirements.columns),
        )

        mandatory = rfq_requirements[
            rfq_requirements["requirement_type"] == "MANDATORY"
        ]
        preferred = rfq_requirements[
            rfq_requirements["requirement_type"] == "PREFERRED"
        ]

        (
            mandatory_matches,
            mandatory_mismatches,
            preferred_matches,
            preferred_mismatches,
        ) = calculate_requirement_satisfaction(
            predicted_config,
            rfq_requirements,
        )

        (
            true_mandatory_matches,
            true_mandatory_mismatches,
        ) = calculate_true_mandatory_satisfaction(
            true_config,
            mandatory,
        )

        # ----------------------------------------------------
        # ERROR CATEGORY
        # ----------------------------------------------------
        if is_top1:
            error_type = "CORRECT_TOP1"
        elif not pd.isna(true_rank) and true_rank <= 3:
            error_type = "TRUE_IN_TOP3"
        elif not pd.isna(true_rank) and true_rank <= 5:
            error_type = "TRUE_IN_TOP5"
        elif not pd.isna(true_rank) and true_rank <= 10:
            error_type = "TRUE_IN_TOP10"
        elif not pd.isna(true_rank):
            error_type = "TRUE_BEYOND_TOP10"
        else:
            error_type = "TRUE_UNRANKED"

        # ----------------------------------------------------
        # ERROR RECORD
        # ----------------------------------------------------
        error_rows.append(
            {
                "rfq_id": rfq_id,
                "target_package": target_package,
                "predicted_configuration_id": predicted_id,
                "true_configuration_id": true_config_id or "",
                "true_rank": true_rank,
                "error_type": error_type,
                "num_changed_characteristics": len(changed),
                "changed_characteristics": "|".join(changed),
                "mandatory_requirements": len(mandatory),
                "mandatory_matches": mandatory_matches,
                "mandatory_mismatches": mandatory_mismatches,
                "preferred_requirements": len(preferred),
                "preferred_matches": preferred_matches,
                "preferred_mismatches": preferred_mismatches,
                "true_mandatory_matches": true_mandatory_matches,
                "true_mandatory_mismatches": true_mandatory_mismatches,
                "predicted_score": row.get("score", np.nan),
                "predicted_package": row.get("package_context", ""),
                "recommendation_package_aware": rec_info.get("package_aware", ""),
            }
        )

        # ----------------------------------------------------
        # RANK RECORD
        # ----------------------------------------------------
        rank_rows.append(
            {
                "rfq_id": rfq_id,
                "target_package": target_package,
                "true_rank": true_rank,
                "top1": int(not pd.isna(true_rank) and true_rank == 1),
                "top3": int(not pd.isna(true_rank) and true_rank <= 3),
                "top5": int(not pd.isna(true_rank) and true_rank <= 5),
                "top10": int(not pd.isna(true_rank) and true_rank <= 10),
                "predicted_configuration_id": predicted_id,
                "error_type": error_type,
            }
        )

        # ----------------------------------------------------
        # CHARACTERISTIC RECORDS
        # ----------------------------------------------------
        for characteristic in CHARACTERISTICS:
            if true_config is None:
                true_value = "UNKNOWN"
            else:
                true_value = norm(true_config.get(characteristic, "NOVALUE"))

            if predicted_config is None:
                predicted_value = "UNKNOWN"
            else:
                predicted_value = norm(
                    predicted_config.get(characteristic, "NOVALUE")
                )

            if true_value == predicted_value:
                relation = "SAME"
            elif predicted_value == "NOVALUE":
                relation = "PREDICTED_NOVALUE"
            elif true_value == "NOVALUE":
                relation = "TRUE_NOVALUE"
            else:
                relation = "VALUE_CONFLICT"

            characteristic_rows.append(
                {
                    "rfq_id": rfq_id,
                    "target_package": target_package,
                    "true_rank": true_rank,
                    "error_type": error_type,
                    "characteristic": characteristic,
                    "true_value": true_value,
                    "predicted_value": predicted_value,
                    "relation": relation,
                    "is_top1": int(is_top1),
                }
            )

        # ----------------------------------------------------
        # TOP-1 COMPETITOR ANALYSIS
        # ----------------------------------------------------
        # Compare the selected true configuration with the ranked candidates
        # immediately above it. This is useful for understanding whether the
        # error is a near-neighbour ranking problem or a larger mismatch.
        if true_rank is not None and not pd.isna(true_rank):
            competitors = rfq_oracle[
                rfq_oracle["rank"] < true_rank
            ].sort_values("rank")
        else:
            competitors = rfq_oracle.sort_values("rank")

        if not competitors.empty:
            competitor = competitors.iloc[0]
            competitor_id = clean_id(competitor["canonical_configuration_id"])
            competitor_config = get_config_row(configurations, competitor_id)
            competitor_changed = compare_configurations(
                true_config,
                competitor_config,
            )

            competitor_rows.append(
                {
                    "rfq_id": rfq_id,
                    "target_package": target_package,
                    "true_configuration_id": true_config_id or "",
                    "true_rank": true_rank,
                    "competitor_rank": competitor.get("rank", np.nan),
                    "competitor_configuration_id": competitor_id,
                    "competitor_score": competitor.get("score", np.nan),
                    "competitor_package": competitor.get("package_context", ""),
                    "num_changed_characteristics_vs_true": len(competitor_changed),
                    "changed_characteristics_vs_true": "|".join(competitor_changed),
                }
            )

    # ========================================================
    # CREATE DATAFRAMES
    # ========================================================
    errors = pd.DataFrame(error_rows)
    ranks = pd.DataFrame(rank_rows)
    characteristic_errors = pd.DataFrame(characteristic_rows)
    competitors = pd.DataFrame(competitor_rows)

    if errors.empty:
        raise ValueError("No error-analysis records were generated.")

    # ========================================================
    # METRICS
    # ========================================================
    top1_accuracy = (errors["error_type"] == "CORRECT_TOP1").mean()

    valid_ranks = ranks.loc[ranks["true_rank"].notna(), "true_rank"]

    if len(valid_ranks) > 0:
        mean_rank = valid_ranks.mean()
        median_rank = valid_ranks.median()
        mrr = (1.0 / valid_ranks).mean()
    else:
        mean_rank = np.nan
        median_rank = np.nan
        mrr = np.nan

    top3 = (
        ranks["true_rank"].notna() & (ranks["true_rank"] <= 3)
    ).mean()
    top5 = (
        ranks["true_rank"].notna() & (ranks["true_rank"] <= 5)
    ).mean()
    top10 = (
        ranks["true_rank"].notna() & (ranks["true_rank"] <= 10)
    ).mean()

    failed = errors[errors["error_type"] != "CORRECT_TOP1"].copy()

    # ========================================================
    # PRINT ERROR SUMMARY
    # ========================================================
    print("\n" + "=" * 100)
    print("V2 ORACLE ERROR SUMMARY")
    print("=" * 100)

    print("\nError categories:")
    print(errors["error_type"].value_counts().to_string())

    print(f"\nTop-1 accuracy: {top1_accuracy:.4f}")
    print(f"Top-3 accuracy: {top3:.4f}")
    print(f"Top-5 accuracy: {top5:.4f}")
    print(f"Top-10 accuracy: {top10:.4f}")
    print(f"MRR: {mrr:.4f}")
    print(f"Mean true rank: {mean_rank:.2f}")
    print(f"Median true rank: {median_rank:.2f}")

    print("\nTrue rank distribution:")
    print(
        ranks["true_rank"]
        .value_counts()
        .sort_index()
        .head(50)
        .to_string()
    )

    # ========================================================
    # CHANGED CHARACTERISTICS
    # ========================================================
    print("\n" + "=" * 100)
    print("CONFIGURATION DIFFERENCES IN FAILED TOP-1 CASES")
    print("=" * 100)

    if not failed.empty:
        print(
            failed["num_changed_characteristics"]
            .value_counts()
            .sort_index()
            .to_string()
        )

        print("\nMost common changed-characteristic combinations:")
        print(
            failed["changed_characteristics"]
            .value_counts()
            .head(50)
            .to_string()
        )

    # ========================================================
    # CHARACTERISTIC CONFLICTS
    # ========================================================
    print("\n" + "=" * 100)
    print("CHARACTERISTICS MOST OFTEN DIFFERING")
    print("=" * 100)

    conflicts = characteristic_errors[
        characteristic_errors["relation"] != "SAME"
    ].copy()

    if not conflicts.empty:
        print(
            conflicts["characteristic"]
            .value_counts()
            .to_string()
        )

        print("\nConflict type by characteristic:")
        print(
            conflicts.groupby(
                ["characteristic", "relation"]
            ).size()
            .sort_values(ascending=False)
            .head(100)
            .to_string()
        )

    # ========================================================
    # REQUIREMENT SATISFACTION
    # ========================================================
    print("\n" + "=" * 100)
    print("PREDICTED TOP-1 REQUIREMENT SATISFACTION")
    print("=" * 100)

    if not failed.empty:
        cols = [
            "mandatory_requirements",
            "mandatory_matches",
            "mandatory_mismatches",
            "preferred_requirements",
            "preferred_matches",
            "preferred_mismatches",
        ]
        print(failed[cols].describe().to_string())

        print("\nTrue configuration mandatory satisfaction:")
        print(
            failed[
                [
                    "true_mandatory_matches",
                    "true_mandatory_mismatches",
                ]
            ]
            .describe()
            .to_string()
        )

    # ========================================================
    # PACKAGE ANALYSIS
    # ========================================================
    print("\n" + "=" * 100)
    print("ERRORS BY PACKAGE")
    print("=" * 100)

    package_summary = (
        errors.groupby("target_package")
        .agg(
            rfqs=("rfq_id", "count"),
            top1=(
                "error_type",
                lambda values: (values == "CORRECT_TOP1").mean(),
            ),
            median_rank=("true_rank", "median"),
            mean_rank=("true_rank", "mean"),
        )
        .sort_values("top1")
    )

    print(package_summary.to_string())

    # ========================================================
    # RANK BANDS
    # ========================================================
    ranks["rank_band"] = ranks["true_rank"].apply(get_rank_band)

    band_order = [
        "1",
        "2-3",
        "4-5",
        "6-10",
        "11-20",
        "21-50",
        "51+",
        "UNRANKED",
    ]

    band_summary = (
        ranks["rank_band"]
        .value_counts()
        .reindex(band_order)
        .fillna(0)
        .astype(int)
    )

    print("\n" + "=" * 100)
    print("PERFORMANCE BY TRUE-RANK BAND")
    print("=" * 100)
    print(band_summary.to_string())

    # ========================================================
    # WORST CASES
    # ========================================================
    print("\n" + "=" * 100)
    print("WORST V2 ORACLE CASES")
    print("=" * 100)

    if not failed.empty:
        worst = failed.copy()
        worst["_rank_sort"] = worst["true_rank"].fillna(999999)
        worst = (
            worst.sort_values(
                ["_rank_sort", "num_changed_characteristics"],
                ascending=[False, False],
            )
            .head(30)
            .drop(columns=["_rank_sort"])
        )

        print(
            worst[
                [
                    "rfq_id",
                    "target_package",
                    "true_rank",
                    "error_type",
                    "num_changed_characteristics",
                    "changed_characteristics",
                    "mandatory_mismatches",
                    "preferred_mismatches",
                    "predicted_configuration_id",
                    "true_configuration_id",
                ]
            ].to_string(index=False)
        )

    # ========================================================
    # DEVELOPMENT DIAGNOSTICS
    # ========================================================
    print("\n" + "=" * 100)
    print("DEVELOPMENT DIAGNOSTICS")
    print("=" * 100)

    print(f"Failed Top-1 cases: {len(failed):,}")

    if not failed.empty:
        zero_change = failed[failed["num_changed_characteristics"] == 0]
        five_plus = failed[failed["num_changed_characteristics"] >= 5]

        print(
            "Failed cases with zero changed characteristics: "
            f"{len(zero_change):,}"
        )
        print(
            "Failed cases with >=5 changed characteristics: "
            f"{len(five_plus):,}"
        )
        print(
            "Median mandatory mismatch count: "
            f"{failed['mandatory_mismatches'].median():.2f}"
        )
        print(
            "Median preferred mismatch count: "
            f"{failed['preferred_mismatches'].median():.2f}"
        )

    # ========================================================
    # SUMMARY DATAFRAME
    # ========================================================
    summary = pd.DataFrame(
        {
            "metric": [
                "RFQs analyzed",
                "Top-1 accuracy",
                "Top-3 accuracy",
                "Top-5 accuracy",
                "Top-10 accuracy",
                "MRR",
                "Mean true rank",
                "Median true rank",
                "Unranked RFQs",
                "Mean changed characteristics for failed Top-1",
                "Median changed characteristics for failed Top-1",
                "Failed Top-1 cases",
            ],
            "value": [
                len(errors),
                top1_accuracy,
                top3,
                top5,
                top10,
                mrr,
                mean_rank,
                median_rank,
                ranks["true_rank"].isna().sum(),
                (
                    failed["num_changed_characteristics"].mean()
                    if not failed.empty
                    else 0
                ),
                (
                    failed["num_changed_characteristics"].median()
                    if not failed.empty
                    else 0
                ),
                len(failed),
            ],
        }
    )

    # ========================================================
    # SAVE OUTPUTS
    # ========================================================
    errors.to_csv(ERROR_FILE, index=False)
    characteristic_errors.to_csv(CHARACTERISTIC_FILE, index=False)
    ranks.to_csv(RANK_FILE, index=False)
    summary.to_csv(SUMMARY_FILE, index=False)
    competitors.to_csv(COMPETITOR_FILE, index=False)

    print("\n" + "=" * 100)
    print("FILES SAVED")
    print("=" * 100)
    print(f"Error analysis:       {ERROR_FILE}")
    print(f"Characteristic errors: {CHARACTERISTIC_FILE}")
    print(f"Rank analysis:        {RANK_FILE}")
    print(f"Summary:              {SUMMARY_FILE}")
    print(f"Competitor analysis:  {COMPETITOR_FILE}")

    print("\n" + "=" * 100)
    print("731 V2 ORACLE ERROR ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
