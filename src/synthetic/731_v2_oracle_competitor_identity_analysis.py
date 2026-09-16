import os
import numpy as np
import pandas as pd

# ============================================================
# 731 V2 ORACLE COMPETITOR + CONFIGURATION IDENTITY ANALYSIS
# ============================================================
#
# Purpose
# -------
# Diagnose why the Oracle recommendation engine does not always
# place the true configuration at Rank 1.
#
# This script compares:
#   1. True configuration
#   2. Oracle Top-1 predicted configuration
#   3. Their scores
#   4. Their characteristic-level differences
#   5. RFQ requirement context
#   6. Configuration identity / duplicate canonical IDs
#
# IMPORTANT:
# The 731 Excel/configuration data represents product/rule knowledge.
# It must NOT be interpreted as historical engineer decisions.
#
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

GROUND_TRUTH_FILE = os.path.join(
    BASE,
    "data/synthetic/rfqs/731_rfq_ground_truth_v2.csv"
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
    "731_rfq_v2_oracle_true_vs_top1_pairs.csv"
)

DIFF_FILE = os.path.join(
    OUTPUT_DIR,
    "731_rfq_v2_oracle_true_vs_top1_characteristic_differences.csv"
)

CHAR_SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "731_rfq_v2_oracle_competitor_characteristic_summary.csv"
)

IDENTITY_FILE = os.path.join(
    OUTPUT_DIR,
    "731_rfq_v2_configuration_identity_analysis.csv"
)

DUPLICATE_FILE = os.path.join(
    OUTPUT_DIR,
    "731_rfq_v2_duplicate_configuration_ids.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "731_rfq_v2_oracle_competitor_diagnostic_summary.csv"
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
# HELPER FUNCTIONS
# ============================================================

def norm(value):
    """
    Normalize configuration values so comparisons are consistent.

    Missing values are represented as NOVALUE.
    """
    if pd.isna(value):
        return "NOVALUE"

    value = str(value).strip()

    if value == "":
        return "NOVALUE"

    return value


def require_file(path):
    """
    Stop execution with a clear error if an input file is missing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}\n"
        )


def load_configurations():
    """
    Load the V5 synthetic configuration table.

    This table may contain multiple rows with the same canonical
    configuration ID. We preserve those rows for identity analysis.
    """

    require_file(CONFIGURATIONS_FILE)

    df = pd.read_csv(
        CONFIGURATIONS_FILE,
        low_memory=False
    )

    if "canonical_configuration_id" not in df.columns:
        raise ValueError(
            "Configuration file does not contain "
            "'canonical_configuration_id'."
        )

    df["canonical_configuration_id"] = (
        df["canonical_configuration_id"]
        .astype(str)
        .str.strip()
    )

    for characteristic in CHARACTERISTICS:

        if characteristic in df.columns:

            df[characteristic] = (
                df[characteristic]
                .apply(norm)
            )

    if "package_context" in df.columns:

        df["package_context"] = (
            df["package_context"]
            .astype(str)
            .str.strip()
        )

    return df


def build_configuration_lookup(configurations):
    """
    Build a lookup from canonical configuration ID to one
    representative configuration row.

    IMPORTANT:
    Duplicate canonical IDs are not silently discarded from the
    identity analysis. They are separately analyzed below.

    The representative row is used only for pairwise comparison.
    """

    unique_rows = (
        configurations
        .drop_duplicates(
            subset=["canonical_configuration_id"],
            keep="first"
        )
    )

    lookup = {}

    for _, row in unique_rows.iterrows():

        configuration_id = (
            str(row["canonical_configuration_id"])
            .strip()
        )

        lookup[configuration_id] = row

    return lookup


def get_configuration_value(
    configuration,
    characteristic
):
    """
    Safely retrieve a normalized characteristic value.
    """

    if configuration is None:
        return "NOVALUE"

    if characteristic not in configuration.index:
        return "NOVALUE"

    return norm(configuration[characteristic])


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print(
        "731 V2 ORACLE COMPETITOR + CONFIGURATION IDENTITY ANALYSIS"
    )
    print("=" * 100)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # ========================================================
    # 1. LOAD TOP-K RESULTS
    # ========================================================

    print("\n[1/7] Loading Oracle Top-K results...")

    require_file(TOPK_FILE)

    topk = pd.read_csv(
        TOPK_FILE,
        low_memory=False
    )

    required_topk_columns = {
        "rfq_id",
        "mode",
        "rank",
        "canonical_configuration_id",
        "package_context",
        "score",
    }

    missing_columns = (
        required_topk_columns
        - set(topk.columns)
    )

    if missing_columns:

        raise ValueError(
            "Top-K file is missing columns: "
            + str(sorted(missing_columns))
        )

    topk["rfq_id"] = (
        topk["rfq_id"]
        .astype(str)
        .str.strip()
    )

    topk["mode"] = (
        topk["mode"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    topk["canonical_configuration_id"] = (
        topk["canonical_configuration_id"]
        .astype(str)
        .str.strip()
    )

    topk["rank"] = pd.to_numeric(
        topk["rank"],
        errors="coerce"
    )

    topk["score"] = pd.to_numeric(
        topk["score"],
        errors="coerce"
    )

    oracle = (
        topk[
            topk["mode"] == "ORACLE"
        ]
        .copy()
    )

    print(
        f"Total Top-K rows: {len(topk):,}"
    )

    print(
        f"Oracle rows: {len(oracle):,}"
    )

    print(
        f"Oracle RFQs: "
        f"{oracle['rfq_id'].nunique():,}"
    )

    # ========================================================
    # 2. LOAD RECOMMENDATIONS
    # ========================================================

    print("\n[2/7] Loading recommendation metadata...")

    require_file(RECOMMENDATIONS_FILE)

    recommendations = pd.read_csv(
        RECOMMENDATIONS_FILE,
        low_memory=False
    )

    recommendations["rfq_id"] = (
        recommendations["rfq_id"]
        .astype(str)
        .str.strip()
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

            recommendations[column] = (
                recommendations[column]
                .astype(str)
                .str.strip()
            )

    required_recommendation_columns = {
        "rfq_id",
        "mode",
        "true_configuration_id",
    }

    missing_recommendation_columns = (
        required_recommendation_columns
        - set(recommendations.columns)
    )

    if missing_recommendation_columns:
        raise ValueError(
            "Recommendation file is missing columns: "
            + str(sorted(missing_recommendation_columns))
        )

    oracle_recommendations = (
        recommendations[
            recommendations["mode"] == "ORACLE"
        ]
        .copy()
    )

    print(
        f"Oracle recommendation rows: "
        f"{len(oracle_recommendations):,}"
    )

    # --------------------------------------------------------
    # Build one true-configuration record per RFQ.
    #
    # Prefer package-aware records when available.
    # --------------------------------------------------------

    if "package_aware" in oracle_recommendations.columns:

        package_aware_string = (
            oracle_recommendations["package_aware"]
            .astype(str)
            .str.lower()
        )

        package_aware = (
            oracle_recommendations[
                package_aware_string == "true"
            ]
            .copy()
        )

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
        f"RFQs with true configuration metadata: "
        f"{len(true_map):,}"
    )

    # ========================================================
    # 3. LOAD CONFIGURATIONS
    # ========================================================

    print("\n[3/7] Loading V5 configurations...")

    configurations = load_configurations()

    print(
        f"Configuration rows: "
        f"{len(configurations):,}"
    )

    unique_configuration_count = (
        configurations[
            "canonical_configuration_id"
        ]
        .nunique()
    )

    print(
        f"Unique canonical configuration IDs: "
        f"{unique_configuration_count:,}"
    )

    # --------------------------------------------------------
    # Configuration identity / duplicate analysis
    # --------------------------------------------------------

    id_counts = (
        configurations[
            "canonical_configuration_id"
        ]
        .value_counts()
        .rename_axis(
            "canonical_configuration_id"
        )
        .reset_index(
            name="row_count"
        )
    )

    duplicate_ids = (
        id_counts[
            id_counts["row_count"] > 1
        ]
        .copy()
    )

    duplicate_ids["duplicate_rows"] = (
        duplicate_ids["row_count"] - 1
    )

    duplicate_ids.to_csv(
        DUPLICATE_FILE,
        index=False
    )

    print(
        f"Canonical IDs with duplicate rows: "
        f"{len(duplicate_ids):,}"
    )

    duplicate_row_total = (
        int(duplicate_ids["duplicate_rows"].sum())
        if len(duplicate_ids) > 0
        else 0
    )

    print(
        f"Total duplicate configuration rows: "
        f"{duplicate_row_total:,}"
    )

    configuration_lookup = (
        build_configuration_lookup(
            configurations
        )
    )

    # ========================================================
    # 4. LOAD REQUIREMENTS
    # ========================================================

    print("\n[4/7] Loading structured RFQ requirements...")

    requirements = None

    if os.path.exists(REQUIREMENTS_FILE):

        requirements = pd.read_csv(
            REQUIREMENTS_FILE,
            low_memory=False
        )

        requirements["rfq_id"] = (
            requirements["rfq_id"]
            .astype(str)
            .str.strip()
        )

        requirements["characteristic"] = (
            requirements["characteristic"]
            .astype(str)
            .str.strip()
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
            f"Requirement rows: "
            f"{len(requirements):,}"
        )

    else:

        print(
            "WARNING: Requirements file not found."
        )

    # ========================================================
    # 5. EXTRACT ORACLE TOP-1
    # ========================================================

    print("\n[5/7] Building true-vs-Top-1 comparisons...")

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

    # --------------------------------------------------------
    # Iterate through Top-1 predictions
    # --------------------------------------------------------

    for _, top1_row in top1.iterrows():

        rfq_id = str(
            top1_row["rfq_id"]
        ).strip()

        # ----------------------------------------------------
        # Need true configuration metadata
        # ----------------------------------------------------

        if rfq_id not in true_map:
            continue

        truth = true_map[rfq_id]

        # ----------------------------------------------------
        # True configuration
        # ----------------------------------------------------

        true_id = str(
            truth.get(
                "true_configuration_id",
                ""
            )
        ).strip()

        # ----------------------------------------------------
        # Predicted configuration
        # ----------------------------------------------------

        predicted_id = str(
            top1_row[
                "canonical_configuration_id"
            ]
        ).strip()

        # ----------------------------------------------------
        # Configuration rows
        # ----------------------------------------------------

        true_configuration = (
            configuration_lookup.get(
                true_id
            )
        )

        predicted_configuration = (
            configuration_lookup.get(
                predicted_id
            )
        )

        # ----------------------------------------------------
        # Find score assigned to TRUE configuration
        # ----------------------------------------------------

        true_score_rows = oracle[
            (oracle["rfq_id"] == rfq_id)
            &
            (
                oracle[
                    "canonical_configuration_id"
                ]
                == true_id
            )
        ]

        if len(true_score_rows) > 0:

            true_score = (
                true_score_rows["score"]
                .max()
            )

        else:

            true_score = np.nan

        predicted_score = pd.to_numeric(
            top1_row["score"],
            errors="coerce"
        )

        # ----------------------------------------------------
        # Exact configuration identity
        # ----------------------------------------------------

        is_top1_correct = (
            predicted_id == true_id
        )

        # ----------------------------------------------------
        # Compare all characteristics
        # ----------------------------------------------------

        changed_characteristics = []

        for characteristic in CHARACTERISTICS:

            true_value = get_configuration_value(
                true_configuration,
                characteristic
            )

            predicted_value = get_configuration_value(
                predicted_configuration,
                characteristic
            )

            if true_value == predicted_value:
                continue

            changed_characteristics.append(
                characteristic
            )

            # -----------------------------------------------
            # Difference type
            # -----------------------------------------------

            if (
                true_value == "NOVALUE"
                and
                predicted_value != "NOVALUE"
            ):

                relation = (
                    "PREDICTED_ADDITIONAL_VALUE"
                )

            elif (
                predicted_value == "NOVALUE"
                and
                true_value != "NOVALUE"
            ):

                relation = (
                    "PREDICTED_MISSING_VALUE"
                )

            else:

                relation = (
                    "VALUE_CONFLICT"
                )

            # -----------------------------------------------
            # Requirement context
            # -----------------------------------------------

            requirement_type = (
                "NOT_MENTIONED"
            )

            required_value = (
                "NOVALUE"
            )

            if requirements is not None:

                matching_requirements = (
                    requirements[
                        (requirements["rfq_id"] == rfq_id)
                        &
                        (
                            requirements[
                                "characteristic"
                            ]
                            == characteristic
                        )
                    ]
                )

                if len(matching_requirements) > 0:

                    # If multiple requirement rows exist,
                    # prefer mandatory over preferred.
                    if (
                        "requirement_type"
                        in matching_requirements.columns
                    ):

                        types = (
                            matching_requirements[
                                "requirement_type"
                            ]
                            .astype(str)
                            .str.upper()
                            .tolist()
                        )

                        if "MANDATORY" in types:

                            requirement_type = (
                                "MANDATORY"
                            )

                            row = (
                                matching_requirements[
                                    matching_requirements[
                                        "requirement_type"
                                    ]
                                    == "MANDATORY"
                                ]
                                .iloc[0]
                            )

                        elif "PREFERRED" in types:

                            requirement_type = (
                                "PREFERRED"
                            )

                            row = (
                                matching_requirements[
                                    matching_requirements[
                                        "requirement_type"
                                    ]
                                    == "PREFERRED"
                                ]
                                .iloc[0]
                            )

                        else:

                            row = (
                                matching_requirements.iloc[0]
                            )

                            requirement_type = (
                                str(
                                    row.get(
                                        "requirement_type",
                                        "UNKNOWN"
                                    )
                                )
                                .upper()
                            )

                    else:

                        row = (
                            matching_requirements.iloc[0]
                        )

                    if (
                        "internal_value"
                        in row.index
                    ):

                        required_value = norm(
                            row[
                                "internal_value"
                            ]
                        )

            # -----------------------------------------------
            # Is predicted value an unnecessary extra?
            # -----------------------------------------------

            predicted_extra_not_required = 0

            if (
                relation
                ==
                "PREDICTED_ADDITIONAL_VALUE"
                and
                requirement_type
                ==
                "NOT_MENTIONED"
            ):

                predicted_extra_not_required = 1

            # -----------------------------------------------
            # Save difference row
            # -----------------------------------------------

            difference_rows.append({

                "rfq_id": rfq_id,

                "target_package": truth.get(
                    "target_package",
                    ""
                ),

                "true_configuration_id": true_id,

                "predicted_configuration_id": predicted_id,

                "true_score": true_score,

                "predicted_score": predicted_score,

                "score_difference_predicted_minus_true": (
                    predicted_score - true_score
                    if pd.notna(true_score)
                    else np.nan
                ),

                "is_top1_correct": int(
                    is_top1_correct
                ),

                "characteristic": characteristic,

                "true_value": true_value,

                "predicted_value": predicted_value,

                "relation": relation,

                "requirement_presence": (
                    requirement_type
                ),

                "required_value": required_value,

                "predicted_extra_not_required": (
                    predicted_extra_not_required
                ),
            })

        # ----------------------------------------------------
        # Save pair-level row
        # ----------------------------------------------------

        pair_rows.append({

            "rfq_id": rfq_id,

            "target_package": truth.get(
                "target_package",
                ""
            ),

            "true_configuration_id": true_id,

            "predicted_configuration_id": predicted_id,

            "true_score": true_score,

            "predicted_score": predicted_score,

            "score_difference_predicted_minus_true": (
                predicted_score - true_score
                if pd.notna(true_score)
                else np.nan
            ),

            "is_top1_correct": int(
                is_top1_correct
            ),

            "num_changed_characteristics": (
                len(changed_characteristics)
            ),

            "changed_characteristics": (
                "|".join(
                    changed_characteristics
                )
            ),
        })

    pairs = pd.DataFrame(
        pair_rows
    )

    differences = pd.DataFrame(
        difference_rows
    )

    # ========================================================
    # 6. CHARACTERISTIC SUMMARY
    # ========================================================

    print(
        "\n[6/7] Building characteristic-level diagnostics..."
    )

    if len(differences) > 0:

        characteristic_summary_rows = []

        for characteristic, group in (
            differences
            .groupby("characteristic")
        ):

            characteristic_summary_rows.append({

                "characteristic": characteristic,

                "changed_pairs": len(group),

                "additional_predicted_value": int(
                    (
                        group["relation"]
                        ==
                        "PREDICTED_ADDITIONAL_VALUE"
                    ).sum()
                ),

                "missing_predicted_value": int(
                    (
                        group["relation"]
                        ==
                        "PREDICTED_MISSING_VALUE"
                    ).sum()
                ),

                "value_conflict": int(
                    (
                        group["relation"]
                        ==
                        "VALUE_CONFLICT"
                    ).sum()
                ),

                "mandatory_differences": int(
                    (
                        group[
                            "requirement_presence"
                        ]
                        ==
                        "MANDATORY"
                    ).sum()
                ),

                "preferred_differences": int(
                    (
                        group[
                            "requirement_presence"
                        ]
                        ==
                        "PREFERRED"
                    ).sum()
                ),

                "not_mentioned_differences": int(
                    (
                        group[
                            "requirement_presence"
                        ]
                        ==
                        "NOT_MENTIONED"
                    ).sum()
                ),

                "extra_not_required": int(
                    group[
                        "predicted_extra_not_required"
                    ].sum()
                ),
            })

        characteristic_summary = (
            pd.DataFrame(
                characteristic_summary_rows
            )
            .sort_values(
                "changed_pairs",
                ascending=False
            )
            .reset_index(drop=True)
        )

    else:

        characteristic_summary = pd.DataFrame(
            columns=[
                "characteristic",
                "changed_pairs",
                "additional_predicted_value",
                "missing_predicted_value",
                "value_conflict",
                "mandatory_differences",
                "preferred_differences",
                "not_mentioned_differences",
                "extra_not_required",
            ]
        )

    # ========================================================
    # 7. CONFIGURATION IDENTITY ANALYSIS
    # ========================================================

    print(
        "\n[7/7] Analyzing configuration identity..."
    )

    identity_rows = []

    identity_characteristics = [
        c
        for c in CHARACTERISTICS
        if c in configurations.columns
    ]

    for configuration_id, group in (
        configurations
        .groupby(
            "canonical_configuration_id"
        )
    ):

        # ----------------------------------------------------
        # Number of package contexts
        # ----------------------------------------------------

        if "package_context" in group.columns:

            package_contexts = sorted(
                group[
                    "package_context"
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

        else:

            package_contexts = []

        # ----------------------------------------------------
        # Characteristic signatures
        # ----------------------------------------------------

        if len(identity_characteristics) > 0:

            signatures = (
                group[
                    identity_characteristics
                ]
                .astype(str)
                .drop_duplicates()
            )

            unique_signature_count = (
                len(signatures)
            )

        else:

            unique_signature_count = 0

        # ----------------------------------------------------
        # Identity interpretation
        # ----------------------------------------------------

        if len(group) == 1:

            identity_status = (
                "UNIQUE_ROW"
            )

        elif (
            len(package_contexts) > 1
            and
            unique_signature_count == 1
        ):

            identity_status = (
                "DUPLICATE_ID_DIFFERENT_PACKAGE_CONTEXT"
            )

        elif (
            unique_signature_count > 1
        ):

            identity_status = (
                "DUPLICATE_ID_DIFFERENT_CHARACTERISTIC_SIGNATURE"
            )

        else:

            identity_status = (
                "EXACT_DUPLICATE_ROWS"
            )

        identity_rows.append({

            "canonical_configuration_id": (
                configuration_id
            ),

            "row_count": len(group),

            "unique_package_contexts": (
                len(package_contexts)
            ),

            "package_contexts": (
                "|".join(package_contexts)
            ),

            "unique_characteristic_signatures": (
                unique_signature_count
            ),

            "identity_status": (
                identity_status
            ),
        })

    identity = pd.DataFrame(
        identity_rows
    )

    # ========================================================
    # SUMMARY STATISTICS
    # ========================================================

    total_rfqs = len(pairs)

    if total_rfqs > 0:

        top1_accuracy = (
            pairs["is_top1_correct"]
            .mean()
        )

        failed_pairs = (
            pairs[
                pairs["is_top1_correct"] == 0
            ]
            .copy()
        )

    else:

        top1_accuracy = np.nan

        failed_pairs = pd.DataFrame()

    if len(failed_pairs) > 0:

        mean_changed = (
            failed_pairs[
                "num_changed_characteristics"
            ]
            .mean()
        )

        median_changed = (
            failed_pairs[
                "num_changed_characteristics"
            ]
            .median()
        )

        one_or_two_change_cases = int(
            (
                failed_pairs[
                    "num_changed_characteristics"
                ]
                <= 2
            ).sum()
        )

        one_or_two_change_percentage = (
            one_or_two_change_cases
            /
            len(failed_pairs)
        )

    else:

        mean_changed = np.nan
        median_changed = np.nan
        one_or_two_change_cases = 0
        one_or_two_change_percentage = np.nan

    if len(differences) > 0:

        additional_values = int(
            (
                differences["relation"]
                ==
                "PREDICTED_ADDITIONAL_VALUE"
            ).sum()
        )

        missing_values = int(
            (
                differences["relation"]
                ==
                "PREDICTED_MISSING_VALUE"
            ).sum()
        )

        value_conflicts = int(
            (
                differences["relation"]
                ==
                "VALUE_CONFLICT"
            ).sum()
        )

        extra_not_required = int(
            differences[
                "predicted_extra_not_required"
            ].sum()
        )

    else:

        additional_values = 0
        missing_values = 0
        value_conflicts = 0
        extra_not_required = 0

    # --------------------------------------------------------
    # Duplicate identity statistics
    # --------------------------------------------------------

    duplicate_count = len(
        duplicate_ids
    )

    duplicate_rows = (
        int(
            duplicate_ids[
                "duplicate_rows"
            ].sum()
        )
        if duplicate_count > 0
        else 0
    )

    exact_duplicate_count = int(
        (
            identity[
                "identity_status"
            ]
            ==
            "EXACT_DUPLICATE_ROWS"
        ).sum()
    )

    package_context_duplicate_count = int(
        (
            identity[
                "identity_status"
            ]
            ==
            "DUPLICATE_ID_DIFFERENT_PACKAGE_CONTEXT"
        ).sum()
    )

    characteristic_duplicate_count = int(
        (
            identity[
                "identity_status"
            ]
            ==
            "DUPLICATE_ID_DIFFERENT_CHARACTERISTIC_SIGNATURE"
        ).sum()
    )

    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    print("\nSaving diagnostic files...")

    pairs.to_csv(
        PAIR_FILE,
        index=False
    )

    differences.to_csv(
        DIFF_FILE,
        index=False
    )

    characteristic_summary.to_csv(
        CHAR_SUMMARY_FILE,
        index=False
    )

    identity.to_csv(
        IDENTITY_FILE,
        index=False
    )

    duplicate_ids.to_csv(
        DUPLICATE_FILE,
        index=False
    )

    summary_rows = [

        (
            "Oracle RFQs analyzed",
            total_rfqs
        ),

        (
            "Oracle Top-1 correct",
            int(
                pairs["is_top1_correct"].sum()
            )
            if total_rfqs > 0
            else 0
        ),

        (
            "Oracle Top-1 accuracy",
            top1_accuracy
        ),

        (
            "Failed Top-1 cases",
            len(failed_pairs)
        ),

        (
            "Mean changed characteristics in failed Top-1",
            mean_changed
        ),

        (
            "Median changed characteristics in failed Top-1",
            median_changed
        ),

        (
            "Failed cases with <=2 characteristic changes",
            one_or_two_change_cases
        ),

        (
            "Failed cases with <=2 characteristic changes (%)",
            one_or_two_change_percentage
        ),

        (
            "Total characteristic differences",
            len(differences)
        ),

        (
            "Predicted additional values",
            additional_values
        ),

        (
            "Predicted missing values",
            missing_values
        ),

        (
            "Value conflicts",
            value_conflicts
        ),

        (
            "Predicted values not mentioned in RFQ",
            extra_not_required
        ),

        (
            "Configuration rows",
            len(configurations)
        ),

        (
            "Unique canonical configuration IDs",
            unique_configuration_count
        ),

        (
            "Canonical IDs with duplicate rows",
            duplicate_count
        ),

        (
            "Total duplicate configuration rows",
            duplicate_rows
        ),

        (
            "Exact duplicate IDs",
            exact_duplicate_count
        ),

        (
            "Duplicate IDs with different package contexts",
            package_context_duplicate_count
        ),

        (
            "Duplicate IDs with different characteristic signatures",
            characteristic_duplicate_count
        ),
    ]

    summary = pd.DataFrame(
        summary_rows,
        columns=[
            "metric",
            "value"
        ]
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n")
    print("=" * 100)
    print("KEY FINDINGS")
    print("=" * 100)

    print(
        f"\nOracle RFQs analyzed: "
        f"{total_rfqs:,}"
    )

    print(
        f"Oracle Top-1 accuracy: "
        f"{top1_accuracy:.4f}"
        if pd.notna(top1_accuracy)
        else
        "Oracle Top-1 accuracy: N/A"
    )

    print(
        f"Failed Top-1 cases: "
        f"{len(failed_pairs):,}"
    )

    print(
        f"Mean changed characteristics "
        f"in failed Top-1: "
        f"{mean_changed:.3f}"
        if pd.notna(mean_changed)
        else
        "Mean changed characteristics: N/A"
    )

    print(
        f"Median changed characteristics "
        f"in failed Top-1: "
        f"{median_changed:.1f}"
        if pd.notna(median_changed)
        else
        "Median changed characteristics: N/A"
    )

    print(
        f"Failed cases with <=2 characteristic changes: "
        f"{one_or_two_change_cases:,}"
    )

    print(
        f"Percentage of failed cases with <=2 changes: "
        f"{one_or_two_change_percentage:.2%}"
        if pd.notna(one_or_two_change_percentage)
        else
        "Percentage of failed cases with <=2 changes: N/A"
    )

    print(
        f"\nTotal characteristic differences: "
        f"{len(differences):,}"
    )

    print(
        f"Predicted additional values: "
        f"{additional_values:,}"
    )

    print(
        f"Predicted missing values: "
        f"{missing_values:,}"
    )

    print(
        f"Value conflicts: "
        f"{value_conflicts:,}"
    )

    print(
        f"Predicted values not mentioned in RFQ: "
        f"{extra_not_required:,}"
    )

    # ========================================================
    # TOP CHARACTERISTIC DIFFERENCES
    # ========================================================

    if len(characteristic_summary) > 0:

        print("\n")
        print("-" * 100)
        print("MOST COMMON CHARACTERISTIC DIFFERENCES")
        print("-" * 100)

        print(
            characteristic_summary
            .head(15)
            .to_string(index=False)
        )

    # ========================================================
    # CONFIGURATION IDENTITY
    # ========================================================

    print("\n")
    print("-" * 100)
    print("CONFIGURATION IDENTITY")
    print("-" * 100)

    print(
        f"Configuration rows: "
        f"{len(configurations):,}"
    )

    print(
        f"Unique canonical IDs: "
        f"{unique_configuration_count:,}"
    )

    print(
        f"Canonical IDs with duplicate rows: "
        f"{duplicate_count:,}"
    )

    print(
        f"Total duplicate rows: "
        f"{duplicate_rows:,}"
    )

    print(
        f"Exact duplicate IDs: "
        f"{exact_duplicate_count:,}"
    )

    print(
        f"Duplicate IDs with different package contexts: "
        f"{package_context_duplicate_count:,}"
    )

    print(
        f"Duplicate IDs with different characteristic signatures: "
        f"{characteristic_duplicate_count:,}"
    )

    # ========================================================
    # TOP SCORE DIFFERENCES
    # ========================================================

    if len(failed_pairs) > 0:

        print("\n")
        print("-" * 100)
        print("LARGEST TOP-1 SCORE ADVANTAGES")
        print("-" * 100)

        score_view = (
            failed_pairs[
                [
                    "rfq_id",
                    "target_package",
                    "true_configuration_id",
                    "predicted_configuration_id",
                    "true_score",
                    "predicted_score",
                    "score_difference_predicted_minus_true",
                    "num_changed_characteristics",
                    "changed_characteristics",
                ]
            ]
            .sort_values(
                "score_difference_predicted_minus_true",
                ascending=False
            )
            .head(20)
        )

        print(
            score_view.to_string(
                index=False
            )
        )

    # ========================================================
    # SAVE COMPLETE SUMMARY
    # ========================================================

    print("\n")
    print("=" * 100)
    print("OUTPUT FILES")
    print("=" * 100)

    print(
        f"\n1. {PAIR_FILE}"
    )

    print(
        f"2. {DIFF_FILE}"
    )

    print(
        f"3. {CHAR_SUMMARY_FILE}"
    )

    print(
        f"4. {IDENTITY_FILE}"
    )

    print(
        f"5. {DUPLICATE_FILE}"
    )

    print(
        f"6. {SUMMARY_FILE}"
    )

    print("\n")
    print("=" * 100)
    print(
        "731 V2 ORACLE COMPETITOR + IDENTITY ANALYSIS COMPLETE"
    )
    print("=" * 100)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()