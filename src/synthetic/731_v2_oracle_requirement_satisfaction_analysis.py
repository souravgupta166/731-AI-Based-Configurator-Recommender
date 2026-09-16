from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TOPK_FILE = (
    PROJECT_ROOT
    / "data/processed/731_rfq_configuration_topk_v2.csv"
)

RECOMMENDATIONS_FILE = (
    PROJECT_ROOT
    / "data/processed/731_rfq_configuration_recommendations_v2.csv"
)

CONFIGURATIONS_FILE = (
    PROJECT_ROOT
    / "data/synthetic/configurations/731_synthetic_configurations_v5.csv"
)

GROUND_TRUTH_FILE = (
    PROJECT_ROOT
    / "data/synthetic/rfqs/731_rfq_ground_truth_v2.csv"
)

REQUIREMENTS_FILE = (
    PROJECT_ROOT
    / "data/synthetic/rfqs/731_rfq_requirements_v2.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data/processed"


# ============================================================
# TECHNICAL CHARACTERISTICS
# ============================================================

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


# ============================================================
# VALUE NORMALIZATION
# ============================================================

def normalize_value(value):
    """
    Normalize configuration and requirement values
    before comparison.
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    if value == "":
        return None

    if value.upper() in {
        "NOVALUE",
        "NAN",
        "NONE",
        "NULL",
    }:
        return None

    return value


def values_equal(value_a, value_b):
    """
    Compare two values after normalization.
    """

    a = normalize_value(value_a)
    b = normalize_value(value_b)

    return a == b


def normalize_requirement_type(value):
    """
    Normalize requirement type labels.
    """

    if pd.isna(value):
        return "UNKNOWN"

    value = str(value).strip().upper()

    if value in {
        "MANDATORY",
        "REQUIRED",
        "MUST",
    }:
        return "MANDATORY"

    if value in {
        "PREFERRED",
        "PREFERENCE",
        "PREFER",
    }:
        return "PREFERRED"

    return value


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("731 V2 ORACLE REQUIREMENT SATISFACTION ANALYSIS")
    print("=" * 80)

    # ========================================================
    # 1. LOAD ALL INPUT FILES
    # ========================================================

    print("\n[1/6] Loading input files...")

    topk = pd.read_csv(TOPK_FILE)

    recommendations = pd.read_csv(
        RECOMMENDATIONS_FILE
    )

    configurations = pd.read_csv(
        CONFIGURATIONS_FILE
    )

    ground_truth = pd.read_csv(
        GROUND_TRUTH_FILE
    )

    requirements = pd.read_csv(
        REQUIREMENTS_FILE
    )

    # --------------------------------------------------------
    # Validate Top-K schema
    # --------------------------------------------------------

    required_topk_columns = {
        "rfq_id",
        "mode",
        "rank",
        "canonical_configuration_id",
        "package_context",
        "score",
    }

    missing = (
        required_topk_columns
        - set(topk.columns)
    )

    if missing:
        raise ValueError(
            "Top-K file is missing required columns: "
            f"{sorted(missing)}"
        )

    # --------------------------------------------------------
    # Validate recommendation schema
    # --------------------------------------------------------

    required_recommendation_columns = {
        "rfq_id",
        "mode",
        "package_aware",
        "target_package",
        "true_configuration_id",
        "recommended_configuration_id",
        "recommended_package",
        "top_score",
        "true_rank",
        "candidate_count",
        "requirement_count",
    }

    missing = (
        required_recommendation_columns
        - set(recommendations.columns)
    )

    if missing:
        raise ValueError(
            "Recommendation file is missing required columns: "
            f"{sorted(missing)}"
        )

    # --------------------------------------------------------
    # Validate ground-truth schema
    # --------------------------------------------------------

    required_ground_truth_columns = {
        "rfq_id",
        "customer_id",
        "canonical_configuration_id",
        "target_package",
        "record_type",
        "evidence_level",
        "nearest_real_distance",
        "source_canonical_configuration_id",
        "ground_truth_hidden_from_customer",
    }

    missing = (
        required_ground_truth_columns
        - set(ground_truth.columns)
    )

    if missing:
        raise ValueError(
            "Ground-truth file is missing required columns: "
            f"{sorted(missing)}"
        )

    # --------------------------------------------------------
    # Validate requirement schema
    # --------------------------------------------------------

    required_requirement_columns = {
        "rfq_id",
        "characteristic",
        "internal_value",
        "requirement_type",
    }

    missing = (
        required_requirement_columns
        - set(requirements.columns)
    )

    if missing:
        raise ValueError(
            "Requirements file is missing required columns: "
            f"{sorted(missing)}"
        )

    # --------------------------------------------------------
    # Validate configuration schema
    # --------------------------------------------------------

    required_configuration_columns = {
        "canonical_configuration_id",
        *TECHNICAL_CHARACTERISTICS,
    }

    missing = (
        required_configuration_columns
        - set(configurations.columns)
    )

    if missing:
        raise ValueError(
            "Configuration file is missing required columns: "
            f"{sorted(missing)}"
        )

    print(
        f"Top-K rows:              {len(topk):,}"
    )

    print(
        f"Recommendation rows:     {len(recommendations):,}"
    )

    print(
        f"Ground-truth rows:       {len(ground_truth):,}"
    )

    print(
        f"Requirement rows:        {len(requirements):,}"
    )

    print(
        f"Configuration rows:      {len(configurations):,}"
    )

    # ========================================================
    # 2. PREPARE GROUND TRUTH
    # ========================================================

    print(
        "\n[2/6] Preparing ground truth..."
    )

    # IMPORTANT:
    #
    # Actual ground-truth schema:
    #
    # canonical_configuration_id
    #
    # We rename it internally so it cannot be confused
    # with the candidate configuration ID.

    ground_truth = ground_truth.rename(
        columns={
            "canonical_configuration_id":
                "ground_truth_configuration_id"
        }
    ).copy()

    ground_truth["rfq_id"] = (
        ground_truth["rfq_id"]
        .astype(str)
        .str.strip()
    )

    ground_truth["ground_truth_configuration_id"] = (
        ground_truth[
            "ground_truth_configuration_id"
        ]
        .astype(str)
        .str.strip()
    )

    # There should be one ground-truth record per RFQ.
    duplicate_gt = (
        ground_truth["rfq_id"]
        .duplicated()
        .sum()
    )

    if duplicate_gt > 0:

        print(
            "WARNING: "
            f"{duplicate_gt:,} duplicate RFQ ground-truth rows found."
        )

        ground_truth = ground_truth.drop_duplicates(
            subset=["rfq_id"],
            keep="first",
        )

    # ========================================================
    # 3. PREPARE CONFIGURATION LOOKUP
    # ========================================================

    print(
        "\n[3/6] Preparing configuration lookup..."
    )

    configurations = configurations.copy()

    configurations["canonical_configuration_id"] = (
        configurations[
            "canonical_configuration_id"
        ]
        .astype(str)
        .str.strip()
    )

    duplicate_configuration_rows = (
        configurations[
            "canonical_configuration_id"
        ]
        .duplicated()
        .sum()
    )

    print(
        "Configuration rows: "
        f"{len(configurations):,}"
    )

    print(
        "Unique canonical configuration IDs: "
        f"{configurations['canonical_configuration_id'].nunique():,}"
    )

    print(
        "Duplicate canonical-ID rows: "
        f"{duplicate_configuration_rows:,}"
    )

    # For requirement satisfaction we need one concrete
    # configuration row for each candidate ID.
    #
    # The separate configuration-equivalence analysis handles
    # duplicate-ID identity questions.
    config_lookup = (
        configurations
        .drop_duplicates(
            subset=["canonical_configuration_id"],
            keep="first",
        )
        .set_index(
            "canonical_configuration_id",
            drop=False,
        )
    )

    # ========================================================
    # 4. BUILD REQUIREMENT LOOKUP
    # ========================================================

    print(
        "\n[4/6] Building RFQ requirement lookup..."
    )

    requirements = requirements.copy()

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

    requirement_lookup = {}

    for _, row in requirements.iterrows():

        rfq_id = row["rfq_id"]

        characteristic = (
            row["characteristic"]
        )

        requirement_record = {
            "internal_value":
                normalize_value(
                    row["internal_value"]
                ),
            "requirement_type":
                normalize_requirement_type(
                    row["requirement_type"]
                ),
        }

        if rfq_id not in requirement_lookup:

            requirement_lookup[rfq_id] = {}

        if characteristic not in requirement_lookup[rfq_id]:

            requirement_lookup[rfq_id][
                characteristic
            ] = []

        requirement_lookup[rfq_id][
            characteristic
        ].append(
            requirement_record
        )

    print(
        "RFQs with requirements: "
        f"{len(requirement_lookup):,}"
    )

    # ========================================================
    # 5. SELECT ORACLE TOP-K
    # ========================================================

    print(
        "\n[5/6] Selecting Oracle Top-K candidates..."
    )

    oracle_topk = topk[
        topk["mode"]
        .astype(str)
        .str.upper()
        == "ORACLE"
    ].copy()

    oracle_topk["rfq_id"] = (
        oracle_topk["rfq_id"]
        .astype(str)
        .str.strip()
    )

    oracle_topk[
        "canonical_configuration_id"
    ] = (
        oracle_topk[
            "canonical_configuration_id"
        ]
        .astype(str)
        .str.strip()
    )

    oracle_topk["rank"] = pd.to_numeric(
        oracle_topk["rank"],
        errors="raise",
    ).astype(int)

    # Join the actual ground truth.
    oracle_topk = oracle_topk.merge(
        ground_truth[
            [
                "rfq_id",
                "ground_truth_configuration_id",
                "target_package",
                "record_type",
                "evidence_level",
            ]
        ],
        on="rfq_id",
        how="inner",
        validate="many_to_one",
    )

    print(
        f"Oracle Top-K rows after GT join: "
        f"{len(oracle_topk):,}"
    )

    print(
        f"Oracle RFQs: "
        f"{oracle_topk['rfq_id'].nunique():,}"
    )

    # ========================================================
    # 6. REQUIREMENT SATISFACTION
    # ========================================================

    print(
        "\n[6/6] Evaluating requirement satisfaction..."
    )

    pair_rows = []

    characteristic_rows = []

    # --------------------------------------------------------
    # Evaluate every Oracle candidate
    # --------------------------------------------------------

    for row in oracle_topk.itertuples(
        index=False
    ):

        rfq_id = row.rfq_id

        configuration_id = (
            row.canonical_configuration_id
        )

        ground_truth_id = (
            row.ground_truth_configuration_id
        )

        rank = int(row.rank)

        # ----------------------------------------------------
        # Make sure candidate configuration exists
        # ----------------------------------------------------

        if configuration_id not in config_lookup.index:

            raise KeyError(
                "Configuration ID "
                f"{configuration_id!r} from Top-K "
                "does not exist in the configuration file."
            )

        config_row = config_lookup.loc[
            configuration_id
        ]

        rfq_requirements = (
            requirement_lookup.get(
                rfq_id,
                {},
            )
        )

        mandatory_total = 0
        mandatory_satisfied = 0
        mandatory_violated = 0

        preferred_total = 0
        preferred_satisfied = 0
        preferred_different = 0

        unmentioned_features = 0

        # ----------------------------------------------------
        # Evaluate all 19 technical characteristics
        # ----------------------------------------------------

        for characteristic in (
            TECHNICAL_CHARACTERISTICS
        ):

            configuration_value = (
                normalize_value(
                    config_row[
                        characteristic
                    ]
                )
            )

            requirement_records = (
                rfq_requirements.get(
                    characteristic,
                    [],
                )
            )

            # =================================================
            # NO REQUIREMENT
            # =================================================

            if not requirement_records:

                requirement_status = (
                    "NO_REQUIREMENT"
                )

                if configuration_value is not None:

                    unmentioned_features += 1

            else:

                mandatory_records = [
                    record
                    for record in requirement_records
                    if record[
                        "requirement_type"
                    ]
                    == "MANDATORY"
                ]

                preferred_records = [
                    record
                    for record in requirement_records
                    if record[
                        "requirement_type"
                    ]
                    == "PREFERRED"
                ]

                # =================================================
                # MANDATORY
                # =================================================

                if mandatory_records:

                    matched = any(
                        values_equal(
                            configuration_value,
                            record[
                                "internal_value"
                            ],
                        )
                        for record
                        in mandatory_records
                    )

                    if matched:

                        requirement_status = (
                            "SATISFIED_MANDATORY"
                        )

                    else:

                        requirement_status = (
                            "VIOLATED_MANDATORY"
                        )

                    for record in mandatory_records:

                        mandatory_total += 1

                        if values_equal(
                            configuration_value,
                            record[
                                "internal_value"
                            ],
                        ):

                            mandatory_satisfied += 1

                        else:

                            mandatory_violated += 1

                # =================================================
                # PREFERRED
                # =================================================

                elif preferred_records:

                    matched = any(
                        values_equal(
                            configuration_value,
                            record[
                                "internal_value"
                            ],
                        )
                        for record
                        in preferred_records
                    )

                    if matched:

                        requirement_status = (
                            "SATISFIED_PREFERRED"
                        )

                    else:

                        requirement_status = (
                            "DIFFERENT_FROM_PREFERRED"
                        )

                    for record in preferred_records:

                        preferred_total += 1

                        if values_equal(
                            configuration_value,
                            record[
                                "internal_value"
                            ],
                        ):

                            preferred_satisfied += 1

                        else:

                            preferred_different += 1

                else:

                    requirement_status = (
                        "UNKNOWN_REQUIREMENT"
                    )

            # ----------------------------------------------------
            # Characteristic-level result
            # ----------------------------------------------------

            characteristic_rows.append(
                {
                    "rfq_id":
                        rfq_id,

                    "rank":
                        rank,

                    "configuration_id":
                        configuration_id,

                    "ground_truth_configuration_id":
                        ground_truth_id,

                    "characteristic":
                        characteristic,

                    "configuration_value":
                        configuration_value,

                    "requirement_status":
                        requirement_status,

                    "has_requirement":
                        bool(
                            requirement_records
                        ),
                }
            )

        # ----------------------------------------------------
        # Overall candidate result
        # ----------------------------------------------------

        mandatory_fully_satisfied = (
            mandatory_violated == 0
        )

        if mandatory_violated > 0:

            candidate_status = (
                "MANDATORY_VIOLATION"
            )

        elif preferred_total > 0:

            candidate_status = (
                "MANDATORY_SATISFIED_PREFERRED_ANALYZED"
            )

        else:

            candidate_status = (
                "MANDATORY_SATISFIED"
            )

        pair_rows.append(
            {
                "rfq_id":
                    rfq_id,

                "rank":
                    rank,

                "configuration_id":
                    configuration_id,

                "ground_truth_configuration_id":
                    ground_truth_id,

                "is_ground_truth_configuration":
                    configuration_id
                    == ground_truth_id,

                "package_context":
                    row.package_context,

                "target_package":
                    row.target_package,

                "record_type":
                    row.record_type,

                "evidence_level":
                    row.evidence_level,

                "score":
                    row.score,

                "mandatory_total":
                    mandatory_total,

                "mandatory_satisfied":
                    mandatory_satisfied,

                "mandatory_violated":
                    mandatory_violated,

                "mandatory_fully_satisfied":
                    mandatory_fully_satisfied,

                "preferred_total":
                    preferred_total,

                "preferred_satisfied":
                    preferred_satisfied,

                "preferred_different":
                    preferred_different,

                "unmentioned_features":
                    unmentioned_features,

                "candidate_status":
                    candidate_status,
            }
        )

    pair_results = pd.DataFrame(
        pair_rows
    )

    characteristic_results = pd.DataFrame(
        characteristic_rows
    )

    # ========================================================
    # RFQ-LEVEL SUMMARY
    # ========================================================

    rfq_rows = []

    for rfq_id, group in pair_results.groupby(
        "rfq_id",
        sort=False,
    ):

        group = group.sort_values(
            "rank"
        )

        top1 = group.iloc[0]

        # First candidate satisfying every mandatory
        # requirement.

        satisfying_candidates = group[
            group[
                "mandatory_fully_satisfied"
            ]
        ]

        if len(satisfying_candidates) > 0:

            first_satisfying_rank = int(
                satisfying_candidates.iloc[0][
                    "rank"
                ]
            )

        else:

            first_satisfying_rank = np.nan

        # Locate ground-truth configuration in Top-K.

        ground_truth_rows = group[
            group[
                "is_ground_truth_configuration"
            ]
        ]

        if len(ground_truth_rows) > 0:

            true_row = ground_truth_rows.iloc[0]

            ground_truth_rank = int(
                true_row["rank"]
            )

            ground_truth_mandatory_violated = int(
                true_row[
                    "mandatory_violated"
                ]
            )

            ground_truth_mandatory_satisfied = int(
                true_row[
                    "mandatory_satisfied"
                ]
            )

            ground_truth_preferred_total = int(
                true_row[
                    "preferred_total"
                ]
            )

            ground_truth_preferred_satisfied = int(
                true_row[
                    "preferred_satisfied"
                ]
            )

        else:

            ground_truth_rank = np.nan

            ground_truth_mandatory_violated = (
                np.nan
            )

            ground_truth_mandatory_satisfied = (
                np.nan
            )

            ground_truth_preferred_total = (
                np.nan
            )

            ground_truth_preferred_satisfied = (
                np.nan
            )

        rfq_rows.append(
            {
                "rfq_id":
                    rfq_id,

                "top1_configuration_id":
                    top1[
                        "configuration_id"
                    ],

                "ground_truth_configuration_id":
                    top1[
                        "ground_truth_configuration_id"
                    ],

                "top1_mandatory_total":
                    int(
                        top1[
                            "mandatory_total"
                        ]
                    ),

                "top1_mandatory_satisfied":
                    int(
                        top1[
                            "mandatory_satisfied"
                        ]
                    ),

                "top1_mandatory_violated":
                    int(
                        top1[
                            "mandatory_violated"
                        ]
                    ),

                "top1_mandatory_fully_satisfied":
                    bool(
                        top1[
                            "mandatory_fully_satisfied"
                        ]
                    ),

                "top1_preferred_total":
                    int(
                        top1[
                            "preferred_total"
                        ]
                    ),

                "top1_preferred_satisfied":
                    int(
                        top1[
                            "preferred_satisfied"
                        ]
                    ),

                "top1_unmentioned_features":
                    int(
                        top1[
                            "unmentioned_features"
                        ]
                    ),

                "first_mandatory_satisfying_rank":
                    first_satisfying_rank,

                "ground_truth_rank":
                    ground_truth_rank,

                "ground_truth_mandatory_violated":
                    ground_truth_mandatory_violated,

                "ground_truth_mandatory_satisfied":
                    ground_truth_mandatory_satisfied,

                "ground_truth_preferred_total":
                    ground_truth_preferred_total,

                "ground_truth_preferred_satisfied":
                    ground_truth_preferred_satisfied,
            }
        )

    rfq_summary = pd.DataFrame(
        rfq_rows
    )

    # ========================================================
    # METRICS
    # ========================================================

    total_rfqs = len(
        rfq_summary
    )

    # --------------------------------------------------------
    # Top-1 mandatory satisfaction
    # --------------------------------------------------------

    top1_mandatory_satisfaction_rate = (
        rfq_summary[
            "top1_mandatory_fully_satisfied"
        ]
        .astype(float)
        .mean()
    )

    # --------------------------------------------------------
    # Top-1 mandatory violation
    # --------------------------------------------------------

    top1_mandatory_violation_rate = (
        rfq_summary[
            "top1_mandatory_violated"
        ]
        .gt(0)
        .astype(float)
        .mean()
    )

    # --------------------------------------------------------
    # Top-K mandatory satisfaction
    # --------------------------------------------------------

    first_satisfying_rank = (
        rfq_summary[
            "first_mandatory_satisfying_rank"
        ]
    )

    top3_mandatory_satisfaction_rate = (
        first_satisfying_rank
        .le(3)
        .mean()
    )

    top5_mandatory_satisfaction_rate = (
        first_satisfying_rank
        .le(5)
        .mean()
    )

    top10_mandatory_satisfaction_rate = (
        first_satisfying_rank
        .le(10)
        .mean()
    )

    # --------------------------------------------------------
    # Mean first satisfying rank
    # --------------------------------------------------------

    if first_satisfying_rank.notna().any():

        mean_first_mandatory_satisfying_rank = (
            first_satisfying_rank
            .mean()
        )

    else:

        mean_first_mandatory_satisfying_rank = (
            np.nan
        )

    # --------------------------------------------------------
    # Ground-truth mandatory satisfaction
    # --------------------------------------------------------

    gt_mandatory = (
        rfq_summary[
            "ground_truth_mandatory_violated"
        ]
        .dropna()
    )

    if len(gt_mandatory) > 0:

        ground_truth_mandatory_satisfaction_rate = (
            gt_mandatory
            .eq(0)
            .astype(float)
            .mean()
        )

    else:

        ground_truth_mandatory_satisfaction_rate = (
            np.nan
        )

    # --------------------------------------------------------
    # Top-1 preferred satisfaction
    # --------------------------------------------------------

    preferred_top1 = rfq_summary[
        rfq_summary[
            "top1_preferred_total"
        ] > 0
    ].copy()

    if len(preferred_top1) > 0:

        top1_preferred_satisfaction_rate = (
            preferred_top1[
                "top1_preferred_satisfied"
            ]
            /
            preferred_top1[
                "top1_preferred_total"
            ]
        ).mean()

    else:

        top1_preferred_satisfaction_rate = (
            np.nan
        )

    # --------------------------------------------------------
    # Ground-truth preferred satisfaction
    # --------------------------------------------------------

    preferred_gt = rfq_summary[
        rfq_summary[
            "ground_truth_preferred_total"
        ] > 0
    ].copy()

    if len(preferred_gt) > 0:

        ground_truth_preferred_satisfaction_rate = (
            preferred_gt[
                "ground_truth_preferred_satisfied"
            ]
            /
            preferred_gt[
                "ground_truth_preferred_total"
            ]
        ).mean()

    else:

        ground_truth_preferred_satisfaction_rate = (
            np.nan
        )

    # ========================================================
    # METRICS DATAFRAME
    # ========================================================

    metrics_rows = [
        {
            "metric":
                "total_rfqs",
            "value":
                total_rfqs,
        },
        {
            "metric":
                "top1_mandatory_satisfaction_rate",
            "value":
                top1_mandatory_satisfaction_rate,
        },
        {
            "metric":
                "top1_mandatory_violation_rate",
            "value":
                top1_mandatory_violation_rate,
        },
        {
            "metric":
                "top3_mandatory_satisfaction_rate",
            "value":
                top3_mandatory_satisfaction_rate,
        },
        {
            "metric":
                "top5_mandatory_satisfaction_rate",
            "value":
                top5_mandatory_satisfaction_rate,
        },
        {
            "metric":
                "top10_mandatory_satisfaction_rate",
            "value":
                top10_mandatory_satisfaction_rate,
        },
        {
            "metric":
                "mean_first_mandatory_satisfying_rank",
            "value":
                mean_first_mandatory_satisfying_rank,
        },
        {
            "metric":
                "ground_truth_mandatory_satisfaction_rate",
            "value":
                ground_truth_mandatory_satisfaction_rate,
        },
        {
            "metric":
                "top1_preferred_satisfaction_rate",
            "value":
                top1_preferred_satisfaction_rate,
        },
        {
            "metric":
                "ground_truth_preferred_satisfaction_rate",
            "value":
                ground_truth_preferred_satisfaction_rate,
        },
    ]

    metrics = pd.DataFrame(
        metrics_rows
    )

    # ========================================================
    # CHARACTERISTIC-LEVEL SUMMARY
    # ========================================================

    characteristic_summary = (
        characteristic_results
        .groupby(
            "characteristic"
        )
        .agg(
            evaluated_candidates=(
                "configuration_id",
                "size",
            ),
            candidates_with_requirement=(
                "has_requirement",
                "sum",
            ),
        )
        .reset_index()
    )

    # Mandatory evaluations.

    mandatory_characteristics = (
        characteristic_results[
            characteristic_results[
                "requirement_status"
            ].isin(
                [
                    "SATISFIED_MANDATORY",
                    "VIOLATED_MANDATORY",
                ]
            )
        ]
    )

    mandatory_counts = (
        mandatory_characteristics
        .groupby(
            "characteristic"
        )
        .size()
        .rename(
            "mandatory_evaluations"
        )
    )

    mandatory_violations = (
        characteristic_results[
            characteristic_results[
                "requirement_status"
            ]
            == "VIOLATED_MANDATORY"
        ]
        .groupby(
            "characteristic"
        )
        .size()
        .rename(
            "mandatory_violations"
        )
    )

    preferred_satisfied = (
        characteristic_results[
            characteristic_results[
                "requirement_status"
            ]
            == "SATISFIED_PREFERRED"
        ]
        .groupby(
            "characteristic"
        )
        .size()
        .rename(
            "preferred_satisfied"
        )
    )

    characteristic_summary = (
        characteristic_summary
        .merge(
            mandatory_counts,
            on="characteristic",
            how="left",
        )
        .merge(
            mandatory_violations,
            on="characteristic",
            how="left",
        )
        .merge(
            preferred_satisfied,
            on="characteristic",
            how="left",
        )
    )

    for column in [
        "mandatory_evaluations",
        "mandatory_violations",
        "preferred_satisfied",
    ]:

        characteristic_summary[
            column
        ] = (
            characteristic_summary[
                column
            ]
            .fillna(0)
            .astype(int)
        )

    characteristic_summary[
        "mandatory_violation_rate"
    ] = np.where(
        characteristic_summary[
            "mandatory_evaluations"
        ] > 0,
        characteristic_summary[
            "mandatory_violations"
        ]
        /
        characteristic_summary[
            "mandatory_evaluations"
        ],
        np.nan,
    )

    # ========================================================
    # SCORE VS SATISFACTION
    # ========================================================

    score_satisfaction_summary = (
        pair_results
        .groupby(
            "mandatory_fully_satisfied"
        )
        .agg(
            candidate_count=(
                "configuration_id",
                "size",
            ),
            mean_score=(
                "score",
                "mean",
            ),
            median_score=(
                "score",
                "median",
            ),
            mean_mandatory_violations=(
                "mandatory_violated",
                "mean",
            ),
            mean_preferred_satisfied=(
                "preferred_satisfied",
                "mean",
            ),
        )
        .reset_index()
    )

    # ========================================================
    # TOP-1 FAILURE ANALYSIS
    # ========================================================

    top1_failures = rfq_summary[
        rfq_summary[
            "top1_mandatory_violated"
        ] > 0
    ].copy()

    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    print(
        "\nSaving output files..."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_files = {
        "731_v2_oracle_requirement_satisfaction_pairs.csv":
            pair_results,

        "731_v2_oracle_requirement_satisfaction_rfq_summary.csv":
            rfq_summary,

        "731_v2_oracle_requirement_satisfaction_characteristics.csv":
            characteristic_results,

        "731_v2_oracle_requirement_satisfaction_characteristic_summary.csv":
            characteristic_summary,

        "731_v2_oracle_requirement_satisfaction_score_summary.csv":
            score_satisfaction_summary,

        "731_v2_oracle_requirement_satisfaction_top1_failures.csv":
            top1_failures,

        "731_v2_oracle_requirement_satisfaction_metrics.csv":
            metrics,
    }

    for filename, dataframe in output_files.items():

        output_path = (
            OUTPUT_DIR
            / filename
        )

        dataframe.to_csv(
            output_path,
            index=False,
        )

        print(
            f"  Saved: {output_path}"
        )

    # ========================================================
    # FINAL CONSOLE SUMMARY
    # ========================================================

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(
        f"\nOracle RFQs analysed: "
        f"{total_rfqs:,}"
    )

    print(
        "Top-1 mandatory satisfaction: "
        f"{top1_mandatory_satisfaction_rate:.4f}"
    )

    print(
        "Top-1 mandatory violation rate: "
        f"{top1_mandatory_violation_rate:.4f}"
    )

    print(
        "Top-3 mandatory satisfaction: "
        f"{top3_mandatory_satisfaction_rate:.4f}"
    )

    print(
        "Top-5 mandatory satisfaction: "
        f"{top5_mandatory_satisfaction_rate:.4f}"
    )

    print(
        "Top-10 mandatory satisfaction: "
        f"{top10_mandatory_satisfaction_rate:.4f}"
    )

    if pd.notna(
        mean_first_mandatory_satisfying_rank
    ):

        print(
            "Mean first mandatory-satisfying rank: "
            f"{mean_first_mandatory_satisfying_rank:.2f}"
        )

    else:

        print(
            "Mean first mandatory-satisfying rank: "
            "N/A"
        )

    if pd.notna(
        ground_truth_mandatory_satisfaction_rate
    ):

        print(
            "Ground-truth mandatory satisfaction: "
            f"{ground_truth_mandatory_satisfaction_rate:.4f}"
        )

    else:

        print(
            "Ground-truth mandatory satisfaction: "
            "N/A"
        )

    if pd.notna(
        top1_preferred_satisfaction_rate
    ):

        print(
            "Top-1 preferred satisfaction: "
            f"{top1_preferred_satisfaction_rate:.4f}"
        )

    else:

        print(
            "Top-1 preferred satisfaction: "
            "N/A"
        )

    if pd.notna(
        ground_truth_preferred_satisfaction_rate
    ):

        print(
            "Ground-truth preferred satisfaction: "
            f"{ground_truth_preferred_satisfaction_rate:.4f}"
        )

    else:

        print(
            "Ground-truth preferred satisfaction: "
            "N/A"
        )

    print(
        "\nTop-1 candidates violating at least one "
        "mandatory requirement: "
        f"{len(top1_failures):,}"
    )

    print(
        "\nAnalysis complete."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()