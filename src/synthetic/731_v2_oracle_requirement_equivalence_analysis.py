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
    Normalize values before comparison.
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
    Normalize requirement-type labels.
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
# BUILD REQUIREMENT LOOKUP
# ============================================================

def build_requirement_lookup(requirements):
    """
    Build:

        rfq_id
            characteristic
                requirement records
    """

    lookup = {}

    for row in requirements.itertuples(
        index=False
    ):

        rfq_id = str(row.rfq_id).strip()

        characteristic = (
            str(row.characteristic).strip()
        )

        requirement_type = (
            normalize_requirement_type(
                row.requirement_type
            )
        )

        internal_value = normalize_value(
            row.internal_value
        )

        if rfq_id not in lookup:
            lookup[rfq_id] = {}

        if characteristic not in lookup[rfq_id]:
            lookup[rfq_id][characteristic] = []

        lookup[rfq_id][characteristic].append(
            {
                "requirement_type":
                    requirement_type,
                "internal_value":
                    internal_value,
            }
        )

    return lookup


# ============================================================
# EXTRACT REQUIREMENTS
# ============================================================

def split_requirements(rfq_requirements):
    """
    Split one RFQ's requirements into mandatory and preferred
    lookup dictionaries.

    If multiple requirements exist for the same characteristic,
    a candidate satisfies the characteristic when its value
    matches at least one requirement value.
    """

    mandatory = {}
    preferred = {}

    for characteristic, records in (
        rfq_requirements.items()
    ):

        mandatory_values = []
        preferred_values = []

        for record in records:

            requirement_type = (
                record["requirement_type"]
            )

            internal_value = (
                record["internal_value"]
            )

            if requirement_type == "MANDATORY":

                mandatory_values.append(
                    internal_value
                )

            elif requirement_type == "PREFERRED":

                preferred_values.append(
                    internal_value
                )

        if mandatory_values:
            mandatory[characteristic] = (
                mandatory_values
            )

        if preferred_values:
            preferred[characteristic] = (
                preferred_values
            )

    return mandatory, preferred


# ============================================================
# CANDIDATE SATISFACTION
# ============================================================

def evaluate_configuration(
    config_row,
    mandatory_requirements,
    preferred_requirements,
):
    """
    Evaluate one configuration against one RFQ.

    Returns:

        mandatory_satisfied
        preferred_satisfied_count
        preferred_total
        mandatory_violation_count
        preferred_mismatch_count
    """

    mandatory_satisfied = True

    mandatory_violation_count = 0

    for characteristic, required_values in (
        mandatory_requirements.items()
    ):

        configuration_value = normalize_value(
            config_row[characteristic]
        )

        matched = any(
            values_equal(
                configuration_value,
                required_value,
            )
            for required_value
            in required_values
        )

        if not matched:

            mandatory_satisfied = False

            mandatory_violation_count += 1

    preferred_total = 0

    preferred_satisfied_count = 0

    preferred_mismatch_count = 0

    for characteristic, required_values in (
        preferred_requirements.items()
    ):

        configuration_value = normalize_value(
            config_row[characteristic]
        )

        preferred_total += len(
            required_values
        )

        matched = any(
            values_equal(
                configuration_value,
                required_value,
            )
            for required_value
            in required_values
        )

        if matched:

            preferred_satisfied_count += len(
                required_values
            )

        else:

            preferred_mismatch_count += len(
                required_values
            )

    return (
        mandatory_satisfied,
        preferred_satisfied_count,
        preferred_total,
        mandatory_violation_count,
        preferred_mismatch_count,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "731 V2 ORACLE REQUIREMENT EQUIVALENCE ANALYSIS"
    )
    print("=" * 80)

    # ========================================================
    # 1. LOAD FILES
    # ========================================================

    print("\n[1/7] Loading input files...")

    topk = pd.read_csv(TOPK_FILE)

    configurations = pd.read_csv(
        CONFIGURATIONS_FILE
    )

    ground_truth = pd.read_csv(
        GROUND_TRUTH_FILE
    )

    requirements = pd.read_csv(
        REQUIREMENTS_FILE
    )

    print(
        f"Top-K rows:         {len(topk):,}"
    )

    print(
        f"Configuration rows: {len(configurations):,}"
    )

    print(
        f"Ground-truth rows:  {len(ground_truth):,}"
    )

    print(
        f"Requirement rows:   {len(requirements):,}"
    )

    # ========================================================
    # 2. VALIDATE SCHEMAS
    # ========================================================

    print("\n[2/7] Validating schemas...")

    required_topk_columns = {
        "rfq_id",
        "mode",
        "rank",
        "canonical_configuration_id",
        "package_context",
        "score",
    }

    required_configuration_columns = {
        "canonical_configuration_id",
        *TECHNICAL_CHARACTERISTICS,
    }

    required_ground_truth_columns = {
        "rfq_id",
        "canonical_configuration_id",
        "target_package",
        "record_type",
        "evidence_level",
        "source_canonical_configuration_id",
    }

    required_requirement_columns = {
        "rfq_id",
        "characteristic",
        "internal_value",
        "requirement_type",
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

    missing = (
        required_configuration_columns
        - set(configurations.columns)
    )

    if missing:
        raise ValueError(
            "Configuration file is missing required columns: "
            f"{sorted(missing)}"
        )

    missing = (
        required_ground_truth_columns
        - set(ground_truth.columns)
    )

    if missing:
        raise ValueError(
            "Ground-truth file is missing required columns: "
            f"{sorted(missing)}"
        )

    missing = (
        required_requirement_columns
        - set(requirements.columns)
    )

    if missing:
        raise ValueError(
            "Requirements file is missing required columns: "
            f"{sorted(missing)}"
        )

    print("Schema validation passed.")

    # ========================================================
    # 3. PREPARE DATA
    # ========================================================

    print("\n[3/7] Preparing data...")

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    ground_truth = ground_truth.copy()

    ground_truth["rfq_id"] = (
        ground_truth["rfq_id"]
        .astype(str)
        .str.strip()
    )

    ground_truth[
        "ground_truth_configuration_id"
    ] = (
        ground_truth[
            "canonical_configuration_id"
        ]
        .astype(str)
        .str.strip()
    )

    # Ensure one GT row per RFQ.
    if ground_truth["rfq_id"].duplicated().any():

        duplicate_count = (
            ground_truth["rfq_id"]
            .duplicated()
            .sum()
        )

        print(
            "WARNING: "
            f"{duplicate_count:,} duplicate RFQ "
            "ground-truth rows detected. "
            "Keeping first row per RFQ."
        )

        ground_truth = ground_truth.drop_duplicates(
            subset=["rfq_id"],
            keep="first",
        )

    # --------------------------------------------------------
    # Top-K
    # --------------------------------------------------------

    topk = topk.copy()

    topk["rfq_id"] = (
        topk["rfq_id"]
        .astype(str)
        .str.strip()
    )

    topk["canonical_configuration_id"] = (
        topk[
            "canonical_configuration_id"
        ]
        .astype(str)
        .str.strip()
    )

    topk["rank"] = pd.to_numeric(
        topk["rank"],
        errors="raise",
    ).astype(int)

    # Only Oracle candidates.
    oracle_topk = topk[
        topk["mode"]
        .astype(str)
        .str.upper()
        == "ORACLE"
    ].copy()

    # --------------------------------------------------------
    # Configurations
    # --------------------------------------------------------

    configurations = configurations.copy()

    configurations[
        "canonical_configuration_id"
    ] = (
        configurations[
            "canonical_configuration_id"
        ]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Requirements
    # --------------------------------------------------------

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

    requirement_lookup = (
        build_requirement_lookup(
            requirements
        )
    )

    print(
        f"Oracle RFQs: "
        f"{oracle_topk['rfq_id'].nunique():,}"
    )

    print(
        f"Unique configurations: "
        f"{configurations['canonical_configuration_id'].nunique():,}"
    )

    # ========================================================
    # 4. BUILD CONFIGURATION RECORDS
    # ========================================================

    print(
        "\n[4/7] Preparing configuration records..."
    )

    # IMPORTANT:
    #
    # We retain duplicate configuration rows here.
    #
    # The analysis will report:
    #   - configuration rows
    #   - unique canonical IDs
    #
    # This avoids silently collapsing the configuration space.
    #
    # Candidate identity is evaluated using canonical IDs.

    configuration_records = (
        configurations.to_dict(
            orient="records"
        )
    )

    print(
        f"Configuration records available: "
        f"{len(configuration_records):,}"
    )

    # ========================================================
    # 5. PREPARE RFQ METADATA
    # ========================================================

    print(
        "\n[5/7] Preparing RFQ metadata..."
    )

    gt_lookup = (
        ground_truth
        .set_index("rfq_id")
    )

    rfq_ids = sorted(
        set(
            oracle_topk["rfq_id"]
        )
        &
        set(
            ground_truth["rfq_id"]
        )
    )

    print(
        f"RFQs available for analysis: "
        f"{len(rfq_ids):,}"
    )

    # ========================================================
    # 6. ENUMERATE VALID CONFIGURATIONS
    # ========================================================

    print(
        "\n[6/7] Finding technically valid configurations..."
    )

    print(
        "This evaluates every RFQ against the full "
        "configuration space."
    )

    all_rfq_rows = []

    valid_configuration_rows = []

    # --------------------------------------------------------
    # Main RFQ loop
    # --------------------------------------------------------

    for rfq_index, rfq_id in enumerate(
        rfq_ids,
        start=1,
    ):

        if (
            rfq_index == 1
            or rfq_index % 500 == 0
            or rfq_index == len(rfq_ids)
        ):

            print(
                f"  Processing RFQ "
                f"{rfq_index:,}/{len(rfq_ids):,}"
            )

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        gt_row = gt_lookup.loc[
            rfq_id
        ]

        ground_truth_id = str(
            gt_row[
                "ground_truth_configuration_id"
            ]
        ).strip()

        target_package = str(
            gt_row["target_package"]
        ).strip()

        record_type = str(
            gt_row["record_type"]
        ).strip()

        evidence_level = str(
            gt_row["evidence_level"]
        ).strip()

        source_configuration_id = str(
            gt_row[
                "source_canonical_configuration_id"
            ]
        ).strip()

        # ----------------------------------------------------
        # Requirements
        # ----------------------------------------------------

        rfq_requirements = (
            requirement_lookup.get(
                rfq_id,
                {},
            )
        )

        mandatory_requirements, preferred_requirements = (
            split_requirements(
                rfq_requirements
            )
        )

        mandatory_count = sum(
            len(values)
            for values
            in mandatory_requirements.values()
        )

        preferred_count = sum(
            len(values)
            for values
            in preferred_requirements.values()
        )

        # ----------------------------------------------------
        # Candidate configurations
        # ----------------------------------------------------

        valid_mandatory_ids = set()

        valid_mandatory_preferred_ids = set()

        # Keep row-level valid candidates too.
        row_valid_mandatory = []

        row_valid_mandatory_preferred = []

        for config_row in configuration_records:

            configuration_id = str(
                config_row[
                    "canonical_configuration_id"
                ]
            ).strip()

            # Package filtering:
            #
            # For this analysis we use target_package as the
            # package context. The configuration data contains
            # package information through packageFixed_dev
            # where available.
            #
            # However, packageFixed_dev is not assumed to be
            # equivalent to target_package automatically.
            #
            # Therefore we do NOT discard configurations here.
            # Package-specific analysis is reported separately
            # using the Top-K candidate set.
            #
            # Requirement equivalence is technical requirement
            # equivalence, not package equivalence.

            (
                mandatory_satisfied,
                preferred_satisfied_count,
                preferred_total,
                mandatory_violation_count,
                preferred_mismatch_count,
            ) = evaluate_configuration(
                config_row,
                mandatory_requirements,
                preferred_requirements,
            )

            if mandatory_satisfied:

                valid_mandatory_ids.add(
                    configuration_id
                )

                row_valid_mandatory.append(
                    {
                        "rfq_id":
                            rfq_id,
                        "configuration_id":
                            configuration_id,
                    }
                )

                if (
                    preferred_mismatch_count == 0
                ):

                    valid_mandatory_preferred_ids.add(
                        configuration_id
                    )

                    row_valid_mandatory_preferred.append(
                        {
                            "rfq_id":
                                rfq_id,
                            "configuration_id":
                                configuration_id,
                        }
                    )

        # ----------------------------------------------------
        # Unique valid IDs
        # ----------------------------------------------------

        mandatory_valid_count = len(
            valid_mandatory_ids
        )

        mandatory_preferred_valid_count = len(
            valid_mandatory_preferred_ids
        )

        ground_truth_mandatory_valid = (
            ground_truth_id
            in valid_mandatory_ids
        )

        ground_truth_mandatory_preferred_valid = (
            ground_truth_id
            in valid_mandatory_preferred_ids
        )

        # ----------------------------------------------------
        # Top-K recommendation positions
        # ----------------------------------------------------

        rfq_topk = oracle_topk[
            oracle_topk["rfq_id"]
            == rfq_id
        ].sort_values(
            "rank"
        )

        top1_id = None

        top1_rank = None

        if len(rfq_topk) > 0:

            top1_id = str(
                rfq_topk.iloc[0][
                    "canonical_configuration_id"
                ]
            ).strip()

            top1_rank = int(
                rfq_topk.iloc[0]["rank"]
            )

        top1_mandatory_equivalent = (
            top1_id in valid_mandatory_ids
            if top1_id is not None
            else False
        )

        top1_mandatory_preferred_equivalent = (
            top1_id
            in valid_mandatory_preferred_ids
            if top1_id is not None
            else False
        )

        # ----------------------------------------------------
        # Top-K equivalence
        # ----------------------------------------------------

        def first_equivalent_rank(
            valid_ids,
            max_rank=None,
        ):

            candidates = rfq_topk

            if max_rank is not None:

                candidates = candidates[
                    candidates["rank"]
                    <= max_rank
                ]

            matches = candidates[
                candidates[
                    "canonical_configuration_id"
                ].isin(valid_ids)
            ]

            if len(matches) == 0:

                return np.nan

            return int(
                matches.iloc[0]["rank"]
            )

        mandatory_equivalent_rank = (
            first_equivalent_rank(
                valid_mandatory_ids
            )
        )

        mandatory_preferred_equivalent_rank = (
            first_equivalent_rank(
                valid_mandatory_preferred_ids
            )
        )

        # ----------------------------------------------------
        # Exact-ID rank
        # ----------------------------------------------------

        exact_matches = rfq_topk[
            rfq_topk[
                "canonical_configuration_id"
            ]
            == ground_truth_id
        ]

        if len(exact_matches) > 0:

            exact_rank = int(
                exact_matches.iloc[0]["rank"]
            )

        else:

            exact_rank = np.nan

        # ----------------------------------------------------
        # Number of valid configurations
        # ----------------------------------------------------

        if mandatory_valid_count == 1:

            alternative_bucket = "1"

        elif mandatory_valid_count <= 5:

            alternative_bucket = "2-5"

        elif mandatory_valid_count <= 10:

            alternative_bucket = "6-10"

        elif mandatory_valid_count <= 50:

            alternative_bucket = "11-50"

        elif mandatory_valid_count <= 100:

            alternative_bucket = "51-100"

        elif mandatory_valid_count <= 500:

            alternative_bucket = "101-500"

        else:

            alternative_bucket = ">500"

        all_rfq_rows.append(
            {
                "rfq_id":
                    rfq_id,

                "ground_truth_configuration_id":
                    ground_truth_id,

                "source_canonical_configuration_id":
                    source_configuration_id,

                "target_package":
                    target_package,

                "record_type":
                    record_type,

                "evidence_level":
                    evidence_level,

                "mandatory_requirement_count":
                    mandatory_count,

                "preferred_requirement_count":
                    preferred_count,

                "mandatory_valid_configuration_count":
                    mandatory_valid_count,

                "mandatory_preferred_valid_configuration_count":
                    mandatory_preferred_valid_count,

                "alternative_count_bucket":
                    alternative_bucket,

                "ground_truth_mandatory_valid":
                    ground_truth_mandatory_valid,

                "ground_truth_mandatory_preferred_valid":
                    ground_truth_mandatory_preferred_valid,

                "top1_configuration_id":
                    top1_id,

                "top1_rank":
                    top1_rank,

                "top1_exact_match":
                    bool(
                        top1_id
                        == ground_truth_id
                    ),

                "top1_mandatory_equivalent":
                    top1_mandatory_equivalent,

                "top1_mandatory_preferred_equivalent":
                    top1_mandatory_preferred_equivalent,

                "exact_ground_truth_rank":
                    exact_rank,

                "first_mandatory_equivalent_rank":
                    mandatory_equivalent_rank,

                "first_mandatory_preferred_equivalent_rank":
                    mandatory_preferred_equivalent_rank,
            }
        )

        # ----------------------------------------------------
        # Valid configuration IDs
        # ----------------------------------------------------

        for configuration_id in sorted(
            valid_mandatory_ids
        ):

            valid_configuration_rows.append(
                {
                    "rfq_id":
                        rfq_id,

                    "configuration_id":
                        configuration_id,

                    "validity_type":
                        "MANDATORY",

                    "is_ground_truth":
                        configuration_id
                        == ground_truth_id,

                    "is_top1":
                        configuration_id
                        == top1_id,

                    "target_package":
                        target_package,

                    "record_type":
                        record_type,

                    "evidence_level":
                        evidence_level,
                }
            )

        for configuration_id in sorted(
            valid_mandatory_preferred_ids
        ):

            valid_configuration_rows.append(
                {
                    "rfq_id":
                        rfq_id,

                    "configuration_id":
                        configuration_id,

                    "validity_type":
                        "MANDATORY_AND_PREFERRED",

                    "is_ground_truth":
                        configuration_id
                        == ground_truth_id,

                    "is_top1":
                        configuration_id
                        == top1_id,

                    "target_package":
                        target_package,

                    "record_type":
                        record_type,

                    "evidence_level":
                        evidence_level,
                }
            )

    # ========================================================
    # 7. CREATE OUTPUTS AND METRICS
    # ========================================================

    print(
        "\n[7/7] Calculating metrics and saving outputs..."
    )

    rfq_summary = pd.DataFrame(
        all_rfq_rows
    )

    valid_configurations = pd.DataFrame(
        valid_configuration_rows
    )

    # ========================================================
    # CORE METRICS
    # ========================================================

    total_rfqs = len(
        rfq_summary
    )

    exact_top1_accuracy = (
        rfq_summary[
            "top1_exact_match"
        ]
        .mean()
    )

    mandatory_equivalent_top1 = (
        rfq_summary[
            "top1_mandatory_equivalent"
        ]
        .mean()
    )

    mandatory_preferred_equivalent_top1 = (
        rfq_summary[
            "top1_mandatory_preferred_equivalent"
        ]
        .mean()
    )

    gt_mandatory_valid_rate = (
        rfq_summary[
            "ground_truth_mandatory_valid"
        ]
        .mean()
    )

    gt_mandatory_preferred_valid_rate = (
        rfq_summary[
            "ground_truth_mandatory_preferred_valid"
        ]
        .mean()
    )

    # --------------------------------------------------------
    # Top-K mandatory-equivalence
    # --------------------------------------------------------

    top3_mandatory_equivalent = (
        rfq_summary[
            "first_mandatory_equivalent_rank"
        ]
        .le(3)
        .mean()
    )

    top5_mandatory_equivalent = (
        rfq_summary[
            "first_mandatory_equivalent_rank"
        ]
        .le(5)
        .mean()
    )

    top10_mandatory_equivalent = (
        rfq_summary[
            "first_mandatory_equivalent_rank"
        ]
        .le(10)
        .mean()
    )

    # --------------------------------------------------------
    # Top-K mandatory + preferred equivalence
    # --------------------------------------------------------

    top3_mandatory_preferred_equivalent = (
        rfq_summary[
            "first_mandatory_preferred_equivalent_rank"
        ]
        .le(3)
        .mean()
    )

    top5_mandatory_preferred_equivalent = (
        rfq_summary[
            "first_mandatory_preferred_equivalent_rank"
        ]
        .le(5)
        .mean()
    )

    top10_mandatory_preferred_equivalent = (
        rfq_summary[
            "first_mandatory_preferred_equivalent_rank"
        ]
        .le(10)
        .mean()
    )

    # ========================================================
    # ALTERNATIVE COUNT STATISTICS
    # ========================================================

    alternative_counts = (
        rfq_summary[
            "mandatory_valid_configuration_count"
        ]
    )

    alternative_statistics = {
        "mean":
            alternative_counts.mean(),

        "median":
            alternative_counts.median(),

        "min":
            alternative_counts.min(),

        "max":
            alternative_counts.max(),

        "p25":
            alternative_counts.quantile(
                0.25
            ),

        "p75":
            alternative_counts.quantile(
                0.75,

            ),

        "p90":
            alternative_counts.quantile(
                0.90
            ),

        "p95":
            alternative_counts.quantile(
                0.95
            ),
    }

    # ========================================================
    # ALTERNATIVE BUCKET SUMMARY
    # ========================================================

    bucket_order = [
        "1",
        "2-5",
        "6-10",
        "11-50",
        "51-100",
        "101-500",
        ">500",
    ]

    bucket_rows = []

    for bucket in bucket_order:

        subset = rfq_summary[
            rfq_summary[
                "alternative_count_bucket"
            ]
            == bucket
        ]

        if len(subset) == 0:
            continue

        bucket_rows.append(
            {
                "alternative_count_bucket":
                    bucket,

                "rfq_count":
                    len(subset),

                "rfq_share":
                    len(subset)
                    / total_rfqs,

                "exact_top1_accuracy":
                    subset[
                        "top1_exact_match"
                    ].mean(),

                "mandatory_equivalent_top1":
                    subset[
                        "top1_mandatory_equivalent"
                    ].mean(),

                "mandatory_preferred_equivalent_top1":
                    subset[
                        "top1_mandatory_preferred_equivalent"
                    ].mean(),

                "mean_valid_configuration_count":
                    subset[
                        "mandatory_valid_configuration_count"
                    ].mean(),
            }
        )

    bucket_summary = pd.DataFrame(
        bucket_rows
    )

    # ========================================================
    # RECORD TYPE SUMMARY
    # ========================================================

    record_type_summary = (
        rfq_summary
        .groupby(
            "record_type"
        )
        .agg(
            rfq_count=(
                "rfq_id",
                "size",
            ),
            exact_top1_accuracy=(
                "top1_exact_match",
                "mean",
            ),
            mandatory_equivalent_top1=(
                "top1_mandatory_equivalent",
                "mean",
            ),
            mandatory_preferred_equivalent_top1=(
                "top1_mandatory_preferred_equivalent",
                "mean",
            ),
            mean_valid_configurations=(
                "mandatory_valid_configuration_count",
                "mean",
            ),
            median_valid_configurations=(
                "mandatory_valid_configuration_count",
                "median",
            ),
        )
        .reset_index()
    )

    # ========================================================
    # EVIDENCE LEVEL SUMMARY
    # ========================================================

    evidence_summary = (
        rfq_summary
        .groupby(
            "evidence_level"
        )
        .agg(
            rfq_count=(
                "rfq_id",
                "size",
            ),
            exact_top1_accuracy=(
                "top1_exact_match",
                "mean",
            ),
            mandatory_equivalent_top1=(
                "top1_mandatory_equivalent",
                "mean",
            ),
            mandatory_preferred_equivalent_top1=(
                "top1_mandatory_preferred_equivalent",
                "mean",
            ),
            mean_valid_configurations=(
                "mandatory_valid_configuration_count",
                "mean",
            ),
            median_valid_configurations=(
                "mandatory_valid_configuration_count",
                "median",
            ),
        )
        .reset_index()
    )

    # ========================================================
    # PACKAGE SUMMARY
    # ========================================================

    package_summary = (
        rfq_summary
        .groupby(
            "target_package"
        )
        .agg(
            rfq_count=(
                "rfq_id",
                "size",
            ),
            exact_top1_accuracy=(
                "top1_exact_match",
                "mean",
            ),
            mandatory_equivalent_top1=(
                "top1_mandatory_equivalent",
                "mean",
            ),
            mandatory_preferred_equivalent_top1=(
                "top1_mandatory_preferred_equivalent",
                "mean",
            ),
            mean_valid_configurations=(
                "mandatory_valid_configuration_count",
                "mean",
            ),
            median_valid_configurations=(
                "mandatory_valid_configuration_count",
                "median",
            ),
        )
        .reset_index()
    )

    # ========================================================
    # METRICS TABLE
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
                "exact_top1_accuracy",
            "value":
                exact_top1_accuracy,
        },
        {
            "metric":
                "mandatory_equivalent_top1_accuracy",
            "value":
                mandatory_equivalent_top1,
        },
        {
            "metric":
                "mandatory_preferred_equivalent_top1_accuracy",
            "value":
                mandatory_preferred_equivalent_top1,
        },
        {
            "metric":
                "mandatory_equivalent_top3_accuracy",
            "value":
                top3_mandatory_equivalent,
        },
        {
            "metric":
                "mandatory_equivalent_top5_accuracy",
            "value":
                top5_mandatory_equivalent,
        },
        {
            "metric":
                "mandatory_equivalent_top10_accuracy",
            "value":
                top10_mandatory_equivalent,
        },
        {
            "metric":
                "mandatory_preferred_equivalent_top3_accuracy",
            "value":
                top3_mandatory_preferred_equivalent,
        },
        {
            "metric":
                "mandatory_preferred_equivalent_top5_accuracy",
            "value":
                top5_mandatory_preferred_equivalent,
        },
        {
            "metric":
                "mandatory_preferred_equivalent_top10_accuracy",
            "value":
                top10_mandatory_preferred_equivalent,
        },
        {
            "metric":
                "ground_truth_mandatory_valid_rate",
            "value":
                gt_mandatory_valid_rate,
        },
        {
            "metric":
                "ground_truth_mandatory_preferred_valid_rate",
            "value":
                gt_mandatory_preferred_valid_rate,
        },
        {
            "metric":
                "mean_mandatory_valid_configurations",
            "value":
                alternative_statistics["mean"],
        },
        {
            "metric":
                "median_mandatory_valid_configurations",
            "value":
                alternative_statistics["median"],
        },
        {
            "metric":
                "minimum_mandatory_valid_configurations",
            "value":
                alternative_statistics["min"],
        },
        {
            "metric":
                "maximum_mandatory_valid_configurations",
            "value":
                alternative_statistics["max"],
        },
        {
            "metric":
                "p25_mandatory_valid_configurations",
            "value":
                alternative_statistics["p25"],
        },
        {
            "metric":
                "p75_mandatory_valid_configurations",
            "value":
                alternative_statistics["p75"],
        },
        {
            "metric":
                "p90_mandatory_valid_configurations",
            "value":
                alternative_statistics["p90"],
        },
        {
            "metric":
                "p95_mandatory_valid_configurations",
            "value":
                alternative_statistics["p95"],
        },
    ]

    metrics = pd.DataFrame(
        metrics_rows
    )

    # ========================================================
    # SAVE FILES
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_files = {
        "731_v2_oracle_requirement_equivalence_rfq_summary.csv":
            rfq_summary,

        "731_v2_oracle_requirement_equivalence_valid_configurations.csv":
            valid_configurations,

        "731_v2_oracle_requirement_equivalence_metrics.csv":
            metrics,

        "731_v2_oracle_requirement_equivalence_alternative_buckets.csv":
            bucket_summary,

        "731_v2_oracle_requirement_equivalence_record_type_summary.csv":
            record_type_summary,

        "731_v2_oracle_requirement_equivalence_evidence_summary.csv":
            evidence_summary,

        "731_v2_oracle_requirement_equivalence_package_summary.csv":
            package_summary,
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
    # CONSOLE RESULTS
    # ========================================================

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(
        f"\nRFQs analysed: "
        f"{total_rfqs:,}"
    )

    print(
        "\n--- Exact configuration identity ---"
    )

    print(
        "Exact-ID Top-1 accuracy: "
        f"{exact_top1_accuracy:.4f}"
    )

    print(
        "\n--- Mandatory requirement equivalence ---"
    )

    print(
        "Top-1 mandatory-equivalent accuracy: "
        f"{mandatory_equivalent_top1:.4f}"
    )

    print(
        "Top-3 mandatory-equivalent accuracy: "
        f"{top3_mandatory_equivalent:.4f}"
    )

    print(
        "Top-5 mandatory-equivalent accuracy: "
        f"{top5_mandatory_equivalent:.4f}"
    )

    print(
        "Top-10 mandatory-equivalent accuracy: "
        f"{top10_mandatory_equivalent:.4f}"
    )

    print(
        "\n--- Mandatory + preferred equivalence ---"
    )

    print(
        "Top-1 mandatory+preferred-equivalent accuracy: "
        f"{mandatory_preferred_equivalent_top1:.4f}"
    )

    print(
        "Top-3 mandatory+preferred-equivalent accuracy: "
        f"{top3_mandatory_preferred_equivalent:.4f}"
    )

    print(
        "Top-5 mandatory+preferred-equivalent accuracy: "
        f"{top5_mandatory_preferred_equivalent:.4f}"
    )

    print(
        "Top-10 mandatory+preferred-equivalent accuracy: "
        f"{top10_mandatory_preferred_equivalent:.4f}"
    )

    print(
        "\n--- Ground truth validity ---"
    )

    print(
        "Ground-truth mandatory-valid rate: "
        f"{gt_mandatory_valid_rate:.4f}"
    )

    print(
        "Ground-truth mandatory+preferred-valid rate: "
        f"{gt_mandatory_preferred_valid_rate:.4f}"
    )

    print(
        "\n--- Number of valid alternatives ---"
    )

    print(
        "Mean valid configurations: "
        f"{alternative_statistics['mean']:.2f}"
    )

    print(
        "Median valid configurations: "
        f"{alternative_statistics['median']:.2f}"
    )

    print(
        "Minimum valid configurations: "
        f"{alternative_statistics['min']:.0f}"
    )

    print(
        "Maximum valid configurations: "
        f"{alternative_statistics['max']:.0f}"
    )

    print(
        "P25 valid configurations: "
        f"{alternative_statistics['p25']:.2f}"
    )

    print(
        "P75 valid configurations: "
        f"{alternative_statistics['p75']:.2f}"
    )

    print(
        "P90 valid configurations: "
        f"{alternative_statistics['p90']:.2f}"
    )

    print(
        "P95 valid configurations: "
        f"{alternative_statistics['p95']:.2f}"
    )

    print(
        "\n--- Alternative-count distribution ---"
    )

    if len(bucket_summary) > 0:

        print(
            bucket_summary[
                [
                    "alternative_count_bucket",
                    "rfq_count",
                    "rfq_share",
                    "exact_top1_accuracy",
                    "mandatory_equivalent_top1",
                ]
            ]
            .to_string(
                index=False
            )
        )

    print(
        "\nAnalysis complete."
    )


if __name__ == "__main__":
    main()