from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 731 V2 ORACLE PACKAGE-CONSTRAINED
# REQUIREMENT EQUIVALENCE ANALYSIS
#
# Purpose:
#   1. Evaluate Oracle recommendations against the full
#      synthetic configuration space.
#   2. Restrict valid configurations to the RFQ target package.
#   3. Separate exact configuration identity from
#      requirement-equivalent validity.
#   4. Quantify the number of technically valid alternatives.
#   5. Classify Oracle Top-1 outcomes.
#
# Important:
#   package_context is the explicit package field in
#   731_synthetic_configurations_v5.csv.
#
#   We do NOT infer package membership from technical
#   characteristics or from packageFixed_dev.
# ============================================================


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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/processed"
)


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
# NORMALIZATION
# ============================================================

def normalize_value(value):
    """
    Normalize configuration / requirement values.

    NOVALUE, NaN, None and empty strings are treated as
    unconfigured / missing.
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


def normalize_requirement_type(value):
    """
    Normalize requirement labels.
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


def normalize_package(value):
    """
    Normalize package identifiers.
    """

    value = normalize_value(value)

    if value is None:
        return None

    return str(value).strip()


def values_equal(a, b):
    """
    Exact normalized value comparison.
    """

    return (
        normalize_value(a)
        == normalize_value(b)
    )


# ============================================================
# PACKAGE MATCHING
# ============================================================

def package_matches(
    configuration_package,
    target_package,
):
    """
    V5 contains an explicit package_context field.

    Package compatibility therefore uses exact normalized
    equality.

    No package membership is inferred from technical
    characteristics.
    """

    configuration_package = (
        normalize_package(
            configuration_package
        )
    )

    target_package = (
        normalize_package(
            target_package
        )
    )

    if (
        configuration_package is None
        or target_package is None
    ):
        return False

    return (
        configuration_package
        == target_package
    )


# ============================================================
# REQUIREMENT LOOKUP
# ============================================================

def build_requirement_lookup(
    requirements
):
    """
    Build:

        RFQ
          -> characteristic
             -> requirement records
    """

    lookup = {}

    for row in requirements.itertuples(
        index=False
    ):

        rfq_id = str(
            row.rfq_id
        ).strip()

        characteristic = str(
            row.characteristic
        ).strip()

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

        if (
            characteristic
            not in lookup[rfq_id]
        ):

            lookup[rfq_id][
                characteristic
            ] = []

        lookup[rfq_id][
            characteristic
        ].append(
            {
                "requirement_type":
                    requirement_type,

                "internal_value":
                    internal_value,
            }
        )

    return lookup


def split_requirements(
    rfq_requirements
):
    """
    Split requirements into mandatory and preferred.
    """

    mandatory = {}

    preferred = {}

    for (
        characteristic,
        records,
    ) in rfq_requirements.items():

        for record in records:

            requirement_type = (
                record[
                    "requirement_type"
                ]
            )

            value = record[
                "internal_value"
            ]

            if (
                requirement_type
                == "MANDATORY"
            ):

                mandatory.setdefault(
                    characteristic,
                    [],
                ).append(
                    value
                )

            elif (
                requirement_type
                == "PREFERRED"
            ):

                preferred.setdefault(
                    characteristic,
                    [],
                ).append(
                    value
                )

    return (
        mandatory,
        preferred,
    )


# ============================================================
# CONFIGURATION SATISFACTION
# ============================================================

def evaluate_configuration(
    config,
    mandatory_requirements,
    preferred_requirements,
):
    """
    Evaluate one configuration against one RFQ.

    Mandatory:
        Every mandatory requirement must be satisfied.

    Preferred:
        Preferred requirements are scored separately.

    Unmentioned configuration features are NOT treated
    as violations.
    """

    mandatory_satisfied = True

    mandatory_violations = 0

    preferred_satisfied = 0

    preferred_total = 0

    preferred_mismatches = 0

    # --------------------------------------------------------
    # Mandatory requirements
    # --------------------------------------------------------

    for (
        characteristic,
        required_values,
    ) in mandatory_requirements.items():

        configuration_value = (
            normalize_value(
                config.get(
                    characteristic
                )
            )
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

            mandatory_violations += 1

    # --------------------------------------------------------
    # Preferred requirements
    # --------------------------------------------------------

    for (
        characteristic,
        required_values,
    ) in preferred_requirements.items():

        configuration_value = (
            normalize_value(
                config.get(
                    characteristic
                )
            )
        )

        preferred_total += (
            len(required_values)
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

            preferred_satisfied += (
                len(required_values)
            )

        else:

            preferred_mismatches += (
                len(required_values)
            )

    return {
        "mandatory_satisfied":
            mandatory_satisfied,

        "mandatory_violations":
            mandatory_violations,

        "preferred_satisfied":
            preferred_satisfied,

        "preferred_total":
            preferred_total,

        "preferred_mismatches":
            preferred_mismatches,

        "fully_preferred_satisfied":
            preferred_mismatches == 0,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "731 V2 ORACLE PACKAGE-CONSTRAINED "
        "REQUIREMENT EQUIVALENCE ANALYSIS"
    )

    print("=" * 80)

    # ========================================================
    # 1. LOAD INPUT FILES
    # ========================================================

    print(
        "\n[1/8] Loading files..."
    )

    topk = pd.read_csv(
        TOPK_FILE
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

    print(
        f"Top-K rows:         "
        f"{len(topk):,}"
    )

    print(
        f"Configuration rows: "
        f"{len(configurations):,}"
    )

    print(
        f"Ground-truth rows:  "
        f"{len(ground_truth):,}"
    )

    print(
        f"Requirement rows:   "
        f"{len(requirements):,}"
    )

    # ========================================================
    # 2. VALIDATE SCHEMAS
    # ========================================================

    print(
        "\n[2/8] Validating schemas..."
    )

    required_topk = {
        "rfq_id",
        "mode",
        "rank",
        "canonical_configuration_id",
        "package_context",
        "score",
    }

    required_config = {
        "canonical_configuration_id",
        "package_context",
        *TECHNICAL_CHARACTERISTICS,
    }

    required_gt = {
        "rfq_id",
        "canonical_configuration_id",
        "target_package",
        "record_type",
        "evidence_level",
        "source_canonical_configuration_id",
    }

    required_req = {
        "rfq_id",
        "characteristic",
        "internal_value",
        "requirement_type",
    }

    schema_definitions = [
        (
            "Top-K",
            topk,
            required_topk,
        ),
        (
            "Configurations",
            configurations,
            required_config,
        ),
        (
            "Ground truth",
            ground_truth,
            required_gt,
        ),
        (
            "Requirements",
            requirements,
            required_req,
        ),
    ]

    for (
        name,
        dataframe,
        required_columns,
    ) in schema_definitions:

        missing = (
            required_columns
            - set(dataframe.columns)
        )

        if missing:

            raise ValueError(
                f"{name} file is missing "
                f"required columns: "
                f"{sorted(missing)}"
            )

    print(
        "Schema validation passed."
    )

    # ========================================================
    # 3. PREPARE DATA
    # ========================================================

    print(
        "\n[3/8] Preparing data..."
    )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

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

    ground_truth[
        "target_package"
    ] = (
        ground_truth[
            "target_package"
        ]
        .astype(str)
        .str.strip()
    )

    ground_truth = (
        ground_truth
        .drop_duplicates(
            subset=["rfq_id"],
            keep="first",
        )
    )

    # --------------------------------------------------------
    # Top-K
    # --------------------------------------------------------

    topk["rfq_id"] = (
        topk["rfq_id"]
        .astype(str)
        .str.strip()
    )

    topk[
        "canonical_configuration_id"
    ] = (
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

    topk["mode"] = (
        topk["mode"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    oracle_topk = topk[
        topk["mode"] == "ORACLE"
    ].copy()

    # --------------------------------------------------------
    # Configurations
    # --------------------------------------------------------

    configurations[
        "canonical_configuration_id"
    ] = (
        configurations[
            "canonical_configuration_id"
        ]
        .astype(str)
        .str.strip()
    )

    configurations[
        "package_context"
    ] = (
        configurations[
            "package_context"
        ]
        .apply(
            normalize_package
        )
    )

    configuration_records = (
        configurations.to_dict(
            orient="records"
        )
    )

    # --------------------------------------------------------
    # Requirements
    # --------------------------------------------------------

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

    gt_lookup = (
        ground_truth
        .set_index("rfq_id")
    )

    rfq_ids = sorted(
        set(
            oracle_topk[
                "rfq_id"
            ]
        )
        &
        set(
            ground_truth[
                "rfq_id"
            ]
        )
    )

    print(
        f"Oracle RFQs: "
        f"{len(rfq_ids):,}"
    )

    print(
        f"Unique configuration IDs: "
        f"{configurations['canonical_configuration_id'].nunique():,}"
    )

    # ========================================================
    # 4. PACKAGE AUDIT
    # ========================================================

    print(
        "\n[4/8] Auditing package compatibility..."
    )

    package_values = (
        ground_truth[
            "target_package"
        ]
        .dropna()
        .astype(str)
        .str.strip()
        .value_counts()
    )

    package_audit_rows = []

    for target_package in (
        package_values.index
    ):

        matching_rows = [
            config
            for config
            in configuration_records
            if package_matches(
                config[
                    "package_context"
                ],
                target_package,
            )
        ]

        matching_ids = {
            config[
                "canonical_configuration_id"
            ]
            for config
            in matching_rows
        }

        package_audit_rows.append(
            {
                "target_package":
                    target_package,

                "rfq_count":
                    int(
                        package_values[
                            target_package
                        ]
                    ),

                "matching_configuration_rows":
                    len(matching_rows),

                "matching_unique_configuration_ids":
                    len(matching_ids),
            }
        )

    package_audit = pd.DataFrame(
        package_audit_rows
    )

    print(
        package_audit.to_string(
            index=False
        )
    )

    # ========================================================
    # 5. PACKAGE CACHE
    # ========================================================

    print(
        "\n[5/8] Preparing package-constrained "
        "configuration pools..."
    )

    package_configuration_cache = {}

    unique_target_packages = (
        ground_truth[
            "target_package"
        ]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    for target_package in (
        unique_target_packages
    ):

        matching = [
            config
            for config
            in configuration_records
            if package_matches(
                config[
                    "package_context"
                ],
                target_package,
            )
        ]

        package_configuration_cache[
            target_package
        ] = matching

    print(
        f"Package contexts prepared: "
        f"{len(package_configuration_cache):,}"
    )

    # ========================================================
    # 6. EVALUATE RFQs
    # ========================================================

    print(
        "\n[6/8] Evaluating package-constrained "
        "configuration space..."
    )

    print(
        "This evaluates every RFQ against "
        "all configurations in its target package."
    )

    rfq_results = []

    valid_configuration_results = []

    # --------------------------------------------------------
    # Process RFQs
    # --------------------------------------------------------

    for index, rfq_id in enumerate(
        rfq_ids,
        start=1,
    ):

        if (
            index == 1
            or index % 500 == 0
            or index == len(rfq_ids)
        ):

            print(
                f"  Processing "
                f"{index:,}/{len(rfq_ids):,}"
            )

        # ----------------------------------------------------
        # Ground truth metadata
        # ----------------------------------------------------

        gt = gt_lookup.loc[
            rfq_id
        ]

        ground_truth_id = str(
            gt[
                "ground_truth_configuration_id"
            ]
        ).strip()

        target_package = str(
            gt[
                "target_package"
            ]
        ).strip()

        record_type = str(
            gt[
                "record_type"
            ]
        ).strip()

        evidence_level = str(
            gt[
                "evidence_level"
            ]
        ).strip()

        source_id = str(
            gt[
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

        mandatory, preferred = (
            split_requirements(
                rfq_requirements
            )
        )

        mandatory_count = sum(
            len(values)
            for values
            in mandatory.values()
        )

        preferred_count = sum(
            len(values)
            for values
            in preferred.values()
        )

        # ----------------------------------------------------
        # Package candidate pool
        # ----------------------------------------------------

        package_configs = (
            package_configuration_cache.get(
                target_package,
                [],
            )
        )

        # ----------------------------------------------------
        # Valid configuration IDs
        # ----------------------------------------------------

        mandatory_valid_ids = set()

        mandatory_preferred_valid_ids = set()

        # ----------------------------------------------------
        # Evaluate every package configuration
        # ----------------------------------------------------

        for config in package_configs:

            configuration_id = str(
                config[
                    "canonical_configuration_id"
                ]
            ).strip()

            result = evaluate_configuration(
                config,
                mandatory,
                preferred,
            )

            if result[
                "mandatory_satisfied"
            ]:

                mandatory_valid_ids.add(
                    configuration_id
                )

                if result[
                    "fully_preferred_satisfied"
                ]:

                    (
                        mandatory_preferred_valid_ids
                        .add(
                            configuration_id
                        )
                    )

        # ----------------------------------------------------
        # Oracle Top-K
        # ----------------------------------------------------

        rfq_topk = (
            oracle_topk[
                oracle_topk[
                    "rfq_id"
                ]
                == rfq_id
            ]
            .sort_values(
                "rank"
            )
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
                rfq_topk.iloc[0][
                    "rank"
                ]
            )

        # ----------------------------------------------------
        # Exact ground-truth rank
        # ----------------------------------------------------

        exact_rows = rfq_topk[
            rfq_topk[
                "canonical_configuration_id"
            ]
            == ground_truth_id
        ]

        if len(exact_rows) > 0:

            exact_rank = int(
                exact_rows.iloc[0][
                    "rank"
                ]
            )

        else:

            exact_rank = np.nan

        # ----------------------------------------------------
        # First valid rank
        # ----------------------------------------------------

        def first_valid_rank(
            valid_ids
        ):

            if len(
                valid_ids
            ) == 0:

                return np.nan

            matches = rfq_topk[
                rfq_topk[
                    "canonical_configuration_id"
                ].isin(
                    valid_ids
                )
            ]

            if len(matches) == 0:

                return np.nan

            return int(
                matches.iloc[0][
                    "rank"
                ]
            )

        mandatory_equivalent_rank = (
            first_valid_rank(
                mandatory_valid_ids
            )
        )

        mandatory_preferred_equivalent_rank = (
            first_valid_rank(
                mandatory_preferred_valid_ids
            )
        )

        # ----------------------------------------------------
        # Top-1 validity
        # ----------------------------------------------------

        top1_exact = (
            top1_id
            == ground_truth_id
        )

        top1_mandatory_valid = (
            top1_id
            in mandatory_valid_ids
        )

        top1_mandatory_preferred_valid = (
            top1_id
            in mandatory_preferred_valid_ids
        )

        # ----------------------------------------------------
        # Compare Top-1 to ground truth
        # ----------------------------------------------------

        difference_count = np.nan

        differing_characteristics = ""

        gt_config_rows = configurations[
            configurations[
                "canonical_configuration_id"
            ]
            == ground_truth_id
        ]

        top1_config_rows = configurations[
            configurations[
                "canonical_configuration_id"
            ]
            == top1_id
        ]

        if (
            len(gt_config_rows) > 0
            and len(top1_config_rows) > 0
            and not top1_exact
        ):

            gt_config = (
                gt_config_rows.iloc[0]
            )

            top1_config = (
                top1_config_rows.iloc[0]
            )

            differences = []

            for characteristic in (
                TECHNICAL_CHARACTERISTICS
            ):

                gt_value = (
                    normalize_value(
                        gt_config[
                            characteristic
                        ]
                    )
                )

                top1_value = (
                    normalize_value(
                        top1_config[
                            characteristic
                        ]
                    )
                )

                if (
                    gt_value
                    != top1_value
                ):

                    differences.append(
                        characteristic
                    )

            difference_count = len(
                differences
            )

            differing_characteristics = (
                "|".join(
                    differences
                )
            )

        # ----------------------------------------------------
        # Failure category
        # ----------------------------------------------------

        if top1_exact:

            failure_category = (
                "EXACT_MATCH"
            )

        elif top1_id is None:

            failure_category = (
                "NO_TOP1_CANDIDATE"
            )

        elif top1_mandatory_preferred_valid:

            if (
                pd.notna(
                    difference_count
                )
                and difference_count == 0
            ):

                failure_category = (
                    "REQUIREMENT_EQUIVALENT"
                )

            else:

                failure_category = (
                    "REQUIREMENT_EQUIVALENT_EXTRA_DIFFERENCES"
                )

        elif top1_mandatory_valid:

            failure_category = (
                "MANDATORY_ONLY_VALID"
            )

        else:

            failure_category = (
                "NON_COMPLIANT_TOP1"
            )

        # ----------------------------------------------------
        # Alternative count bucket
        # ----------------------------------------------------

        valid_count = len(
            mandatory_valid_ids
        )

        if valid_count == 0:

            bucket = "0"

        elif valid_count == 1:

            bucket = "1"

        elif valid_count <= 5:

            bucket = "2-5"

        elif valid_count <= 10:

            bucket = "6-10"

        elif valid_count <= 50:

            bucket = "11-50"

        elif valid_count <= 100:

            bucket = "51-100"

        elif valid_count <= 500:

            bucket = "101-500"

        else:

            bucket = ">500"

        # ----------------------------------------------------
        # RFQ summary record
        # ----------------------------------------------------

        rfq_results.append(
            {
                "rfq_id":
                    rfq_id,

                "target_package":
                    target_package,

                "record_type":
                    record_type,

                "evidence_level":
                    evidence_level,

                "ground_truth_configuration_id":
                    ground_truth_id,

                "source_canonical_configuration_id":
                    source_id,

                "mandatory_requirement_count":
                    mandatory_count,

                "preferred_requirement_count":
                    preferred_count,

                "package_configuration_rows":
                    len(package_configs),

                "package_unique_configuration_ids":
                    len(
                        {
                            config[
                                "canonical_configuration_id"
                            ]
                            for config
                            in package_configs
                        }
                    ),

                "mandatory_valid_configuration_count":
                    valid_count,

                "mandatory_preferred_valid_configuration_count":
                    len(
                        mandatory_preferred_valid_ids
                    ),

                "alternative_count_bucket":
                    bucket,

                "ground_truth_mandatory_valid":
                    (
                        ground_truth_id
                        in mandatory_valid_ids
                    ),

                "ground_truth_mandatory_preferred_valid":
                    (
                        ground_truth_id
                        in mandatory_preferred_valid_ids
                    ),

                "top1_configuration_id":
                    top1_id,

                "top1_rank":
                    top1_rank,

                "exact_ground_truth_rank":
                    exact_rank,

                "top1_exact_match":
                    top1_exact,

                "top1_mandatory_valid":
                    top1_mandatory_valid,

                "top1_mandatory_preferred_valid":
                    top1_mandatory_preferred_valid,

                "first_mandatory_equivalent_rank":
                    mandatory_equivalent_rank,

                "first_mandatory_preferred_equivalent_rank":
                    mandatory_preferred_equivalent_rank,

                "top1_difference_count":
                    difference_count,

                "top1_differing_characteristics":
                    differing_characteristics,

                "failure_category":
                    failure_category,
            }
        )

        # ----------------------------------------------------
        # Save mandatory-valid configurations
        # ----------------------------------------------------

        for configuration_id in sorted(
            mandatory_valid_ids
        ):

            valid_configuration_results.append(
                {
                    "rfq_id":
                        rfq_id,

                    "target_package":
                        target_package,

                    "configuration_id":
                        configuration_id,

                    "validity_type":
                        "MANDATORY",

                    "is_ground_truth":
                        (
                            configuration_id
                            == ground_truth_id
                        ),

                    "is_top1":
                        (
                            configuration_id
                            == top1_id
                        ),
                }
            )

        # ----------------------------------------------------
        # Save mandatory + preferred configurations
        # ----------------------------------------------------

        for configuration_id in sorted(
            mandatory_preferred_valid_ids
        ):

            valid_configuration_results.append(
                {
                    "rfq_id":
                        rfq_id,

                    "target_package":
                        target_package,

                    "configuration_id":
                        configuration_id,

                    "validity_type":
                        "MANDATORY_AND_PREFERRED",

                    "is_ground_truth":
                        (
                            configuration_id
                            == ground_truth_id
                        ),

                    "is_top1":
                        (
                            configuration_id
                            == top1_id
                        ),
                }
            )

    # ========================================================
    # 7. BUILD OUTPUTS
    # ========================================================

    print(
        "\n[7/8] Building summaries..."
    )

    rfq_summary = pd.DataFrame(
        rfq_results
    )

    valid_configurations = (
        pd.DataFrame(
            valid_configuration_results
        )
    )

    total_rfqs = len(
        rfq_summary
    )

    # --------------------------------------------------------
    # Core accuracy
    # --------------------------------------------------------

    exact_top1 = (
        rfq_summary[
            "top1_exact_match"
        ].mean()
    )

    mandatory_top1 = (
        rfq_summary[
            "top1_mandatory_valid"
        ].mean()
    )

    mandatory_preferred_top1 = (
        rfq_summary[
            "top1_mandatory_preferred_valid"
        ].mean()
    )

    # --------------------------------------------------------
    # Top-K requirement equivalence
    # --------------------------------------------------------

    mandatory_top3 = (
        rfq_summary[
            "first_mandatory_equivalent_rank"
        ]
        .le(3)
        .mean()
    )

    mandatory_top5 = (
        rfq_summary[
            "first_mandatory_equivalent_rank"
        ]
        .le(5)
        .mean()
    )

    mandatory_top10 = (
        rfq_summary[
            "first_mandatory_equivalent_rank"
        ]
        .le(10)
        .mean()
    )

    mandatory_preferred_top3 = (
        rfq_summary[
            "first_mandatory_preferred_equivalent_rank"
        ]
        .le(3)
        .mean()
    )

    mandatory_preferred_top5 = (
        rfq_summary[
            "first_mandatory_preferred_equivalent_rank"
        ]
        .le(5)
        .mean()
    )

    mandatory_preferred_top10 = (
        rfq_summary[
            "first_mandatory_preferred_equivalent_rank"
        ]
        .le(10)
        .mean()
    )

    # --------------------------------------------------------
    # Ground-truth validity
    # --------------------------------------------------------

    gt_mandatory_valid = (
        rfq_summary[
            "ground_truth_mandatory_valid"
        ].mean()
    )

    gt_mandatory_preferred_valid = (
        rfq_summary[
            "ground_truth_mandatory_preferred_valid"
        ].mean()
    )

    # --------------------------------------------------------
    # Valid configuration statistics
    # --------------------------------------------------------

    valid_counts = (
        rfq_summary[
            "mandatory_valid_configuration_count"
        ]
    )

    # --------------------------------------------------------
    # Failure categories
    # --------------------------------------------------------

    failure_summary = (
        rfq_summary[
            "failure_category"
        ]
        .value_counts()
        .rename_axis(
            "failure_category"
        )
        .reset_index(
            name="rfq_count"
        )
    )

    failure_summary[
        "rfq_share"
    ] = (
        failure_summary[
            "rfq_count"
        ]
        / total_rfqs
    )

    # --------------------------------------------------------
    # Alternative-count distribution
    # --------------------------------------------------------

    bucket_summary = (
        rfq_summary
        .groupby(
            "alternative_count_bucket"
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

            mandatory_top1_accuracy=(
                "top1_mandatory_valid",
                "mean",
            ),

            mandatory_preferred_top1_accuracy=(
                "top1_mandatory_preferred_valid",
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

    bucket_order = [
        "0",
        "1",
        "2-5",
        "6-10",
        "11-50",
        "51-100",
        "101-500",
        ">500",
    ]

    bucket_summary[
        "bucket_order"
    ] = (
        bucket_summary[
            "alternative_count_bucket"
        ].apply(
            lambda x:
                (
                    bucket_order.index(x)
                    if x in bucket_order
                    else 999
                )
        )
    )

    bucket_summary = (
        bucket_summary
        .sort_values(
            "bucket_order"
        )
        .drop(
            columns=[
                "bucket_order"
            ]
        )
    )

    bucket_summary[
        "rfq_share"
    ] = (
        bucket_summary[
            "rfq_count"
        ]
        / total_rfqs
    )

    # --------------------------------------------------------
    # Package summary
    # --------------------------------------------------------

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

            mandatory_top1_accuracy=(
                "top1_mandatory_valid",
                "mean",
            ),

            mandatory_preferred_top1_accuracy=(
                "top1_mandatory_preferred_valid",
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

            min_valid_configurations=(
                "mandatory_valid_configuration_count",
                "min",
            ),

            max_valid_configurations=(
                "mandatory_valid_configuration_count",
                "max",
            ),

            mean_package_configuration_rows=(
                "package_configuration_rows",
                "mean",
            ),

            mean_unique_package_configurations=(
                "package_unique_configuration_ids",
                "mean",
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Record-type summary
    # --------------------------------------------------------

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

            mandatory_top1_accuracy=(
                "top1_mandatory_valid",
                "mean",
            ),

            mandatory_preferred_top1_accuracy=(
                "top1_mandatory_preferred_valid",
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

    # --------------------------------------------------------
    # Evidence-level summary
    # --------------------------------------------------------

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

            mandatory_top1_accuracy=(
                "top1_mandatory_valid",
                "mean",
            ),

            mandatory_preferred_top1_accuracy=(
                "top1_mandatory_preferred_valid",
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

    # --------------------------------------------------------
    # Difference summary
    # --------------------------------------------------------

    failed_exact = rfq_summary[
        ~rfq_summary[
            "top1_exact_match"
        ]
    ].copy()

    if len(failed_exact) > 0:

        difference_summary = pd.DataFrame(
            {
                "metric": [
                    "failed_exact_rfqs",

                    "mean_top1_characteristic_differences",

                    "median_top1_characteristic_differences",

                    "max_top1_characteristic_differences",

                    "failed_cases_with_1_or_less_difference",

                    "failed_cases_with_2_or_less_differences",

                    "failed_cases_with_top1_mandatory_valid",

                    "failed_cases_with_top1_mandatory_preferred_valid",
                ],

                "value": [
                    len(
                        failed_exact
                    ),

                    failed_exact[
                        "top1_difference_count"
                    ].mean(),

                    failed_exact[
                        "top1_difference_count"
                    ].median(),

                    failed_exact[
                        "top1_difference_count"
                    ].max(),

                    (
                        failed_exact[
                            "top1_difference_count"
                        ]
                        <= 1
                    ).sum(),

                    (
                        failed_exact[
                            "top1_difference_count"
                        ]
                        <= 2
                    ).sum(),

                    failed_exact[
                        "top1_mandatory_valid"
                    ].sum(),

                    failed_exact[
                        "top1_mandatory_preferred_valid"
                    ].sum(),
                ],
            }
        )

    else:

        difference_summary = pd.DataFrame(
            {
                "metric": [
                    "failed_exact_rfqs",
                ],

                "value": [
                    0,
                ],
            }
        )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    non_compliant_count = (
        rfq_summary[
            "failure_category"
        ]
        == "NON_COMPLIANT_TOP1"
    ).sum()

    metrics = pd.DataFrame(
        {
            "metric": [
                "total_rfqs",

                "exact_top1_accuracy",

                "mandatory_equivalent_top1_accuracy",

                "mandatory_equivalent_top3_accuracy",

                "mandatory_equivalent_top5_accuracy",

                "mandatory_equivalent_top10_accuracy",

                "mandatory_preferred_equivalent_top1_accuracy",

                "mandatory_preferred_equivalent_top3_accuracy",

                "mandatory_preferred_equivalent_top5_accuracy",

                "mandatory_preferred_equivalent_top10_accuracy",

                "ground_truth_mandatory_valid_rate",

                "ground_truth_mandatory_preferred_valid_rate",

                "mean_valid_mandatory_configurations",

                "median_valid_mandatory_configurations",

                "min_valid_mandatory_configurations",

                "max_valid_mandatory_configurations",

                "p25_valid_mandatory_configurations",

                "p75_valid_mandatory_configurations",

                "p90_valid_mandatory_configurations",

                "p95_valid_mandatory_configurations",

                "non_compliant_top1_count",

                "non_compliant_top1_rate",
            ],

            "value": [
                total_rfqs,

                exact_top1,

                mandatory_top1,

                mandatory_top3,

                mandatory_top5,

                mandatory_top10,

                mandatory_preferred_top1,

                mandatory_preferred_top3,

                mandatory_preferred_top5,

                mandatory_preferred_top10,

                gt_mandatory_valid,

                gt_mandatory_preferred_valid,

                valid_counts.mean(),

                valid_counts.median(),

                valid_counts.min(),

                valid_counts.max(),

                valid_counts.quantile(
                    0.25
                ),

                valid_counts.quantile(
                    0.75
                ),

                valid_counts.quantile(
                    0.90
                ),

                valid_counts.quantile(
                    0.95
                ),

                non_compliant_count,

                (
                    non_compliant_count
                    / total_rfqs
                ),
            ],
        }
    )

    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    print(
        "\n[8/8] Saving outputs..."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {

        "731_v2_oracle_package_equivalence_rfq_summary.csv":
            rfq_summary,

        "731_v2_oracle_package_equivalence_valid_configurations.csv":
            valid_configurations,

        "731_v2_oracle_package_equivalence_metrics.csv":
            metrics,

        "731_v2_oracle_package_equivalence_failure_summary.csv":
            failure_summary,

        "731_v2_oracle_package_equivalence_alternative_buckets.csv":
            bucket_summary,

        "731_v2_oracle_package_equivalence_package_summary.csv":
            package_summary,

        "731_v2_oracle_package_equivalence_record_type_summary.csv":
            record_type_summary,

        "731_v2_oracle_package_equivalence_evidence_summary.csv":
            evidence_summary,

        "731_v2_oracle_package_equivalence_difference_summary.csv":
            difference_summary,

        "731_v2_oracle_package_equivalence_package_audit.csv":
            package_audit,
    }

    for (
        filename,
        dataframe,
    ) in outputs.items():

        path = (
            OUTPUT_DIR
            / filename
        )

        dataframe.to_csv(
            path,
            index=False,
        )

        print(
            f"  Saved: {path}"
        )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "RESULTS"
    )

    print(
        "=" * 80
    )

    print(
        f"\nRFQs analysed: "
        f"{total_rfqs:,}"
    )

    # --------------------------------------------------------
    # Exact identity
    # --------------------------------------------------------

    print(
        "\n--- Exact configuration identity ---"
    )

    print(
        f"Exact-ID Top-1 accuracy: "
        f"{exact_top1:.4f}"
    )

    # --------------------------------------------------------
    # Mandatory equivalence
    # --------------------------------------------------------

    print(
        "\n--- Package-constrained mandatory "
        "requirement equivalence ---"
    )

    print(
        f"Top-1: "
        f"{mandatory_top1:.4f}"
    )

    print(
        f"Top-3: "
        f"{mandatory_top3:.4f}"
    )

    print(
        f"Top-5: "
        f"{mandatory_top5:.4f}"
    )

    print(
        f"Top-10: "
        f"{mandatory_top10:.4f}"
    )

    # --------------------------------------------------------
    # Mandatory + preferred
    # --------------------------------------------------------

    print(
        "\n--- Package-constrained mandatory + "
        "preferred equivalence ---"
    )

    print(
        f"Top-1: "
        f"{mandatory_preferred_top1:.4f}"
    )

    print(
        f"Top-3: "
        f"{mandatory_preferred_top3:.4f}"
    )

    print(
        f"Top-5: "
        f"{mandatory_preferred_top5:.4f}"
    )

    print(
        f"Top-10: "
        f"{mandatory_preferred_top10:.4f}"
    )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    print(
        "\n--- Ground truth validity ---"
    )

    print(
        f"Ground-truth mandatory-valid rate: "
        f"{gt_mandatory_valid:.4f}"
    )

    print(
        f"Ground-truth mandatory+preferred-valid rate: "
        f"{gt_mandatory_preferred_valid:.4f}"
    )

    # --------------------------------------------------------
    # Valid configuration space
    # --------------------------------------------------------

    print(
        "\n--- Number of valid alternatives ---"
    )

    print(
        f"Mean valid configurations: "
        f"{valid_counts.mean():.2f}"
    )

    print(
        f"Median valid configurations: "
        f"{valid_counts.median():.2f}"
    )

    print(
        f"Minimum valid configurations: "
        f"{valid_counts.min():.0f}"
    )

    print(
        f"Maximum valid configurations: "
        f"{valid_counts.max():.0f}"
    )

    print(
        f"P25 valid configurations: "
        f"{valid_counts.quantile(0.25):.2f}"
    )

    print(
        f"P75 valid configurations: "
        f"{valid_counts.quantile(0.75):.2f}"
    )

    print(
        f"P90 valid configurations: "
        f"{valid_counts.quantile(0.90):.2f}"
    )

    print(
        f"P95 valid configurations: "
        f"{valid_counts.quantile(0.95):.2f}"
    )

    # --------------------------------------------------------
    # Failure categories
    # --------------------------------------------------------

    print(
        "\n--- Failure categories ---"
    )

    print(
        failure_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Alternative buckets
    # --------------------------------------------------------

    print(
        "\n--- Alternative-count distribution ---"
    )

    print(
        bucket_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Package summary
    # --------------------------------------------------------

    print(
        "\n--- Package summary ---"
    )

    print(
        package_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Record type
    # --------------------------------------------------------

    print(
        "\n--- Record type summary ---"
    )

    print(
        record_type_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Evidence
    # --------------------------------------------------------

    print(
        "\n--- Evidence-level summary ---"
    )

    print(
        evidence_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Difference summary
    # --------------------------------------------------------

    print(
        "\n--- Exact-ID failure difference summary ---"
    )

    print(
        difference_summary.to_string(
            index=False
        )
    )

    print(
        "\nAnalysis complete."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()