import os
import math
import numpy as np
import pandas as pd


# ============================================================
# 731 RFQ CONFIGURATION RECOMMENDATION — V4
#
# V4 improvements:
#   1. Mandatory requirements receive higher weight
#   2. Preferred requirements receive lower weight
#   3. Exact characteristic/value matches are rewarded
#   4. Explicit value conflicts are strongly penalized
#   5. Unspecified optional features receive a small penalty
#   6. Known difficult characteristics receive refinement weights
#   7. Package-aware candidate refinement
#   8. Near-neighbour configuration similarity
#   9. V8 extracted vs ORACLE evaluation
#
# IMPORTANT:
# V4 does NOT modify RFQ extraction.
# It operates on V8 extracted requirements.
# ============================================================


BASE = "/home/e1546562/thesis-configurator"

V8_FILE = os.path.join(
    BASE,
    "data/processed/731_rfq_extraction_results_v8.csv"
)

CONFIG_FILE = os.path.join(
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


# ============================================================
# OUTPUT FILES
# ============================================================

RECOMMENDATION_FILE = os.path.join(
    OUTPUT_DIR,
    "731_rfq_configuration_recommendations_v4.csv"
)

METRICS_FILE = os.path.join(
    OUTPUT_DIR,
    "731_rfq_configuration_recommendation_metrics_v4.csv"
)

TOPK_FILE = os.path.join(
    OUTPUT_DIR,
    "731_rfq_configuration_topk_v4.csv"
)

ERROR_FILE = os.path.join(
    OUTPUT_DIR,
    "731_rfq_configuration_recommendation_errors_v4.csv"
)

PACKAGE_FILE = os.path.join(
    OUTPUT_DIR,
    "731_rfq_configuration_package_metrics_v4.csv"
)

SCORING_FILE = os.path.join(
    OUTPUT_DIR,
    "731_rfq_configuration_scoring_statistics_v4.csv"
)

CHARACTERISTIC_FILE = os.path.join(
    OUTPUT_DIR,
    "731_rfq_configuration_characteristic_metrics_v4.csv"
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
# DIFFICULT / DISCRIMINATIVE FEATURES
# ============================================================

REFINEMENT_WEIGHTS = {

    "stromSchaltbar": 1.80,

    "temperaturEingaenge": 1.80,

    "stromEingaenge": 1.80,

    "dev_advMeterVerification": 1.60,

    "customUserFluid": 1.60,

    "binaerOpenColl_MP": 1.60,

    "binaerDigitalOpenColl_MN": 1.60,

    "dynamicGasMaster": 1.50,

    "waveInjector": 1.40,

    "dataInterface": 1.30,

    "explosionApproval": 1.30,

    "protectionArea": 1.20,

    "certification": 1.20,

    "powerSupply": 1.20,

    "numberOfChannels": 1.20,

    "housing": 1.00,

    "devCategory": 1.00,

    "characteristic": 1.00,

    "steamApplication": 1.00,
}


# ============================================================
# REQUIREMENT WEIGHTS
# ============================================================

MANDATORY_WEIGHT = 5.0

PREFERRED_WEIGHT = 2.0

# Strong penalty when the configuration explicitly conflicts
# with a mandatory requirement.
MANDATORY_CONFLICT_PENALTY = 8.0

# Smaller penalty for conflict with preferred requirements.
PREFERRED_CONFLICT_PENALTY = 2.5

# Small penalty when the RFQ does not specify a feature but the
# candidate activates it.
UNSPECIFIED_OPTIONAL_PENALTY = 0.08


# ============================================================
# HELPERS
# ============================================================

def normalize(value):

    if pd.isna(value):
        return "NOVALUE"

    value = str(value).strip()

    if value == "":
        return "NOVALUE"

    return value


def load_csv(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    return pd.read_csv(
        path,
        low_memory=False
    )


def prepare_configuration_table(configurations):

    configurations = configurations.copy()

    required = [
        "canonical_configuration_id"
    ]

    missing = [
        c for c in required
        if c not in configurations.columns
    ]

    if missing:

        raise ValueError(
            "Configuration file is missing columns: "
            f"{missing}"
        )

    configurations[
        "canonical_configuration_id"
    ] = (
        configurations[
            "canonical_configuration_id"
        ]
        .astype(str)
        .str.strip()
    )

    if "package_context" in configurations.columns:

        configurations[
            "package_context"
        ] = (
            configurations[
                "package_context"
            ]
            .astype(str)
            .str.strip()
        )

    for characteristic in CHARACTERISTICS:

        if characteristic in configurations.columns:

            configurations[
                characteristic
            ] = configurations[
                characteristic
            ].apply(normalize)

        else:

            configurations[
                characteristic
            ] = "NOVALUE"

    return configurations


def build_config_lookup(configurations):

    lookup = {}

    for _, row in configurations.iterrows():

        config_id = str(
            row[
                "canonical_configuration_id"
            ]
        )

        lookup[config_id] = row

    return lookup


def build_requirement_lookup(requirements):

    result = {}

    for rfq_id, group in requirements.groupby(
        "rfq_id"
    ):

        result[str(rfq_id)] = group.copy()

    return result


def build_extracted_requirement_lookup(
    predictions
):

    result = {}

    for rfq_id, group in predictions.groupby(
        "rfq_id"
    ):

        result[str(rfq_id)] = group.copy()

    return result


def build_ground_truth_lookup(
    ground_truth
):

    lookup = {}

    for rfq_id, group in ground_truth.groupby(
        "rfq_id"
    ):

        rfq_id = str(rfq_id)

        config_ids = set(
            group[
                "canonical_configuration_id"
            ]
            .astype(str)
            .str.strip()
        )

        packages = set(
            group[
                "target_package"
            ]
            .astype(str)
            .str.strip()
        )

        lookup[rfq_id] = {

            "configuration_ids":
                config_ids,

            "packages":
                packages,
        }

    return lookup


# ============================================================
# REQUIREMENT EXTRACTION NORMALIZATION
# ============================================================

def prepare_predictions(predictions):

    predictions = predictions.copy()

    required = [
        "rfq_id",
        "characteristic",
        "internal_value",
        "confidence",
    ]

    missing = [
        c for c in required
        if c not in predictions.columns
    ]

    if missing:

        raise ValueError(
            "V8 prediction file is missing columns: "
            f"{missing}"
        )

    predictions["rfq_id"] = (
        predictions["rfq_id"]
        .astype(str)
        .str.strip()
    )

    predictions["characteristic"] = (
        predictions["characteristic"]
        .astype(str)
        .str.strip()
    )

    predictions["internal_value"] = (
        predictions["internal_value"]
        .apply(normalize)
    )

    predictions["confidence"] = pd.to_numeric(
        predictions["confidence"],
        errors="coerce"
    ).fillna(0.0)

    # Remove duplicate requirement predictions.
    #
    # Keep the highest-confidence occurrence.
    predictions = (
        predictions
        .sort_values(
            "confidence",
            ascending=False
        )
        .drop_duplicates(
            subset=[
                "rfq_id",
                "characteristic",
                "internal_value",
            ]
        )
        .copy()
    )

    return predictions


# ============================================================
# PACKAGE DETECTION
# ============================================================

def infer_package_from_config(
    config_row
):

    if config_row is None:
        return "UNKNOWN"

    if "package_context" in config_row.index:

        package = str(
            config_row[
                "package_context"
            ]
        ).strip()

        if package not in [
            "",
            "nan",
            "None",
            "NOVALUE",
        ]:

            return package

    return "UNKNOWN"


# ============================================================
# REQUIREMENT SCORE
# ============================================================

def score_configuration(
    config_row,
    requirements,
    package_context=None,
):
    """
    Calculate V4 score for one configuration.

    Positive evidence:
        exact mandatory match
        exact preferred match
        confidence

    Negative evidence:
        mandatory conflict
        preferred conflict
        optional activation without evidence

    Refinement:
        difficult-characteristic weights
    """

    if config_row is None:

        return {
            "score": -999999.0,
            "positive_score": 0.0,
            "negative_score": 999999.0,
            "mandatory_matches": 0,
            "mandatory_conflicts": 0,
            "preferred_matches": 0,
            "preferred_conflicts": 0,
            "unspecified_active": 0,
            "matched_requirements": 0,
        }

    score = 0.0

    positive_score = 0.0

    negative_score = 0.0

    mandatory_matches = 0

    mandatory_conflicts = 0

    preferred_matches = 0

    preferred_conflicts = 0

    unspecified_active = 0

    matched_requirements = 0

    specified_characteristics = set()

    # --------------------------------------------------------
    # REQUIREMENT MATCHING
    # --------------------------------------------------------

    for _, req in requirements.iterrows():

        characteristic = str(
            req["characteristic"]
        ).strip()

        required_value = normalize(
            req["internal_value"]
        )

        requirement_type = str(
            req.get(
                "requirement_type",
                "MANDATORY"
            )
        ).strip().upper()

        confidence = float(
            req.get(
                "confidence",
                1.0
            )
        )

        specified_characteristics.add(
            characteristic
        )

        if characteristic not in config_row.index:

            candidate_value = "NOVALUE"

        else:

            candidate_value = normalize(
                config_row[
                    characteristic
                ]
            )

        feature_weight = REFINEMENT_WEIGHTS.get(
            characteristic,
            1.0
        )

        if requirement_type == "MANDATORY":

            base_weight = MANDATORY_WEIGHT

        else:

            base_weight = PREFERRED_WEIGHT

        # Confidence is capped so that extremely large
        # extraction confidence values cannot dominate.
        confidence_factor = min(
            max(confidence, 0.25),
            2.0
        )

        effective_weight = (
            base_weight
            * feature_weight
            * confidence_factor
        )

        # ----------------------------------------------------
        # EXACT MATCH
        # ----------------------------------------------------

        if candidate_value == required_value:

            score += effective_weight

            positive_score += effective_weight

            matched_requirements += 1

            if requirement_type == "MANDATORY":

                mandatory_matches += 1

            else:

                preferred_matches += 1

        # ----------------------------------------------------
        # EXPLICIT CONFLICT
        # ----------------------------------------------------

        else:

            if requirement_type == "MANDATORY":

                penalty = (
                    MANDATORY_CONFLICT_PENALTY
                    * feature_weight
                    * confidence_factor
                )

                score -= penalty

                negative_score += penalty

                mandatory_conflicts += 1

            else:

                penalty = (
                    PREFERRED_CONFLICT_PENALTY
                    * feature_weight
                    * confidence_factor
                )

                score -= penalty

                negative_score += penalty

                preferred_conflicts += 1

    # --------------------------------------------------------
    # UNSPECIFIED OPTIONAL FEATURE PENALTY
    # --------------------------------------------------------

    for characteristic in CHARACTERISTICS:

        if characteristic in specified_characteristics:

            continue

        candidate_value = normalize(
            config_row.get(
                characteristic,
                "NOVALUE"
            )
        )

        if candidate_value != "NOVALUE":

            # Only apply a very small penalty.
            # This prevents unnecessary suppression of
            # legitimate configurations that have additional
            # features not explicitly mentioned in the RFQ.
            feature_weight = REFINEMENT_WEIGHTS.get(
                characteristic,
                1.0
            )

            penalty = (
                UNSPECIFIED_OPTIONAL_PENALTY
                * feature_weight
            )

            score -= penalty

            negative_score += penalty

            unspecified_active += 1

    # --------------------------------------------------------
    # PACKAGE BONUS
    # --------------------------------------------------------

    package_bonus = 0.0

    if (
        package_context is not None
        and package_context != ""
        and package_context != "UNKNOWN"
    ):

        candidate_package = (
            infer_package_from_config(
                config_row
            )
        )

        if candidate_package == package_context:

            package_bonus = 1.5

            score += package_bonus

            positive_score += package_bonus

        elif candidate_package != "UNKNOWN":

            score -= 0.75

            negative_score += 0.75

    return {

        "score": score,

        "positive_score":
            positive_score,

        "negative_score":
            negative_score,

        "mandatory_matches":
            mandatory_matches,

        "mandatory_conflicts":
            mandatory_conflicts,

        "preferred_matches":
            preferred_matches,

        "preferred_conflicts":
            preferred_conflicts,

        "unspecified_active":
            unspecified_active,

        "matched_requirements":
            matched_requirements,
    }


# ============================================================
# REFINEMENT DISTANCE
# ============================================================

def configuration_distance(
    config_a,
    config_b
):

    if config_a is None or config_b is None:

        return len(CHARACTERISTICS)

    distance = 0.0

    for characteristic in CHARACTERISTICS:

        value_a = normalize(
            config_a.get(
                characteristic,
                "NOVALUE"
            )
        )

        value_b = normalize(
            config_b.get(
                characteristic,
                "NOVALUE"
            )
        )

        if value_a != value_b:

            weight = REFINEMENT_WEIGHTS.get(
                characteristic,
                1.0
            )

            distance += weight

    return distance


# ============================================================
# CANDIDATE REFINEMENT
# ============================================================

def refine_candidates(
    scored_candidates,
    configurations_lookup,
    requirements,
    top_n=25,
):

    if len(scored_candidates) == 0:

        return scored_candidates

    candidates = scored_candidates.copy()

    candidates = (
        candidates
        .sort_values(
            "score",
            ascending=False
        )
        .head(top_n)
        .copy()
    )

    # --------------------------------------------------------
    # Determine a requirement-compatible reference profile.
    #
    # This profile is NOT an invented configuration.
    # It simply contains the values explicitly required by
    # the RFQ.
    # --------------------------------------------------------

    requested_values = {}

    for _, req in requirements.iterrows():

        characteristic = str(
            req["characteristic"]
        ).strip()

        required_value = normalize(
            req["internal_value"]
        )

        requested_values[
            characteristic
        ] = required_value

    # --------------------------------------------------------
    # Refine each candidate using weighted distance from
    # explicitly requested characteristics.
    # --------------------------------------------------------

    refinement_scores = []

    for _, candidate in candidates.iterrows():

        config_id = str(
            candidate[
                "canonical_configuration_id"
            ]
        )

        config = configurations_lookup.get(
            config_id
        )

        distance = 0.0

        exact_matches = 0

        conflicts = 0

        for characteristic, required_value in (
            requested_values.items()
        ):

            if config is None:

                candidate_value = "NOVALUE"

            else:

                candidate_value = normalize(
                    config.get(
                        characteristic,
                        "NOVALUE"
                    )
                )

            weight = REFINEMENT_WEIGHTS.get(
                characteristic,
                1.0
            )

            if candidate_value == required_value:

                exact_matches += 1

            else:

                conflicts += 1

                distance += weight

        # Refinement is deliberately small compared with
        # the primary evidence score.
        refinement_bonus = (
            exact_matches * 0.50
        )

        refinement_penalty = (
            distance * 0.35
        )

        final_score = (
            candidate["score"]
            + refinement_bonus
            - refinement_penalty
        )

        refinement_scores.append({

            "canonical_configuration_id":
                config_id,

            "refinement_distance":
                distance,

            "refinement_exact_matches":
                exact_matches,

            "refinement_conflicts":
                conflicts,

            "refinement_bonus":
                refinement_bonus,

            "refinement_penalty":
                refinement_penalty,

            "final_score":
                final_score,
        })

    refinement = pd.DataFrame(
        refinement_scores
    )

    candidates = candidates.merge(
        refinement,
        on="canonical_configuration_id",
        how="left"
    )

    candidates = (
        candidates
        .sort_values(
            [
                "final_score",
                "mandatory_matches",
                "matched_requirements",
                "positive_score",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    return candidates


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("731 RFQ CONFIGURATION RECOMMENDATION — V4")
    print("=" * 100)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # ========================================================
    # LOAD V8
    # ========================================================

    print()
    print("=" * 100)
    print("LOADING V8 EXTRACTED REQUIREMENTS")
    print("=" * 100)

    v8 = load_csv(
        V8_FILE
    )

    v8 = prepare_predictions(
        v8
    )

    print(
        "V8 predictions:",
        len(v8)
    )

    print(
        "RFQs:",
        v8["rfq_id"].nunique()
    )

    # ========================================================
    # LOAD CONFIGURATIONS
    # ========================================================

    print()
    print("=" * 100)
    print("LOADING V5 TECHNICAL CONFIGURATIONS")
    print("=" * 100)

    configurations = load_csv(
        CONFIG_FILE
    )

    configurations = (
        prepare_configuration_table(
            configurations
        )
    )

    print(
        "Configurations:",
        len(configurations)
    )

    print(
        "Unique configuration IDs:",
        configurations[
            "canonical_configuration_id"
        ].nunique()
    )

    config_lookup = build_config_lookup(
        configurations
    )

    # ========================================================
    # LOAD GROUND TRUTH
    # ========================================================

    print()
    print("=" * 100)
    print("LOADING RFQ GROUND TRUTH")
    print("=" * 100)

    ground_truth = load_csv(
        GROUND_TRUTH_FILE
    )

    ground_truth_lookup = (
        build_ground_truth_lookup(
            ground_truth
        )
    )

    print(
        "Ground-truth RFQs:",
        len(ground_truth_lookup)
    )

    # ========================================================
    # LOAD STRUCTURED REQUIREMENTS
    # ========================================================

    print()
    print("=" * 100)
    print("LOADING STRUCTURED REQUIREMENTS")
    print("=" * 100)

    requirements = load_csv(
        REQUIREMENTS_FILE
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

    requirements["internal_value"] = (
        requirements["internal_value"]
        .apply(normalize)
    )

    requirements["requirement_type"] = (
        requirements[
            "requirement_type"
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    requirement_lookup = (
        build_requirement_lookup(
            requirements
        )
    )

    extracted_lookup = (
        build_extracted_requirement_lookup(
            v8
        )
    )

    print(
        "Ground-truth requirements:",
        len(requirements)
    )

    # ========================================================
    # MODES
    # ========================================================

    modes = [
        "V8_EXTRACTED",
        "ORACLE",
    ]

    results = []

    recommendation_rows = []

    topk_rows = []

    error_rows = []

    scoring_rows = []

    characteristic_rows = []

    # ========================================================
    # EVALUATE EACH MODE
    # ========================================================

    for mode in modes:

        for package_aware in [
            False,
            True,
        ]:

            print()
            print("=" * 100)
            print(
                f"MODE: {mode} | "
                f"package_aware={package_aware}"
            )
            print("=" * 100)

            mode_recommendations = []

            rfq_ids = sorted(
                ground_truth_lookup.keys()
            )

            total = len(rfq_ids)

            for counter, rfq_id in enumerate(
                rfq_ids,
                start=1
            ):

                if counter % 500 == 0:

                    print(
                        f"Processed {counter:,} / "
                        f"{total:,}"
                    )

                truth = (
                    ground_truth_lookup[
                        rfq_id
                    ]
                )

                true_config_ids = (
                    truth[
                        "configuration_ids"
                    ]
                )

                target_packages = (
                    truth[
                        "packages"
                    ]
                )

                target_package = (
                    sorted(
                        target_packages
                    )[0]
                    if len(target_packages)
                    else "UNKNOWN"
                )

                # ------------------------------------------------
                # SELECT REQUIREMENTS
                # ------------------------------------------------

                if mode == "ORACLE":

                    rfq_requirements = (
                        requirement_lookup.get(
                            rfq_id,
                            pd.DataFrame()
                        )
                    ).copy()

                    # Oracle requirements should have full
                    # requirement type information.
                    rfq_requirements[
                        "confidence"
                    ] = 1.0

                else:

                    rfq_requirements = (
                        extracted_lookup.get(
                            rfq_id,
                            pd.DataFrame()
                        )
                    ).copy()

                    if len(
                        rfq_requirements
                    ) > 0:

                        # V8 predictions do not necessarily
                        # contain requirement_type.
                        #
                        # Use ground truth requirement type
                        # only as a weighting reference for
                        # evaluation of the extracted signal.
                        #
                        # If a prediction has no corresponding
                        # ground-truth characteristic/value,
                        # treat it as preferred rather than
                        # silently making it mandatory.

                        type_lookup = {}

                        oracle_req = (
                            requirement_lookup.get(
                                rfq_id,
                                pd.DataFrame()
                            )
                        )

                        for _, req in (
                            oracle_req.iterrows()
                        ):

                            key = (
                                str(
                                    req[
                                        "characteristic"
                                    ]
                                ),
                                normalize(
                                    req[
                                        "internal_value"
                                    ]
                                ),
                            )

                            type_lookup[key] = (
                                str(
                                    req[
                                        "requirement_type"
                                    ]
                                )
                                .upper()
                            )

                        rfq_requirements[
                            "requirement_type"
                        ] = rfq_requirements.apply(
                            lambda row:
                            type_lookup.get(
                                (
                                    str(
                                        row[
                                            "characteristic"
                                        ]
                                    ),
                                    normalize(
                                        row[
                                            "internal_value"
                                        ]
                                    ),
                                ),
                                "PREFERRED",
                            ),
                            axis=1
                        )

                # ------------------------------------------------
                # EMPTY RFQ
                # ------------------------------------------------

                if len(
                    rfq_requirements
                ) == 0:

                    continue

                # ------------------------------------------------
                # CANDIDATE SCORING
                # ------------------------------------------------

                candidates = []

                for _, config in (
                    configurations.iterrows()
                ):

                    config_id = str(
                        config[
                            "canonical_configuration_id"
                        ]
                    )

                    candidate_package = (
                        infer_package_from_config(
                            config
                        )
                    )

                    # Package-aware mode:
                    #
                    # Do NOT completely eliminate other
                    # packages. Instead apply package context
                    # as a ranking signal.
                    #
                    # This prevents candidate starvation.
                    score_package = (
                        target_package
                        if package_aware
                        else None
                    )

                    score_info = (
                        score_configuration(
                            config,
                            rfq_requirements,
                            score_package,
                        )
                    )

                    candidates.append({

                        "canonical_configuration_id":
                            config_id,

                        "package_context":
                            candidate_package,

                        **score_info,
                    })

                scored = pd.DataFrame(
                    candidates
                )

                # ------------------------------------------------
                # REFINEMENT
                # ------------------------------------------------

                refined = refine_candidates(
                    scored,
                    config_lookup,
                    rfq_requirements,
                    top_n=50,
                )

                if len(refined) == 0:

                    continue

                # ------------------------------------------------
                # FINAL RANK
                # ------------------------------------------------

                refined = (
                    refined
                    .sort_values(
                        [
                            "final_score",
                            "mandatory_matches",
                            "matched_requirements",
                            "positive_score",
                        ],
                        ascending=[
                            False,
                            False,
                            False,
                            False,
                        ],
                    )
                    .reset_index(
                        drop=True
                    )
                )

                refined[
                    "rank"
                ] = (
                    np.arange(
                        len(refined)
                    )
                    + 1
                )

                # ------------------------------------------------
                # TRUE RANK
                # ------------------------------------------------

                true_rows = refined[
                    refined[
                        "canonical_configuration_id"
                    ].isin(
                        true_config_ids
                    )
                ]

                if len(true_rows) > 0:

                    true_rank = int(
                        true_rows[
                            "rank"
                        ].min()
                    )

                else:

                    true_rank = np.nan

                recommended = refined.iloc[0]

                recommended_id = str(
                    recommended[
                        "canonical_configuration_id"
                    ]
                )

                is_top1 = (
                    recommended_id
                    in true_config_ids
                )

                # ------------------------------------------------
                # TOP-K
                # ------------------------------------------------

                top_k = refined.head(10)

                for _, candidate in (
                    top_k.iterrows()
                ):

                    candidate_id = str(
                        candidate[
                            "canonical_configuration_id"
                        ]
                    )

                    topk_rows.append({

                        "rfq_id":
                            rfq_id,

                        "mode":
                            mode,

                        "package_aware":
                            package_aware,

                        "rank":
                            int(
                                candidate[
                                    "rank"
                                ]
                            ),

                        "canonical_configuration_id":
                            candidate_id,

                        "package_context":
                            candidate[
                                "package_context"
                            ],

                        "score":
                            candidate[
                                "score"
                            ],

                        "final_score":
                            candidate[
                                "final_score"
                            ],

                        "positive_score":
                            candidate[
                                "positive_score"
                            ],

                        "negative_score":
                            candidate[
                                "negative_score"
                            ],

                        "mandatory_matches":
                            candidate[
                                "mandatory_matches"
                            ],

                        "mandatory_conflicts":
                            candidate[
                                "mandatory_conflicts"
                            ],

                        "preferred_matches":
                            candidate[
                                "preferred_matches"
                            ],

                        "preferred_conflicts":
                            candidate[
                                "preferred_conflicts"
                            ],

                        "refinement_distance":
                            candidate[
                                "refinement_distance"
                            ],

                        "refinement_exact_matches":
                            candidate[
                                "refinement_exact_matches"
                            ],

                        "matched_requirements":
                            candidate[
                                "matched_requirements"
                            ],
                    })

                # ------------------------------------------------
                # RECOMMENDATION RECORD
                # ------------------------------------------------

                recommendation_rows.append({

                    "rfq_id":
                        rfq_id,

                    "mode":
                        mode,

                    "package_aware":
                        package_aware,

                    "target_package":
                        target_package,

                    "true_configuration_id":
                        sorted(
                            true_config_ids
                        )[0]
                        if len(
                            true_config_ids
                        )
                        else "",

                    "recommended_configuration_id":
                        recommended_id,

                    "recommended_package":
                        recommended[
                            "package_context"
                        ],

                    "top_score":
                        recommended[
                            "final_score"
                        ],

                    "true_rank":
                        true_rank,

                    "candidate_count":
                        len(refined),

                    "requirement_count":
                        len(
                            rfq_requirements
                        ),

                    "mandatory_matches":
                        recommended[
                            "mandatory_matches"
                        ],

                    "mandatory_conflicts":
                        recommended[
                            "mandatory_conflicts"
                        ],

                    "preferred_matches":
                        recommended[
                            "preferred_matches"
                        ],

                    "preferred_conflicts":
                        recommended[
                            "preferred_conflicts"
                        ],

                    "refinement_distance":
                        recommended[
                            "refinement_distance"
                        ],

                    "is_top1":
                        is_top1,
                })

                # ------------------------------------------------
                # ERROR RECORD
                # ------------------------------------------------

                if is_top1:

                    error_type = (
                        "CORRECT_TOP1"
                    )

                elif (
                    not pd.isna(true_rank)
                    and true_rank <= 3
                ):

                    error_type = (
                        "TRUE_IN_TOP3"
                    )

                elif (
                    not pd.isna(true_rank)
                    and true_rank <= 5
                ):

                    error_type = (
                        "TRUE_IN_TOP5"
                    )

                elif (
                    not pd.isna(true_rank)
                    and true_rank <= 10
                ):

                    error_type = (
                        "TRUE_IN_TOP10"
                    )

                elif not pd.isna(true_rank):

                    error_type = (
                        "TRUE_BEYOND_TOP10"
                    )

                else:

                    error_type = (
                        "TRUE_UNRANKED"
                    )

                error_rows.append({

                    "rfq_id":
                        rfq_id,

                    "mode":
                        mode,

                    "package_aware":
                        package_aware,

                    "target_package":
                        target_package,

                    "recommended_configuration_id":
                        recommended_id,

                    "true_configuration_id":
                        sorted(
                            true_config_ids
                        )[0]
                        if len(
                            true_config_ids
                        )
                        else "",

                    "true_rank":
                        true_rank,

                    "error_type":
                        error_type,

                    "top_score":
                        recommended[
                            "final_score"
                        ],

                    "mandatory_matches":
                        recommended[
                            "mandatory_matches"
                        ],

                    "mandatory_conflicts":
                        recommended[
                            "mandatory_conflicts"
                        ],

                    "preferred_matches":
                        recommended[
                            "preferred_matches"
                        ],

                    "preferred_conflicts":
                        recommended[
                            "preferred_conflicts"
                        ],

                    "refinement_distance":
                        recommended[
                            "refinement_distance"
                        ],
                })

                # ------------------------------------------------
                # SCORING STATISTICS
                # ------------------------------------------------

                scoring_rows.append({

                    "rfq_id":
                        rfq_id,

                    "mode":
                        mode,

                    "package_aware":
                        package_aware,

                    "candidate_count":
                        len(refined),

                    "requirement_count":
                        len(
                            rfq_requirements
                        ),

                    "top_score":
                        recommended[
                            "final_score"
                        ],

                    "positive_score":
                        recommended[
                            "positive_score"
                        ],

                    "negative_score":
                        recommended[
                            "negative_score"
                        ],

                    "mandatory_matches":
                        recommended[
                            "mandatory_matches"
                        ],

                    "mandatory_conflicts":
                        recommended[
                            "mandatory_conflicts"
                        ],

                    "preferred_matches":
                        recommended[
                            "preferred_matches"
                        ],

                    "preferred_conflicts":
                        recommended[
                            "preferred_conflicts"
                        ],

                    "refinement_distance":
                        recommended[
                            "refinement_distance"
                        ],

                    "refinement_exact_matches":
                        recommended[
                            "refinement_exact_matches"
                        ],
                })

                # ------------------------------------------------
                # CHARACTERISTIC-LEVEL TOP-1
                # ------------------------------------------------

                recommended_config = (
                    config_lookup.get(
                        recommended_id
                    )
                )

                # Select first true configuration.
                true_config_id = (
                    sorted(
                        true_config_ids
                    )[0]
                    if len(
                        true_config_ids
                    )
                    else None
                )

                true_config = (
                    config_lookup.get(
                        true_config_id
                    )
                    if true_config_id
                    else None
                )

                for characteristic in (
                    CHARACTERISTICS
                ):

                    predicted_value = normalize(
                        recommended_config.get(
                            characteristic,
                            "NOVALUE"
                        )
                        if recommended_config
                        is not None
                        else "NOVALUE"
                    )

                    true_value = normalize(
                        true_config.get(
                            characteristic,
                            "NOVALUE"
                        )
                        if true_config
                        is not None
                        else "NOVALUE"
                    )

                    characteristic_rows.append({

                        "rfq_id":
                            rfq_id,

                        "mode":
                            mode,

                        "package_aware":
                            package_aware,

                        "characteristic":
                            characteristic,

                        "true_value":
                            true_value,

                        "predicted_value":
                            predicted_value,

                        "correct":
                            int(
                                predicted_value
                                == true_value
                            ),

                        "is_top1":
                            int(
                                is_top1
                            ),
                    })

            # ====================================================
            # MODE / PACKAGE METRICS
            # ====================================================

            mode_df = pd.DataFrame(
                [
                    row
                    for row in recommendation_rows
                    if (
                        row["mode"] == mode
                        and row[
                            "package_aware"
                        ]
                        == package_aware
                    )
                ]
            )

            if len(mode_df) == 0:

                continue

            true_ranks = pd.to_numeric(
                mode_df[
                    "true_rank"
                ],
                errors="coerce"
            )

            top1 = (
                true_ranks
                .notna()
                &
                (true_ranks <= 1)
            ).mean()

            top3 = (
                true_ranks
                .notna()
                &
                (true_ranks <= 3)
            ).mean()

            top5 = (
                true_ranks
                .notna()
                &
                (true_ranks <= 5)
            ).mean()

            top10 = (
                true_ranks
                .notna()
                &
                (true_ranks <= 10)
            ).mean()

            valid_ranks = (
                true_ranks[
                    true_ranks.notna()
                ]
            )

            if len(valid_ranks) > 0:

                mrr = (
                    1.0 / valid_ranks
                ).mean()

                mean_rank = (
                    valid_ranks.mean()
                )

                median_rank = (
                    valid_ranks.median()
                )

            else:

                mrr = 0.0

                mean_rank = np.nan

                median_rank = np.nan

            unranked = (
                true_ranks.isna()
                .sum()
            )

            results.append({

                "mode":
                    mode,

                "package_aware":
                    package_aware,

                "rfqs_evaluated":
                    len(mode_df),

                "top1_accuracy":
                    top1,

                "top3_accuracy":
                    top3,

                "top5_accuracy":
                    top5,

                "top10_accuracy":
                    top10,

                "mrr":
                    mrr,

                "mean_true_rank":
                    mean_rank,

                "median_true_rank":
                    median_rank,

                "unranked":
                    unranked,
            })

    # ========================================================
    # DATAFRAMES
    # ========================================================

    recommendations_df = pd.DataFrame(
        recommendation_rows
    )

    topk_df = pd.DataFrame(
        topk_rows
    )

    errors_df = pd.DataFrame(
        error_rows
    )

    scoring_df = pd.DataFrame(
        scoring_rows
    )

    characteristic_df = pd.DataFrame(
        characteristic_rows
    )

    metrics_df = pd.DataFrame(
        results
    )

    # ========================================================
    # PACKAGE METRICS
    # ========================================================

    package_rows = []

    for (
        mode,
        package_aware
    ), group in (
        recommendations_df
        .groupby(
            [
                "mode",
                "package_aware",
            ]
        )
    ):

        group = group.copy()

        group[
            "true_rank"
        ] = pd.to_numeric(
            group[
                "true_rank"
            ],
            errors="coerce"
        )

        for package, package_group in (
            group.groupby(
                "target_package"
            )
        ):

            ranks = package_group[
                "true_rank"
            ]

            valid = ranks[
                ranks.notna()
            ]

            package_rows.append({

                "mode":
                    mode,

                "package_aware":
                    package_aware,

                "target_package":
                    package,

                "rfqs":
                    len(
                        package_group
                    ),

                "top1_accuracy":
                    (
                        (
                            ranks == 1
                        ).mean()
                    ),

                "top3_accuracy":
                    (
                        (
                            ranks <= 3
                        ).mean()
                        if len(valid)
                        else 0
                    ),

                "top5_accuracy":
                    (
                        (
                            ranks <= 5
                        ).mean()
                        if len(valid)
                        else 0
                    ),

                "top10_accuracy":
                    (
                        (
                            ranks <= 10
                        ).mean()
                        if len(valid)
                        else 0
                    ),

                "mean_true_rank":
                    (
                        valid.mean()
                        if len(valid)
                        else np.nan
                    ),

                "median_true_rank":
                    (
                        valid.median()
                        if len(valid)
                        else np.nan
                    ),
            })

    package_df = pd.DataFrame(
        package_rows
    )

    # ========================================================
    # CHARACTERISTIC METRICS
    # ========================================================

    characteristic_metrics = []

    for (
        mode,
        package_aware,
        characteristic
    ), group in (
        characteristic_df
        .groupby(
            [
                "mode",
                "package_aware",
                "characteristic",
            ]
        )
    ):

        correct = (
            group["correct"]
            .sum()
        )

        total = len(group)

        accuracy = (
            correct / total
            if total
            else 0
        )

        top1_group = group[
            group["is_top1"] == 1
        ]

        top1_accuracy = (
            top1_group[
                "correct"
            ].mean()
            if len(top1_group)
            else 0
        )

        characteristic_metrics.append({

            "mode":
                mode,

            "package_aware":
                package_aware,

            "characteristic":
                characteristic,

            "records":
                total,

            "value_accuracy":
                accuracy,

            "top1_case_value_accuracy":
                top1_accuracy,
        })

    characteristic_metrics_df = (
        pd.DataFrame(
            characteristic_metrics
        )
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 100)
    print("731 RFQ CONFIGURATION RECOMMENDATION V4 RESULTS")
    print("=" * 100)

    for _, metric in metrics_df.iterrows():

        print()
        print(
            f"{metric['mode']} | "
            f"package_aware="
            f"{bool(metric['package_aware'])}"
        )

        print(
            "RFQs evaluated:",
            f"{int(metric['rfqs_evaluated']):,}"
        )

        print(
            "Top-1 accuracy:",
            f"{metric['top1_accuracy']:.4f}"
        )

        print(
            "Top-3 accuracy:",
            f"{metric['top3_accuracy']:.4f}"
        )

        print(
            "Top-5 accuracy:",
            f"{metric['top5_accuracy']:.4f}"
        )

        print(
            "Top-10 accuracy:",
            f"{metric['top10_accuracy']:.4f}"
        )

        print(
            "MRR:",
            f"{metric['mrr']:.4f}"
        )

        print(
            "Mean true rank:",
            f"{metric['mean_true_rank']:.2f}"
        )

        print(
            "Median true rank:",
            f"{metric['median_true_rank']:.2f}"
        )

        print(
            "Unranked:",
            int(
                metric["unranked"]
            )
        )

    # ========================================================
    # PRINT PACKAGE METRICS
    # ========================================================

    print()
    print("=" * 100)
    print("V4 PACKAGE PERFORMANCE")
    print("=" * 100)

    if len(package_df) > 0:

        print(
            package_df[
                [
                    "mode",
                    "package_aware",
                    "target_package",
                    "rfqs",
                    "top1_accuracy",
                    "top3_accuracy",
                    "top5_accuracy",
                    "top10_accuracy",
                    "mean_true_rank",
                ]
            ]
            .to_string(
                index=False
            )
        )

    # ========================================================
    # PRINT CHARACTERISTIC RESULTS
    # ========================================================

    print()
    print("=" * 100)
    print("V4 CHARACTERISTIC PERFORMANCE")
    print("=" * 100)

    extracted_characteristics = (
        characteristic_metrics_df[
            characteristic_metrics_df[
                "mode"
            ]
            == "V8_EXTRACTED"
        ]
    )

    if len(
        extracted_characteristics
    ) > 0:

        print(
            extracted_characteristics[
                [
                    "characteristic",
                    "value_accuracy",
                    "top1_case_value_accuracy",
                ]
            ]
            .sort_values(
                "value_accuracy"
            )
            .to_string(
                index=False
            )
        )

    # ========================================================
    # SAVE FILES
    # ========================================================

    recommendations_df.to_csv(
        RECOMMENDATION_FILE,
        index=False
    )

    metrics_df.to_csv(
        METRICS_FILE,
        index=False
    )

    topk_df.to_csv(
        TOPK_FILE,
        index=False
    )

    errors_df.to_csv(
        ERROR_FILE,
        index=False
    )

    package_df.to_csv(
        PACKAGE_FILE,
        index=False
    )

    scoring_df.to_csv(
        SCORING_FILE,
        index=False
    )

    characteristic_metrics_df.to_csv(
        CHARACTERISTIC_FILE,
        index=False
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 100)
    print("V4 FILES SAVED")
    print("=" * 100)

    print()
    print(
        "Recommendations:"
    )
    print(
        RECOMMENDATION_FILE
    )

    print()
    print(
        "Metrics:"
    )
    print(
        METRICS_FILE
    )

    print()
    print(
        "Top-K rankings:"
    )
    print(
        TOPK_FILE
    )

    print()
    print(
        "Recommendation errors:"
    )
    print(
        ERROR_FILE
    )

    print()
    print(
        "Package metrics:"
    )
    print(
        PACKAGE_FILE
    )

    print()
    print(
        "Scoring statistics:"
    )
    print(
        SCORING_FILE
    )

    print()
    print(
        "Characteristic metrics:"
    )
    print(
        CHARACTERISTIC_FILE
    )

    print()
    print("=" * 100)
    print(
        "731 RFQ CONFIGURATION RECOMMENDATION V4 COMPLETE"
    )
    print("=" * 100)


if __name__ == "__main__":

    main()