#!/usr/bin/env python3
"""
731 CPQ ML RANKER
=================

Application inference layer for the 731 configurator.

Architecture
------------

    Deterministic CPQ / constraint engine
                    |
                    v
          technically valid candidates
                    |
                    v
             V7 feature adapter
                    |
                    v
       V7 HGB inference pipeline
                    |
                    v
              ranked candidates

Important
---------

This module DOES NOT determine technical validity.

The caller must provide candidates that have already passed the
deterministic 731 configuration / constraint workflow.

ML only ranks those candidates.

This module also does NOT:

- generate Emerson model codes
- infer model-code positions
- modify candidate eligibility
- use engineer ground truth
- use ML target
- use engineer scores/ranks/utilities
- retrain the model

The serialized model is:

    data/processed/ml/v7_10_algorithms/
    731_v7_HIST_GRADIENT_BOOSTING_REALISTIC_BASELINE.joblib
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT
    / "data"
    / "processed"
    / "ml"
    / "v7_10_algorithms"
    / "731_v7_HIST_GRADIENT_BOOSTING_REALISTIC_BASELINE.joblib"
)

FEATURE_METADATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "ml"
    / "v7_10_algorithms"
    / "731_v7_HIST_GRADIENT_BOOSTING_REALISTIC_BASELINE_features.json"
)

V41_DATASET = (
    ROOT
    / "data"
    / "processed"
    / "ml"
    / "731_v4_1_ml_candidate_dataset.csv"
)


# =============================================================================
# EXACT V7 REALISTIC-BASELINE FEATURES
# =============================================================================

V7_FEATURES = [
    "mandatory_total",
    "mandatory_satisfied",
    "preferred_total",
    "preferred_satisfied",

    "config_devCategory",
    "config_characteristic",
    "config_powerSupply",
    "config_numberOfChannels",
    "config_explosionApproval",
    "config_protectionArea",
    "config_certification",
    "config_dataInterface",
    "config_stromSchaltbar",
    "config_stromEingaenge",
    "config_temperaturEingaenge",
    "config_binaerDigitalOpenColl_MN",
    "config_binaerOpenColl_MP",
    "config_waveInjector",
    "config_dev_advMeterVerification",
    "config_dynamicGasMaster",
    "config_customUserFluid",
    "config_steamApplication",
]


TECHNICAL_FIELDS = [
    "devCategory",
    "characteristic",
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


# =============================================================================
# SAFETY / GOVERNANCE
# =============================================================================

FORBIDDEN_COLUMNS = {
    "ml_target",
    "engineer_rank",
    "engineer_score",
    "engineer_utility",
    "synthetic_engineer_rank",
    "synthetic_engineer_score",
    "synthetic_engineer_utility",
    "ground_truth_configuration_id",
    "ground_truth_canonical_configuration_id",
    "is_ground_truth_configuration",
    "tie_break_jitter",
}


def norm(value) -> str:
    """
    Normalise a value for deterministic requirement comparison.

    Missing values are represented by a stable sentinel.
    """

    if pd.isna(value):
        return "__MISSING__"

    text = str(value).strip()

    if text == "":
        return "__MISSING__"

    if text.lower() in {
        "nan",
        "none",
        "null",
    }:
        return "__MISSING__"

    return text


# =============================================================================
# MODEL LOADING
# =============================================================================

class CPQMLRanker:
    """
    Load and apply the V7 HGB inference artifact.

    The object is intentionally stateless with respect to candidate validity.
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        feature_metadata_path: Path = FEATURE_METADATA_PATH,
    ):
        self.model_path = Path(model_path)
        self.feature_metadata_path = Path(
            feature_metadata_path
        )

        self._validate_artifacts()

        self.model = joblib.load(
            self.model_path
        )

        with open(
            self.feature_metadata_path,
            "r",
            encoding="utf-8",
        ) as handle:
            self.metadata = json.load(handle)

        self.feature_columns = list(
            self.metadata.get(
                "features",
                V7_FEATURES,
            )
        )

        self._validate_feature_contract()

    # -------------------------------------------------------------------------
    # ARTIFACT VALIDATION
    # -------------------------------------------------------------------------

    def _validate_artifacts(self) -> None:

        if not self.model_path.exists():

            raise FileNotFoundError(
                "V7 HGB inference artifact not found:\n"
                f"{self.model_path}"
            )

        if not self.feature_metadata_path.exists():

            raise FileNotFoundError(
                "V7 HGB feature metadata not found:\n"
                f"{self.feature_metadata_path}"
            )

    def _validate_feature_contract(self) -> None:

        if self.feature_columns != V7_FEATURES:

            raise ValueError(
                "V7 inference feature contract mismatch.\n\n"
                f"Expected:\n{V7_FEATURES}\n\n"
                f"Artifact contains:\n{self.feature_columns}"
            )

        if not hasattr(
            self.model,
            "predict_proba",
        ):

            raise TypeError(
                "Serialized V7 model does not expose "
                "predict_proba()."
            )

    # -------------------------------------------------------------------------
    # CANDIDATE NORMALISATION
    # -------------------------------------------------------------------------

    @staticmethod
    def _ensure_config_columns(
        candidates: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert deterministic CPQ candidate fields into the exact
        config_* representation used by V7.

        Both representations are accepted:

            devCategory
            config_devCategory

        The function never overwrites an existing config_* field.
        """

        out = candidates.copy()

        for field in TECHNICAL_FIELDS:

            config_column = (
                f"config_{field}"
            )

            if (
                config_column not in out.columns
                and field in out.columns
            ):

                out[config_column] = (
                    out[field]
                )

        return out

    # -------------------------------------------------------------------------
    # REQUIREMENT FEATURES
    # -------------------------------------------------------------------------

    @staticmethod
    def calculate_requirement_features(
        candidates: pd.DataFrame,
        requirements: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """
        Calculate the four requirement features actually used by the
        V7 REALISTIC_BASELINE inference model.

        Expected requirement columns:

            characteristic
            value
            type

        type must be:

            MANDATORY
            PREFERRED

        Logic matches the existing configurator feature construction:
        each requirement is compared against the corresponding candidate
        config_* field.
        """

        out = candidates.copy()

        if requirements is None:

            requirements = pd.DataFrame(
                columns=[
                    "characteristic",
                    "value",
                    "type",
                ]
            )

        if requirements.empty:

            out["mandatory_total"] = 0
            out["mandatory_satisfied"] = 0
            out["preferred_total"] = 0
            out["preferred_satisfied"] = 0

            return out

        required_columns = {
            "characteristic",
            "value",
            "type",
        }

        missing = (
            required_columns
            - set(requirements.columns)
        )

        if missing:

            raise ValueError(
                "Requirements are missing columns: "
                f"{sorted(missing)}"
            )

        req = requirements.copy()

        req["type"] = (
            req["type"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        req["characteristic"] = (
            req["characteristic"]
            .astype(str)
            .str.strip()
        )

        mandatory = req[
            req["type"]
            == "MANDATORY"
        ]

        preferred = req[
            req["type"]
            == "PREFERRED"
        ]

        mandatory_total = len(
            mandatory
        )

        preferred_total = len(
            preferred
        )

        mandatory_satisfied = np.zeros(
            len(out),
            dtype=int,
        )

        preferred_satisfied = np.zeros(
            len(out),
            dtype=int,
        )

        for _, requirement in (
            mandatory.iterrows()
        ):

            characteristic = str(
                requirement[
                    "characteristic"
                ]
            ).strip()

            column = (
                f"config_{characteristic}"
            )

            if column not in out.columns:

                raise ValueError(
                    "Requirement references a "
                    "characteristic that is not "
                    f"available in candidates: "
                    f"{characteristic}"
                )

            requested = norm(
                requirement["value"]
            )

            matches = (
                out[column]
                .map(norm)
                .eq(requested)
                .to_numpy(
                    dtype=int
                )
            )

            mandatory_satisfied += (
                matches
            )

        for _, requirement in (
            preferred.iterrows()
        ):

            characteristic = str(
                requirement[
                    "characteristic"
                ]
            ).strip()

            column = (
                f"config_{characteristic}"
            )

            if column not in out.columns:

                raise ValueError(
                    "Requirement references a "
                    "characteristic that is not "
                    f"available in candidates: "
                    f"{characteristic}"
                )

            requested = norm(
                requirement["value"]
            )

            matches = (
                out[column]
                .map(norm)
                .eq(requested)
                .to_numpy(
                    dtype=int
                )
            )

            preferred_satisfied += (
                matches
            )

        out["mandatory_total"] = (
            mandatory_total
        )

        out["mandatory_satisfied"] = (
            mandatory_satisfied
        )

        out["preferred_total"] = (
            preferred_total
        )

        out["preferred_satisfied"] = (
            preferred_satisfied
        )

        return out

    # -------------------------------------------------------------------------
    # VALIDITY GUARD
    # -------------------------------------------------------------------------

    @staticmethod
    def _check_candidate_frame(
        candidates: pd.DataFrame,
    ) -> None:
        """
        Safety check.

        This function deliberately does not validate technical compatibility.
        That belongs to the deterministic CPQ engine.

        It only checks that the caller supplied enough information for
        ranking.
        """

        if candidates.empty:

            raise ValueError(
                "No candidates were supplied to the ML ranker."
            )

        if (
            "canonical_configuration_id"
            not in candidates.columns
        ):

            raise ValueError(
                "Candidates must contain "
                "'canonical_configuration_id'."
            )

        duplicate_ids = (
            candidates[
                "canonical_configuration_id"
            ]
            .astype(str)
            .duplicated()
        )

        if duplicate_ids.any():

            duplicates = (
                candidates.loc[
                    duplicate_ids,
                    "canonical_configuration_id",
                ]
                .astype(str)
                .tolist()
            )

            raise ValueError(
                "Duplicate candidate IDs supplied "
                "to the ranker: "
                f"{duplicates[:10]}"
            )

        forbidden_found = (
            set(candidates.columns)
            & FORBIDDEN_COLUMNS
        )

        if forbidden_found:

            raise ValueError(
                "Forbidden ML fields were supplied "
                "to the inference layer: "
                f"{sorted(forbidden_found)}"
            )

    # -------------------------------------------------------------------------
    # FEATURE FRAME
    # -------------------------------------------------------------------------

    def build_feature_frame(
        self,
        candidates: pd.DataFrame,
        requirements: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Build the exact 22-column V7 REALISTIC_BASELINE feature frame.
        """

        self._check_candidate_frame(
            candidates
        )

        out = self._ensure_config_columns(
            candidates
        )

        out = (
            self.calculate_requirement_features(
                out,
                requirements,
            )
        )

        missing = [
            column
            for column in self.feature_columns
            if column not in out.columns
        ]

        if missing:

            raise ValueError(
                "Unable to construct the exact V7 "
                "feature frame. Missing columns:\n"
                + "\n".join(
                    f"  - {column}"
                    for column in missing
                )
            )

        features = out[
            self.feature_columns
        ].copy()

        return features

    # -------------------------------------------------------------------------
    # RANK
    # -------------------------------------------------------------------------

    def rank_valid_candidates(
        self,
        candidates: pd.DataFrame,
        requirements: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Rank candidates that have ALREADY passed deterministic CPQ validity.

        Returns the original candidate columns plus:

            mandatory_total
            mandatory_satisfied
            preferred_total
            preferred_satisfied
            ml_score
            rank

        Ranking order:

            1. model score descending
            2. mandatory_satisfied descending
            3. preferred_satisfied descending
            4. canonical_configuration_id ascending

        The latter fields provide deterministic tie handling.

        They do not change technical validity.
        """

        candidates = candidates.copy()

        features = self.build_feature_frame(
            candidates,
            requirements,
        )

        probabilities = (
            self.model.predict_proba(
                features
            )[:, 1]
        )

        if not np.isfinite(
            probabilities
        ).all():

            raise ValueError(
                "The ML model produced non-finite "
                "ranking scores."
            )

        ranked = (
            candidates.copy()
        )

        # Preserve requirement features for UI/audit.
        ranked[
            "mandatory_total"
        ] = features[
            "mandatory_total"
        ].to_numpy()

        ranked[
            "mandatory_satisfied"
        ] = features[
            "mandatory_satisfied"
        ].to_numpy()

        ranked[
            "preferred_total"
        ] = features[
            "preferred_total"
        ].to_numpy()

        ranked[
            "preferred_satisfied"
        ] = features[
            "preferred_satisfied"
        ].to_numpy()

        ranked[
            "ml_score"
        ] = probabilities.astype(float)

        ranked = ranked.sort_values(
            [
                "ml_score",
                "mandatory_satisfied",
                "preferred_satisfied",
                "canonical_configuration_id",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
            kind="mergesort",
        ).reset_index(
            drop=True
        )

        ranked["rank"] = (
            np.arange(
                1,
                len(ranked) + 1,
            )
        )

        # Final integrity check.
        expected_ranks = list(
            range(
                1,
                len(ranked) + 1,
            )
        )

        actual_ranks = (
            ranked["rank"]
            .astype(int)
            .tolist()
        )

        if actual_ranks != expected_ranks:

            raise RuntimeError(
                "Ranking sequence integrity check failed."
            )

        return ranked

    # -------------------------------------------------------------------------
    # RESULT SUMMARY
    # -------------------------------------------------------------------------

    @staticmethod
    def ranking_summary(
        ranked: pd.DataFrame,
    ) -> dict:
        """
        Return a compact audit summary.
        """

        if ranked.empty:

            return {
                "candidate_count": 0,
                "top_configuration_id": None,
                "top_score": None,
                "second_score": None,
                "ranking_gap": None,
            }

        top = ranked.iloc[0]

        second_score = None

        if len(ranked) > 1:

            second_score = float(
                ranked.iloc[1]["ml_score"]
            )

        top_score = float(
            top["ml_score"]
        )

        gap = None

        if second_score is not None:

            gap = (
                top_score
                - second_score
            )

        return {
            "candidate_count": int(
                len(ranked)
            ),
            "top_configuration_id": str(
                top[
                    "canonical_configuration_id"
                ]
            ),
            "top_score": top_score,
            "second_score": second_score,
            "ranking_gap": gap,
        }


# =============================================================================
# SELF TEST
# =============================================================================

def self_test() -> None:

    print("=" * 80)
    print("731 CPQ ML RANKER — SELF TEST")
    print("=" * 80)

    print()

    print("[1] Loading V7 HGB artifact")

    ranker = CPQMLRanker()

    print(
        f"  Model: {ranker.model_path}"
    )

    print(
        f"  Features: "
        f"{len(ranker.feature_columns)}"
    )

    print()

    print("[2] Loading V4.1 candidate data")

    if not V41_DATASET.exists():

        raise FileNotFoundError(
            f"Missing V4.1 dataset: "
            f"{V41_DATASET}"
        )

    v41 = pd.read_csv(
        V41_DATASET,
        low_memory=False,
    )

    print(
        f"  Rows: {len(v41):,}"
    )

    print()

    print(
        "[3] Creating deterministic test subset"
    )

    # Use one RFQ from the existing V4.1 dataset.
    rfq_id = (
        v41["rfq_id"]
        .astype(str)
        .iloc[0]
    )

    candidates = v41[
        v41["rfq_id"]
        .astype(str)
        .eq(rfq_id)
    ].copy()

    print(
        f"  RFQ: {rfq_id}"
    )

    print(
        f"  Candidate rows: "
        f"{len(candidates)}"
    )

    # Remove benchmark-only fields before sending the frame through
    # the application inference interface.
    candidates = candidates.drop(
        columns=[
            column
            for column in FORBIDDEN_COLUMNS
            if column in candidates.columns
        ],
        errors="ignore",
    )

    requirements = pd.DataFrame(
        columns=[
            "characteristic",
            "value",
            "type",
        ]
    )

    print()

    print("[4] Building V7 application features")

    features = (
        ranker.build_feature_frame(
            candidates,
            requirements,
        )
    )

    print(
        f"  Feature rows: "
        f"{len(features)}"
    )

    print(
        f"  Feature columns: "
        f"{len(features.columns)}"
    )

    if list(features.columns) != V7_FEATURES:

        raise AssertionError(
            "Application feature contract "
            "does not match V7."
        )

    print(
        "  Feature contract: PASS"
    )

    print()

    print("[5] Ranking")

    ranked = (
        ranker.rank_valid_candidates(
            candidates,
            requirements,
        )
    )

    summary = (
        ranker.ranking_summary(
            ranked
        )
    )

    print(
        f"  Candidates ranked: "
        f"{summary['candidate_count']}"
    )

    print(
        f"  Top configuration: "
        f"{summary['top_configuration_id']}"
    )

    print(
        f"  Top score: "
        f"{summary['top_score']:.6f}"
    )

    if (
        summary["ranking_gap"]
        is not None
    ):

        print(
            f"  Top-1 vs Top-2 gap: "
            f"{summary['ranking_gap']:.6f}"
        )

    print()

    print("[6] Result integrity")

    if (
        ranked["rank"]
        .astype(int)
        .tolist()
        != list(
            range(
                1,
                len(ranked) + 1,
            )
        )
    ):

        raise AssertionError(
            "Invalid rank sequence."
        )

    if (
        ranked[
            "canonical_configuration_id"
        ]
        .duplicated()
        .any()
    ):

        raise AssertionError(
            "Duplicate configuration IDs "
            "after ranking."
        )

    print(
        "  Rank sequence: PASS"
    )

    print(
        "  Unique configuration IDs: PASS"
    )

    print()

    print(
        "TOP 5"
    )

    display_columns = [
        "rank",
        "canonical_configuration_id",
        "mandatory_satisfied",
        "preferred_satisfied",
        "ml_score",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in ranked.columns
    ]

    print(
        ranked[
            display_columns
        ]
        .head(5)
        .to_string(
            index=False
        )
    )

    print()

    print("=" * 80)
    print(
        "SELF TEST: PASS"
    )
    print("=" * 80)


# =============================================================================
# CLI
# =============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Rank already-valid 731 CPQ "
            "configurations using the V7 HGB model."
        )
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the built-in inference test.",
    )

    args = parser.parse_args()

    if args.self_test:

        self_test()

        return

    parser.print_help()

    print()

    print(
        "No inference was executed."
    )

    print(
        "Use --self-test to verify the "
        "serialized V7 HGB artifact."
    )


if __name__ == "__main__":
    main()
