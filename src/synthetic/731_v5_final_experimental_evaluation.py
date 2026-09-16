#!/usr/bin/env python3
"""
731 V5 — Final Experimental Evaluation & Model Selection
=========================================================

Purpose
-------
Freeze the V4.5 pairwise learning-to-rank experiment and produce a
thesis-ready final comparison of the three feature-governance regimes.

Primary questions
-----------------
1. Which pairwise model/regime gives the strongest RFQ-level ranking?
2. Does provenance add a statistically meaningful improvement over the
   realistic baseline?
3. Does the mechanistic upper bound add meaningful improvement beyond
   provenance?
4. How does performance change with candidate-set difficulty and package?
5. How often is a prediction an exact synthetic-engineer choice versus a
   requirement-equivalent alternative?
6. Are the reported conclusions robust under RFQ-level bootstrap?

Important methodological rule
------------------------------
The labels are synthetic engineer preferences. Results measure recovery of
the synthetic preference-generation mechanism, NOT historical human engineer
behaviour.

Primary deployment interpretation:
    REALISTIC_BASELINE

Sensitivity:
    PROVENANCE_AWARE

Upper bound:
    MECHANISTIC_UPPER_BOUND

This script is evaluation-only. It does not retrain models.
It consumes the frozen V4.5 prediction output and performs final analysis.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
ML = ROOT / "data" / "processed" / "ml"
SYN_RFQ = ROOT / "data" / "synthetic" / "rfqs"

PRED = ML / "731_v4_5_pairwise_predictions.csv"
RANKING = ML / "731_v4_5_pairwise_ranking_metrics.csv"
MODEL_METRICS = ML / "731_v4_5_pairwise_model_metrics.csv"
MODEL_SUMMARY = ML / "731_v4_5_pairwise_model_summary.csv"
CANDIDATES = ML / "731_v4_1_ml_candidate_dataset.csv"
GT = SYN_RFQ / "731_rfq_ground_truth_v2.csv"
REQ = SYN_RFQ / "731_rfq_requirements_v2.csv"

OUT_COMPARISON = ML / "731_v5_final_model_comparison.csv"
OUT_REGIME = ML / "731_v5_regime_comparison.csv"
OUT_DIFFICULTY = ML / "731_v5_candidate_difficulty.csv"
OUT_PACKAGE = ML / "731_v5_package_performance.csv"
OUT_EQUIV = ML / "731_v5_requirement_equivalence.csv"
OUT_BOOT = ML / "731_v5_bootstrap_confidence_intervals.csv"
OUT_PAIRED = ML / "731_v5_paired_statistical_comparisons.csv"
OUT_ERRORS = ML / "731_v5_error_analysis.csv"
OUT_SELECTION = ML / "731_v5_final_model_selection.csv"
OUT_SUMMARY = ML / "731_v5_thesis_results_summary.csv"
OUT_REPORT = ML / "731_v5_final_report.txt"


def need(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")


def norm_text(x):
    if pd.isna(x):
        return "__MISSING__"
    s = str(x).strip()
    return "__MISSING__" if not s or s.lower() in {"nan", "none", "null"} else s


def ndcg5(rank):
    if rank is None or not np.isfinite(rank) or rank > 5:
        return 0.0
    return 1.0 / np.log2(rank + 1.0)


def ranking_metrics(g):
    rows = []

    if g.empty:
        return {
            "rfqs": 0,
            "top1": np.nan,
            "top3": np.nan,
            "top5": np.nan,
            "top10": np.nan,
            "mrr": np.nan,
            "ndcg5": np.nan,
            "mean_true_rank": np.nan,
            "median_true_rank": np.nan,
        }, pd.DataFrame()

    required = {"rfq_id", "canonical_configuration_id", "ml_target", "model_score"}
    missing = required - set(g.columns)
    if missing:
        raise ValueError(f"Ranking data missing columns: {sorted(missing)}")

    for rfq, rg in g.groupby("rfq_id", sort=False):
        rg = rg.sort_values(
            ["model_score", "canonical_configuration_id"],
            ascending=[False, True],
            kind="mergesort",
        )

        pos = np.flatnonzero(rg["ml_target"].to_numpy() == 1)
        if len(pos) != 1:
            raise ValueError(
                f"RFQ {rfq} has {len(pos)} positive candidates; expected exactly one."
            )

        rank = int(pos[0]) + 1

        rows.append({
            "rfq_id": rfq,
            "candidate_count": len(rg),
            "true_rank": rank,
            "top1": int(rank == 1),
            "top3": int(rank <= 3),
            "top5": int(rank <= 5),
            "top10": int(rank <= 10),
            "mrr": 1.0 / rank,
            "ndcg5": ndcg5(rank),
            "true_configuration_id": rg.iloc[pos[0]]["canonical_configuration_id"],
            "predicted_configuration_id": rg.iloc[0]["canonical_configuration_id"],
        })

    r = pd.DataFrame(rows)

    metrics = {
        "rfqs": len(r),
        "top1": r["top1"].mean(),
        "top3": r["top3"].mean(),
        "top5": r["top5"].mean(),
        "top10": r["top10"].mean(),
        "mrr": r["mrr"].mean(),
        "ndcg5": r["ndcg5"].mean(),
        "mean_true_rank": r["true_rank"].mean(),
        "median_true_rank": r["true_rank"].median(),
    }

    return metrics, r


def bootstrap_metrics(score_df, n=5000, seed=731):
    """RFQ-level bootstrap. Each RFQ is the independent resampling unit."""
    rng = np.random.default_rng(seed)
    rfqs = score_df["rfq_id"].unique()

    if len(rfqs) == 0:
        return pd.DataFrame()

    by = {rfq: g for rfq, g in score_df.groupby("rfq_id")}
    records = []

    for i in range(n):
        sample = rng.choice(rfqs, size=len(rfqs), replace=True)
        top1 = []
        mrr = []
        ndcg = []

        for rfq in sample:
            g = by[rfq].sort_values(
                ["model_score", "canonical_configuration_id"],
                ascending=[False, True],
            )
            pos = np.flatnonzero(g["ml_target"].to_numpy() == 1)

            if len(pos) != 1:
                continue

            rank = int(pos[0]) + 1
            top1.append(int(rank == 1))
            mrr.append(1.0 / rank)
            ndcg.append(ndcg5(rank))

        if top1:
            records.append({
                "replicate": i,
                "top1": np.mean(top1),
                "mrr": np.mean(mrr),
                "ndcg5": np.mean(ndcg),
            })

    return pd.DataFrame(records)


def bootstrap_ci(s, alpha=0.05):
    return (
        float(s.quantile(alpha / 2)),
        float(s.quantile(1 - alpha / 2)),
    )


def paired_bootstrap_top1(a, b, n=5000, seed=731):
    """Bootstrap RFQ-level Top-1 difference: b minus a."""
    common = sorted(set(a.rfq_id) & set(b.rfq_id))
    if not common:
        return None

    def top1_map(df):
        result = {}
        for rfq, g in df.groupby("rfq_id"):
            m, _ = ranking_metrics(g)
            result[rfq] = m["top1"]
        return result

    ma = top1_map(a)
    mb = top1_map(b)

    diffs_obs = np.array([mb[r] - ma[r] for r in common], dtype=float)
    observed = float(diffs_obs.mean())

    rng = np.random.default_rng(seed)
    boot = np.empty(n)

    for i in range(n):
        idx = rng.integers(0, len(common), len(common))
        boot[i] = diffs_obs[idx].mean()

    lo, hi = np.quantile(boot, [0.025, 0.975])

    # Two-sided bootstrap sign probability.
    p = 2 * min(np.mean(boot <= 0), np.mean(boot >= 0))
    p = float(min(1.0, p))

    return {
        "common_rfqs": len(common),
        "observed_top1_difference": observed,
        "bootstrap_ci_low": float(lo),
        "bootstrap_ci_high": float(hi),
        "bootstrap_p_two_sided": p,
    }


def requirement_equivalence(pred_df, cand, gt, req):
    """Compare Top-1 predictions against the synthetic engineer choice."""
    req_cols = {"rfq_id", "characteristic", "internal_value", "requirement_type"}
    if not req_cols.issubset(req.columns):
        raise ValueError(
            f"Requirements file missing columns: {sorted(req_cols - set(req.columns))}"
        )

    req = req.copy()
    req["requirement_type"] = req["requirement_type"].astype(str).str.upper()

    cfg_lookup = (
        cand.drop_duplicates(["rfq_id", "canonical_configuration_id"])
        .set_index(["rfq_id", "canonical_configuration_id"])
    )

    gt_lookup = gt[
        ["rfq_id", "canonical_configuration_id"]
    ].drop_duplicates("rfq_id").rename(
        columns={"canonical_configuration_id": "engineer_configuration_id"}
    )

    gt_lookup = gt_lookup.set_index("rfq_id")

    rows = []

    for (regime, model), g in pred_df.groupby(["regime", "model"]):
        top = (
            g.sort_values(
                ["rfq_id", "model_score", "canonical_configuration_id"],
                ascending=[True, False, True],
            )
            .groupby("rfq_id", as_index=False)
            .first()
        )

        for _, row in top.iterrows():
            rfq = row["rfq_id"]
            pred_id = row["canonical_configuration_id"]

            if rfq not in gt_lookup.index:
                continue

            gt_id = gt_lookup.loc[rfq, "engineer_configuration_id"]

            try:
                pred_cfg = cfg_lookup.loc[(rfq, pred_id)]
            except KeyError:
                continue

            reqs = req[req["rfq_id"].eq(rfq)]

            mandatory_ok = True
            preferred_ok = True
            mandatory_count = 0
            preferred_count = 0
            mandatory_violations = 0
            preferred_differences = 0

            for _, rr in reqs.iterrows():
                c = rr["characteristic"]
                rt = str(rr["requirement_type"]).upper()
                val = norm_text(rr["internal_value"])
                col = f"config_{c}"

                if col not in pred_cfg.index:
                    continue

                same = norm_text(pred_cfg[col]) == val

                if rt == "MANDATORY":
                    mandatory_count += 1
                    if not same:
                        mandatory_ok = False
                        mandatory_violations += 1

                elif rt == "PREFERRED":
                    preferred_count += 1
                    if not same:
                        preferred_ok = False
                        preferred_differences += 1

            if pred_id == gt_id:
                category = "EXACT_ENGINEER_CHOICE"
            elif mandatory_ok and preferred_ok:
                category = "MANDATORY_AND_PREFERRED_EQUIVALENT"
            elif mandatory_ok:
                category = "MANDATORY_VALID_PREFERRED_DIFFERENCE"
            else:
                category = "MANDATORY_VIOLATION"

            rows.append({
                "regime": regime,
                "model": model,
                "rfq_id": rfq,
                "predicted_configuration_id": pred_id,
                "engineer_configuration_id": gt_id,
                "category": category,
                "mandatory_count": mandatory_count,
                "mandatory_violations": mandatory_violations,
                "preferred_count": preferred_count,
                "preferred_differences": preferred_differences,
            })

    return pd.DataFrame(rows)


def main():
    print("=" * 90)
    print("731 V5 — FINAL EXPERIMENTAL EVALUATION & MODEL SELECTION")
    print("=" * 90)

    for p in [PRED, RANKING, MODEL_METRICS, MODEL_SUMMARY, CANDIDATES, GT, REQ]:
        need(p)

    print("\n[1] Loading frozen V4.5 outputs")
    pred = pd.read_csv(PRED, low_memory=False)
    ranking_file = pd.read_csv(RANKING, low_memory=False)
    model_metrics = pd.read_csv(MODEL_METRICS, low_memory=False)
    model_summary = pd.read_csv(MODEL_SUMMARY, low_memory=False)
    cand = pd.read_csv(CANDIDATES, low_memory=False)
    gt = pd.read_csv(GT, low_memory=False)
    req = pd.read_csv(REQ, low_memory=False)

    print(f"Predictions:       {len(pred):,}")
    print(f"Prediction RFQs:   {pred.rfq_id.nunique():,}")
    print(f"Candidates:        {len(cand):,}")
    print(f"Candidate RFQs:    {cand.rfq_id.nunique():,}")
    print(f"Ground truth:      {len(gt):,}")

    required_pred = {
        "rfq_id", "regime", "model", "canonical_configuration_id",
        "ml_target", "model_score"
    }
    missing = required_pred - set(pred.columns)
    if missing:
        raise ValueError(f"Prediction file missing columns: {sorted(missing)}")

    combos = pred[["regime", "model"]].drop_duplicates()
    expected = pred.rfq_id.nunique() * len(combos)

    group_counts = (
        pred.groupby(["rfq_id", "regime", "model"])["ml_target"]
        .sum()
    )

    if not (group_counts == 1).all():
        bad = group_counts[group_counts != 1]
        raise ValueError(
            f"V5 target integrity failed: {len(bad):,} RFQ/regime/model groups "
            "do not contain exactly one positive."
        )

    actual_groups = len(group_counts)
    if actual_groups != expected:
        raise ValueError(
            f"Unexpected prediction group structure: {actual_groups:,} "
            f"groups; expected {expected:,}."
        )

    print("Prediction integrity: PASS")

    # ------------------------------------------------------------
    # [2] Model/regime ranking comparison
    # ------------------------------------------------------------
    print("\n[2] Final model/regime comparison")

    ranking_rows = []

    for (regime, model), g in pred.groupby(["regime", "model"]):
        metrics, per_rfq = ranking_metrics(g)

        ranking_rows.append({
            "regime": regime,
            "model": model,
            **metrics,
        })

        print(
            f"{regime:28s} {model:24s} "
            f"Top1={metrics['top1']:.4f} "
            f"Top3={metrics['top3']:.4f} "
            f"Top5={metrics['top5']:.4f} "
            f"MRR={metrics['mrr']:.4f} "
            f"NDCG5={metrics['ndcg5']:.4f}"
        )

    comparison = pd.DataFrame(ranking_rows)
    comparison = comparison.sort_values(
        ["top1", "mrr", "ndcg5"],
        ascending=False,
    ).reset_index(drop=True)
    comparison["overall_rank"] = np.arange(1, len(comparison) + 1)

    # ------------------------------------------------------------
    # [3] Regime comparison
    # ------------------------------------------------------------
    print("\n[3] Regime comparison")

    regime_rows = []
    for regime, g in comparison.groupby("regime"):
        best = g.sort_values(
            ["top1", "mrr", "ndcg5"], ascending=False
        ).iloc[0]

        regime_rows.append({
            "regime": regime,
            "best_model": best["model"],
            "top1": best["top1"],
            "top3": best["top3"],
            "top5": best["top5"],
            "top10": best["top10"],
            "mrr": best["mrr"],
            "ndcg5": best["ndcg5"],
        })

    regime_df = pd.DataFrame(regime_rows)

    # ------------------------------------------------------------
    # [4] Candidate difficulty
    # ------------------------------------------------------------
    print("\n[4] Candidate-set difficulty")

    test_cands = cand[cand["ml_split"].eq("TEST")].copy()

    candidate_counts = (
        test_cands.groupby("rfq_id")["canonical_configuration_id"]
        .nunique()
        .rename("candidate_count")
        .reset_index()
    )

    candidate_counts["difficulty_bucket"] = pd.cut(
        candidate_counts["candidate_count"],
        bins=[0, 1, 2, 5, 10, 50, np.inf],
        labels=["1", "2", "3-5", "6-10", "11-50", "51+"],
    )

    diff_pred = pred.merge(candidate_counts, on="rfq_id", how="left")

    diff_rows = []

    for (regime, model, bucket), g in diff_pred.groupby(
        ["regime", "model", "difficulty_bucket"],
        observed=True,
    ):
        if g.empty:
            continue

        metrics, _ = ranking_metrics(g)

        diff_rows.append({
            "regime": regime,
            "model": model,
            "difficulty_bucket": str(bucket),
            **metrics,
        })

    difficulty_df = pd.DataFrame(diff_rows)

    # ------------------------------------------------------------
    # [5] Package robustness
    # ------------------------------------------------------------
    print("\n[5] Package robustness")

    gt_pkg = (
        gt[["rfq_id", "target_package"]]
        .drop_duplicates("rfq_id")
    )

    package_pred = pred.merge(
        gt_pkg,
        on="rfq_id",
        how="left",
        validate="many_to_one",
    )

    package_rows = []

    for (regime, model, package_name), g in package_pred.groupby(
        ["regime", "model", "target_package"],
        dropna=False,
    ):
        if g.empty:
            continue

        metrics, _ = ranking_metrics(g)

        package_rows.append({
            "regime": regime,
            "model": model,
            "target_package": package_name,
            **metrics,
        })

    package_df = pd.DataFrame(package_rows)

    # ------------------------------------------------------------
    # [6] Requirement equivalence
    # ------------------------------------------------------------
    print("\n[6] Requirement equivalence")

    equivalence_df = requirement_equivalence(
        pred_df=pred,
        cand=cand,
        gt=gt,
        req=req,
    )

    if not equivalence_df.empty:
        eq_summary = (
            equivalence_df.groupby(["regime", "model", "category"])
            .size()
            .rename("rfqs")
            .reset_index()
        )
        eq_summary["share"] = (
            eq_summary.groupby(["regime", "model"])["rfqs"]
            .transform(lambda s: s / s.sum())
        )
    else:
        eq_summary = pd.DataFrame()

    # ------------------------------------------------------------
    # [7] Bootstrap confidence intervals
    # ------------------------------------------------------------
    print("\n[7] RFQ-level bootstrap confidence intervals — 5,000 replicates")

    bootstrap_rows = []

    for (regime, model), g in pred.groupby(["regime", "model"]):
        boot = bootstrap_metrics(g, n=5000, seed=731)

        if boot.empty:
            continue

        top1_lo, top1_hi = bootstrap_ci(boot["top1"])
        mrr_lo, mrr_hi = bootstrap_ci(boot["mrr"])
        ndcg_lo, ndcg_hi = bootstrap_ci(boot["ndcg5"])

        bootstrap_rows.append({
            "regime": regime,
            "model": model,
            "replicates": len(boot),
            "top1_ci_low": top1_lo,
            "top1_ci_high": top1_hi,
            "mrr_ci_low": mrr_lo,
            "mrr_ci_high": mrr_hi,
            "ndcg5_ci_low": ndcg_lo,
            "ndcg5_ci_high": ndcg_hi,
        })

        print(
            f"{regime:28s} {model:24s} "
            f"Top1 CI [{top1_lo:.4f}, {top1_hi:.4f}] "
            f"MRR CI [{mrr_lo:.4f}, {mrr_hi:.4f}]"
        )

    bootstrap_df = pd.DataFrame(bootstrap_rows)

    # ------------------------------------------------------------
    # [8] Paired regime comparisons — HGB
    # ------------------------------------------------------------
    print("\n[8] Paired statistical comparisons — HGB")

    hgb = {
        regime: pred[
            (pred["regime"] == regime) &
            (pred["model"] == "HIST_GRADIENT_BOOSTING")
        ].copy()
        for regime in [
            "REALISTIC_BASELINE",
            "PROVENANCE_AWARE",
            "MECHANISTIC_UPPER_BOUND",
        ]
    }

    comparison_pairs = [
        ("REALISTIC_BASELINE", "PROVENANCE_AWARE"),
        ("PROVENANCE_AWARE", "MECHANISTIC_UPPER_BOUND"),
        ("REALISTIC_BASELINE", "MECHANISTIC_UPPER_BOUND"),
    ]

    paired_rows = []

    for a, b in comparison_pairs:
        result = paired_bootstrap_top1(
            hgb[a], hgb[b], n=5000, seed=731
        )

        if result is None:
            continue

        row = {
            "comparison": f"{b} HGB minus {a} HGB",
            **result,
        }
        paired_rows.append(row)

        print(
            f"{row['comparison']}: "
            f"diff={row['observed_top1_difference']:+.4f} "
            f"CI=[{row['bootstrap_ci_low']:+.4f}, "
            f"{row['bootstrap_ci_high']:+.4f}] "
            f"p={row['bootstrap_p_two_sided']:.4f}"
        )

    paired_df = pd.DataFrame(paired_rows)

    # ------------------------------------------------------------
    # [9] Error analysis for best overall model
    # ------------------------------------------------------------
    print("\n[9] Best-model error analysis")

    best = comparison.iloc[0]
    best_regime = best["regime"]
    best_model = best["model"]

    best_pred = pred[
        (pred["regime"] == best_regime) &
        (pred["model"] == best_model)
    ].copy()

    best_metrics, per_rfq = ranking_metrics(best_pred)

    gt_cfg = (
        gt[["rfq_id", "canonical_configuration_id"]]
        .drop_duplicates("rfq_id")
        .rename(columns={"canonical_configuration_id": "engineer_configuration_id"})
    )

    errors = per_rfq[per_rfq["top1"].eq(0)].copy()
    errors = errors.merge(gt_cfg, on="rfq_id", how="left")

    errors["error_type"] = np.where(
        errors["true_configuration_id"].eq(errors["engineer_configuration_id"]),
        "RANKING_MISMATCH",
        "IDENTITY_REFERENCE_MISMATCH",
    )

    errors = errors.merge(candidate_counts, on="rfq_id", how="left")

    # ------------------------------------------------------------
    # [10] Final model selection
    # ------------------------------------------------------------
    print("\n[10] Final model selection")

    realistic = comparison[
        comparison["regime"].eq("REALISTIC_BASELINE")
    ].sort_values(
        ["top1", "mrr", "ndcg5"],
        ascending=False,
    ).iloc[0]

    provenance = comparison[
        comparison["regime"].eq("PROVENANCE_AWARE")
    ].sort_values(
        ["top1", "mrr", "ndcg5"],
        ascending=False,
    ).iloc[0]

    mechanistic = comparison[
        comparison["regime"].eq("MECHANISTIC_UPPER_BOUND")
    ].sort_values(
        ["top1", "mrr", "ndcg5"],
        ascending=False,
    ).iloc[0]

    selection_rows = [{
        "primary_model": "HIST_GRADIENT_BOOSTING",
        "primary_regime": "REALISTIC_BASELINE",
        "primary_top1": realistic["top1"],
        "primary_mrr": realistic["mrr"],
        "primary_ndcg5": realistic["ndcg5"],
        "provenance_best_model": provenance["model"],
        "provenance_top1": provenance["top1"],
        "provenance_mrr": provenance["mrr"],
        "mechanistic_best_model": mechanistic["model"],
        "mechanistic_top1": mechanistic["top1"],
        "mechanistic_mrr": mechanistic["mrr"],
        "overall_best_regime": best_regime,
        "overall_best_model": best_model,
        "overall_best_top1": best["top1"],
        "overall_best_mrr": best["mrr"],
        "selection_rationale": (
            "REALISTIC_BASELINE/HIST_GRADIENT_BOOSTING is the primary "
            "scientific/deployment model because it avoids synthetic "
            "preference-generation metadata. PROVENANCE_AWARE and "
            "MECHANISTIC_UPPER_BOUND are sensitivity/upper-bound analyses."
        ),
    }]

    selection_df = pd.DataFrame(selection_rows)

    # ------------------------------------------------------------
    # [11] Thesis summary
    # ------------------------------------------------------------
    eq_best = eq_summary[
        (eq_summary["regime"] == best_regime) &
        (eq_summary["model"] == best_model)
    ].copy()

    exact_share = float(
        eq_best.loc[
            eq_best["category"].eq("EXACT_ENGINEER_CHOICE"),
            "share",
        ].sum()
    ) if not eq_best.empty else np.nan

    req_equiv_share = float(
        eq_best.loc[
            eq_best["category"].isin([
                "EXACT_ENGINEER_CHOICE",
                "MANDATORY_AND_PREFERRED_EQUIVALENT",
            ]),
            "share",
        ].sum()
    ) if not eq_best.empty else np.nan

    thesis_summary = pd.DataFrame([{
        "test_rfqs": pred.rfq_id.nunique(),
        "candidate_rows": len(cand),
        "pairwise_prediction_rows": len(pred),
        "best_regime": best_regime,
        "best_model": best_model,
        "best_top1": best["top1"],
        "best_top3": best["top3"],
        "best_top5": best["top5"],
        "best_top10": best["top10"],
        "best_mrr": best["mrr"],
        "best_ndcg5": best["ndcg5"],
        "best_mean_true_rank": best["mean_true_rank"],
        "best_median_true_rank": best["median_true_rank"],
        "exact_engineer_choice_share": exact_share,
        "exact_or_mandatory_preferred_equivalent_share": req_equiv_share,
        "realistic_hgb_top1": realistic["top1"],
        "realistic_hgb_mrr": realistic["mrr"],
        "provenance_hgb_top1": provenance["top1"],
        "provenance_hgb_mrr": provenance["mrr"],
        "mechanistic_hgb_top1": mechanistic["top1"],
        "mechanistic_hgb_mrr": mechanistic["mrr"],
        "note": (
            "All labels are synthetic engineer preferences. Results "
            "measure recovery of the synthetic preference mechanism, "
            "not historical human engineer decisions."
        ),
    }])

    # ------------------------------------------------------------
    # [12] Report
    # ------------------------------------------------------------
    report = f"""
731 V5 — FINAL EXPERIMENTAL EVALUATION REPORT
=============================================

Purpose
-------
Final evaluation and model-selection stage following the frozen V4.5
pairwise learning-to-rank experiment.

Dataset
-------
Test RFQs: {pred.rfq_id.nunique():,}
Candidate rows: {len(cand):,}
Prediction rows: {len(pred):,}

Best overall frozen pairwise model
----------------------------------
Regime: {best_regime}
Model: {best_model}
Top-1: {best["top1"]:.4f}
Top-3: {best["top3"]:.4f}
Top-5: {best["top5"]:.4f}
Top-10: {best["top10"]:.4f}
MRR: {best["mrr"]:.4f}
NDCG@5: {best["ndcg5"]:.4f}
Mean true rank: {best["mean_true_rank"]:.4f}
Median true rank: {best["median_true_rank"]:.4f}

Governance-selected primary model
---------------------------------
Regime: REALISTIC_BASELINE
Model: HIST_GRADIENT_BOOSTING
Top-1: {realistic["top1"]:.4f}
MRR: {realistic["mrr"]:.4f}
NDCG@5: {realistic["ndcg5"]:.4f}

Sensitivity
-----------
PROVENANCE_AWARE best model:
{provenance["model"]}
Top-1: {provenance["top1"]:.4f}
MRR: {provenance["mrr"]:.4f}

MECHANISTIC_UPPER_BOUND best model:
{mechanistic["model"]}
Top-1: {mechanistic["top1"]:.4f}
MRR: {mechanistic["mrr"]:.4f}

Interpretation
--------------
The realistic baseline is the preferred primary model because it does not
depend on synthetic preference-generation metadata. Provenance-aware and
mechanistic models are retained as sensitivity and upper-bound analyses.

The exact configuration identity is evaluated separately from requirement
equivalence. A configuration may differ from the synthetic engineer's
selected ID while still satisfying the mandatory and preferred RFQ
requirements.

The RFQ-level bootstrap treats RFQs, not candidate rows, as the independent
sampling unit.

Synthetic-label caveat
----------------------
All engineer-preference labels are synthetic. Therefore these results
measure recovery of the synthetic preference-generation mechanism, not
historical human engineer behaviour.

Final recommendation
--------------------
Freeze REALISTIC_BASELINE / HIST_GRADIENT_BOOSTING as the primary
decision-support model unless the paired statistical comparison demonstrates
a clearly justified reason to prefer an alternative regime.

Do not add further ML algorithms merely to increase the numerical score.
The next work should be thesis figures, results tables, discussion, and
limitations.
""".strip()

    # ------------------------------------------------------------
    # [13] Write outputs
    # ------------------------------------------------------------
    OUT_COMPARISON.parent.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(OUT_COMPARISON, index=False)
    regime_df.to_csv(OUT_REGIME, index=False)
    difficulty_df.to_csv(OUT_DIFFICULTY, index=False)
    package_df.to_csv(OUT_PACKAGE, index=False)
    equivalence_df.to_csv(OUT_EQUIV, index=False)
    bootstrap_df.to_csv(OUT_BOOT, index=False)
    paired_df.to_csv(OUT_PAIRED, index=False)
    errors.to_csv(OUT_ERRORS, index=False)
    selection_df.to_csv(OUT_SELECTION, index=False)
    thesis_summary.to_csv(OUT_SUMMARY, index=False)
    OUT_REPORT.write_text(report + "\n", encoding="utf-8")

    print("\n" + "=" * 90)
    print("V5 FINAL EVALUATION COMPLETE")
    print("=" * 90)
    print(
        f"Overall best: {best_regime} / {best_model} | "
        f"Top-1={best['top1']:.4f} | "
        f"MRR={best['mrr']:.4f}"
    )
    print(
        f"Primary model: REALISTIC_BASELINE / HIST_GRADIENT_BOOSTING | "
        f"Top-1={realistic['top1']:.4f} | "
        f"MRR={realistic['mrr']:.4f}"
    )
    print(f"Outputs written to: {ML}")


if __name__ == "__main__":
    main()
