#!/usr/bin/env python3
"""
731 V6 — Thesis Results, Explainability & Final Evidence Package
===============================================================

This script is evaluation-only. It consumes frozen V4.5/V5 outputs and
creates thesis-ready tables, figures, explainability summaries, error
analysis, and a final results report.

No model is retrained.

Primary scientific model:
    REALISTIC_BASELINE / LOGISTIC_REGRESSION

Sensitivity:
    PROVENANCE_AWARE / HIST_GRADIENT_BOOSTING

Upper bound:
    MECHANISTIC_UPPER_BOUND / HIST_GRADIENT_BOOSTING

All preference labels are synthetic. Results measure recovery of the
synthetic engineer preference mechanism, not historical human decisions.
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
IMPORTANCE = ML / "731_v4_5_pairwise_feature_importance.csv"
MODEL_METRICS = ML / "731_v4_5_pairwise_model_metrics.csv"
CAND = ML / "731_v4_1_ml_candidate_dataset.csv"
GT = SYN_RFQ / "731_rfq_ground_truth_v2.csv"
REQ = SYN_RFQ / "731_rfq_requirements_v2.csv"

OUT_DIR = ML / "v6_thesis"
FIG_DIR = OUT_DIR / "figures"

OUT_ALL = OUT_DIR / "731_v6_all_model_results.csv"
OUT_PRIMARY = OUT_DIR / "731_v6_primary_model_results.csv"
OUT_CI = OUT_DIR / "731_v6_confidence_intervals.csv"
OUT_DIFF = OUT_DIR / "731_v6_candidate_difficulty.csv"
OUT_PACKAGE = OUT_DIR / "731_v6_package_performance.csv"
OUT_EQ = OUT_DIR / "731_v6_requirement_equivalence.csv"
OUT_ERRORS = OUT_DIR / "731_v6_error_analysis.csv"
OUT_IMPORTANCE = OUT_DIR / "731_v6_feature_importance.csv"
OUT_STATS = OUT_DIR / "731_v6_statistical_comparisons.csv"
OUT_THESIS = OUT_DIR / "731_v6_thesis_table.csv"
OUT_REPORT = OUT_DIR / "731_v6_thesis_results_report.txt"

TECHNICAL = [
    "config_devCategory",
    "config_characteristic",
    "config_housing",
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

REQ_FEATURES = [
    "mandatory_total",
    "mandatory_satisfied",
    "preferred_total",
    "preferred_satisfied",
]


def need(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")


def ndcg5(rank):
    if rank is None or not np.isfinite(rank) or rank > 5:
        return 0.0
    return 1.0 / np.log2(rank + 1.0)


def rank_table(g):
    rows = []
    for rfq, rg in g.groupby("rfq_id", sort=False):
        rg = rg.sort_values(
            ["model_score", "canonical_configuration_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        pos = np.flatnonzero(rg["ml_target"].to_numpy() == 1)
        if len(pos) != 1:
            raise ValueError(
                f"RFQ {rfq} has {len(pos)} positives; expected exactly one."
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
            "predicted_configuration_id": rg.iloc[0]["canonical_configuration_id"],
            "true_configuration_id": rg.iloc[pos[0]]["canonical_configuration_id"],
        })
    return pd.DataFrame(rows)


def metrics_from_rank(r):
    if r.empty:
        return {}
    return {
        "rfqs": len(r),
        "top1": r.top1.mean(),
        "top3": r.top3.mean(),
        "top5": r.top5.mean(),
        "top10": r.top10.mean(),
        "mrr": r.mrr.mean(),
        "ndcg5": r.ndcg5.mean(),
        "mean_true_rank": r.true_rank.mean(),
        "median_true_rank": r.true_rank.median(),
    }


def bootstrap_rank(r, n=5000, seed=731):
    rng = np.random.default_rng(seed)
    rfqs = r.rfq_id.unique()
    by = {x: g for x, g in r.groupby("rfq_id")}
    vals = []
    for _ in range(n):
        sample = rng.choice(rfqs, size=len(rfqs), replace=True)
        b = pd.concat([by[x] for x in sample], ignore_index=True)
        vals.append([
            b.top1.mean(),
            b.mrr.mean(),
            b.ndcg5.mean(),
        ])
    return np.asarray(vals)


def ci(a):
    return float(np.quantile(a, .025)), float(np.quantile(a, .975))


def make_figures(all_results, difficulty, package, importance):
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Overall Top-1
    x = all_results.copy()
    x["label"] = x["regime"].str.replace("_", " ") + "\n" + x["model"].str.replace("_", " ")
    x = x.sort_values("top1", ascending=True)

    plt.figure(figsize=(11, 7))
    plt.barh(x["label"], x["top1"] * 100)
    plt.xlabel("RFQ-level Top-1 (%)")
    plt.ylabel("Model / feature regime")
    plt.title("V4.5 Pairwise Learning-to-Rank: Top-1 Performance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_overall_top1.png", dpi=220)
    plt.close()

    # 2. Metric comparison for selected primary/sensitivity/upper models
    selected = all_results[
        ((all_results.regime == "REALISTIC_BASELINE") &
         (all_results.model == "LOGISTIC_REGRESSION")) |
        ((all_results.regime == "PROVENANCE_AWARE") &
         (all_results.model == "HIST_GRADIENT_BOOSTING")) |
        ((all_results.regime == "MECHANISTIC_UPPER_BOUND") &
         (all_results.model == "HIST_GRADIENT_BOOSTING"))
    ].copy()

    long = selected.melt(
        id_vars=["regime", "model"],
        value_vars=["top1", "top3", "top5", "top10"],
        var_name="metric",
        value_name="value",
    )
    long["label"] = long["regime"].str.replace("_", " ") + "\n" + long["model"].str.replace("_", " ")

    plt.figure(figsize=(12, 7))
    for label, g in long.groupby("label"):
        plt.plot(g["metric"], g["value"] * 100, marker="o", label=label)
    plt.ylabel("Performance (%)")
    plt.xlabel("Ranking metric")
    plt.title("Primary and Sensitivity Model Ranking Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_primary_sensitivity_metrics.png", dpi=220)
    plt.close()

    # 3. Difficulty
    d = difficulty[
        (difficulty.regime == "REALISTIC_BASELINE") &
        (difficulty.model == "LOGISTIC_REGRESSION")
    ].copy()

    if not d.empty:
        order = ["1", "2", "3-5", "6-10", "11-50", "51+"]
        d["difficulty_bucket"] = pd.Categorical(
            d["difficulty_bucket"], categories=order, ordered=True
        )
        d = d.sort_values("difficulty_bucket")

        plt.figure(figsize=(10, 6))
        plt.plot(
            d["difficulty_bucket"].astype(str),
            d["top1"] * 100,
            marker="o",
        )
        plt.xlabel("Candidate-set size")
        plt.ylabel("Top-1 (%)")
        plt.title("Primary Model Performance by Candidate-Set Difficulty")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "03_candidate_difficulty.png", dpi=220)
        plt.close()

    # 4. Package performance
    p = package[
        (package.regime == "REALISTIC_BASELINE") &
        (package.model == "LOGISTIC_REGRESSION")
    ].copy()

    if not p.empty:
        p = p.sort_values("top1")

        plt.figure(figsize=(11, 7))
        plt.barh(p["target_package"].astype(str), p["top1"] * 100)
        plt.xlabel("Top-1 (%)")
        plt.ylabel("Package")
        plt.title("Primary Model Performance by Package")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "04_package_performance.png", dpi=220)
        plt.close()

    # 5. Pairwise feature importance — realistic logistic
    imp = importance[
        (importance.regime == "REALISTIC_BASELINE") &
        (importance.model == "LOGISTIC_REGRESSION")
    ].copy()

    if not imp.empty:
        imp = imp.sort_values("importance", ascending=False).head(20)
        imp = imp.sort_values("importance")

        plt.figure(figsize=(11, 8))
        plt.barh(imp["feature"], imp["importance"])
        plt.xlabel("Absolute pairwise coefficient / importance")
        plt.ylabel("Feature")
        plt.title("Top Realistic-Baseline Pairwise Features")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "05_realistic_feature_importance.png", dpi=220)
        plt.close()


def main():
    print("=" * 90)
    print("731 V6 — THESIS RESULTS, EXPLAINABILITY & FINAL EVIDENCE PACKAGE")
    print("=" * 90)

    for p in [PRED, RANKING, IMPORTANCE, MODEL_METRICS, CAND, GT, REQ]:
        need(p)

    pred = pd.read_csv(PRED, low_memory=False)
    ranking_file = pd.read_csv(RANKING, low_memory=False)
    importance = pd.read_csv(IMPORTANCE, low_memory=False)
    model_metrics = pd.read_csv(MODEL_METRICS, low_memory=False)
    cand = pd.read_csv(CAND, low_memory=False)
    gt = pd.read_csv(GT, low_memory=False)
    req = pd.read_csv(REQ, low_memory=False)

    print(f"\nPredictions: {len(pred):,}")
    print(f"Test RFQs:   {pred.rfq_id.nunique():,}")

    # ---------------------------------------------------------
    # 1. Overall results
    # ---------------------------------------------------------
    print("\n[1] Overall model results")

    all_rows = []
    rank_tables = {}

    for (regime, model), g in pred.groupby(["regime", "model"]):
        r = rank_table(g)
        rank_tables[(regime, model)] = r
        m = metrics_from_rank(r)
        all_rows.append({
            "regime": regime,
            "model": model,
            **m,
        })

    all_results = pd.DataFrame(all_rows).sort_values(
        ["top1", "mrr", "ndcg5"], ascending=False
    ).reset_index(drop=True)

    print(
        all_results[
            ["regime", "model", "top1", "top3", "top5", "top10", "mrr", "ndcg5"]
        ].to_string(index=False)
    )

    # ---------------------------------------------------------
    # 2. Bootstrap
    # ---------------------------------------------------------
    print("\n[2] Bootstrap confidence intervals")

    ci_rows = []

    for key, r in rank_tables.items():
        boot = bootstrap_rank(r)
        tlo, thi = ci(boot[:, 0])
        mlo, mhi = ci(boot[:, 1])
        nlo, nhi = ci(boot[:, 2])

        ci_rows.append({
            "regime": key[0],
            "model": key[1],
            "top1_ci_low": tlo,
            "top1_ci_high": thi,
            "mrr_ci_low": mlo,
            "mrr_ci_high": mhi,
            "ndcg5_ci_low": nlo,
            "ndcg5_ci_high": nhi,
        })

        print(
            f"{key[0]:28s} {key[1]:24s} "
            f"Top1 [{tlo:.4f}, {thi:.4f}] "
            f"MRR [{mlo:.4f}, {mhi:.4f}]"
        )

    ci_df = pd.DataFrame(ci_rows)

    # ---------------------------------------------------------
    # 3. Candidate difficulty
    # ---------------------------------------------------------
    print("\n[3] Candidate-set difficulty")

    test_cand = cand[cand.ml_split.eq("TEST")].copy()
    counts = (
        test_cand.groupby("rfq_id")["canonical_configuration_id"]
        .nunique()
        .rename("candidate_count")
        .reset_index()
    )
    counts["difficulty_bucket"] = pd.cut(
        counts.candidate_count,
        bins=[0, 1, 2, 5, 10, 50, np.inf],
        labels=["1", "2", "3-5", "6-10", "11-50", "51+"],
    )

    difficulty_pred = pred.merge(counts, on="rfq_id", how="left")
    diff_rows = []

    for (regime, model, bucket), g in difficulty_pred.groupby(
        ["regime", "model", "difficulty_bucket"], observed=True
    ):
        if g.empty:
            continue
        r = rank_table(g)
        m = metrics_from_rank(r)
        diff_rows.append({
            "regime": regime,
            "model": model,
            "difficulty_bucket": str(bucket),
            **m,
        })

    difficulty_df = pd.DataFrame(diff_rows)

    # ---------------------------------------------------------
    # 4. Package performance
    # ---------------------------------------------------------
    print("\n[4] Package robustness")

    gt_pkg = gt[["rfq_id", "target_package"]].drop_duplicates("rfq_id")
    package_pred = pred.merge(
        gt_pkg, on="rfq_id", how="left", validate="many_to_one"
    )

    package_rows = []

    for (regime, model, package), g in package_pred.groupby(
        ["regime", "model", "target_package"], dropna=False
    ):
        if g.empty:
            continue
        r = rank_table(g)
        m = metrics_from_rank(r)
        package_rows.append({
            "regime": regime,
            "model": model,
            "target_package": package,
            **m,
        })

    package_df = pd.DataFrame(package_rows)

    # ---------------------------------------------------------
    # 5. Requirement equivalence
    # ---------------------------------------------------------
    print("\n[5] Requirement equivalence")

    req["requirement_type"] = req["requirement_type"].astype(str).str.upper()
    cfg = (
        cand.drop_duplicates(["rfq_id", "canonical_configuration_id"])
        .set_index(["rfq_id", "canonical_configuration_id"])
    )
    gt_ids = gt[
        ["rfq_id", "canonical_configuration_id"]
    ].drop_duplicates("rfq_id")

    eq_rows = []

    for key, r in rank_tables.items():
        regime, model = key
        for _, rr in r.iterrows():
            rfq = rr.rfq_id
            pred_id = rr.predicted_configuration_id
            true_id = rr.true_configuration_id

            try:
                pred_cfg = cfg.loc[(rfq, pred_id)]
            except KeyError:
                continue

            reqs = req[req.rfq_id.eq(rfq)]
            mandatory_ok = True
            preferred_ok = True

            for _, q in reqs.iterrows():
                col = f"config_{q.characteristic}"
                if col not in pred_cfg.index:
                    continue
                same = str(pred_cfg[col]).strip() == str(q.internal_value).strip()
                if q.requirement_type == "MANDATORY" and not same:
                    mandatory_ok = False
                if q.requirement_type == "PREFERRED" and not same:
                    preferred_ok = False

            if pred_id == true_id:
                category = "EXACT_ENGINEER_CHOICE"
            elif mandatory_ok and preferred_ok:
                category = "MANDATORY_AND_PREFERRED_EQUIVALENT"
            elif mandatory_ok:
                category = "MANDATORY_VALID_PREFERRED_DIFFERENCE"
            else:
                category = "MANDATORY_VIOLATION"

            eq_rows.append({
                "regime": regime,
                "model": model,
                "rfq_id": rfq,
                "predicted_configuration_id": pred_id,
                "engineer_configuration_id": true_id,
                "candidate_count": rr.candidate_count,
                "true_rank": rr.true_rank,
                "category": category,
            })

    equivalence_df = pd.DataFrame(eq_rows)

    # ---------------------------------------------------------
    # 6. Error analysis
    # ---------------------------------------------------------
    print("\n[6] Primary-model error analysis")

    primary_key = ("REALISTIC_BASELINE", "LOGISTIC_REGRESSION")
    primary_rank = rank_tables[primary_key]
    errors = primary_rank[primary_rank.top1.eq(0)].copy()

    errors["error_bucket"] = pd.cut(
        errors.candidate_count,
        bins=[0, 1, 2, 5, 10, 50, np.inf],
        labels=["1", "2", "3-5", "6-10", "11-50", "51+"],
    )

    error_summary = (
        errors.groupby("error_bucket", observed=True)
        .agg(
            failed_rfqs=("rfq_id", "count"),
            mean_true_rank=("true_rank", "mean"),
            median_true_rank=("true_rank", "median"),
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # 7. Statistical comparisons
    # ---------------------------------------------------------
    print("\n[7] Paired statistical comparisons")

    def paired(a_key, b_key, n=5000):
        a = rank_tables[a_key].set_index("rfq_id")
        b = rank_tables[b_key].set_index("rfq_id")
        common = sorted(set(a.index) & set(b.index))
        diffs = (
            b.loc[common, "top1"].to_numpy(dtype=float)
            - a.loc[common, "top1"].to_numpy(dtype=float)
        )

        rng = np.random.default_rng(731)
        boot = np.empty(n)
        for i in range(n):
            idx = rng.integers(0, len(diffs), len(diffs))
            boot[i] = diffs[idx].mean()

        lo, hi = np.quantile(boot, [.025, .975])
        p = 2 * min(np.mean(boot <= 0), np.mean(boot >= 0))
        return {
            "comparison": f"{b_key[0]}/{b_key[1]} minus {a_key[0]}/{a_key[1]}",
            "common_rfqs": len(common),
            "observed_top1_difference": diffs.mean(),
            "ci_low": lo,
            "ci_high": hi,
            "p_two_sided": min(1.0, p),
        }

    stat_rows = [
        paired(
            ("REALISTIC_BASELINE", "LOGISTIC_REGRESSION"),
            ("PROVENANCE_AWARE", "HIST_GRADIENT_BOOSTING"),
        ),
        paired(
            ("PROVENANCE_AWARE", "HIST_GRADIENT_BOOSTING"),
            ("MECHANISTIC_UPPER_BOUND", "HIST_GRADIENT_BOOSTING"),
        ),
        paired(
            ("REALISTIC_BASELINE", "LOGISTIC_REGRESSION"),
            ("MECHANISTIC_UPPER_BOUND", "HIST_GRADIENT_BOOSTING"),
        ),
    ]
    stats_df = pd.DataFrame(stat_rows)

    # ---------------------------------------------------------
    # 8. Feature importance
    # ---------------------------------------------------------
    imp = importance.copy()
    imp = imp[
        imp["regime"].isin([
            "REALISTIC_BASELINE",
            "PROVENANCE_AWARE",
            "MECHANISTIC_UPPER_BOUND",
        ])
    ].copy()

    # Keep the top 30 per model/regime for the thesis package.
    imp["abs_importance"] = imp["importance"].abs()
    imp = (
        imp.sort_values(
            ["regime", "model", "abs_importance"],
            ascending=[True, True, False],
        )
        .groupby(["regime", "model"], group_keys=False)
        .head(30)
    )

    # ---------------------------------------------------------
    # 9. Final model-selection table
    # ---------------------------------------------------------
    primary = all_results[
        (all_results.regime == "REALISTIC_BASELINE") &
        (all_results.model == "LOGISTIC_REGRESSION")
    ].iloc[0]

    sensitivity = all_results[
        (all_results.regime == "PROVENANCE_AWARE") &
        (all_results.model == "HIST_GRADIENT_BOOSTING")
    ].iloc[0]

    upper = all_results[
        (all_results.regime == "MECHANISTIC_UPPER_BOUND") &
        (all_results.model == "HIST_GRADIENT_BOOSTING")
    ].iloc[0]

    thesis_table = pd.DataFrame([
        {
            "model_role": "PRIMARY",
            "regime": primary.regime,
            "model": primary.model,
            "top1": primary.top1,
            "top3": primary.top3,
            "top5": primary.top5,
            "top10": primary.top10,
            "mrr": primary.mrr,
            "ndcg5": primary.ndcg5,
            "interpretation": "Best clean realistic-information model; recommended primary model.",
        },
        {
            "model_role": "SENSITIVITY",
            "regime": sensitivity.regime,
            "model": sensitivity.model,
            "top1": sensitivity.top1,
            "top3": sensitivity.top3,
            "top5": sensitivity.top5,
            "top10": sensitivity.top10,
            "mrr": sensitivity.mrr,
            "ndcg5": sensitivity.ndcg5,
            "interpretation": "Provenance-aware sensitivity analysis.",
        },
        {
            "model_role": "UPPER_BOUND",
            "regime": upper.regime,
            "model": upper.model,
            "top1": upper.top1,
            "top3": upper.top3,
            "top5": upper.top5,
            "top10": upper.top10,
            "mrr": upper.mrr,
            "ndcg5": upper.ndcg5,
            "interpretation": "Mechanistic upper bound; no gain over provenance-aware model.",
        },
    ])

    # ---------------------------------------------------------
    # 10. Report
    # ---------------------------------------------------------
    p_eq = equivalence_df[
        (equivalence_df.regime == primary.regime) &
        (equivalence_df.model == primary.model)
    ]

    exact_share = (
        p_eq.category.eq("EXACT_ENGINEER_CHOICE").mean()
        if not p_eq.empty else np.nan
    )
    req_equiv_share = (
        p_eq.category.isin([
            "EXACT_ENGINEER_CHOICE",
            "MANDATORY_AND_PREFERRED_EQUIVALENT",
        ]).mean()
        if not p_eq.empty else np.nan
    )

    prov_gain = float(sensitivity.top1 - primary.top1)
    mech_gain = float(upper.top1 - sensitivity.top1)

    report = f"""
731 V6 — THESIS RESULTS, EXPLAINABILITY & FINAL EVIDENCE PACKAGE
================================================================

Scope
-----
V6 is an evaluation-only stage using the frozen V4.5 pairwise predictions.
No model is retrained.

Test RFQs: {pred.rfq_id.nunique():,}
Candidate rows: {len(cand):,}
Prediction rows: {len(pred):,}

PRIMARY MODEL
-------------
REALISTIC_BASELINE / LOGISTIC_REGRESSION

Top-1: {primary.top1:.4f}
Top-3: {primary.top3:.4f}
Top-5: {primary.top5:.4f}
Top-10: {primary.top10:.4f}
MRR: {primary.mrr:.4f}
NDCG@5: {primary.ndcg5:.4f}

SENSITIVITY MODEL
-----------------
PROVENANCE_AWARE / HIST_GRADIENT_BOOSTING

Top-1: {sensitivity.top1:.4f}
MRR: {sensitivity.mrr:.4f}

Improvement over realistic primary:
Top-1 difference: {prov_gain:+.4f}
Percentage-point improvement: {prov_gain * 100:+.2f}

UPPER-BOUND MODEL
-----------------
MECHANISTIC_UPPER_BOUND / HIST_GRADIENT_BOOSTING

Top-1: {upper.top1:.4f}
MRR: {upper.mrr:.4f}

Additional improvement over provenance-aware:
Top-1 difference: {mech_gain:+.4f}

The mechanistic upper-bound model provides no additional Top-1 improvement
over the provenance-aware model in the frozen V4.5 results.

STATISTICAL INTERPRETATION
--------------------------
The paired bootstrap comparison between the realistic primary model and
the provenance-aware HGB model is reported separately in the statistical
comparison table. The provenance-aware gain should be described as
statistically supported only according to that paired confidence interval
and p-value.

REQUIREMENT EQUIVALENCE
-----------------------
Primary-model exact synthetic-engineer-choice share:
{exact_share:.4f}

Primary-model exact-or-mandatory+preferred-equivalent share:
{req_equiv_share:.4f}

Exact ID recovery and technical requirement compliance are distinct
evaluation concepts.

CANDIDATE DIFFICULTY
--------------------
Candidate-set difficulty is evaluated using the number of distinct candidate
configuration IDs per RFQ in the TEST split. Performance is reported by
candidate-count bucket.

PACKAGE ROBUSTNESS
------------------
Package-level performance is reported using the RFQ target package for
post-hoc evaluation.

EXPLAINABILITY
--------------
Feature importance is reported for all regimes/models and a dedicated
realistic-baseline importance table is produced for the primary model.

METHODOLOGICAL LIMITATION
-------------------------
The engineer-preference labels are synthetic. Therefore the results measure
recovery of the synthetic preference-generation mechanism, not historical
human engineer behaviour.

FINAL MODEL RECOMMENDATION
--------------------------
Use REALISTIC_BASELINE / LOGISTIC_REGRESSION as the primary model for the
thesis and decision-support framing because it uses the cleanest realistic
information and is the strongest model within that regime.

Retain PROVENANCE_AWARE / HIST_GRADIENT_BOOSTING as a sensitivity analysis
and MECHANISTIC_UPPER_BOUND / HIST_GRADIENT_BOOSTING as an upper-bound
experiment.

Do not add further ML algorithms unless a thesis reviewer identifies a
specific methodological gap. The next work is interpretation, discussion,
limitations, and thesis writing.
""".strip()

    # ---------------------------------------------------------
    # 11. Figures
    # ---------------------------------------------------------
    print("\n[8] Creating thesis figures")
    make_figures(all_results, difficulty_df, package_df, imp)

    # ---------------------------------------------------------
    # 12. Save
    # ---------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results.to_csv(OUT_ALL, index=False)
    thesis_table.to_csv(OUT_PRIMARY, index=False)
    ci_df.to_csv(OUT_CI, index=False)
    difficulty_df.to_csv(OUT_DIFF, index=False)
    package_df.to_csv(OUT_PACKAGE, index=False)
    equivalence_df.to_csv(OUT_EQ, index=False)
    errors.to_csv(OUT_ERRORS, index=False)
    imp.to_csv(OUT_IMPORTANCE, index=False)
    stats_df.to_csv(OUT_STATS, index=False)
    thesis_table.to_csv(OUT_THESIS, index=False)
    OUT_REPORT.write_text(report + "\n", encoding="utf-8")

    print("\n" + "=" * 90)
    print("V6 THESIS EVIDENCE PACKAGE COMPLETE")
    print("=" * 90)
    print(f"Primary: {primary.regime} / {primary.model}")
    print(
        f"Top-1={primary.top1:.4f} | "
        f"Top-3={primary.top3:.4f} | "
        f"Top-5={primary.top5:.4f} | "
        f"MRR={primary.mrr:.4f}"
    )
    print(f"Output directory: {OUT_DIR}")
    print(f"Figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
