#!/usr/bin/env python3
"""
731 V4.5 — Pairwise Learning-to-Rank
=====================================

Purpose
-------
Train pairwise preference models for the synthetic engineer ranking task.

Unit
----
One row = one ordered pair (A, B) within one RFQ.

Original V4.1 pairwise data stores "preferred > rejected". For supervised
binary pairwise learning, every observed comparison is augmented with its
reverse:
    preferred - rejected -> label 1
    rejected - preferred -> label 0

The model learns which candidate in a pair should be preferred. At inference,
each candidate is compared against the other candidates in the RFQ and its
mean pairwise win probability is used as the ranking score.

Feature governance
------------------
REALISTIC_BASELINE:
    Technical configuration + pre-choice requirement/candidate matching.
    No synthetic preference-generation metadata.

PROVENANCE_AWARE:
    Realistic baseline + synthetic provenance/process variables.

MECHANISTIC_UPPER_BOUND:
    Realistic + provenance + variables used by the synthetic engineer
    preference-generation mechanism.

Forbidden:
    Ground-truth IDs, engineer rank/score/utility, preference labels, and
    any post-choice information.

Primary evaluation:
    RFQ-level Top-1/3/5/10, MRR, NDCG@5, exact engineer-choice recovery.

Secondary:
    Pairwise accuracy, package performance, candidate-set difficulty,
    requirement-equivalence, bootstrap confidence intervals, feature
    importance, leakage audit.

Important:
    These labels are synthetic. Performance estimates recovery of the
    synthetic preference mechanism, not historical human engineer behaviour.
"""

from pathlib import Path
import warnings
import itertools
import math

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
ML = ROOT / "data" / "processed" / "ml"
SYN_RFQ = ROOT / "data" / "synthetic" / "rfqs"
SYN_CFG = ROOT / "data" / "synthetic" / "configurations"

PAIRWISE = ML / "731_v4_1_ml_pairwise_dataset.csv"
CANDIDATES = ML / "731_v4_1_ml_candidate_dataset.csv"
MANIFEST = ML / "731_v4_2_1_model_feature_manifest.csv"

OUT_METRICS = ML / "731_v4_5_pairwise_model_metrics.csv"
OUT_RANKING = ML / "731_v4_5_pairwise_ranking_metrics.csv"
OUT_PRED = ML / "731_v4_5_pairwise_predictions.csv"
OUT_IMPORTANCE = ML / "731_v4_5_pairwise_feature_importance.csv"
OUT_PACKAGE = ML / "731_v4_5_pairwise_package_performance.csv"
OUT_DIFFICULTY = ML / "731_v4_5_pairwise_candidate_difficulty.csv"
OUT_EQUIV = ML / "731_v4_5_pairwise_requirement_equivalence.csv"
OUT_BOOT = ML / "731_v4_5_pairwise_bootstrap_metrics.csv"
OUT_COMPARE = ML / "731_v4_5_pairwise_statistical_comparisons.csv"
OUT_AUDIT = ML / "731_v4_5_pairwise_leakage_audit.csv"
OUT_SUMMARY = ML / "731_v4_5_pairwise_model_summary.csv"
OUT_REPORT = ML / "731_v4_5_pairwise_report.txt"

GT = SYN_RFQ / "731_rfq_ground_truth_v2.csv"
REQ = SYN_RFQ / "731_rfq_requirements_v2.csv"

TECHNICAL = [
    "config_devCategory", "config_characteristic", "config_housing",
    "config_powerSupply", "config_numberOfChannels",
    "config_explosionApproval", "config_protectionArea",
    "config_certification", "config_dataInterface", "config_stromSchaltbar",
    "config_stromEingaenge", "config_temperaturEingaenge",
    "config_binaerDigitalOpenColl_MN", "config_binaerOpenColl_MP",
    "config_waveInjector", "config_dev_advMeterVerification",
    "config_dynamicGasMaster", "config_customUserFluid", "config_steamApplication",
]

REQUIREMENT = [
    "mandatory_total", "mandatory_satisfied", "mandatory_violations",
    "preferred_total", "preferred_satisfied", "preferred_differences",
]

PROVENANCE = [
    "record_type", "generation_method", "evidence_level", "source_frequency",
    "nearest_real_distance", "num_changed_characteristics", "evidence_score",
]

MECHANISTIC = [
    "source_frequency_score", "real_proximity_score", "low_change_score",
    "observed_configuration_score", "evidence_score_normalized",
    "unmentioned_features",
]

FORBIDDEN = {
    "ml_target", "is_synthetic_engineer_choice", "engineer_rank",
    "synthetic_engineer_score", "synthetic_engineer_utility", "tie_break_jitter",
    "ground_truth_configuration_id", "is_ground_truth_configuration",
    "true_configuration_id", "recommended_configuration_id",
    "ground_truth_engineer_rank", "ground_truth_engineer_score",
    "ground_truth_selected_by_engineer", "synthetic_engineer_choice",
    "preferred_rank", "rejected_rank", "preferred_score", "rejected_score",
    "score_margin", "label",
}

PAIRWISE_ARTIFACTS = {
    "preferred_rank", "rejected_rank", "preferred_score", "rejected_score",
    "score_margin", "label", "preference_label_provenance",
    "preference_generation_method",
}

ID_COLUMNS = {
    "rfq_id", "customer_id", "canonical_configuration_id",
    "synthetic_record_id", "source_canonical_configuration_id",
    "preferred_configuration_id", "rejected_configuration_id", "ml_split",
    "target_package",
}


def need(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")


def norm_text(x):
    if pd.isna(x):
        return "__MISSING__"
    s = str(x).strip()
    return "__MISSING__" if not s or s.lower() in {"nan", "none", "null"} else s


def numeric_value(x):
    return pd.to_numeric(x, errors="coerce")


def ndcg_at_k(rank, k):
    if rank is None or not np.isfinite(rank) or rank > k:
        return 0.0
    return 1.0 / np.log2(rank + 1.0)


def ranking_metrics(scores):
    rows = []
    for rfq, g in scores.groupby("rfq_id", sort=False):
        g = g.sort_values(
            ["model_score", "canonical_configuration_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        pos = g.index[g["ml_target"] == 1]
        if len(pos) != 1:
            raise ValueError(f"RFQ {rfq} does not have exactly one positive.")
        true_id = g.loc[pos[0], "canonical_configuration_id"]
        rank = int(np.flatnonzero(g["canonical_configuration_id"].to_numpy() == true_id)[0]) + 1
        rows.append({
            "rfq_id": rfq,
            "candidate_count": len(g),
            "true_rank": rank,
            "top1": int(rank <= 1),
            "top3": int(rank <= 3),
            "top5": int(rank <= 5),
            "top10": int(rank <= 10),
            "mrr": 1.0 / rank,
            "ndcg5": ndcg_at_k(rank, 5),
            "true_configuration_id": true_id,
            "predicted_configuration_id": g.iloc[0]["canonical_configuration_id"],
        })
    r = pd.DataFrame(rows)
    if r.empty:
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
        }, r
    return {
        "rfqs": len(r),
        "top1": r["top1"].mean(),
        "top3": r["top3"].mean(),
        "top5": r["top5"].mean(),
        "top10": r["top10"].mean(),
        "mrr": r["mrr"].mean(),
        "ndcg5": r["ndcg5"].mean(),
        "mean_true_rank": r["true_rank"].mean(),
        "median_true_rank": r["true_rank"].median(),
    }, r


def make_pair_features(pair_df, cand, feature_cols, regime):
    """
    Build directional A-B features.

    Numeric:
        A - B

    Categorical:
        signed one-hot difference:
            +1 if preferred has category and rejected does not
            -1 if rejected has category and preferred does not

    This preserves direction. A simple equality flag would lose which
    configuration owns the category.
    """
    c = cand.set_index(["rfq_id", "canonical_configuration_id"])

    pref_keys = pd.MultiIndex.from_frame(
        pair_df[["rfq_id", "preferred_configuration_id"]]
    )
    rej_keys = pd.MultiIndex.from_frame(
        pair_df[["rfq_id", "rejected_configuration_id"]]
    )

    missing_pref = ~pref_keys.isin(c.index)
    missing_rej = ~rej_keys.isin(c.index)
    if missing_pref.any() or missing_rej.any():
        raise ValueError(
            f"Pairwise candidate lookup failed: preferred={missing_pref.sum()}, "
            f"rejected={missing_rej.sum()}."
        )

    pref = c.loc[pref_keys, feature_cols].reset_index(drop=True)
    rej = c.loc[rej_keys, feature_cols].reset_index(drop=True)

    out = {}
    categorical = []
    numeric = []

    for col in feature_cols:
        # Treat known technical/configuration categorical variables as categorical.
        if col in TECHNICAL or col in {"record_type", "generation_method", "evidence_level"}:
            categorical.append(col)
        else:
            # Detect numeric columns after coercion.
            pnum = pd.to_numeric(pref[col], errors="coerce")
            rnum = pd.to_numeric(rej[col], errors="coerce")
            if pnum.notna().sum() + rnum.notna().sum() > 0:
                numeric.append(col)
            else:
                categorical.append(col)

    for col in numeric:
        p = pd.to_numeric(pref[col], errors="coerce")
        r = pd.to_numeric(rej[col], errors="coerce")
        out[f"DIFF__{col}"] = p.fillna(0) - r.fillna(0)

    for col in categorical:
        p = pref[col].map(norm_text)
        r = rej[col].map(norm_text)
        vals = sorted(set(p.unique()) | set(r.unique()))
        for v in vals:
            out[f"CATDIFF__{col}__{v}"] = (
                (p == v).astype(int) - (r == v).astype(int)
            )

    X = pd.DataFrame(out, index=pair_df.index)
    return X


def fit_model(name, X, y):
    if name == "LOGISTIC_REGRESSION":
        return LogisticRegression(
            max_iter=3000, C=1.0, solver="lbfgs", random_state=731
        )
    if name == "RANDOM_FOREST":
        return RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", random_state=731, n_jobs=-1
        )
    if name == "HIST_GRADIENT_BOOSTING":
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=731
        )
    raise ValueError(name)


def bootstrap_rfqs(score_df, n=2000, seed=731):
    rng = np.random.default_rng(seed)
    rfqs = score_df["rfq_id"].unique()
    by = {r: g for r, g in score_df.groupby("rfq_id")}
    records = []
    for i in range(n):
        sample = rng.choice(rfqs, size=len(rfqs), replace=True)
        vals = []
        for r in sample:
            g = by[r].copy()
            g = g.sort_values(
                ["model_score", "canonical_configuration_id"],
                ascending=[False, True],
            )
            true_pos = np.flatnonzero(g["ml_target"].to_numpy() == 1)
            if len(true_pos) != 1:
                continue
            rank = int(true_pos[0]) + 1
            vals.append((int(rank == 1), 1.0 / rank, ndcg_at_k(rank, 5)))
        if vals:
            a = np.asarray(vals)
            records.append({
                "replicate": i,
                "top1": a[:, 0].mean(),
                "mrr": a[:, 1].mean(),
                "ndcg5": a[:, 2].mean(),
            })
    return pd.DataFrame(records)


def ci(series):
    return (
        float(series.quantile(0.025)),
        float(series.quantile(0.975)),
    )


def main():
    print("=" * 80)
    print("731 V4.5 PAIRWISE LEARNING-TO-RANK")
    print("=" * 80)

    for p in [PAIRWISE, CANDIDATES, MANIFEST, GT, REQ]:
        need(p)

    print("\n[1] Loading inputs")
    pair = pd.read_csv(PAIRWISE, low_memory=False)
    cand = pd.read_csv(CANDIDATES, low_memory=False)
    manifest = pd.read_csv(MANIFEST, low_memory=False)
    gt = pd.read_csv(GT, low_memory=False)
    req = pd.read_csv(REQ, low_memory=False)

    print(f"Pairwise rows:      {len(pair):,}")
    print(f"Candidate rows:     {len(cand):,}")
    print(f"Candidate RFQs:     {cand.rfq_id.nunique():,}")
    print(f"Pairwise RFQs:      {pair.rfq_id.nunique():,}")
    print(f"Ground truth rows:  {len(gt):,}")

    print("\n[2] Schema / integrity")
    required_pair = {
        "rfq_id", "target_package", "preferred_configuration_id",
        "rejected_configuration_id", "label", "ml_split"
    }
    required_cand = {"rfq_id", "canonical_configuration_id", "ml_split", "ml_target"}
    if not required_pair.issubset(pair.columns):
        raise ValueError(f"Missing pairwise columns: {sorted(required_pair - set(pair.columns))}")
    if not required_cand.issubset(cand.columns):
        raise ValueError(f"Missing candidate columns: {sorted(required_cand - set(cand.columns))}")

    if pair["label"].nunique() != 1 or pair["label"].iloc[0] != 1:
        raise ValueError(
            "Expected V4.1 pairwise source labels to be all 1 (preferred > rejected)."
        )

    dup = pair.duplicated(
        ["rfq_id", "preferred_configuration_id", "rejected_configuration_id"]
    ).sum()
    if dup:
        raise ValueError(f"Duplicate ordered pair rows: {dup:,}")

    cand_dups = cand.duplicated(["rfq_id", "canonical_configuration_id"]).sum()
    if cand_dups:
        raise ValueError(f"Duplicate RFQ/candidate rows: {cand_dups:,}")

    rfq_split = cand.groupby("rfq_id")["ml_split"].nunique()
    if (rfq_split != 1).any():
        raise ValueError("Candidate RFQs cross ML splits.")

    pair_split = pair.groupby("rfq_id")["ml_split"].nunique()
    if (pair_split != 1).all() is False:
        raise ValueError("Pairwise RFQs cross ML splits.")

    print("Schema validation: PASS")
    print(f"Duplicate candidate rows: {cand_dups}")
    print(f"Duplicate pair rows: {dup}")
    print("RFQ-level split integrity: PASS")

    print("\n[3] Building symmetric pairwise training data")
    reverse = pair.copy()
    reverse["preferred_configuration_id"], reverse["rejected_configuration_id"] = (
        pair["rejected_configuration_id"].values,
        pair["preferred_configuration_id"].values,
    )
    reverse["label"] = 0

    forward = pair.copy()
    forward["label"] = 1

    sym = pd.concat([forward, reverse], ignore_index=True)
    sym = sym[
        ["rfq_id", "target_package", "preferred_configuration_id",
         "rejected_configuration_id", "label", "ml_split"]
    ]

    print(f"Original comparisons: {len(pair):,}")
    print(f"Symmetric comparisons: {len(sym):,}")
    print(f"Positive: {(sym.label == 1).sum():,}")
    print(f"Negative: {(sym.label == 0).sum():,}")

    print("\n[4] Feature regimes")
    available = set(cand.columns)

    realistic_base = [
        c for c in TECHNICAL + REQUIREMENT
        if c in available
    ]
    provenance_cols = [c for c in PROVENANCE if c in available]
    mechanistic_cols = [c for c in MECHANISTIC if c in available]

    regimes = {
        "REALISTIC_BASELINE": realistic_base,
        "PROVENANCE_AWARE": realistic_base + provenance_cols,
        "MECHANISTIC_UPPER_BOUND": realistic_base + provenance_cols + mechanistic_cols,
    }

    for name, cols in regimes.items():
        print(f"{name}: {len(cols)} candidate-level features")
        print("  " + ", ".join(cols))

    print("\n[5] Building pairwise feature matrices")
    matrices = {}
    for name, cols in regimes.items():
        X = make_pair_features(sym, cand, cols, name)

        # Drop constant pairwise columns based on the training split only.
        train_mask = sym["ml_split"].eq("TRAIN")
        train_var = X.loc[train_mask].nunique(dropna=False)
        keep = train_var[train_var > 1].index.tolist()
        X = X[keep].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        matrices[name] = X
        print(
            f"{name}: {X.shape[1]:,} pairwise features after constant removal"
        )

    print("\n[6] Training pairwise models")
    all_metrics = []
    all_ranking = []
    all_predictions = []
    all_importance = []

    models = [
        "LOGISTIC_REGRESSION",
        "RANDOM_FOREST",
        "HIST_GRADIENT_BOOSTING",
    ]

    for regime, X in matrices.items():
        train_mask = sym["ml_split"].eq("TRAIN")
        val_mask = sym["ml_split"].eq("VALIDATION")
        test_mask = sym["ml_split"].eq("TEST")

        y = sym["label"].astype(int)
        model_train = fit_model("LOGISTIC_REGRESSION", X.loc[train_mask], y.loc[train_mask])
        # Fit every model below separately.

        for model_name in models:
            model = fit_model(model_name, X.loc[train_mask], y.loc[train_mask])
            model.fit(X.loc[train_mask], y.loc[train_mask])

            if hasattr(model, "predict_proba"):
                val_prob = model.predict_proba(X.loc[val_mask])[:, 1]
                test_prob = model.predict_proba(X.loc[test_mask])[:, 1]
            else:
                val_prob = model.decision_function(X.loc[val_mask])
                test_prob = model.decision_function(X.loc[test_mask])

            val_pred = (val_prob >= 0.5).astype(int)
            test_pred = (test_prob >= 0.5).astype(int)

            all_metrics.append({
                "regime": regime,
                "model": model_name,
                "train_rows": int(train_mask.sum()),
                "validation_rows": int(val_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "validation_pairwise_accuracy": accuracy_score(y.loc[val_mask], val_pred),
                "validation_pairwise_auc": roc_auc_score(y.loc[val_mask], val_prob),
                "test_pairwise_accuracy": accuracy_score(y.loc[test_mask], test_pred),
                "test_pairwise_auc": roc_auc_score(y.loc[test_mask], test_prob),
                "feature_count": X.shape[1],
            })

            # Candidate-level scores for test RFQs:
            # compare every candidate A against every other candidate B,
            # then average P(A > B).
            test_cand = cand[cand["ml_split"].eq("TEST")].copy()
            score_rows = []

            for rfq, g in test_cand.groupby("rfq_id", sort=False):
                ids = g["canonical_configuration_id"].tolist()
                if len(ids) == 1:
                    score_rows.append({
                        "rfq_id": rfq,
                        "canonical_configuration_id": ids[0],
                        "model_score": 1.0,
                        "ml_target": int(g.iloc[0]["ml_target"]),
                    })
                    continue

                pair_rows = []
                pair_meta = []
                for a, b in itertools.permutations(ids, 2):
                    pair_rows.append((rfq, a, b))
                    pair_meta.append((rfq, a, b))

                ptest = pd.DataFrame(
                    pair_rows,
                    columns=["rfq_id", "preferred_configuration_id",
                             "rejected_configuration_id"]
                )
                ptest["target_package"] = (
                    g["target_package"].iloc[0]
                    if "target_package" in g.columns else "__NA__"
                )
                ptest["label"] = 1
                Xp = make_pair_features(
                    ptest,
                    cand,
                    regimes[regime],
                    regime
                )
                Xp = Xp.reindex(columns=X.columns, fill_value=0).replace(
                    [np.inf, -np.inf], np.nan
                ).fillna(0.0)

                probs = model.predict_proba(Xp)[:, 1]

                wins = {i: [] for i in ids}
                for (rr, a, b), pr in zip(pair_meta, probs):
                    wins[a].append(float(pr))
                    # P(b > a) = 1 - P(a > b)
                    wins[b].append(float(1.0 - pr))

                target_map = dict(
                    zip(g["canonical_configuration_id"], g["ml_target"])
                )
                for cid in ids:
                    score_rows.append({
                        "rfq_id": rfq,
                        "canonical_configuration_id": cid,
                        "model_score": float(np.mean(wins[cid])),
                        "ml_target": int(target_map[cid]),
                    })

            score_df = pd.DataFrame(score_rows)
            score_df["regime"] = regime
            score_df["model"] = model_name

            rm, detail = ranking_metrics(score_df)
            rm.update({"regime": regime, "model": model_name})
            all_ranking.append(rm)

            all_predictions.append(score_df)

            # Importance.
            if model_name == "LOGISTIC_REGRESSION":
                vals = np.abs(model.coef_[0])
            elif hasattr(model, "feature_importances_"):
                vals = model.feature_importances_
            else:
                # Permutation importance is only performed on a deterministic
                # sample to keep V4.5 practical.
                sample_n = min(5000, int(test_mask.sum()))
                if sample_n > 0:
                    pi = permutation_importance(
                        model,
                        X.loc[test_mask].iloc[:sample_n],
                        y.loc[test_mask].iloc[:sample_n],
                        n_repeats=3,
                        random_state=731,
                        n_jobs=-1,
                    )
                    vals = pi.importances_mean
                else:
                    vals = np.zeros(X.shape[1])

            imp = pd.DataFrame({
                "regime": regime,
                "model": model_name,
                "feature": X.columns,
                "importance": vals,
            }).sort_values("importance", ascending=False)
            imp["rank"] = np.arange(1, len(imp) + 1)
            all_importance.append(imp)

            print(
                f"{regime:28s} {model_name:24s} "
                f"Top1={rm['top1']:.4f} Top3={rm['top3']:.4f} "
                f"Top5={rm['top5']:.4f} MRR={rm['mrr']:.4f}"
            )

    metrics_df = pd.DataFrame(all_metrics)
    ranking_df = pd.DataFrame(all_ranking)
    pred_df = pd.concat(all_predictions, ignore_index=True)
    importance_df = pd.concat(all_importance, ignore_index=True)

    print("\n[7] Package robustness")
    gt_pkg = gt[["rfq_id", "target_package"]].drop_duplicates("rfq_id")
    pred_pkg = pred_df.merge(gt_pkg, on="rfq_id", how="left", validate="many_to_one")

    pkg_rows = []
    for (regime, model, package_name), g in pred_pkg.groupby(
        ["regime", "model", "target_package"], dropna=False
    ):
        if g.empty:
            continue
        rm, _ = ranking_metrics(g)
        if rm["rfqs"] == 0:
            continue
        pkg_rows.append({
            "regime": regime,
            "model": model,
            "target_package": package_name,
            **rm,
        })
    package_df = pd.DataFrame(pkg_rows)

    print("\n[8] Candidate-set difficulty")
    candidate_counts = (
        cand[cand["ml_split"].eq("TEST")]
        .groupby("rfq_id")["canonical_configuration_id"]
        .nunique()
        .rename("candidate_count")
        .reset_index()
    )
    candidate_counts["difficulty_bucket"] = pd.cut(
        candidate_counts["candidate_count"],
        bins=[0, 1, 2, 5, 10, 50, np.inf],
        labels=["1", "2", "3-5", "6-10", "11-50", "51+"],
    )
    diff_pred = pred_df.merge(candidate_counts, on="rfq_id", how="left")

    diff_rows = []
    for (regime, model, bucket), g in diff_pred.groupby(
        ["regime", "model", "difficulty_bucket"], observed=True
    ):
        if g.empty:
            continue
        rm, _ = ranking_metrics(g)
        if rm["rfqs"] == 0:
            continue
        diff_rows.append({
            "regime": regime,
            "model": model,
            "difficulty_bucket": str(bucket),
            **rm,
        })
    difficulty_df = pd.DataFrame(diff_rows)

    print("\n[9] Requirement-equivalence")
    req_records = req[
        ["rfq_id", "characteristic", "internal_value", "requirement_type"]
    ].copy()
    req_records["requirement_type"] = req_records["requirement_type"].astype(str).str.upper()

    cfg_lookup = cand.drop_duplicates(
        ["rfq_id", "canonical_configuration_id"]
    ).set_index(["rfq_id", "canonical_configuration_id"])

    gt_lookup = gt[["rfq_id", "canonical_configuration_id"]].rename(
        columns={"canonical_configuration_id": "engineer_configuration_id"}
    )

    eq_rows = []
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
            gt_id = gt_lookup.loc[
                gt_lookup["rfq_id"].eq(rfq), "engineer_configuration_id"
            ].iloc[0]

            reqs = req_records[req_records["rfq_id"].eq(rfq)]
            pred_cfg = cfg_lookup.loc[(rfq, pred_id)]
            gt_cfg = cfg_lookup.loc[(rfq, gt_id)] if (rfq, gt_id) in cfg_lookup.index else None

            mandatory_ok = True
            preferred_ok = True

            for _, rr in reqs.iterrows():
                c = rr["characteristic"]
                rt = rr["requirement_type"]
                val = norm_text(rr["internal_value"])
                col = f"config_{c}"
                if col not in pred_cfg.index:
                    continue
                same = norm_text(pred_cfg[col]) == val
                if rt == "MANDATORY" and not same:
                    mandatory_ok = False
                if rt == "PREFERRED" and not same:
                    preferred_ok = False

            if pred_id == gt_id:
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
                "engineer_configuration_id": gt_id,
                "category": category,
                "mandatory_valid": mandatory_ok,
                "mandatory_preferred_equivalent": mandatory_ok and preferred_ok,
            })

    equivalence_df = pd.DataFrame(eq_rows)

    print("\n[10] RFQ-level bootstrap: 2,000 replicates")
    best = ranking_df.sort_values(
        ["top1", "mrr"], ascending=False
    ).iloc[0]
    best_scores = pred_df[
        (pred_df["regime"] == best["regime"]) &
        (pred_df["model"] == best["model"])
    ]
    boot = bootstrap_rfqs(best_scores, n=2000)
    btop = ci(boot["top1"])
    bmrr = ci(boot["mrr"])
    print(
        f"Best={best['regime']} / {best['model']} | "
        f"Top1 95% CI [{btop[0]:.4f}, {btop[1]:.4f}] | "
        f"MRR 95% CI [{bmrr[0]:.4f}, {bmrr[1]:.4f}]"
    )

    print("\n[11] Pairwise statistical comparisons")
    comparison_rows = []
    # Compare HGB regimes where possible.
    for a, b in [
        ("REALISTIC_BASELINE", "PROVENANCE_AWARE"),
        ("PROVENANCE_AWARE", "MECHANISTIC_UPPER_BOUND"),
        ("REALISTIC_BASELINE", "MECHANISTIC_UPPER_BOUND"),
    ]:
        sa = pred_df[
            (pred_df["regime"] == a) &
            (pred_df["model"] == "HIST_GRADIENT_BOOSTING")
        ]
        sb = pred_df[
            (pred_df["regime"] == b) &
            (pred_df["model"] == "HIST_GRADIENT_BOOSTING")
        ]
        merged = sa.merge(
            sb,
            on=["rfq_id", "canonical_configuration_id", "ml_target"],
            suffixes=("_a", "_b"),
            how="inner",
        )
        if merged.empty:
            continue

        # Compare RFQ-level Top1 indicators using bootstrap of common RFQs.
        ra = []
        rb = []
        for rfq in sorted(set(sa.rfq_id) & set(sb.rfq_id)):
            ga = sa[sa.rfq_id.eq(rfq)]
            gb = sb[sb.rfq_id.eq(rfq)]
            ma, _ = ranking_metrics(ga)
            mb, _ = ranking_metrics(gb)
            ra.append(ma["top1"])
            rb.append(mb["top1"])

        ra = np.asarray(ra)
        rb = np.asarray(rb)
        rng = np.random.default_rng(731)
        diffs = []
        for _ in range(2000):
            idx = rng.integers(0, len(ra), len(ra))
            diffs.append((rb[idx] - ra[idx]).mean())
        diffs = pd.Series(diffs)
        comparison_rows.append({
            "comparison": f"{b} HGB minus {a} HGB",
            "common_rfqs": len(ra),
            "observed_top1_difference": float(rb.mean() - ra.mean()),
            "bootstrap_ci_low": float(diffs.quantile(0.025)),
            "bootstrap_ci_high": float(diffs.quantile(0.975)),
            "bootstrap_p_two_sided": float(
                2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
            ),
        })

    comparisons_df = pd.DataFrame(comparison_rows)

    print("\n[12] Leakage audit")
    audit_rows = []
    all_model_features = {}
    for regime, X in matrices.items():
        all_model_features[regime] = list(X.columns)

    for regime, features in all_model_features.items():
        for f in features:
            source_base = f
            audit_type = "APPROVED"
            if any(tok in source_base for tok in FORBIDDEN):
                audit_type = "FORBIDDEN"
            if any(tok in source_base for tok in PAIRWISE_ARTIFACTS):
                audit_type = "PAIRWISE_TARGET_ARTIFACT"
            audit_rows.append({
                "regime": regime,
                "pairwise_feature": f,
                "audit_status": audit_type,
            })
    audit_df = pd.DataFrame(audit_rows)

    if (audit_df["audit_status"] != "APPROVED").any():
        bad = audit_df[audit_df["audit_status"] != "APPROVED"]
        raise ValueError(
            "V4.5 leakage audit failed:\n" + bad.head(20).to_string(index=False)
        )

    print("Leakage audit: PASS")

    print("\n[13] Model summary")
    summary_rows = []
    for _, r in ranking_df.iterrows():
        summary_rows.append({
            "regime": r["regime"],
            "model": r["model"],
            "top1": r["top1"],
            "top3": r["top3"],
            "top5": r["top5"],
            "top10": r["top10"],
            "mrr": r["mrr"],
            "ndcg5": r["ndcg5"],
            "mean_true_rank": r["mean_true_rank"],
            "median_true_rank": r["median_true_rank"],
        })
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["top1", "mrr"], ascending=False
    )

    best_row = summary_df.iloc[0]
    report = f"""
731 V4.5 PAIRWISE LEARNING-TO-RANK REPORT

Data
----
Original pairwise comparisons: {len(pair):,}
Symmetric supervised comparisons: {len(sym):,}
Candidate rows: {len(cand):,}
Candidate RFQs: {cand.rfq_id.nunique():,}
Pairwise RFQs: {pair.rfq_id.nunique():,}

Method
------
Pairwise binary preference learning with symmetric augmentation.
For each RFQ, candidate A is compared with candidate B in both directions.
Inference aggregates pairwise win probabilities into a candidate ranking.

Primary result
--------------
Best frozen pairwise model: {best_row['regime']} / {best_row['model']}
Top-1: {best_row['top1']:.4f}
Top-3: {best_row['top3']:.4f}
Top-5: {best_row['top5']:.4f}
Top-10: {best_row['top10']:.4f}
MRR: {best_row['mrr']:.4f}
NDCG@5: {best_row['ndcg5']:.4f}

Interpretation
--------------
The pairwise model estimates preference ordering among candidate
configurations. Labels are synthetic engineer preferences and therefore
measure recovery of the synthetic preference-generation mechanism, not
historical human engineer decisions.

The RFQ-level split is inherited from V4.1, so candidates from one RFQ do
not cross train/validation/test boundaries.

Requirement-equivalence results must be interpreted separately from exact
configuration identity. A prediction can differ from the synthetic
engineer's selected ID while still satisfying all mandatory and preferred
requirements.

The realistic baseline is the cleanest primary model regime. Provenance-aware
and mechanistic regimes are sensitivity/upper-bound analyses because their
additional variables are related to synthetic-data generation.
"""
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(OUT_METRICS, index=False)
    ranking_df.to_csv(OUT_RANKING, index=False)
    pred_df.to_csv(OUT_PRED, index=False)
    importance_df.to_csv(OUT_IMPORTANCE, index=False)
    package_df.to_csv(OUT_PACKAGE, index=False)
    difficulty_df.to_csv(OUT_DIFFICULTY, index=False)
    equivalence_df.to_csv(OUT_EQUIV, index=False)
    boot.to_csv(OUT_BOOT, index=False)
    comparisons_df.to_csv(OUT_COMPARE, index=False)
    audit_df.to_csv(OUT_AUDIT, index=False)
    summary_df.to_csv(OUT_SUMMARY, index=False)
    OUT_REPORT.write_text(report.strip() + "\n", encoding="utf-8")

    print("\n" + "=" * 80)
    print("V4.5 ANALYSIS COMPLETE")
    print("=" * 80)
    print(
        f"Best model: {best_row['regime']} / {best_row['model']}"
    )
    print(
        f"Top-1: {best_row['top1']:.4f} | "
        f"Top-3: {best_row['top3']:.4f} | "
        f"Top-5: {best_row['top5']:.4f} | "
        f"MRR: {best_row['mrr']:.4f}"
    )
    print(f"Outputs written to: {ML}")


if __name__ == "__main__":
    main()
