#!/usr/bin/env python3
"""
731 V4.4 — Model Interpretability, Robustness, Equivalence & Error Analysis

Consumes frozen V4.3 predictions. Does not retrain models.

Analyses:
- model/regime comparison
- feature importance
- package robustness
- candidate-count difficulty
- exact preference vs requirement-equivalent validity
- error taxonomy
- RFQ-level bootstrap confidence intervals
- paired bootstrap model/regime comparisons
- leakage stress test
- thesis-ready summary/report/figures

Synthetic-data rule:
The target is a synthetic engineer choice. Results measure recovery of the
synthetic preference mechanism, not historical human engineer behaviour.
"""

from __future__ import annotations

from pathlib import Path
import math
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
ML = ROOT / "data/processed/ml"
OUT = ML
FIG = OUT / "figures"

FILES = {
    "dataset": ML / "731_v4_1_ml_candidate_dataset.csv",
    "pred": ML / "731_v4_3_predictions.csv",
    "ranking": ML / "731_v4_3_ranking_metrics.csv",
    "importance": ML / "731_v4_3_feature_importance.csv",
    "summary": ML / "731_v4_3_model_summary.csv",
    "manifest": ML / "731_v4_2_1_model_feature_manifest.csv",
    "req": ROOT / "data/synthetic/rfqs/731_rfq_requirements_v2.csv",
    "gt": ROOT / "data/synthetic/rfqs/731_rfq_ground_truth_v2.csv",
    "config": ROOT / "data/synthetic/configurations/731_synthetic_configurations_v5.csv",
}

O = {
    "model": OUT / "731_v4_4_model_comparison.csv",
    "importance": OUT / "731_v4_4_feature_importance.csv",
    "package": OUT / "731_v4_4_package_performance.csv",
    "difficulty": OUT / "731_v4_4_candidate_difficulty.csv",
    "errors": OUT / "731_v4_4_error_analysis.csv",
    "error_summary": OUT / "731_v4_4_error_summary.csv",
    "equivalence": OUT / "731_v4_4_requirement_equivalence.csv",
    "ablation": OUT / "731_v4_4_regime_ablation.csv",
    "bootstrap": OUT / "731_v4_4_bootstrap_metrics.csv",
    "stats": OUT / "731_v4_4_statistical_comparisons.csv",
    "leakage": OUT / "731_v4_4_leakage_stress_test.csv",
    "thesis": OUT / "731_v4_4_thesis_summary.csv",
    "report": OUT / "731_v4_4_report.txt",
}

CHARS = [
    "devCategory","characteristic","housing","powerSupply","numberOfChannels",
    "explosionApproval","protectionArea","certification","dataInterface",
    "stromSchaltbar","stromEingaenge","temperaturEingaenge",
    "binaerDigitalOpenColl_MN","binaerOpenColl_MP","waveInjector",
    "dev_advMeterVerification","dynamicGasMaster","customUserFluid",
    "steamApplication",
]

FORBIDDEN = {
    "ml_target","ground_truth_configuration_id",
    "is_ground_truth_configuration","engineer_rank",
    "synthetic_engineer_score","synthetic_engineer_utility",
    "tie_break_jitter","is_synthetic_engineer_choice",
}

EVAL_ONLY = {
    "rfq_id","customer_id","canonical_configuration_id",
    "synthetic_record_id","source_canonical_configuration_id",
    "ml_split","candidate_count","pairwise_comparison_count",
    "target_package","package_context","config_package_context",
}

MECHANISTIC = {
    "unmentioned_features","source_frequency_score",
    "real_proximity_score","low_change_score",
    "observed_configuration_score","evidence_score_normalized",
}

PROVENANCE = {
    "record_type","generation_method","evidence_level",
    "source_frequency","nearest_real_distance",
    "num_changed_characteristics","evidence_score",
}

MANDATORY = {"MANDATORY","REQUIRED","MUST"}
PREFERRED = {"PREFERRED","PREFERENCE","PREFER"}
BOOT_REPS = 2000
SEED = 73144


def need(p):
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {p}")


def norm(x):
    if pd.isna(x):
        return "NOVALUE"
    s = str(x).strip()
    return "NOVALUE" if not s or s.lower() in {"nan","none","null"} else s


def upper(x):
    return norm(x).upper()


def package(x):
    return upper(x)


def load():
    # Package robustness is a post-hoc evaluation/stratification analysis.
    # target_package is already available in the frozen ground-truth file.
    # It is NOT used as a predictive model feature.
    for p in FILES.values():
        need(p)
    frames = {k: pd.read_csv(v, low_memory=False) for k,v in FILES.items()}
    for f in frames.values():
        if "rfq_id" in f:
            f["rfq_id"] = f["rfq_id"].astype(str).str.strip()
    for k in ["dataset","pred","gt","config"]:
        f = frames[k]
        if "canonical_configuration_id" in f:
            f["canonical_configuration_id"] = (
                f["canonical_configuration_id"].astype(str).str.strip()
            )
    return frames


def ranking_metrics(g):
    hits = {1: [],3: [],5: [],10: []}
    rr, ndcg = [], []
    for _, x in g.groupby("rfq_id", sort=False):
        x = x.sort_values(
            ["model_probability","canonical_configuration_id"],
            ascending=[False,True]
        )
        pos = np.flatnonzero(x["ml_target"].astype(int).to_numpy() == 1)
        if len(pos) != 1:
            continue
        rank = int(pos[0]) + 1
        for k in hits:
            hits[k].append(int(rank <= k))
        rr.append(1/rank)
        ndcg.append(1/math.log2(rank+1) if rank <= 5 else 0)
    return {
        "rfqs":len(rr),
        "top1":np.mean(hits[1]) if rr else np.nan,
        "top3":np.mean(hits[3]) if rr else np.nan,
        "top5":np.mean(hits[5]) if rr else np.nan,
        "top10":np.mean(hits[10]) if rr else np.nan,
        "mrr":np.mean(rr) if rr else np.nan,
        "ndcg5":np.mean(ndcg) if rr else np.nan,
    }


def bootstrap(g, reps=BOOT_REPS, seed=SEED):
    ids = g.rfq_id.drop_duplicates().to_numpy()
    groups = {i:x for i,x in g.groupby("rfq_id", sort=False)}
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(reps):
        sample = rng.choice(ids, len(ids), replace=True)
        x = pd.concat([groups[i] for i in sample], ignore_index=True)
        m = ranking_metrics(x)
        rows.append({"bootstrap":b+1, **m})
    return pd.DataFrame(rows)


def ci(s):
    s = pd.Series(s).dropna()
    return (s.quantile(.025), s.quantile(.975)) if len(s) else (np.nan,np.nan)


def paired_bootstrap(a,b,reps=BOOT_REPS,seed=SEED):
    common = sorted(set(a.rfq_id)&set(b.rfq_id))
    ag = {i:x for i,x in a.groupby("rfq_id",sort=False)}
    bg = {i:x for i,x in b.groupby("rfq_id",sort=False)}
    rng = np.random.default_rng(seed)
    rows=[]
    for j in range(reps):
        ids=rng.choice(common,len(common),replace=True)
        aa=pd.concat([ag[i] for i in ids],ignore_index=True)
        bb=pd.concat([bg[i] for i in ids],ignore_index=True)
        ma,mb=ranking_metrics(aa),ranking_metrics(bb)
        rows.append({
            "bootstrap":j+1,
            "top1":ma["top1"]-mb["top1"],
            "top3":ma["top3"]-mb["top3"],
            "top5":ma["top5"]-mb["top5"],
            "mrr":ma["mrr"]-mb["mrr"],
            "ndcg5":ma["ndcg5"]-mb["ndcg5"],
        })
    return pd.DataFrame(rows)


def req_lookup(req):
    required={"rfq_id","characteristic","internal_value","requirement_type"}
    missing=sorted(required-set(req.columns))
    if missing:
        raise ValueError(f"Requirements file missing expected columns: {missing}")
    out={}
    for rid,g in req.groupby("rfq_id",sort=False):
        rows=[]
        for _,r in g.iterrows():
            t=upper(r["requirement_type"])
            typ="MANDATORY" if t in MANDATORY else "PREFERRED" if t in PREFERRED else t
            rows.append({
                "characteristic":str(r["characteristic"]).strip(),
                "value":norm(r["internal_value"]),
                "type":typ
            })
        out[rid]=rows
    return out


def cfg_lookup(cfg):
    d={}
    for _,r in cfg.iterrows():
        cid=str(r["canonical_configuration_id"])
        if cid not in d:
            d[cid]=r
    return d


def satisfies(row, requirements):
    mt=ms=mv=pt=ps=pdiff=0
    for r in requirements:
        ch=r["characteristic"]
        val=norm(row.get(ch,"NOVALUE")) if row is not None else "NOVALUE"
        if r["type"]=="MANDATORY":
            mt+=1
            if val==r["value"]: ms+=1
            else: mv+=1
        elif r["type"]=="PREFERRED":
            pt+=1
            if val==r["value"]: ps+=1
            else: pdiff+=1
    return {
        "mandatory_total":mt,"mandatory_satisfied":ms,
        "mandatory_violations":mv,"preferred_total":pt,
        "preferred_satisfied":ps,"preferred_differences":pdiff,
        "mandatory_valid":mv==0,
        "mandatory_preferred_valid":mv==0 and pdiff==0,
    }


def difficulty(n):
    if n<=1:return "1"
    if n<=5:return "2-5"
    if n<=10:return "6-10"
    if n<=50:return "11-50"
    if n<=100:return "51-100"
    return "101+"


def top1(pred):
    return (
        pred.sort_values(
            ["regime","model","rfq_id","model_probability",
             "canonical_configuration_id"],
            ascending=[True,True,True,False,True]
        )
        .groupby(["regime","model","rfq_id"],as_index=False,sort=False)
        .head(1).copy()
    )


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    print("="*80)
    print("731 V4.4 MODEL INTERPRETABILITY / ROBUSTNESS / ERROR ANALYSIS")
    print("="*80)

    f=load()
    df,pred,old_rank,imp,summary,manifest,req,gt,cfg=[f[k] for k in
        ["dataset","pred","ranking","importance","summary","manifest",
         "req","gt","config"]]

    print("\n[1] Inputs")
    print(f"Dataset rows:      {len(df):,}")
    print(f"Prediction rows:   {len(pred):,}")
    print(f"Prediction RFQs:   {pred.rfq_id.nunique():,}")
    print(f"Regimes:           {pred.regime.nunique():,}")
    print(f"Models:            {pred.model.nunique():,}")
    print(f"Requirements:      {len(req):,}")
    print(f"Configurations:    {len(cfg):,}")

    # Validation
    required={"rfq_id","canonical_configuration_id","ml_target",
              "regime","model","model_probability"}
    missing=sorted(required-set(pred.columns))
    if missing: raise ValueError(f"Prediction schema missing: {missing}")
    # V4.3 predictions contain one prediction set for every
    # RFQ × regime × model combination. Therefore each RFQ has
    # multiple positives across the complete prediction file
    # (one positive for each of the 9 regime/model combinations).
    # The correct integrity check is at RFQ × regime × model level.
    target_counts = (
        pred.groupby(["rfq_id", "regime", "model"])["ml_target"]
        .sum()
    )
    if not (target_counts == 1).all():
        bad = target_counts[target_counts != 1]
        raise ValueError(
            "Prediction target integrity failed at RFQ × regime × model level. "
            f"Bad groups: {len(bad):,}"
        )
    expected_groups = (
        pred[["rfq_id", "regime", "model"]]
        .drop_duplicates()
        .shape[0]
    )
    expected_rfqs = pred["rfq_id"].nunique()
    if expected_groups != expected_rfqs * pred["regime"].nunique() * pred["model"].nunique():
        raise ValueError(
            "Prediction group structure is inconsistent: expected one "
            "RFQ × regime × model group for every combination."
        )
    split_cross=df.groupby("rfq_id")["ml_split"].nunique()
    if (split_cross>1).any():
        raise ValueError("RFQ split leakage detected.")

    # Model comparison
    print("\n[2] Model comparison")
    rows=[]
    for (reg,model),g in pred.groupby(["regime","model"],sort=False):
        m=ranking_metrics(g)
        rows.append({"regime":reg,"model":model,**m})
    model_df=pd.DataFrame(rows)
    model_df.to_csv(O["model"],index=False)
    print(model_df.to_string(index=False))

    # Importance aggregation
    print("\n[3] Feature importance")
    if not imp.empty:
        imp=imp.copy()
        imp["base_feature"]=(
            imp["transformed_feature"].astype(str)
            .str.replace(r"^(num|cat)__","",regex=True)
            .str.split("__").str[0]
        )
        imp2=(imp.groupby(["regime","model","base_feature"],as_index=False)
              .agg(transformed_feature_count=("transformed_feature","count"),
                   total_absolute_importance=("absolute_importance","sum"),
                   mean_absolute_importance=("absolute_importance","mean"),
                   max_absolute_importance=("absolute_importance","max")))
        imp2["importance_rank"]=(
            imp2.groupby(["regime","model"])["total_absolute_importance"]
            .rank(method="first",ascending=False).astype(int)
        )
        imp2=imp2.sort_values(["regime","model","importance_rank"])
    else:
        imp2=pd.DataFrame()
    imp2.to_csv(O["importance"],index=False)

    # Package
    print("\n[4] Package robustness")
    print("Package context source: ground-truth target_package (post-hoc only)")
    # target_package is intentionally absent from the V4.1 ML dataset.
    # Recover it from the frozen ground-truth file for post-hoc package
    # stratification only. It is never used as a predictive feature.
    if "target_package" not in gt.columns:
        raise ValueError(
            "The ground-truth file is missing target_package; "
            "package robustness analysis cannot be performed."
        )
    pkg_source = gt[["rfq_id","target_package"]].copy()
    pkgmap=pkg_source.drop_duplicates("rfq_id").copy()
    pkgmap["package"]=pkgmap["target_package"].map(package)
    pp=pred.merge(pkgmap[["rfq_id","package"]],on="rfq_id",
                  how="left",validate="many_to_one")
    rows=[]
    for (reg,model,pkg),g in pp.groupby(["regime","model","package"],sort=False):
        m=ranking_metrics(g)
        counts=g.groupby("rfq_id").size()
        rows.append({"regime":reg,"model":model,"package":pkg,**m,
                     "candidate_rows":len(g),
                     "mean_candidates":counts.mean(),
                     "median_candidates":counts.median()})
    package_df=pd.DataFrame(rows)
    package_df.to_csv(O["package"],index=False)

    # Difficulty
    print("\n[5] Candidate-set difficulty")
    # V4.3 predictions contain 3 regimes × 3 models for each RFQ.
    # Counting prediction rows would inflate the true candidate pool by 9×.
    # The V4.1 ML dataset has one row per RFQ × candidate.
    candidate_counts = (
        df.groupby("rfq_id")["canonical_configuration_id"]
          .nunique()
          .rename("candidate_count")
          .reset_index()
    )
    counts = candidate_counts.copy()
    counts["difficulty_bucket"] = counts["candidate_count"].map(difficulty)

    pred["candidate_count"] = pred["rfq_id"].map(
        candidate_counts.set_index("rfq_id")["candidate_count"]
    )
    if pred["candidate_count"].isna().any():
        raise ValueError(
            "Candidate-count mapping failed for one or more prediction RFQs."
        )
    pdiff=pred.merge(counts,on="rfq_id",validate="many_to_one")
    order=["1","2-5","6-10","11-50","51-100","101+"]
    rows=[]
    for (reg,model,bucket),g in pdiff.groupby(
        ["regime","model","difficulty_bucket"],sort=False):
        m=ranking_metrics(g)
        rows.append({"regime":reg,"model":model,
                     "difficulty_bucket":bucket,
                     "bucket_order":order.index(bucket),
                     **m,"rfq_count":g.rfq_id.nunique(),
                     "mean_candidates":g.groupby("rfq_id").size().mean()})
    difficulty_df=pd.DataFrame(rows).sort_values(
        ["model","regime","bucket_order"])
    difficulty_df.to_csv(O["difficulty"],index=False)

    # Equivalence / error analysis
    print("\n[6] Requirement-equivalence analysis")
    reqs=req_lookup(req)
    cfgs=cfg_lookup(cfg)
    gtmap=(gt[["rfq_id","canonical_configuration_id","target_package",
               "record_type","evidence_level"]]
           .rename(columns={"canonical_configuration_id":
                            "synthetic_engineer_configuration_id",
                            "target_package":"ground_truth_package"})
           .drop_duplicates("rfq_id"))
    t=top1(pred).merge(gtmap,on="rfq_id",how="left",validate="many_to_one")

    eq=[]
    for _,r in t.iterrows():
        rid=r.rfq_id
        pid=str(r.canonical_configuration_id)
        gid=str(r.synthetic_engineer_configuration_id)
        reqr=reqs.get(rid,[])
        pc=cfgs.get(pid)
        gc=cfgs.get(gid)
        ps=satisfies(pc,reqr)
        gs=satisfies(gc,reqr)
        exact=pid==gid
        mand=bool(ps["mandatory_valid"])
        pref=bool(ps["mandatory_preferred_valid"])
        if exact: outc="EXACT_SYNTHETIC_ENGINEER_CHOICE"
        elif pref: outc="REQUIREMENT_EQUIVALENT_ALTERNATIVE"
        elif mand: outc="MANDATORY_VALID_BUT_PREFERRED_DIFFERENCE"
        else: outc="MANDATORY_VIOLATION"
        eq.append({
            "rfq_id":rid,"regime":r.regime,"model":r.model,
            "package":package(r.ground_truth_package),
            "predicted_configuration_id":pid,
            "synthetic_engineer_configuration_id":gid,
            "model_probability":r.model_probability,
            "candidate_count":int(counts.loc[
                counts.rfq_id==rid,"candidate_count"].iloc[0]),
            "exact_choice":exact,
            "mandatory_equivalent":mand,
            "mandatory_preferred_equivalent":pref,
            "predicted_mandatory_total":ps["mandatory_total"],
            "predicted_mandatory_satisfied":ps["mandatory_satisfied"],
            "predicted_mandatory_violations":ps["mandatory_violations"],
            "predicted_preferred_total":ps["preferred_total"],
            "predicted_preferred_satisfied":ps["preferred_satisfied"],
            "predicted_preferred_differences":ps["preferred_differences"],
            "ground_truth_mandatory_valid":gs.get("mandatory_valid",np.nan),
            "ground_truth_mandatory_preferred_valid":
                gs.get("mandatory_preferred_valid",np.nan),
            "outcome_category":outc
        })
    eqdf=pd.DataFrame(eq)
    eqdf.to_csv(O["equivalence"],index=False)

    errors=eqdf[~eqdf.exact_choice].copy()
    def diffs(r):
        a=cfgs.get(str(r.synthetic_engineer_configuration_id))
        b=cfgs.get(str(r.predicted_configuration_id))
        if a is None or b is None:return ""
        return "|".join(ch for ch in CHARS if norm(a.get(ch))!=norm(b.get(ch)))
    errors["changed_characteristics"]=errors.apply(diffs,axis=1)
    errors["changed_characteristic_count"]=errors.changed_characteristics.map(
        lambda x:0 if not x else len(x.split("|")))
    errors.to_csv(O["errors"],index=False)
    es=(eqdf.groupby(["regime","model","outcome_category"],as_index=False)
        .agg(rfqs=("rfq_id","nunique"),
             mean_candidate_count=("candidate_count","mean")))
    totals=(eqdf.groupby(["regime","model"]).rfq_id.nunique()
            .rename("total_rfqs").reset_index())
    es=es.merge(totals,on=["regime","model"])
    es["rate"]=es.rfqs/es.total_rfqs
    es.to_csv(O["error_summary"],index=False)

    # HGB ablation
    print("\n[7] HGB regime ablation")
    h=model_df[model_df.model=="HIST_GRADIENT_BOOSTING"].set_index("regime")
    ab=[]
    pairs=[("REALISTIC_BASELINE","PROVENANCE_AWARE"),
           ("PROVENANCE_AWARE","MECHANISTIC_UPPER_BOUND"),
           ("REALISTIC_BASELINE","MECHANISTIC_UPPER_BOUND")]
    for lo,hi in pairs:
        ab.append({
            "model":"HIST_GRADIENT_BOOSTING",
            "lower_regime":lo,"upper_regime":hi,
            "top1_lower":h.loc[lo,"top1"],"top1_upper":h.loc[hi,"top1"],
            "delta_top1":h.loc[hi,"top1"]-h.loc[lo,"top1"],
            "top3_delta":h.loc[hi,"top3"]-h.loc[lo,"top3"],
            "mrr_delta":h.loc[hi,"mrr"]-h.loc[lo,"mrr"],
            "ndcg5_delta":h.loc[hi,"ndcg5"]-h.loc[lo,"ndcg5"]})
    abdf=pd.DataFrame(ab)
    abdf.to_csv(O["ablation"],index=False)

    # Bootstrap
    print(f"\n[8] RFQ-level bootstrap: {BOOT_REPS:,} replicates")
    br=[]
    for (reg,model),g in pred.groupby(["regime","model"],sort=False):
        b=bootstrap(g)
        for metric in ["top1","top3","top5","top10","mrr","ndcg5"]:
            lo,hi=ci(b[metric])
            br.append({"regime":reg,"model":model,"metric":metric,
                       "replicates":len(b),"mean":b[metric].mean(),
                       "ci95_lower":lo,"ci95_upper":hi})
    bootdf=pd.DataFrame(br)
    bootdf.to_csv(O["bootstrap"],index=False)

    # Paired comparisons
    print("\n[9] Paired bootstrap comparisons")
    lookup={(r,m):g for (r,m),g in pred.groupby(["regime","model"],sort=False)}
    specs=[
        (("PROVENANCE_AWARE","HIST_GRADIENT_BOOSTING"),
         ("REALISTIC_BASELINE","HIST_GRADIENT_BOOSTING")),
        (("MECHANISTIC_UPPER_BOUND","HIST_GRADIENT_BOOSTING"),
         ("PROVENANCE_AWARE","HIST_GRADIENT_BOOSTING")),
        (("MECHANISTIC_UPPER_BOUND","HIST_GRADIENT_BOOSTING"),
         ("REALISTIC_BASELINE","HIST_GRADIENT_BOOSTING")),
        (("REALISTIC_BASELINE","HIST_GRADIENT_BOOSTING"),
         ("REALISTIC_BASELINE","RANDOM_FOREST")),
        (("REALISTIC_BASELINE","HIST_GRADIENT_BOOSTING"),
         ("REALISTIC_BASELINE","LOGISTIC_REGRESSION")),
    ]
    sr=[]
    for ak,bk in specs:
        if ak not in lookup or bk not in lookup: continue
        a,b=lookup[ak],lookup[bk]
        observed_a,observed_b=ranking_metrics(a),ranking_metrics(b)
        pb=paired_bootstrap(a,b,seed=SEED+31)
        for metric in ["top1","top3","top5","mrr","ndcg5"]:
            lo,hi=ci(pb[metric])
            sr.append({
                "model_a":f"{ak[0]} / {ak[1]}",
                "model_b":f"{bk[0]} / {bk[1]}",
                "metric":metric,
                "observed_difference":observed_a[metric]-observed_b[metric],
                "bootstrap_ci95_lower":lo,
                "bootstrap_ci95_upper":hi,
                "probability_a_better":(pb[metric]>0).mean()
            })
    stats=pd.DataFrame(sr)
    stats.to_csv(O["stats"],index=False)

    # Leakage stress test
    print("\n[10] Leakage stress test")
    sets={
        "REALISTIC_BASELINE":set(manifest.loc[
            manifest.final_status=="REALISTIC_BASELINE","feature"]),
        "PROVENANCE_AWARE":set(manifest.loc[
            manifest.final_status.isin(["REALISTIC_BASELINE","PROVENANCE_AWARE"]),
            "feature"]),
        "MECHANISTIC_UPPER_BOUND":set(manifest.loc[
            manifest.final_status.isin(
                ["REALISTIC_BASELINE","PROVENANCE_AWARE","MECHANISTIC_ONLY"]),
            "feature"])
    }
    lr=[]
    for reg,s in sets.items():
        fh=sorted(s&FORBIDDEN)
        eh=sorted(s&EVAL_ONLY)
        th=sorted(s&{"ml_target","is_synthetic_engineer_choice"})
        lr.append({"regime":reg,"feature_count":len(s),
                   "forbidden_hits":"|".join(fh),
                   "evaluation_only_hits":"|".join(eh),
                   "target_copy_hits":"|".join(th),
                   "mechanistic_features":len(s&MECHANISTIC),
                   "provenance_features":len(s&PROVENANCE),
                   "governance_pass":not(fh or eh or th)})
    leak=pd.DataFrame(lr)
    leak.to_csv(O["leakage"],index=False)

    # Thesis summary
    best=model_df.sort_values(["top1","mrr","ndcg5"],ascending=False).iloc[0]
    rh=model_df[(model_df.regime=="REALISTIC_BASELINE")&
                (model_df.model=="HIST_GRADIENT_BOOSTING")].iloc[0]
    ph=model_df[(model_df.regime=="PROVENANCE_AWARE")&
                (model_df.model=="HIST_GRADIENT_BOOSTING")].iloc[0]
    mh=model_df[(model_df.regime=="MECHANISTIC_UPPER_BOUND")&
                (model_df.model=="HIST_GRADIENT_BOOSTING")].iloc[0]
    beq=eqdf[(eqdf.regime==best.regime)&(eqdf.model==best.model)]
    thesis=pd.DataFrame([
        ["best_model",f"{best.regime} / {best.model}",
         "Best frozen V4.3 test model by Top-1, then MRR/NDCG@5."],
        ["best_top1",best.top1,
         "Exact recovery of the synthetic engineer-selected candidate."],
        ["realistic_hgb_top1",rh.top1,
         "Primary thesis result under the strict realistic feature regime."],
        ["provenance_gain_over_realistic",ph.top1-rh.top1,
         "Incremental Top-1 gain from provenance/process information."],
        ["mechanistic_gain_over_provenance",mh.top1-ph.top1,
         "Incremental gain from direct synthetic-mechanism variables."],
        ["best_mandatory_equivalent_rate",beq.mandatory_equivalent.mean(),
         "Top-1 predictions satisfying all mandatory requirements."],
        ["best_mandatory_preferred_equivalent_rate",
         beq.mandatory_preferred_equivalent.mean(),
         "Top-1 predictions satisfying mandatory and preferred requirements."],
        ["test_rfqs",pred.rfq_id.nunique(),
         "RFQs evaluated in frozen V4.3 predictions."],
        ["bootstrap_replicates",BOOT_REPS,
         "RFQ-level bootstrap replicates."]
    ],columns=["finding","value","interpretation"])
    thesis.to_csv(O["thesis"],index=False)

    # Figures
    figure_paths=[]
    try:
        import matplotlib.pyplot as plt
        FIG.mkdir(parents=True,exist_ok=True)

        x=model_df.copy()
        x["label"]=x.regime.str.replace("_"," ",regex=False)+"\n"+x.model
        plt.figure(figsize=(12,6))
        plt.bar(x.label,x.top1*100)
        plt.ylabel("Top-1 (%)"); plt.title("731 V4.4 Model / Regime Comparison")
        plt.xticks(rotation=45,ha="right"); plt.tight_layout()
        p=FIG/"731_v4_4_model_comparison_top1.png"; plt.savefig(p,dpi=180); plt.close()
        figure_paths.append(p)

        x=difficulty_df[difficulty_df.model=="HIST_GRADIENT_BOOSTING"]
        plt.figure(figsize=(10,6))
        for reg,g in x.groupby("regime"):
            g=g.sort_values("bucket_order")
            plt.plot(g.difficulty_bucket,g.top1*100,marker="o",label=reg)
        plt.xlabel("Candidate-count bucket"); plt.ylabel("Top-1 (%)")
        plt.title("V4.4 HGB Performance by Candidate-Set Difficulty")
        plt.legend(); plt.tight_layout()
        p=FIG/"731_v4_4_candidate_difficulty.png"; plt.savefig(p,dpi=180); plt.close()
        figure_paths.append(p)

        x=package_df[package_df.model=="HIST_GRADIENT_BOOSTING"]
        plt.figure(figsize=(12,6))
        x=x.sort_values("package")
        plt.bar(x.package,x.top1*100)
        plt.ylabel("Top-1 (%)"); plt.title("V4.4 HGB Top-1 by Package")
        plt.xticks(rotation=45,ha="right"); plt.tight_layout()
        p=FIG/"731_v4_4_package_performance.png"; plt.savefig(p,dpi=180); plt.close()
        figure_paths.append(p)

        if not imp2.empty:
            x=imp2[(imp2.regime=="MECHANISTIC_UPPER_BOUND")&
                   (imp2.model=="HIST_GRADIENT_BOOSTING")].head(15).copy()
            x=x.sort_values("total_absolute_importance")
            plt.figure(figsize=(10,7))
            plt.barh(x.base_feature,x.total_absolute_importance)
            plt.xlabel("Aggregated transformed-feature importance")
            plt.title("V4.4 Top HGB Features")
            plt.tight_layout()
            p=FIG/"731_v4_4_feature_importance_top15.png"; plt.savefig(p,dpi=180); plt.close()
            figure_paths.append(p)
    except Exception as e:
        print(f"Figure generation skipped: {e}")

    # Report
    bb=bootdf[(bootdf.regime==best.regime)&(bootdf.model==best.model)]
    def bline(m):
        r=bb[bb.metric==m]
        if r.empty:return "n/a"
        z=r.iloc[0]
        return f"{z['mean']:.4f} (95% CI {z['ci95_lower']:.4f}–{z['ci95_upper']:.4f})"

    outc=(beq.outcome_category.value_counts(normalize=True)
          .rename("rate").reset_index())
    report=[
        "="*80,
        "731 V4.4 MODEL INTERPRETABILITY / ROBUSTNESS / ERROR ANALYSIS",
        "="*80,"",
        "V4.4 consumes frozen V4.3 predictions; no models are retrained.","",
        "Methodological interpretation:",
        "The target is a SYNTHETIC engineer choice. Results measure recovery",
        "of the synthetic preference mechanism, not historical human decisions.",
        "Exact configuration identity is separated from requirement-equivalent",
        "validity because multiple technically compliant alternatives can exist.","",
        f"Test RFQs: {pred.rfq_id.nunique():,}",
        f"Bootstrap replicates: {BOOT_REPS:,}","",
        "MODEL COMPARISON"
    ]
    for _,r in model_df.iterrows():
        report.append(
            f"{r.regime} | {r.model} | Top1={r.top1:.4f} | "
            f"Top3={r.top3:.4f} | Top5={r.top5:.4f} | Top10={r.top10:.4f} | "
            f"MRR={r.mrr:.4f} | NDCG@5={r.ndcg5:.4f}")
    report += [
        "","PRIMARY HGB ABLATION",
        f"Realistic Top-1: {rh.top1:.4f}",
        f"Provenance Top-1: {ph.top1:.4f}",
        f"Mechanistic Top-1: {mh.top1:.4f}",
        f"Provenance gain: {ph.top1-rh.top1:+.4f}",
        f"Mechanistic gain over provenance: {mh.top1-ph.top1:+.4f}",
        "","BEST-MODEL BOOTSTRAP",
        f"Top-1: {bline('top1')}",
        f"Top-3: {bline('top3')}",
        f"Top-5: {bline('top5')}",
        f"MRR: {bline('mrr')}",
        f"NDCG@5: {bline('ndcg5')}",
        "","BEST-MODEL OUTCOME TAXONOMY"
    ]
    for _,r in outc.iterrows():
        report.append(f"{r.outcome_category}: {r.rate:.4f}")
    report += [
        "","INTERPRETATION",
        "1. Exact Top-1 measures recovery of the synthetic engineer choice.",
        "2. Requirement-equivalent outcomes identify acceptable alternatives.",
        "3. Package and candidate-count analyses reveal difficult operating regions.",
        "4. RFQ-level bootstrap preserves within-RFQ candidate dependence.",
        "5. Regime ablation separates realistic information from synthetic metadata.",
        "6. Mechanistic variables are an upper-bound recovery experiment, not a",
        "   deployment recommendation.",
        "","LEAKAGE GOVERNANCE"
    ]
    for _,r in leak.iterrows():
        report.append(
            f"{r.regime}: PASS={r.governance_pass}; "
            f"forbidden={r.forbidden_hits or 'none'}; "
            f"evaluation_only={r.evaluation_only_hits or 'none'}")
    report += [
        "","LIMITATION",
        "The synthetic preference labels are not historical engineer decisions.",
        "Real engineer decision logs would be required to establish real-world",
        "human decision prediction performance.",
        "","FIGURES"
    ] + [str(p) for p in figure_paths]
    O["report"].write_text("\n".join(report),encoding="utf-8")

    print("\n"+"="*80)
    print("V4.4 ANALYSIS COMPLETE")
    print("="*80)
    print(f"Best frozen model: {best.regime} / {best.model}")
    print(f"Top-1: {best.top1:.4f} | Top-3: {best.top3:.4f} | "
          f"Top-5: {best.top5:.4f} | MRR: {best.mrr:.4f}")
    print(f"Mandatory-equivalent Top-1: {beq.mandatory_equivalent.mean():.4f}")
    print(f"Mandatory+preferred-equivalent Top-1: "
          f"{beq.mandatory_preferred_equivalent.mean():.4f}")
    print("\nOutputs:")
    for p in O.values(): print(f"  {p}")
    for p in figure_paths: print(f"  {p}")


if __name__ == "__main__":
    main()
