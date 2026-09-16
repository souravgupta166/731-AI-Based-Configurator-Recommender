#!/usr/bin/env python3
"""731 V4.2.1 feature governance and leakage audit."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/processed/ml/731_v4_1_ml_candidate_dataset.csv"
OUT_DIR = ROOT / "data/processed/ml"

OUT = {k: OUT_DIR / f"731_v4_2_1_{v}.csv" for k, v in {
    "audit":"feature_leakage_audit", "regime":"feature_regime_summary",
    "profile":"feature_profile", "corr":"feature_correlations",
    "target":"target_relationships", "manifest":"model_feature_manifest",
    "metrics":"audit_metrics"}.items()}
REPORT = OUT_DIR / "731_v4_2_1_audit_report.txt"

FORBIDDEN = {
    "ml_target", "ground_truth_configuration_id", "is_ground_truth_configuration",
    "engineer_rank", "synthetic_engineer_score", "synthetic_engineer_utility",
    "tie_break_jitter", "synthetic_engineer_choice", "ground_truth_engineer_rank",
    "ground_truth_engineer_score", "ground_truth_selected_by_engineer",
}
GROUND_TRUTH_CONTEXT = {
    "rfq_ground_truth_record_type", "rfq_ground_truth_evidence_level",
    "rfq_ground_truth_nearest_real_distance", "rfq_ground_truth_configuration_id",
    "rfq_ground_truth_package", "ground_truth_record_type",
    "ground_truth_evidence_level", "ground_truth_nearest_real_distance",
}
EVALUATION_ONLY = {
    "rfq_id", "customer_id", "canonical_configuration_id", "synthetic_record_id",
    "source_canonical_configuration_id", "ml_split", "candidate_count",
    "pairwise_comparison_count",
}
REALISTIC_CONTEXT = {"target_package", "package_context"}
TECHNICAL = {
    "config_devCategory", "config_characteristic", "config_housing", "config_powerSupply",
    "config_numberOfChannels", "config_explosionApproval", "config_protectionArea",
    "config_certification", "config_dataInterface", "config_stromSchaltbar",
    "config_stromEingaenge", "config_temperaturEingaenge", "config_binaerDigitalOpenColl_MN",
    "config_binaerOpenColl_MP", "config_waveInjector", "config_dev_advMeterVerification",
    "config_dynamicGasMaster", "config_customUserFluid", "config_steamApplication",
}
REQUIREMENT_DERIVED = {
    "mandatory_total", "mandatory_satisfied", "mandatory_violations", "preferred_total",
    "preferred_satisfied", "preferred_differences", "unmentioned_features",
    "mandatory_valid", "mandatory_preferred_valid",
}
PROVENANCE = {
    "record_type", "generation_method", "evidence_level", "source_frequency",
    "nearest_real_distance", "num_changed_characteristics", "evidence_score",
    "preference_label_provenance", "preference_generation_method",
}
MECHANISTIC = {
    "source_frequency_score", "real_proximity_score", "low_change_score",
    "observed_configuration_score", "evidence_score_normalized", "unmentioned_features",
}
TRUE = {"true","1","yes","y","t"}; FALSE = {"false","0","no","n","f"}

def normalize_bool(s):
    if pd.api.types.is_bool_dtype(s): return s.astype("boolean")
    def f(v):
        if pd.isna(v): return pd.NA
        x=str(v).strip().lower()
        if x in TRUE: return True
        if x in FALSE: return False
        return pd.NA
    return s.map(f).astype("boolean")

def normalize_target(s):
    n=pd.to_numeric(s,errors="coerce")
    if n.notna().all() and set(n.unique()).issubset({0,1}): return n.astype(int)
    x=s.astype(str).str.strip().str.lower().map({"true":1,"false":0,"yes":1,"no":0})
    if x.isna().any(): raise ValueError("ml_target contains non-binary values")
    return x.astype(int)

def numeric_like(s):
    if pd.api.types.is_numeric_dtype(s): return True
    return pd.to_numeric(s,errors="coerce").notna().mean() >= .95

def pearson(a,b):
    x=pd.to_numeric(a,errors="coerce"); y=pd.to_numeric(b,errors="coerce"); m=x.notna()&y.notna()
    if m.sum()<2 or x[m].nunique()<2 or y[m].nunique()<2: return np.nan
    return float(x[m].corr(y[m],method="pearson"))

def spearman(a,b):
    x=pd.to_numeric(a,errors="coerce"); y=pd.to_numeric(b,errors="coerce"); m=x.notna()&y.notna()
    if m.sum()<2: return np.nan
    xr=x[m].rank(method="average"); yr=y[m].rank(method="average")
    if xr.nunique()<2 or yr.nunique()<2: return np.nan
    return float(xr.corr(yr,method="pearson"))

def cramers_v(a,b):
    x=pd.Series(a).astype("string"); y=pd.Series(b).astype("string"); m=x.notna()&y.notna(); x=x[m]; y=y[m]
    if len(x)==0: return np.nan
    tab=pd.crosstab(x,y).to_numpy(dtype=float); n=tab.sum(); r,c=tab.shape
    if r<=1 or c<=1 or n<=1: return 0.0
    exp=np.outer(tab.sum(1),tab.sum(0))/n
    chi=np.nansum(np.where(exp>0,(tab-exp)**2/exp,0)); phi=chi/n
    corr=((c-1)*(r-1))/max(n-1,1); phic=max(0,phi-corr)
    rc=r-(r-1)**2/max(n-1,1); kc=c-(c-1)**2/max(n-1,1); den=min(rc-1,kc-1)
    return 0.0 if den<=0 else float(np.sqrt(phic/den))

def classify(c):
    if c in FORBIDDEN: return "FORBIDDEN"
    if c in GROUND_TRUTH_CONTEXT: return "GROUND_TRUTH_CONTEXT"
    if c in EVALUATION_ONLY: return "EVALUATION_ONLY"
    if c in REALISTIC_CONTEXT: return "REALISTIC_CONTEXT"
    if c in TECHNICAL: return "TECHNICAL"
    if c in REQUIREMENT_DERIVED: return "REQUIREMENT_DERIVED"
    if c in PROVENANCE: return "PROVENANCE"
    if c in MECHANISTIC: return "MECHANISTIC"
    return "UNCLASSIFIED"

def proposed(c):
    x=classify(c)
    if x in {"FORBIDDEN","GROUND_TRUTH_CONTEXT","EVALUATION_ONLY","UNCLASSIFIED"}: return "EXCLUDED"
    if x in {"REALISTIC_CONTEXT","TECHNICAL","REQUIREMENT_DERIVED"}: return "REALISTIC_BASELINE"
    if x=="PROVENANCE": return "PROVENANCE_AWARE"
    if x=="MECHANISTIC": return "MECHANISTIC_UPPER_BOUND"
    return "EXCLUDED"

def main():
    print("="*80); print("731 V4.2.1 FEATURE & LEAKAGE AUDIT"); print("="*80)
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    print("\n[1] Checking input file"); print(INPUT)
    if not INPUT.exists(): raise FileNotFoundError(INPUT)
    print("\n[2] Loading V4.1 dataset")
    df=pd.read_csv(INPUT,low_memory=False); print(f"Rows: {len(df):,}\nColumns: {len(df.columns):,}")
    req={"rfq_id","canonical_configuration_id","ml_target","ml_split"}; miss=sorted(req-set(df.columns))
    if miss: raise ValueError(f"Missing required columns: {miss}")
    print("\n[3] Required columns: PASS")
    df["ml_target"]=normalize_target(df["ml_target"]); pos=int((df.ml_target==1).sum()); neg=int((df.ml_target==0).sum())
    print("\n[4] Target"); print(f"Positive: {pos:,}\nNegative: {neg:,}\nPositive rate: {pos/len(df):.4f}")
    for c in df.columns:
        if pd.api.types.is_bool_dtype(df[c]): df[c]=normalize_bool(df[c])
    print("\n[5] Boolean normalization: PASS")
    rfqs=df.rfq_id.nunique(); dup=int(df.duplicated(["rfq_id","canonical_configuration_id"],keep=False).sum())
    print("\n[6] Structural integrity"); print(f"RFQs: {rfqs:,}\nDuplicate RFQ/candidate pairs: {dup:,}")
    if dup: raise ValueError("Duplicate RFQ/candidate pairs detected")
    print("Structural integrity: PASS")

    rows=[]
    for c in df.columns:
        s=df[c]; rows.append({"feature":c,"governance_class":classify(c),"proposed_regime":proposed(c),"dtype":str(s.dtype),"missing_rate":float(s.isna().mean()),"cardinality":int(s.nunique(dropna=True))})
    audit=pd.DataFrame(rows)
    print("\n[7] Columns classified")

    tr=[]
    for c in df.columns:
        if c=="ml_target": continue
        if numeric_like(df[c]):
            p=pearson(df[c],df.ml_target); sp=spearman(df[c],df.ml_target); cv=np.nan; typ="numeric"
        else:
            p=np.nan; sp=np.nan; cv=cramers_v(df[c],df.ml_target); typ="categorical"
        tr.append({"feature":c,"governance_class":classify(c),"regime":proposed(c),"pearson_with_target":p,"spearman_with_target":sp,"categorical_association":cv,"association_type":typ})
    target=pd.DataFrame(tr); target["association_strength"]=target[["spearman_with_target","categorical_association"]].abs().max(axis=1); target["high_association_flag"]=target.association_strength>=.80
    high=target[target.high_association_flag].sort_values("association_strength",ascending=False)
    print("\n[8] Target associations"); print(f"High-association features: {len(high):,}");
    if len(high): print(high[["feature","governance_class","regime","pearson_with_target","spearman_with_target","categorical_association"]].to_string(index=False))

    constants=[c for c in df.columns if c!="ml_target" and df[c].nunique(dropna=False)<=1]; const=set(constants)
    print("\n[9] Constant features",len(constants)); [print("  -",c) for c in constants]
    baseline=[]; prov=[]; mech=[]
    for c in df.columns:
        if c=="ml_target" or c in const: continue
        r=proposed(c)
        if r=="REALISTIC_BASELINE": baseline.append(c)
        elif r=="PROVENANCE_AWARE": prov.append(c)
        elif r=="MECHANISTIC_UPPER_BOUND": mech.append(c)
    upper=sorted(set(baseline+prov+mech))
    print("\n[10] Feature regimes"); print(f"Realistic baseline:      {len(baseline):,}\nProvenance-aware:        {len(prov):,}\nMechanistic upper-bound: {len(upper):,}")
    print("\nRealistic baseline feature list:"); [print("  -",c) for c in baseline]

    manifest=[]
    for c in df.columns:
        if c=="ml_target": continue
        cls=classify(c)
        if c in const: status="EXCLUDED_CONSTANT"
        elif cls=="FORBIDDEN": status="EXCLUDED_FORBIDDEN"
        elif cls=="GROUND_TRUTH_CONTEXT": status="EXCLUDED_GROUND_TRUTH_CONTEXT"
        elif cls=="EVALUATION_ONLY": status="EXCLUDED_EVALUATION_ONLY"
        elif cls=="UNCLASSIFIED": status="EXCLUDED_UNCLASSIFIED"
        elif c in baseline: status="REALISTIC_BASELINE"
        elif c in prov: status="PROVENANCE_AWARE"
        elif c in mech: status="MECHANISTIC_ONLY"
        else: status="EXCLUDED"
        q=target[target.feature==c].iloc[0]
        manifest.append({"feature":c,"governance_class":cls,"final_status":status,"missing_rate":float(df[c].isna().mean()),"cardinality":int(df[c].nunique(dropna=True)),"target_association_strength":q.association_strength,"high_target_association":bool(q.high_association_flag),"is_constant":c in const})
    manifest=pd.DataFrame(manifest)

    nums=[c for c in upper if numeric_like(df[c]) and c not in const]; nd=pd.DataFrame({c:pd.to_numeric(df[c],errors="coerce") for c in nums}); corrrows=[]
    if len(nums)>=2:
        cm=nd.rank(method="average").corr(method="pearson")
        for i,a in enumerate(nums):
            for b in nums[i+1:]:
                v=cm.loc[a,b]
                if pd.notna(v): corrrows.append({"feature_a":a,"feature_b":b,"spearman_correlation":float(v),"abs_spearman":abs(float(v)),"high_correlation_flag":abs(float(v))>=.90})
    corr=pd.DataFrame(corrrows,columns=["feature_a","feature_b","spearman_correlation","abs_spearman","high_correlation_flag"]).sort_values("abs_spearman",ascending=False) if corrrows else pd.DataFrame(columns=["feature_a","feature_b","spearman_correlation","abs_spearman","high_correlation_flag"])
    print("\n[11] Numeric correlation audit"); print(f"Numeric features: {len(nums):,}\nHigh-correlation pairs: {int(corr.high_correlation_flag.sum()):,}")
    if len(corr): print(corr[corr.high_correlation_flag].head(20).to_string(index=False))

    direct=sorted((FORBIDDEN|GROUND_TRUTH_CONTEXT)&set(df.columns)); copies=[]
    for c in df.columns:
        if c=="ml_target" or not numeric_like(df[c]): continue
        x=pd.to_numeric(df[c],errors="coerce"); m=x.notna()
        if m.sum() and set(np.round(x[m].astype(float).unique(),10)).issubset({0.,1.}):
            agr=float((x[m].astype(int)==df.loc[m,"ml_target"]).mean())
            if agr>=.999: copies.append({"feature":c,"agreement_with_target":agr})
    copies=pd.DataFrame(copies)
    baseline_forbidden=len(set(baseline)&(FORBIDDEN|GROUND_TRUTH_CONTEXT)); prov_forbidden=len(set(prov)&(FORBIDDEN|GROUND_TRUTH_CONTEXT)); upper_forbidden=len(set(upper)&(FORBIDDEN|GROUND_TRUTH_CONTEXT)); baseline_eval=len(set(baseline)&EVALUATION_ONLY)
    leakage=(baseline_forbidden==prov_forbidden==upper_forbidden==baseline_eval==0 and len(copies)==0)
    print("\n[12] Leakage validation"); print(f"Forbidden in baseline: {baseline_forbidden}\nForbidden in provenance: {prov_forbidden}\nForbidden in upper bound: {upper_forbidden}\nEvaluation in baseline: {baseline_eval}\nGround-truth fields excluded: {len(set(baseline)&GROUND_TRUTH_CONTEXT)}\nUnclassified: {sum(classify(c)=='UNCLASSIFIED' for c in df.columns)}\nPotential target copies: {len(copies)}")
    print("Leakage governance:","PASS" if leakage else "FAIL")

    profile=pd.DataFrame([{"feature":c,"governance_class":classify(c),"proposed_regime":proposed(c),"dtype":str(df[c].dtype),"rows":len(df),"missing_count":int(df[c].isna().sum()),"missing_rate":float(df[c].isna().mean()),"unique_count":int(df[c].nunique(dropna=True))} for c in df.columns])
    metrics={"rows":len(df),"rfqs":rfqs,"columns":len(df.columns),"positive_rows":pos,"negative_rows":neg,"positive_rate":pos/len(df),"duplicate_rfq_candidate_pairs":dup,"realistic_baseline_feature_count":len(baseline),"provenance_aware_feature_count":len(prov),"mechanistic_upper_bound_feature_count":len(upper),"constant_feature_count":len(constants),"unclassified_feature_count":sum(classify(c)=="UNCLASSIFIED" for c in df.columns),"high_target_association_count":len(high),"numeric_feature_count":len(nums),"high_correlation_pair_count":int(corr.high_correlation_flag.sum()),"direct_forbidden_or_gt_fields_present":len(direct),"potential_target_copy_count":len(copies),"leakage_governance_pass":leakage}
    metricsdf=pd.DataFrame([{"metric":k,"value":v} for k,v in metrics.items()])
    regimes=pd.DataFrame([
        {"regime":"REALISTIC_BASELINE","feature_count":len(baseline),"contains_forbidden":bool(set(baseline)&(FORBIDDEN|GROUND_TRUTH_CONTEXT)),"contains_evaluation_only":bool(set(baseline)&EVALUATION_ONLY)},
        {"regime":"PROVENANCE_AWARE","feature_count":len(prov),"contains_forbidden":bool(set(prov)&(FORBIDDEN|GROUND_TRUTH_CONTEXT)),"contains_evaluation_only":bool(set(prov)&EVALUATION_ONLY)},
        {"regime":"MECHANISTIC_UPPER_BOUND","feature_count":len(upper),"contains_forbidden":bool(set(upper)&(FORBIDDEN|GROUND_TRUTH_CONTEXT)),"contains_evaluation_only":bool(set(upper)&EVALUATION_ONLY)},
    ])
    report="\n".join(["="*80,"731 V4.2.1 FEATURE & LEAKAGE AUDIT REPORT","="*80,"",f"Rows: {len(df):,}",f"RFQs: {rfqs:,}",f"Realistic baseline features: {len(baseline):,}",f"Provenance-aware features: {len(prov):,}",f"Mechanistic upper-bound features: {len(upper):,}",f"Unclassified features: {sum(classify(c) == 'UNCLASSIFIED' for c in df.columns):,}","","Governance decisions:","- target_package/package_context are pre-choice context.","- hidden ground-truth context is excluded from predictive regimes.","- provenance metadata is sensitivity-only.","- synthetic utility components are mechanistic-only.","- identifiers/post-choice fields are evaluation-only.","",f"Leakage governance: {'PASS' if leakage else 'FAIL'}","","V4.3 should use RFQ-level splits and ranking metrics such as Top-K, MRR and NDCG."])
    audit.to_csv(OUT["audit"],index=False); regimes.to_csv(OUT["regime"],index=False); profile.to_csv(OUT["profile"],index=False); corr.to_csv(OUT["corr"],index=False); target.to_csv(OUT["target"],index=False); manifest.to_csv(OUT["manifest"],index=False); metricsdf.to_csv(OUT["metrics"],index=False); REPORT.write_text(report,encoding="utf-8")
    print("\nOutputs:"); [print(" ",p) for p in list(OUT.values())+[REPORT]]
    if not leakage: raise RuntimeError("V4.2.1 leakage governance failed")
    print("\nV4.2.1 complete. Proceed to V4.3 after reviewing the manifest.")

if __name__=="__main__": main()
