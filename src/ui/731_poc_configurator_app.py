#!/usr/bin/env python3
"""
731 Series Intelligent Configurator — Proof of Concept V7 — Data-Grounded Recommendation UI
==========================================================

Run from ~/thesis-configurator:
    streamlit run src/ui/731_poc_configurator_app.py

Architecture
------------
Customer/product context -> technical requirements -> mandatory constraint
filter -> technically valid candidate pool -> pairwise Logistic Regression
ranking -> recommendation + alternatives + explanation + audit trail.

Research safeguards
-------------------
* V4.1 is the ML training candidate table.
* V5 synthetic configurations are the interactive configuration catalogue.
* The UI does not use ground-truth configuration IDs or engineer-choice labels.
* Industry/application are demonstration context only; they do not silently
  change technical requirements.
* Engineer-preference labels are synthetic and are not historical engineer data.
"""

from itertools import combinations
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
ML = ROOT / "data" / "processed" / "ml"
SYN = ROOT / "data" / "synthetic"

V41_CAND_PATH = ML / "731_v4_1_ml_candidate_dataset.csv"
V41_PAIR_PATH = ML / "731_v4_1_ml_pairwise_dataset.csv"
CONFIG_PATH = SYN / "configurations" / "731_synthetic_configurations_v5.csv"
KG_DIR = ROOT / "graphs" / "731_knowledge_graph"
KG_NODES_PATH = KG_DIR / "nodes.csv"
KG_REL_PATH = KG_DIR / "relationships.csv"

TECHNICAL = [
    "devCategory", "characteristic", "housing", "powerSupply",
    "numberOfChannels", "explosionApproval", "protectionArea",
    "certification", "dataInterface", "stromSchaltbar", "stromEingaenge",
    "temperaturEingaenge", "binaerDigitalOpenColl_MN",
    "binaerOpenColl_MP", "waveInjector", "dev_advMeterVerification",
    "dynamicGasMaster", "customUserFluid", "steamApplication",
]

TECHNICAL_CONFIG = [f"config_{c}" for c in TECHNICAL]
REQUIREMENT = [
    "mandatory_total", "mandatory_satisfied", "mandatory_violations",
    "preferred_total", "preferred_satisfied", "preferred_differences",
]

DISPLAY_NAMES = {
    "devCategory": "Device Category",
    "characteristic": "Device Characteristic",
    "housing": "Housing",
    "powerSupply": "Power Supply",
    "numberOfChannels": "Number of Channels",
    "explosionApproval": "Explosion Approval",
    "protectionArea": "Protection Area",
    "certification": "Certification",
    "dataInterface": "Data Interface",
    "stromSchaltbar": "Switchable Current",
    "stromEingaenge": "Current Inputs",
    "temperaturEingaenge": "Temperature Inputs",
    "binaerDigitalOpenColl_MN": "Digital Open Collector MN",
    "binaerOpenColl_MP": "Open Collector MP",
    "waveInjector": "Wave Injector",
    "dev_advMeterVerification": "Advanced Meter Verification",
    "dynamicGasMaster": "Dynamic Gas Master",
    "customUserFluid": "Custom User Fluid",
    "steamApplication": "Steam Application",
}

# Context presets are intentionally descriptive only. They do not inject
# unsupported technical assumptions into the recommendation.
INDUSTRY_CONTEXT = {
    "Oil & Gas": "Demo context: potentially hazardous process environment.",
    "Chemical & Petrochemical": "Demo context: process measurement in industrial process environments.",
    "Water & Wastewater": "Demo context: utility/process measurement.",
    "Energy & Utilities": "Demo context: utility and industrial measurement.",
    "General Industrial": "Demo context: general industrial measurement.",
}

APPLICATIONS = [
    "Flow measurement",
    "Process monitoring",
    "Energy / utility measurement",
    "General industrial measurement",
]

INSTALLATIONS = ["Stationary", "Portable / field use"]


def norm(x):
    if pd.isna(x):
        return "__MISSING__"
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return "__MISSING__"
    return s


def require_files():
    missing = [p for p in [V41_CAND_PATH, V41_PAIR_PATH, CONFIG_PATH, KG_NODES_PATH, KG_REL_PATH] if not p.exists()]
    if missing:
        st.error("Required PoC data files are missing:")
        for p in missing:
            st.code(str(p))
        st.stop()


def config_to_ml_schema(configs):
    """Map V5 configuration-catalog columns to V4.1's config_* ML schema."""
    out = configs.copy()
    for c in TECHNICAL:
        target = f"config_{c}"
        if target not in out.columns:
            if c in out.columns:
                out[target] = out[c]
            else:
                out[target] = "__MISSING__"
    return out


def build_pair_features(pair_df, cand, feature_cols):
    """Reproduce the directional V4.5 pairwise representation."""
    indexed = cand.set_index(["rfq_id", "canonical_configuration_id"])
    pref_keys = pd.MultiIndex.from_frame(
        pair_df[["rfq_id", "preferred_configuration_id"]]
    )
    rej_keys = pd.MultiIndex.from_frame(
        pair_df[["rfq_id", "rejected_configuration_id"]]
    )

    if not pref_keys.isin(indexed.index).all():
        raise ValueError("Preferred candidate lookup failed during pair-feature construction.")
    if not rej_keys.isin(indexed.index).all():
        raise ValueError("Rejected candidate lookup failed during pair-feature construction.")

    pref = indexed.loc[pref_keys, feature_cols].reset_index(drop=True)
    rej = indexed.loc[rej_keys, feature_cols].reset_index(drop=True)

    out = {}
    categorical = []
    numeric = []
    known_categorical = set(TECHNICAL_CONFIG)

    for col in feature_cols:
        if col in known_categorical or col in {"record_type", "generation_method", "evidence_level"}:
            categorical.append(col)
        else:
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
        p = pref[col].map(norm)
        r = rej[col].map(norm)
        vals = sorted(set(p.unique()) | set(r.unique()))
        for v in vals:
            out[f"CATDIFF__{col}__{v}"] = (p == v).astype(int) - (r == v).astype(int)

    return pd.DataFrame(out, index=pair_df.index)


@st.cache_data(show_spinner=False)
def load_knowledge_graph():
    """Load the existing 731 knowledge graph without rebuilding or modifying it."""
    if not KG_NODES_PATH.exists() or not KG_REL_PATH.exists():
        raise FileNotFoundError(
            "731 Knowledge Graph files are missing. Expected: "
            f"{KG_NODES_PATH} and {KG_REL_PATH}"
        )

    nodes = pd.read_csv(KG_NODES_PATH, low_memory=False)
    rel = pd.read_csv(KG_REL_PATH, low_memory=False)

    required_nodes = {"node_id", "node_type", "label", "canonical_configuration_id", "value"}
    required_rel = {"source_id", "target_id", "relation", "characteristic"}
    if not required_nodes.issubset(nodes.columns):
        raise ValueError(f"Knowledge Graph nodes.csv is missing columns: {sorted(required_nodes - set(nodes.columns))}")
    if not required_rel.issubset(rel.columns):
        raise ValueError(f"Knowledge Graph relationships.csv is missing columns: {sorted(required_rel - set(rel.columns))}")

    node_lookup = nodes.set_index("node_id", drop=False)
    config_nodes = nodes[nodes["node_type"].eq("Configuration")].copy()
    config_nodes["canonical_configuration_id"] = config_nodes["canonical_configuration_id"].astype(str)
    config_by_id = config_nodes.drop_duplicates("canonical_configuration_id").set_index("canonical_configuration_id")

    value_edges = rel[rel["relation"].eq("HAS_VALUE")].copy()
    family_edges = rel[rel["relation"].eq("BELONGS_TO_FAMILY")].copy()

    config_values = {}
    for _, e in value_edges.iterrows():
        cid_node = e["source_id"]
        if cid_node not in node_lookup.index:
            continue
        cid = str(node_lookup.loc[cid_node, "canonical_configuration_id"])
        if cid == "nan":
            continue
        ch = norm(e.get("characteristic", ""))
        target = e["target_id"]
        if target in node_lookup.index:
            val = norm(node_lookup.loc[target, "value"])
            if ch != "__MISSING__" and val != "__MISSING__":
                config_values.setdefault(cid, set()).add((ch, val))

    config_families = {}
    for _, e in family_edges.iterrows():
        src = e["source_id"]
        tgt = e["target_id"]
        if src not in node_lookup.index or tgt not in node_lookup.index:
            continue
        cid = str(node_lookup.loc[src, "canonical_configuration_id"])
        family = norm(node_lookup.loc[tgt, "label"])
        if cid != "nan" and family != "__MISSING__":
            config_families.setdefault(cid, set()).add(family)

    return {
        "nodes": nodes,
        "relationships": rel,
        "node_lookup": node_lookup,
        "config_by_id": config_by_id,
        "config_values": config_values,
        "config_families": config_families,
    }


def kg_validate_candidates(pool, requirements, package, kg):
    """Validate candidates against the existing KG using exact backend values."""
    mandatory = requirements[requirements["type"].eq("MANDATORY")]
    checked = []
    failures = []

    for _, row in pool.iterrows():
        cid = str(row["canonical_configuration_id"])
        if cid not in kg["config_by_id"].index:
            failures.append((cid, "Configuration node not found"))
            continue

        families = kg["config_families"].get(cid, set())
        if norm(package) not in families:
            failures.append((cid, f"Configuration is not linked to family {package}"))
            continue

        values = kg["config_values"].get(cid, set())
        ok = True
        for _, req in mandatory.iterrows():
            key = (norm(req["characteristic"]), norm(req["value"]))
            if key not in values:
                ok = False
                failures.append((cid, f"Missing mandatory KG value: {key[0]}={key[1]}"))
                break
        if ok:
            checked.append(cid)

    return set(checked), failures


def kg_consistency_gate(deterministic_pool, kg_valid_ids):
    """Safety gate: KG-valid and deterministic-valid candidate identities must agree."""
    deterministic_ids = set(deterministic_pool["canonical_configuration_id"].astype(str))
    kg_ids = set(kg_valid_ids)
    return deterministic_ids == kg_ids, deterministic_ids, kg_ids


@st.cache_data(show_spinner=False)
def load_data():
    v41 = pd.read_csv(V41_CAND_PATH, low_memory=False)
    pair = pd.read_csv(V41_PAIR_PATH, low_memory=False)
    configs = pd.read_csv(CONFIG_PATH, low_memory=False)

    v41["rfq_id"] = v41["rfq_id"].astype(str)
    v41["canonical_configuration_id"] = v41["canonical_configuration_id"].astype(str)
    pair["rfq_id"] = pair["rfq_id"].astype(str)
    pair["preferred_configuration_id"] = pair["preferred_configuration_id"].astype(str)
    pair["rejected_configuration_id"] = pair["rejected_configuration_id"].astype(str)
    configs["canonical_configuration_id"] = configs["canonical_configuration_id"].astype(str)

    return v41, pair, config_to_ml_schema(configs)


@st.cache_resource(show_spinner="Training the V4.5-style pairwise model once...")
def train_model():
    v41, pair, _ = load_data()

    feature_cols = [c for c in TECHNICAL_CONFIG + REQUIREMENT if c in v41.columns]
    if len(feature_cols) != len(TECHNICAL_CONFIG) + len(REQUIREMENT):
        missing = sorted(set(TECHNICAL_CONFIG + REQUIREMENT) - set(feature_cols))
        raise ValueError(f"V4.1 ML dataset is missing expected realistic features: {missing}")

    forward = pair.copy()
    forward["label"] = 1
    reverse = pair.copy()
    reverse["preferred_configuration_id"], reverse["rejected_configuration_id"] = (
        pair["rejected_configuration_id"].values,
        pair["preferred_configuration_id"].values,
    )
    reverse["label"] = 0
    sym = pd.concat([forward, reverse], ignore_index=True)
    train = sym[sym["ml_split"].eq("TRAIN")].copy()

    X = build_pair_features(train, v41, feature_cols)
    keep = X.columns[X.nunique(dropna=False) > 1].tolist()
    X = X[keep].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    model = LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs", random_state=731)
    model.fit(X, train["label"].astype(int))
    return model, feature_cols, keep


def calculate_requirement_features(candidates, requirements):
    out = candidates.copy()
    mandatory = requirements[requirements["type"].eq("MANDATORY")]
    preferred = requirements[requirements["type"].eq("PREFERRED")]
    mt, pt = len(mandatory), len(preferred)
    ms = np.zeros(len(out), dtype=int)
    ps = np.zeros(len(out), dtype=int)

    for _, req in mandatory.iterrows():
        col = f"config_{req['characteristic']}"
        if col in out.columns:
            ms += out[col].map(norm).eq(norm(req["value"])).to_numpy(dtype=int)

    for _, req in preferred.iterrows():
        col = f"config_{req['characteristic']}"
        if col in out.columns:
            ps += out[col].map(norm).eq(norm(req["value"])).to_numpy(dtype=int)

    out["mandatory_total"] = mt
    out["mandatory_satisfied"] = ms
    out["mandatory_violations"] = mt - ms
    out["preferred_total"] = pt
    out["preferred_satisfied"] = ps
    out["preferred_differences"] = pt - ps
    return out


def ranking_separation(ranked):
    """Return the top-1 vs top-2 score gap without treating it as probability."""
    if ranked is None or len(ranked) < 2 or "ml_score" not in ranked.columns:
        return None
    return float(ranked.iloc[0]["ml_score"]) - float(ranked.iloc[1]["ml_score"])


def ranking_separation_label(margin):
    """Heuristic display label; this is not a calibrated confidence measure."""
    if margin is None:
        return "Single valid configuration"
    if margin >= 0.15:
        return "Clear separation"
    if margin >= 0.05:
        return "Moderate separation"
    return "Close ranking"


def recommendation_review_state(margin):
    """Return customer-facing wording based only on ranking separation."""
    if margin is None:
        return "SINGLE", "Single valid configuration"
    if margin < 0.05:
        return "REVIEW", "Engineer review recommended"
    if margin < 0.15:
        return "MODERATE", "Moderate ranking separation"
    return "CLEAR", "Clear ranking separation"


def rank_candidates(candidates, model, feature_cols, keep_cols):
    """Rank every candidate by mean pairwise win probability, as in V4.5."""
    c = candidates.copy().reset_index(drop=True)
    c["rfq_id"] = "POC_RFQ"
    ids = c["canonical_configuration_id"].astype(str).tolist()

    if len(ids) == 1:
        c["ml_score"] = 0.5
        c["rank"] = 1
        return c

    pair_rows = []
    for a, b in combinations(ids, 2):
        pair_rows.append({
            "rfq_id": "POC_RFQ",
            "preferred_configuration_id": a,
            "rejected_configuration_id": b,
        })

    pairs = pd.DataFrame(pair_rows)
    X_ab = build_pair_features(pairs, c, feature_cols)
    X_ab = X_ab.reindex(columns=keep_cols, fill_value=0.0).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)

    p_ab = model.predict_proba(X_ab)[:, 1]

    scores = {cid: [] for cid in ids}
    for row, p in zip(pairs.itertuples(index=False), p_ab):
        a, b = row.preferred_configuration_id, row.rejected_configuration_id
        scores[a].append(float(p))
        scores[b].append(float(1.0 - p))

    c["ml_score"] = [np.mean(scores[cid]) for cid in ids]
    c = c.sort_values(
        ["ml_score", "mandatory_satisfied", "preferred_satisfied", "canonical_configuration_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    c["rank"] = np.arange(1, len(c) + 1)
    return c



# ---------------------------------------------------------------------------
# Customer-facing terminology
# ---------------------------------------------------------------------------
# The underlying model keeps the original 731 technical codes. The UI uses
# human-readable labels wherever the source meaning is sufficiently clear.
# For codes whose semantic meaning is not explicitly established in the
# project data, we deliberately keep the code visible and call it a
# "technical option" rather than inventing a business meaning.

DISPLAY_NAMES = {
    "devCategory": "Device category",
    "characteristic": "Product characteristic",
    "housing": "Housing material",
    "powerSupply": "Power supply",
    "numberOfChannels": "Measurement channels",
    "explosionApproval": "Explosion protection",
    "protectionArea": "Protection area",
    "certification": "Certification",
    "dataInterface": "Communication interface",
    "stromSchaltbar": "Switchable current output",
    "stromEingaenge": "Current inputs",
    "temperaturEingaenge": "Temperature inputs",
    "binaerDigitalOpenColl_MN": "Digital open-collector output (MN)",
    "binaerOpenColl_MP": "Open-collector output (MP)",
    "waveInjector": "Wave injector",
    "dev_advMeterVerification": "Advanced meter verification",
    "dynamicGasMaster": "Dynamic gas measurement",
    "customUserFluid": "Custom fluid definition",
    "steamApplication": "Steam application",
}

FIELD_HELP = {
    "devCategory": "Internal 731 product-category code. Only select this when the requirement is explicitly known.",
    "characteristic": "Internal 731 product-characteristic code. Only select this when the requirement is explicitly known.",
    "housing": "Select the required transmitter housing material.",
    "powerSupply": "Select the required power-supply option from the 731 configuration catalogue.",
    "numberOfChannels": "Number of measurement channels required for the application.",
    "explosionApproval": "Explosion-protection requirement. Select only when explicitly specified by the customer or project.",
    "protectionArea": "Required protection-area classification, when applicable.",
    "certification": "Required certification or approval, when applicable.",
    "dataInterface": "Required communication/interface option, when explicitly specified.",
    "stromSchaltbar": "Requirement for switchable current-output functionality.",
    "stromEingaenge": "Required number/type of current inputs.",
    "temperaturEingaenge": "Required temperature-input functionality.",
    "binaerDigitalOpenColl_MN": "Requirement for the MN digital open-collector output option.",
    "binaerOpenColl_MP": "Requirement for the MP open-collector output option.",
    "waveInjector": "Requirement for the wave-injector option.",
    "dev_advMeterVerification": "Requirement for advanced meter-verification functionality.",
    "dynamicGasMaster": "Requirement for the Dynamic Gas Master option.",
    "customUserFluid": "Requirement for a custom user-defined fluid.",
    "steamApplication": "Requirement for steam-application configuration.",
}

# Conservative labels: only meanings that are clear from the project/source
# terminology are expanded. Unknown technical codes remain visible rather
# than being guessed.
VALUE_LABELS = {
    "housing": {
        "ST": "Stainless-steel housing (ST)",
        "AL": "Aluminium housing (AL)",
    },
    "numberOfChannels": {
        "1": "1 measurement channel",
        "2": "2 measurement channels",
    },
}

# DATA-GROUNDED PRESENTATION
# --------------------------
# The UI never invents a product name or semantic meaning for a catalogue code.
# Customer-facing text is built from the selected configuration row, the
# requirement rows, and the exact package_context stored in the backend.
TECHNICAL_CODE_FIELDS = {"devCategory", "characteristic"}

# Customer-visible fields are limited to values whose meanings are explicitly
# established in the current project UI mapping.
CUSTOMER_VISIBLE_FIELDS = {"housing", "numberOfChannels"}

ADVANCED_FIELD_GROUPS = {
    "Safety & compliance": [
        "explosionApproval", "protectionArea", "certification"
    ],
    "Measurement & I/O": [
        "powerSupply", "stromSchaltbar", "stromEingaenge",
        "temperaturEingaenge", "binaerDigitalOpenColl_MN",
        "binaerOpenColl_MP"
    ],
    "Communication & advanced options": [
        "dataInterface", "waveInjector", "dev_advMeterVerification",
        "dynamicGasMaster", "customUserFluid", "steamApplication"
    ],
    "Internal 731 classification codes": [
        "devCategory", "characteristic"
    ],
}



INDUSTRY_CONTEXT = {
    "Oil & Gas": "Application context only — it does not automatically change technical requirements.",
    "Chemical & Petrochemical": "Application context only — it does not automatically change technical requirements.",
    "Water & Wastewater": "Application context only — it does not automatically change technical requirements.",
    "Energy & Utilities": "Application context only — it does not automatically change technical requirements.",
    "General Industrial": "Application context only — it does not automatically change technical requirements.",
}

APPLICATIONS = [
    "Flow measurement",
    "Process monitoring",
    "Energy / utility measurement",
    "General industrial measurement",
]
INSTALLATIONS = ["Stationary installation", "Portable / field use"]

GROUPS = {
    "Safety & compliance": [
        "explosionApproval", "protectionArea", "certification"
    ],
    "Hardware": [
        "housing", "powerSupply", "numberOfChannels"
    ],
    "Measurement & I/O": [
        "stromSchaltbar", "stromEingaenge", "temperaturEingaenge",
        "binaerDigitalOpenColl_MN", "binaerOpenColl_MP"
    ],
    "Communication": [
        "dataInterface"
    ],
    "Advanced options": [
        "waveInjector", "dev_advMeterVerification", "dynamicGasMaster",
        "customUserFluid", "steamApplication"
    ],
    "Advanced technical codes": [
        "devCategory", "characteristic"
    ],
}

GROUP_ICONS = {
    "Safety & compliance": "🛡️",
    "Hardware": "⚙️",
    "Measurement & I/O": "◉",
    "Communication": "⌁",
    "Advanced options": "🔧",
    "Advanced technical codes": "#",
}


def package_label(code):
    """Display the exact backend package_context value."""
    code = norm(code)
    return "Not specified" if code == "__MISSING__" else code


def value_label(characteristic, value):
    """Use only verified display mappings; otherwise preserve the exact value."""
    value = norm(value)
    if value == "__MISSING__":
        return "Not specified"
    if characteristic in VALUE_LABELS and value in VALUE_LABELS[characteristic]:
        return VALUE_LABELS[characteristic][value]
    if characteristic in TECHNICAL_CODE_FIELDS:
        return f"Internal 731 code: {value}"
    return f"Catalogue value: {value}"


def backend_configuration_summary(row):
    """Show the exact populated technical fields from the ranked backend row."""
    rows = []
    for ch in TECHNICAL:
        col = f"config_{ch}"
        if col not in row.index:
            continue
        val = norm(row[col])
        if val == "__MISSING__":
            continue
        rows.append({
            "Configuration field": field_label(ch),
            "Exact catalogue value": value_label(ch, val),
        })
    return pd.DataFrame(rows)


def selected_requirement_rows(req_df):
    rows = []
    for _, req in req_df.iterrows():
        rows.append({
            "Requirement": field_label(req["characteristic"]),
            "Priority": "Required" if req["type"] == "MANDATORY" else "Preferred",
            "Your selection": value_label(req["characteristic"], req["value"]),
        })
    return pd.DataFrame(rows)


def field_label(characteristic):
    return DISPLAY_NAMES.get(characteristic, characteristic.replace("_", " ").title())


def apply_custom_css():
    st.markdown(
        """
        <style>
        :root {
            --navy: #12336b;
            --blue: #1f73e8;
            --blue-light: #eaf3ff;
            --border: #dbe4ef;
            --muted: #66758a;
            --bg: #f7f9fc;
            --green: #1fa97a;
            --purple: #7355d6;
        }
        .stApp { background: var(--bg); }
        .block-container { max-width: 1500px; padding-top: 4.4rem; padding-bottom: 3rem; }
        [data-testid="stSidebar"] { background: #f2f6fb; border-right: 1px solid var(--border); }
        [data-testid="stSidebar"] .block-container { padding: 1.4rem 1.0rem; }
        .brand { color: var(--navy); font-weight: 800; letter-spacing: .2px; }
        .brand-sub { color: #35527f; font-size: 1.05rem; margin-top: -5px; }
        .top-title { color: var(--navy); font-size: 2.15rem; font-weight: 800; margin-bottom: 0; }
        .top-subtitle { color: var(--muted); font-size: 1rem; margin-top: .2rem; margin-bottom: 1.1rem; }
        .poc-badge { background: #eaf3ff; color: #174a94; border-radius: 10px; padding: 9px 14px; font-size: .85rem; text-align: center; }
        .section-card { background: white; border: 1px solid var(--border); border-radius: 14px; padding: 1.1rem 1.2rem; margin: .8rem 0; box-shadow: 0 1px 3px rgba(18,51,107,.04); }
        .section-title { color: var(--navy); font-size: 1.35rem; font-weight: 750; margin-bottom: .05rem; }
        .section-help { color: var(--muted); font-size: .9rem; margin-bottom: .75rem; }
        .step-number { display: inline-flex; align-items: center; justify-content: center; width: 38px; height: 38px; border-radius: 50%; background: var(--blue); color: white; font-weight: 800; margin-right: 10px; }
        .step-number.green { background: var(--green); }
        .step-number.purple { background: var(--purple); }
        .step-title { color: var(--navy); font-size: 1.2rem; font-weight: 750; vertical-align: middle; }
        .side-step { padding: 10px 12px; border-radius: 10px; margin: 5px 0; }
        .side-step.active { background: #e7f1ff; color: #174a94; }
        .side-step .n { display: inline-flex; width: 32px; height: 32px; border-radius: 50%; border: 1px solid #c8d5e6; align-items: center; justify-content: center; margin-right: 8px; font-weight: 700; }
        .side-step.active .n { background: var(--blue); color: white; border-color: var(--blue); }
        .side-step b { color: #1c2e4b; }
        .side-step small { color: var(--muted); display: block; margin-left: 42px; margin-top: -4px; }
        .info-box { border: 1px solid var(--border); border-radius: 10px; background: white; padding: 12px; margin-top: 12px; }
        .field-note { color: var(--muted); font-size: .78rem; margin-top: -4px; margin-bottom: 5px; }
        .result-card { background: white; border: 1px solid var(--border); border-radius: 14px; padding: 1.1rem 1.2rem; }
        .result-id { color: var(--navy); font-size: 1.45rem; font-weight: 800; }
        div.stButton > button[kind="primary"] { border-radius: 10px; min-height: 48px; font-weight: 750; font-size: 1rem; }
        .metric-label { color: var(--muted); font-size: .78rem; }
        .recommendation-header {
            background: white;
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.25rem 1.4rem;
            margin: .8rem 0 1rem;
        }
        .recommendation-label {
            color: #5b6f89;
            font-size: .76rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: .09em;
        }
        .recommendation-product {
            color: var(--navy);
            font-size: 1.8rem;
            line-height: 1.18;
            font-weight: 850;
            margin: .18rem 0 .3rem;
        }
        .recommendation-description {
            color: #52657e;
            font-size: .96rem;
            line-height: 1.45;
            margin-bottom: .8rem;
        }
        .choice-card {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            min-height: 130px;
        }
        .choice-card h4 {
            color: var(--navy);
            margin: 0 0 .5rem;
            font-size: 1rem;
        }
        .choice-line {
            color: #33465f;
            font-size: .9rem;
            margin: .28rem 0;
        }
        .choice-muted {
            color: var(--muted);
            font-size: .82rem;
        }
        .next-action {
            background: #f5f8fc;
            border: 1px solid #dfe7f1;
            border-radius: 12px;
            padding: .85rem 1rem;
            color: #455a73;
            margin-top: .8rem;
        }
        .result-hero {
            background: linear-gradient(135deg, #f7fbff 0%, #eef6ff 100%);
            border: 1px solid #cfe1f7;
            border-radius: 16px;
            padding: 1.35rem 1.45rem;
            margin-top: .8rem;
        }
        .result-kicker {
            color: #4b6585;
            font-size: .78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: .08em;
        }
        .result-title {
            color: var(--navy);
            font-size: 1.65rem;
            font-weight: 800;
            margin: .15rem 0 .25rem;
        }
        .result-summary {
            color: #52657e;
            font-size: .95rem;
            margin-bottom: .85rem;
        }
        .status-chip {
            display: inline-block;
            background: #e7f7f0;
            color: #147252;
            border: 1px solid #bfe8d6;
            border-radius: 999px;
            padding: 5px 10px;
            font-size: .78rem;
            font-weight: 700;
            margin-right: 5px;
        }
        .neutral-chip {
            display: inline-block;
            background: #eef3f9;
            color: #53657b;
            border: 1px solid #dce5ef;
            border-radius: 999px;
            padding: 5px 10px;
            font-size: .78rem;
            font-weight: 650;
            margin-right: 5px;
        }
        .research-note {
            background: #f7f8fa;
            border: 1px solid #e4e8ee;
            border-radius: 10px;
            padding: .8rem 1rem;
            color: #647287;
            font-size: .82rem;
        }

        .code-chip { display:inline-block; padding:2px 7px; border-radius:6px; background:#eef3f9; color:#53657b; font-size:.75rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_step(number, title, subtitle, active=False):
    cls = "side-step active" if active else "side-step"
    st.sidebar.markdown(
        f'<div class="{cls}"><span class="n">{number}</span><b>{title}</b><small>{subtitle}</small></div>',
        unsafe_allow_html=True,
    )


def requirement_widget(configs, characteristic, key_suffix="main", family_configs=None):
    """Render a requirement selector using catalogue values, while flagging
    values that are unavailable in the currently selected family.

    We deliberately keep unavailable catalogue values selectable so that a
    user can test a conflicting mandatory requirement and receive a clear,
    data-grounded explanation rather than silently hiding the conflict.
    """
    vals = unique_values(configs, characteristic)
    if not vals:
        return None

    available = set(unique_values(family_configs, characteristic)) if family_configs is not None else set(vals)

    label = field_label(characteristic)
    help_text = FIELD_HELP.get(characteristic, "Select this only when it is explicitly required.")

    def option_label(x):
        if x == "__NONE__":
            return "No specific requirement"
        base = value_label(characteristic, x)
        if x not in available:
            return f"{base} · Not available in selected family"
        return base

    options = ["__NONE__"] + vals
    selected = st.selectbox(
        label,
        options,
        index=0,
        format_func=option_label,
        key=f"value_{characteristic}_{key_suffix}",
        help=help_text,
    )

    if selected == "__NONE__":
        return None

    importance = st.radio(
        "Importance",
        ["Mandatory", "Preferred"],
        horizontal=True,
        key=f"importance_{characteristic}_{key_suffix}",
        help="Mandatory requirements are hard constraints. Preferred requirements influence ranking but do not remove candidates.",
    )
    return {
        "characteristic": characteristic,
        "value": selected,
        "type": importance.upper(),
    }


def unique_values(configs, characteristic):
    col = f"config_{characteristic}"
    if col not in configs.columns:
        return []
    vals = configs[col].map(norm).unique().tolist()
    vals = [v for v in vals if v != "__MISSING__"]
    return sorted(vals)


def friendly_context(industry, application, installation):
    return (
        f"{industry} · {application} · {installation}. "
        "Context is recorded for the PoC but does not create unsupported technical constraints."
    )


def filter_mandatory(pool, req_df):
    filtered = pool.copy()
    for _, req in req_df[req_df["type"].eq("MANDATORY")].iterrows():
        col = f"config_{req['characteristic']}"
        if col in filtered.columns:
            filtered = filtered[filtered[col].map(norm).eq(norm(req["value"]))]
        else:
            filtered = filtered.iloc[0:0]
    return filtered


def conflict_diagnostics(pool, req_df):
    mandatory = req_df[req_df["type"].eq("MANDATORY")].copy()
    rows = []
    for idx, req in mandatory.iterrows():
        remaining = mandatory.drop(index=idx)
        candidate_count = len(filter_mandatory(pool, remaining))
        rows.append({
            "Requirement": field_label(req["characteristic"]),
            "Selected value": value_label(req["characteristic"], req["value"]),
            "Candidates if relaxed": candidate_count,
        })
    return pd.DataFrame(rows).sort_values("Candidates if relaxed", ascending=False)


def family_value_summary(configs, package, characteristic):
    """Return exact catalogue values available for one field in one family."""
    if configs is None or "package_context" not in configs.columns:
        return []
    pool = configs[configs["package_context"].map(norm).eq(norm(package))]
    return unique_values(pool, characteristic)


def explain_zero_candidate_conflict(pool, req_df, package):
    """Create a concise, catalogue-grounded explanation for zero candidates."""
    rows = []
    mandatory = req_df[req_df["type"].eq("MANDATORY")].copy()
    for _, req in mandatory.iterrows():
        available = family_value_summary(pool, package, req["characteristic"])
        requested = norm(req["value"])
        if requested not in set(available):
            rows.append({
                "Requirement": field_label(req["characteristic"]),
                "Requested value": value_label(req["characteristic"], requested),
                "Available in selected family": (
                    ", ".join(value_label(req["characteristic"], v) for v in available)
                    if available else "No catalogue value available"
                ),
                "Catalogue match": "Not available",
            })
    return pd.DataFrame(rows)


def result_option_name(rank):
    rank = int(rank)
    if rank == 1:
        return "Recommended configuration"
    return f"Alternative {rank}"


def match_quality(rank):
    rank = int(rank)
    if rank == 1:
        return "Best match"
    if rank <= 3:
        return "Excellent alternative"
    if rank <= 5:
        return "Strong alternative"
    return "Alternative"


def requirement_summary(row):
    mt = int(row.get("mandatory_total", 0))
    ms = int(row.get("mandatory_satisfied", 0))
    pt = int(row.get("preferred_total", 0))
    ps = int(row.get("preferred_satisfied", 0))
    return f"{ms}/{mt} mandatory · {ps}/{pt} preferred"


def nonempty_specs(row, characteristics=None):
    chars = characteristics if characteristics is not None else TECHNICAL
    rows = []
    for ch in chars:
        col = f"config_{ch}"
        if col not in row.index:
            continue
        val = norm(row[col])
        if val == "__MISSING__":
            continue
        rows.append({
            "Requirement": field_label(ch),
            "Selected option": value_label(ch, val),
        })
    return pd.DataFrame(rows)



def configuration_differences(reference_row, alternative_row):
    """Return only technical differences directly observable in backend fields."""
    technical_fields = [
        c for c in reference_row.index
        if str(c).startswith("config_") and c in alternative_row.index
    ]
    rows = []
    for col in technical_fields:
        a = norm(reference_row.get(col, ""))
        b = norm(alternative_row.get(col, ""))
        if a != b:
            characteristic = col[len("config_"):]
            rows.append({
                "Configuration field": field_label(characteristic),
                "Recommended (#1)": value_label(characteristic, reference_row.get(col, "")),
                "Alternative": value_label(characteristic, alternative_row.get(col, "")),
            })
    return pd.DataFrame(rows)


def alternative_difference_count(reference_row, alternative_row):
    return int(len(configuration_differences(reference_row, alternative_row)))



def run_result_integrity_audit(ranked, result, kg):
    """Audit the displayed result without changing candidate generation or ranking."""
    checks = []

    ranked_ids = set(ranked["canonical_configuration_id"].astype(str))
    deterministic_ids = set(map(str, result.get("deterministic_ids", [])))
    kg_ids = set(map(str, result.get("kg_valid_ids", [])))

    checks.append({
        "Check": "Candidate-set completeness",
        "Status": "PASS" if ranked_ids == deterministic_ids == kg_ids else "FAIL",
        "Evidence": (
            f"Ranked={len(ranked_ids):,}; "
            f"deterministic={len(deterministic_ids):,}; KG={len(kg_ids):,}"
        ),
    })

    checks.append({
        "Check": "No duplicate canonical configurations",
        "Status": "PASS" if len(ranked) == len(ranked_ids) else "FAIL",
        "Evidence": f"Rows={len(ranked):,}; unique IDs={len(ranked_ids):,}",
    })

    # Every displayed configuration must be linked to the selected family in the KG.
    family_ok = True
    family_failures = 0
    package = norm(result["package"])
    for cid in ranked_ids:
        families = kg["config_families"].get(cid, set())
        if package not in families:
            family_ok = False
            family_failures += 1

    checks.append({
        "Check": "Every result belongs to selected family",
        "Status": "PASS" if family_ok else "FAIL",
        "Evidence": f"Family failures={family_failures}",
    })

    # Every displayed configuration must have the required values.
    req_df = result["requirements"]
    mandatory = req_df[req_df["type"].eq("MANDATORY")]
    mandatory_ok = True
    mandatory_failures = 0
    for _, row in ranked.iterrows():
        for _, req in mandatory.iterrows():
            col = f"config_{req['characteristic']}"
            if col not in row.index or norm(row[col]) != norm(req["value"]):
                mandatory_ok = False
                mandatory_failures += 1

    checks.append({
        "Check": "Every result satisfies mandatory requirements",
        "Status": "PASS" if mandatory_ok else "FAIL",
        "Evidence": f"Requirement mismatches={mandatory_failures}",
    })

    # Every displayed configuration must be represented by the KG.
    kg_representation_ok = ranked_ids.issubset(kg["config_by_id"].index.astype(str))
    missing_from_kg = len(ranked_ids - set(kg["config_by_id"].index.astype(str)))
    checks.append({
        "Check": "Every result exists in Knowledge Graph",
        "Status": "PASS" if kg_representation_ok else "FAIL",
        "Evidence": f"Missing KG configuration nodes={missing_from_kg}",
    })

    # The selected top rank must really be the first row after ranking.
    rank_ok = (
        len(ranked) > 0
        and ranked.iloc[0]["rank"] == 1
        and ranked["rank"].astype(int).tolist() == list(range(1, len(ranked) + 1))
    )
    checks.append({
        "Check": "Ranking sequence integrity",
        "Status": "PASS" if rank_ok else "FAIL",
        "Evidence": f"Ranks 1..{len(ranked):,}",
    })

    # The KG safety gate must be open before a recommendation is presented.
    gate_ok = bool(result.get("kg_gate", False))
    checks.append({
        "Check": "Knowledge Graph safety gate",
        "Status": "PASS" if gate_ok else "FAIL",
        "Evidence": "Recommendation allowed only after KG/deterministic agreement",
    })

    audit = pd.DataFrame(checks)
    return audit, bool(audit["Status"].eq("PASS").all())



def main():
    st.set_page_config(
        page_title="731 Series Intelligent Configurator",
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_custom_css()
    require_files()
    v41, pair, configs = load_data()
    kg = load_knowledge_graph()
    model, feature_cols, keep_cols = train_model()

    package_col = "package_context" if "package_context" in configs.columns else None
    if package_col is None:
        st.error("The V5 configuration catalogue does not contain package_context.")
        st.stop()

    # ------------------------------- Sidebar -------------------------------
    with st.sidebar:
        st.markdown('<div class="brand">EMERSON</div>', unsafe_allow_html=True)
        st.markdown('<div class="brand-sub">731 Series<br><b>Intelligent Configurator</b></div>', unsafe_allow_html=True)
        st.divider()
        sidebar_step(1, "Product", "Select product family", active=True)
        sidebar_step(2, "Customer context", "Define your application")
        sidebar_step(3, "Technical requirements", "Specify key requirements")
        sidebar_step(4, "Recommendation", "AI-powered results")
        st.divider()
        st.markdown("### ❔ Need help?")
        st.caption("Use the help icon on a field for a short explanation. Technical codes are shown only where necessary.")
        st.markdown('<div class="info-box"><b>ⓘ About this tool</b><br><span style="color:#66758a;font-size:.82rem">Research proof of concept using the 731 configuration knowledge and a synthetic-preference ranking model.</span></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("Research PoC · Not for commercial use")

    # ------------------------------- Header --------------------------------
    h1, h2 = st.columns([5, 1])
    with h1:
        st.markdown('<div class="top-title">731 Series Intelligent Configurator</div>', unsafe_allow_html=True)
        st.markdown('<div class="top-subtitle">From requirements to the right configuration — faster, smarter, and easier.</div>', unsafe_allow_html=True)
    with h2:
        st.markdown('<div class="poc-badge"><b>Research PoC</b><br>Not for commercial use</div>', unsafe_allow_html=True)

    # ------------------------------- Product --------------------------------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<span class="step-number">1</span><span class="step-title">Product</span>', unsafe_allow_html=True)
    st.caption("Select the product family you want to configure.")
    product = st.selectbox("Product family", ["731 Series"], label_visibility="collapsed")
    c1, c2 = st.columns([1.0, 1.25])
    with c1:
        st.info("**731 Series**\n\nUltrasonic flow measurement configuration with multiple application and hardware options.")
    with c2:
        st.markdown("**Configuration knowledge base**")
        a, b, c = st.columns(3)
        a.metric("Catalogue rows", f"{len(configs):,}")
        b.metric("Unique configurations", f"{configs.canonical_configuration_id.nunique():,}")
        c.metric("Packages", f"{configs[package_col].nunique():,}")
    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------- Customer context ---------------------------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<span class="step-number green">2</span><span class="step-title">Customer context</span>', unsafe_allow_html=True)
    st.caption("Tell us about the application so the result can be understood in context. These fields do not automatically create technical constraints.")
    a, b, c = st.columns(3)
    with a:
        industry = st.selectbox("Industry", list(INDUSTRY_CONTEXT.keys()), key="industry")
        st.caption(INDUSTRY_CONTEXT[industry])
    with b:
        application = st.selectbox("Application", APPLICATIONS, key="application")
    with c:
        installation = st.selectbox("Installation type", INSTALLATIONS, key="installation")
    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------- Technical requirements ---------------------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    t1, t2 = st.columns([5, 1])
    with t1:
        st.markdown('<span class="step-number purple">3</span><span class="step-title">Technical requirements</span>', unsafe_allow_html=True)
        st.caption("Specify only requirements that are explicitly known. All fields are optional.")
    with t2:
        if st.button("↻ Clear all", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key.startswith("value_") or key.startswith("importance_"):
                    del st.session_state[key]
            st.session_state.pop("result", None)
            st.rerun()

    package_values = sorted(configs[package_col].map(norm).unique().tolist())
    default_package = "x731" if "x731" in package_values else package_values[0]
    package = st.selectbox(
        "Configuration family",
        package_values,
        index=package_values.index(default_package),
        format_func=package_label,
        help="Choose the application/product family. The internal catalogue code is retained for research/audit purposes.",
    )
    st.caption("The selected family is the exact package_context value stored in the configuration catalogue.")

    selected_family_pool = configs[
        configs[package_col].map(norm).eq(norm(package))
    ].copy()

    requirement_rows = []
    st.markdown("**Basic requirements**")
    st.caption("Specify only requirements that are explicitly known. Values unavailable in the selected family remain visible so conflicts can be tested and explained.")
    basic_cols = st.columns(2)
    for i, ch in enumerate(sorted(CUSTOMER_VISIBLE_FIELDS)):
        with basic_cols[i % 2]:
            req = requirement_widget(configs, ch, family_configs=selected_family_pool)
            if req:
                requirement_rows.append(req)

    with st.expander("🔧 Advanced technical requirements — expert users", expanded=False):
        st.warning("The fields below use internal 731 catalogue values. They are explicitly labelled as catalogue/internal codes where the current source does not provide a validated plain-language meaning.")
        adv_cols = st.columns(2)
        for i, (group_name, chars) in enumerate(ADVANCED_FIELD_GROUPS.items()):
            with adv_cols[i % 2]:
                st.markdown(f"**{group_name}**")
                for ch in chars:
                    req = requirement_widget(configs, ch, key_suffix="advanced", family_configs=selected_family_pool)
                    if req:
                        requirement_rows.append(req)
    st.markdown('</div>', unsafe_allow_html=True)

    # ------------------------------ Action ---------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    generate = st.button("✨  Find Best Matching Configurations  →", type="primary", use_container_width=True)

    if generate:
        req_df = pd.DataFrame(requirement_rows, columns=["characteristic", "value", "type"])
        pool = configs[configs[package_col].map(norm).eq(norm(package))].copy()
        pool = pool.drop_duplicates("canonical_configuration_id").reset_index(drop=True)
        initial_count = len(pool)
        deterministic_pool = filter_mandatory(pool, req_df)
        valid_count = len(deterministic_pool)

        kg_valid_ids, kg_failures = kg_validate_candidates(pool, req_df, package, kg)
        gate_pass, deterministic_ids, kg_ids = kg_consistency_gate(deterministic_pool, kg_valid_ids)

        if not gate_pass:
            st.session_state.result = {
                "error": "Knowledge Graph validation blocked the recommendation because the KG-valid candidate set does not exactly match the deterministic constraint-engine candidate set.",
                "error_kind": "kg_gate",
                "requirements": req_df,
                "initial_count": initial_count,
                "conflict": pd.DataFrame(),
                "package": package,
                "kg_gate": False,
                "deterministic_count": len(deterministic_ids),
                "kg_count": len(kg_ids),
                "kg_only_count": len(kg_ids - deterministic_ids),
                "det_only_count": len(deterministic_ids - kg_ids),
            }
        elif deterministic_pool.empty:
            conflict = conflict_diagnostics(pool, req_df) if not req_df.empty else pd.DataFrame()
            catalogue_conflict = (
                explain_zero_candidate_conflict(pool, req_df, package)
                if not req_df.empty else pd.DataFrame()
            )
            st.session_state.result = {
                "error": "No configuration in this package satisfies all selected mandatory requirements.",
                "error_kind": "no_compatible",
                "requirements": req_df,
                "initial_count": initial_count,
                "conflict": conflict,
                "catalogue_conflict": catalogue_conflict,
                "package": package,
            }
        else:
            valid_pool = deterministic_pool[deterministic_pool["canonical_configuration_id"].astype(str).isin(kg_ids)].copy()
            valid_pool = calculate_requirement_features(valid_pool, req_df)
            ranked = rank_candidates(valid_pool, model, feature_cols, keep_cols)
            st.session_state.result = {
                "ranked": ranked,
                "requirements": req_df,
                "package": package,
                "industry": industry,
                "application": application,
                "installation": installation,
                "initial_count": initial_count,
                "valid_count": valid_count,
                "feature_count": len(feature_cols),
                "model": "Pairwise Logistic Regression",
                "kg_gate": True,
                "kg_candidate_count": len(kg_ids),
                "deterministic_ids": sorted(deterministic_ids),
                "kg_valid_ids": sorted(kg_ids),
            }

    if "result" not in st.session_state:
        st.session_state.result = None
    result = st.session_state.result

    # ------------------------------- Landing --------------------------------
    if result is None:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Ready to configure")
        st.write("Enter only what you know from the customer or project. The configurator will identify compatible 731 configurations and recommend the best match.")
        x, y, z = st.columns(3)
        x.info("**1 · Tell us what you need**\n\nChoose the requirements that are explicitly known.")
        y.info("**2 · We check compatibility**\n\nRequired requirements are treated as hard constraints.")
        z.info("**3 · We recommend a configuration**\n\nThe remaining valid options are ranked and explained.")
        st.markdown('<div class="next-action"><b>Tip:</b> You do not need to fill every field. Leaving a requirement blank means the customer did not specify it.</div>', unsafe_allow_html=True)
        st.caption("Research PoC: the ranking model is trained and evaluated on synthetic engineer-preference labels, not historical engineer decisions.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # ------------------------------ Error state -----------------------------
    if "error" in result:
        if result.get("error_kind") == "no_compatible":
            st.warning(result["error"])
        else:
            st.error(result["error"])
        st.write(f"**{result['initial_count']:,}** package candidates were checked.")
        st.info("Try changing one mandatory requirement to Preferred, or clear a requirement that is not explicitly required.")
        catalogue_conflict = result.get("catalogue_conflict", pd.DataFrame())
        if not catalogue_conflict.empty:
            st.markdown("#### Why there are no compatible configurations")
            st.caption("The selected value is not present in the catalogue for this configuration family.")
            st.dataframe(
                catalogue_conflict,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Requirement": st.column_config.TextColumn("Technical area"),
                    "Requested value": st.column_config.TextColumn("Your requirement"),
                    "Available in selected family": st.column_config.TextColumn("Catalogue values available"),
                    "Catalogue match": st.column_config.TextColumn("Compatibility"),
                },
            )
        if not result["conflict"].empty:
            st.markdown("#### Which requirement is causing the conflict?")
            st.caption("Each row shows how many candidates would remain if that one mandatory requirement were relaxed while the other mandatory requirements stayed fixed.")
            display = result["conflict"].copy()
            st.dataframe(display, use_container_width=True, hide_index=True)
        if not result["requirements"].empty:
            st.markdown("#### Selected requirements")
            display = result["requirements"].copy()
            display["characteristic"] = display["characteristic"].map(field_label)
            display["value"] = [value_label(c, v) for c, v in zip(result["requirements"]["characteristic"], result["requirements"]["value"])]
            display.columns = ["Requirement", "Requested value", "Importance"]
            st.dataframe(display, use_container_width=True, hide_index=True)
        return


    # ------------------------------ Results ---------------------------------
    ranked = result["ranked"]
    req_df = result["requirements"]
    top = ranked.iloc[0]

    package = result["package"]
    mandatory_total = int(top["mandatory_total"])
    mandatory_satisfied = int(top["mandatory_satisfied"])
    preferred_total = int(top["preferred_total"])
    preferred_satisfied = int(top["preferred_satisfied"])

    # CUSTOMER-FIRST RESULT: distinguish requirement-driven recommendations
    # from a neutral ranking when the customer has not supplied requirements.
    has_requirements = not req_df.empty
    margin = ranking_separation(ranked)
    review_state, review_label = recommendation_review_state(margin)

    st.markdown('<div class="recommendation-header">', unsafe_allow_html=True)

    if has_requirements:
        st.markdown(
            '<div class="recommendation-label">Your 731 Series recommendation</div>',
            unsafe_allow_html=True,
        )
        recommendation_title = (
            "Top-ranked 731 configuration · Rank #1"
            if review_state == "REVIEW"
            else "Recommended 731 configuration · Rank #1"
        )
        st.markdown(
            f'<div class="recommendation-product">{recommendation_title}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="recommendation-description">'
            f'The backend selected the #1 configuration record from the '
            f'<b>{package}</b> configuration family after applying your required constraints.'
            f'</div>',
            unsafe_allow_html=True,
        )
        if review_state == "REVIEW":
            st.warning(
                "⚠️ **Engineer review recommended:** the top-ranked valid configurations "
                "have very similar ranking scores. The system has identified a leading "
                "option, but the ranking does not provide a strong separation from the runner-up."
            )

        status = (
            "✓ All required technical requirements are satisfied"
            if mandatory_total == mandatory_satisfied
            else "Review required technical requirements"
        )
        st.markdown(
            f'<span class="status-chip">{status}</span>'
            f'<span class="neutral-chip">{mandatory_satisfied}/{mandatory_total} required matched</span>'
            f'<span class="neutral-chip">{preferred_satisfied}/{preferred_total} preferred matched</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="recommendation-label">Top-ranked 731 Series configuration</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="recommendation-product">Top-ranked configuration · Rank #1</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="recommendation-description">'
            f'No technical requirements were specified. The backend ranked the '
            f'KG-valid configurations in the <b>{package}</b> configuration family '
            f'and selected the highest-ranked option.'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<span class="neutral-chip">No technical requirements specified</span>'
            '<span class="status-chip">✓ Knowledge Graph validation passed</span>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # The two most important things for a user: what they asked for and what
    # the configurator selected.
    left, right = st.columns([1, 1.15])

    with left:
        st.markdown("#### What you selected")
        if req_df.empty:
            st.markdown(
                '<div class="choice-card"><h4>No technical requirements specified</h4>'
                '<div class="choice-muted">No customer-specific technical constraint was supplied. '
                'The system therefore ranked the KG-valid configurations within the selected family.</div></div>',
                unsafe_allow_html=True,
            )
        else:
            rows = []
            for _, req in req_df.iterrows():
                rows.append({
                    "Requirement": field_label(req["characteristic"]),
                    "Priority": "Required" if req["type"] == "MANDATORY" else "Preferred",
                    "Selection": value_label(req["characteristic"], req["value"]),
                })
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Requirement": st.column_config.TextColumn("Requirement"),
                    "Priority": st.column_config.TextColumn("Priority"),
                    "Selection": st.column_config.TextColumn("Your selection"),
                },
            )

    with right:
        st.markdown("#### What the backend selected")
        st.markdown(
            f'<div class="choice-card">'
            f'<h4>{("Top-ranked backend configuration" if review_state == "REVIEW" or not has_requirements else "Recommended backend configuration")} · Rank #1</h4>'
            f'<div class="choice-line"><b>Configuration family:</b> {package}</div>'
            f'<div class="choice-line"><b>Required requirements:</b> {mandatory_satisfied}/{mandatory_total} matched</div>'
            f'<div class="choice-line"><b>Preferred requirements:</b> {preferred_satisfied}/{preferred_total} matched</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.success("✓ Knowledge Graph validation passed — the recommendation is restricted to configurations validated by the existing 731 Knowledge Graph.")
    margin_label = ranking_separation_label(margin)
    kgm1, kgm2, kgm3, kgm4 = st.columns(4)
    kgm1.metric("KG-valid candidates", f"{result.get('kg_candidate_count', len(ranked)):,}")
    kgm2.metric("Deterministic-valid candidates", f"{result['valid_count']:,}")
    kgm3.metric("KG consistency", "PASS")
    kgm4.metric("Top-1 vs #2 gap", "N/A" if margin is None else f"{margin:.4f}")
    if margin is None:
        st.info("**Ranking separation:** Only one technically valid configuration remains, so there is no second candidate for comparison.")
    else:
        if review_state == "REVIEW":
            st.warning(
                f"**Ranking separation: {margin_label}** · Top-1 score minus Top-2 score = **{margin:.4f}**. "
                "This is a relative ranking margin, not a probability or percentage of correctness. "
                "Because the gap is small, engineer review is recommended."
            )
        else:
            st.info(
                f"**Ranking separation: {margin_label}** · Top-1 score minus Top-2 score = **{margin:.4f}**. "
                "This is a relative ranking margin, not a probability or percentage of correctness. "
                "The label is a heuristic indicator for review, not a calibrated confidence estimate."
            )

    st.markdown("#### Exact configuration selected by the backend")
    backend_specs = backend_configuration_summary(top)
    if backend_specs.empty:
        st.warning("The ranked backend configuration contains no populated technical fields.")
    else:
        st.dataframe(
            backend_specs,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Configuration field": st.column_config.TextColumn("Configuration field"),
                "Exact catalogue value": st.column_config.TextColumn("Exact catalogue value"),
            },
        )
        st.caption(
            "Every value in this table comes directly from the ranked configuration "
            "record in the V5 synthetic configuration catalogue."
        )

    if has_requirements:
        st.markdown("#### Why this configuration was recommended")
        reason_cols = st.columns(3)
        reason_cols[0].metric("Required requirements", f"{mandatory_satisfied}/{mandatory_total}")
        reason_cols[1].metric("Preferred requirements", f"{preferred_satisfied}/{preferred_total}")
        reason_cols[2].metric("Valid configurations considered", f"{len(ranked):,}")
        st.caption(
            "The configurator first removes configurations that cannot satisfy the "
            "required constraints. It then ranks the technically valid options."
        )
    else:
        st.markdown("#### Why this configuration was ranked first")
        reason_cols = st.columns(2)
        reason_cols[0].metric("KG-valid configurations", f"{len(ranked):,}")
        reason_cols[1].metric("Technical requirements", "None specified")
        st.caption(
            "No customer-specific technical constraints were provided. "
            "The configurator therefore ranked the Knowledge Graph-valid options "
            "within the selected configuration family."
        )

    if not req_df.empty:
        st.markdown("#### Does the recommendation match what you asked for?")
        checks = []
        for _, req in req_df.iterrows():
            col = f"config_{req['characteristic']}"
            actual = norm(top.get(col, "__MISSING__"))
            requested = norm(req["value"])
            satisfied = actual == requested
            checks.append({
                "Requirement": field_label(req["characteristic"]),
                "Priority": "Required" if req["type"] == "MANDATORY" else "Preferred",
                "You selected": value_label(req["characteristic"], requested),
                "Recommendation": value_label(req["characteristic"], actual),
                "Result": "✓ Matches your requirement" if satisfied else "Review",
            })
        st.dataframe(pd.DataFrame(checks), use_container_width=True, hide_index=True, column_config={
            "Requirement": st.column_config.TextColumn("Requirement"),
            "Priority": st.column_config.TextColumn("Priority"),
            "You selected": st.column_config.TextColumn("Your selection"),
            "Recommendation": st.column_config.TextColumn("Recommended configuration"),
            "Result": st.column_config.TextColumn("Result"),
        })

    # --------------------- Transparent alternative comparison ----------------
    st.markdown("#### Other backend configurations that also fit")

    # These are the first five members of the already validated and ranked
    # candidate set. This section does not change eligibility or ranking.
    alternatives = ranked.head(min(5, len(ranked))).copy()

    if len(alternatives) > 1:
        alt_rows = []
        for _, row in alternatives.iterrows():
            rank = int(row["rank"])
            score = float(row.get("ml_score", 0.0))
            top_score = float(top.get("ml_score", 0.0))
            gap = top_score - score
            diff_count = alternative_difference_count(top, row)

            alt_rows.append({
                "Option": result_option_name(rank),
                "Configuration family": package_label(row.get("package_context", package)),
                "Match": match_quality(rank),
                "Technical differences from #1": diff_count,
                "Required": f"{int(row['mandatory_satisfied'])}/{int(row['mandatory_total'])}",
                "Preferred": f"{int(row['preferred_satisfied'])}/{int(row['preferred_total'])}",
                "Ranking score": round(score, 6),
                "Gap vs #1": round(gap, 6),
            })

        st.caption(
            "All options shown here are already technically compatible and "
            "Knowledge Graph validated. Differences are calculated directly "
            "from backend configuration fields."
        )
        st.dataframe(
            pd.DataFrame(alt_rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Technical differences from #1": st.column_config.NumberColumn(
                    "Technical differences from #1", format="%d"
                ),
                "Ranking score": st.column_config.NumberColumn(
                    "Ranking score", format="%.6f"
                ),
                "Gap vs #1": st.column_config.NumberColumn(
                    "Gap vs #1", format="%.6f"
                ),
            },
        )

        for _, row in alternatives.head(3).iterrows():
            rank = int(row["rank"])
            diff_df = configuration_differences(top, row)
            score = float(row.get("ml_score", 0.0))
            top_score = float(top.get("ml_score", 0.0))
            gap = top_score - score

            with st.expander(f"{result_option_name(rank)} · {match_quality(rank)}"):
                st.markdown(
                    f"**Configuration family:** "
                    f"{package_label(row.get('package_context', package))}"
                )

                info_cols = st.columns(4)
                info_cols[0].metric("Rank", f"#{rank}")
                info_cols[1].metric("Ranking score", f"{score:.6f}")
                info_cols[2].metric("Gap vs #1", f"{gap:.6f}")
                info_cols[3].metric("Technical differences", f"{len(diff_df)}")

                if rank == 1:
                    st.success("This is the top-ranked configuration.")
                elif diff_df.empty:
                    st.info(
                        "No technical field differences were found between this "
                        "configuration and Rank #1 in the backend fields inspected."
                    )
                else:
                    st.markdown("**What differs from Rank #1?**")
                    st.dataframe(
                        diff_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                st.markdown("**Complete configuration details**")
                st.dataframe(
                    backend_configuration_summary(row),
                    use_container_width=True,
                    hide_index=True,
                )

                if rank != 1:
                    st.caption(
                        "The configurator reports observable catalogue differences. "
                        "It does not infer cost, availability, engineering effort, "
                        "or business reasons unless those factors are represented "
                        "in the available data."
                    )
    else:
        st.info("Only one technically compatible configuration is available.")

    # --------------------- Formal result-integrity audit ---------------------
    audit_df, audit_pass = run_result_integrity_audit(ranked, result, kg)

    st.markdown("#### Configuration result validation")
    if audit_pass:
        st.success(
            "✓ Result integrity checks passed — the complete displayed set is "
            "consistent with deterministic filtering and the existing Knowledge Graph."
        )
    else:
        st.error(
            "⚠ Result integrity check failed. The recommendation should not be treated "
            "as validated until the failing check is investigated."
        )

    audit_cols = st.columns(3)
    audit_cols[0].metric("Audit checks", f"{len(audit_df)}")
    audit_cols[1].metric(
        "Checks passed",
        f"{int(audit_df['Status'].eq('PASS').sum())}/{len(audit_df)}",
    )
    audit_cols[2].metric(
        "Complete candidate set",
        "PASS" if audit_pass else "FAIL",
    )

    with st.expander("Show validation checks", expanded=False):
        st.dataframe(
            audit_df,
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "These checks validate the current result returned by the PoC. "
            "They do not retrain the model or alter candidate generation."
        )

        audit_download = audit_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download result validation report (CSV)",
            data=audit_download,
            file_name="731_configurator_result_validation.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # --------------------- All compatible configurations ---------------------
    # IMPORTANT: this section only exposes the candidate set that has already
    # passed deterministic mandatory filtering AND the KG consistency gate.
    # It does not change candidate generation, validation, or ranking.
    st.markdown(f"#### All technically compatible configurations · {len(ranked):,}")

    if has_requirements:
        st.caption(
            "These are all unique configurations in the selected family that satisfy "
            "every mandatory requirement and pass Knowledge Graph validation. "
            "Preferred requirements influence ranking but do not remove technically valid options."
        )
    else:
        st.caption(
            "These are all unique configurations in the selected family that passed "
            "both the deterministic validity check and Knowledge Graph validation."
        )

    # Clear separation between technical eligibility and ML preference.
    explorer_a, explorer_b, explorer_c = st.columns(3)
    explorer_a.metric("Technically compatible", f"{len(ranked):,}")
    explorer_b.metric("KG-validated", f"{result.get('kg_candidate_count', len(ranked)):,}")
    explorer_c.metric("Currently ranked", f"{len(ranked):,}")

    st.info(
        "Technical compatibility comes first. The Knowledge Graph and mandatory "
        "constraints determine which configurations are possible; the ranking model "
        "only orders those valid configurations."
    )

    all_cols = [
        "rank",
        "canonical_configuration_id",
        "package_context",
        "config_housing",
        "config_numberOfChannels",
        "config_powerSupply",
        "config_devCategory",
        "config_characteristic",
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
        "mandatory_satisfied",
        "preferred_satisfied",
        "ml_score",
    ]
    all_cols = [c for c in all_cols if c in ranked.columns]
    all_configs = ranked[all_cols].copy()

    # Customer-readable labels only where meanings are explicitly established.
    label_map = {
        "config_housing": "housing",
        "config_numberOfChannels": "numberOfChannels",
        "config_powerSupply": "powerSupply",
        "config_devCategory": "devCategory",
        "config_characteristic": "characteristic",
        "config_explosionApproval": "explosionApproval",
        "config_protectionArea": "protectionArea",
        "config_certification": "certification",
        "config_dataInterface": "dataInterface",
        "config_stromSchaltbar": "stromSchaltbar",
        "config_stromEingaenge": "stromEingaenge",
        "config_temperaturEingaenge": "temperaturEingaenge",
        "config_binaerDigitalOpenColl_MN": "binaerDigitalOpenColl_MN",
        "config_binaerOpenColl_MP": "binaerOpenColl_MP",
        "config_waveInjector": "waveInjector",
        "config_dev_advMeterVerification": "dev_advMeterVerification",
        "config_dynamicGasMaster": "dynamicGasMaster",
        "config_customUserFluid": "customUserFluid",
        "config_steamApplication": "steamApplication",
    }
    if "package_context" in all_configs.columns:
        all_configs["package_context"] = all_configs["package_context"].map(package_label)
    for col, characteristic in label_map.items():
        if col in all_configs.columns:
            all_configs[col] = all_configs[col].map(lambda v, ch=characteristic: value_label(ch, v))
    if "ml_score" in all_configs.columns:
        all_configs["ml_score"] = all_configs["ml_score"].astype(float).round(6)

    display_rename = {
        "rank": "Rank",
        "canonical_configuration_id": "Configuration ID",
        "package_context": "Configuration family",
        "config_housing": "Housing",
        "config_numberOfChannels": "Measurement channels",
        "config_powerSupply": "Power supply",
        "config_devCategory": "Device category",
        "config_characteristic": "Product characteristic",
        "config_explosionApproval": "Explosion protection",
        "config_protectionArea": "Protection area",
        "config_certification": "Certification",
        "config_dataInterface": "Communication interface",
        "config_stromSchaltbar": "Switchable current output",
        "config_stromEingaenge": "Current inputs",
        "config_temperaturEingaenge": "Temperature inputs",
        "config_binaerDigitalOpenColl_MN": "Binary digital input",
        "config_binaerOpenColl_MP": "Binary open-collector output",
        "config_waveInjector": "Wave injector",
        "config_dev_advMeterVerification": "Advanced Meter Verification",
        "config_dynamicGasMaster": "Dynamic Gas Master",
        "config_customUserFluid": "Custom user fluid",
        "config_steamApplication": "Steam application",
        "mandatory_satisfied": "Required matched",
        "preferred_satisfied": "Preferred matched",
        "ml_score": "Ranking score",
    }
    all_configs = all_configs.rename(columns=display_rename)

    # Compact overview table for the normal user / Configurator Lead.
    overview_cols = [
        "Rank", "Configuration ID", "Housing", "Measurement channels",
        "Power supply", "Device category", "Product characteristic",
        "Required matched", "Preferred matched", "Ranking score"
    ]
    overview_cols = [c for c in overview_cols if c in all_configs.columns]

    st.markdown("**Complete compatible set**")
    st.caption(
        "Use the overview to compare every compatible configuration. "
        "The detailed technical values are available by inspecting any row below."
    )
    st.dataframe(
        all_configs[overview_cols],
        use_container_width=True,
        hide_index=True,
        height=min(650, 140 + 35 * min(len(all_configs), 15)),
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", format="%d"),
            "Configuration ID": st.column_config.TextColumn("Configuration ID"),
            "Ranking score": st.column_config.NumberColumn("Ranking score", format="%.6f"),
        },
    )

    st.download_button(
        "⬇ Download complete compatible configuration set (CSV)",
        data=ranked.to_csv(index=False).encode("utf-8"),
        file_name=f"731_all_compatible_configurations_{norm(package)}.csv",
        mime="text/csv",
        use_container_width=True,
        help=(
            "Downloads every unique configuration that passed the KG and deterministic "
            "validity checks, including all backend technical fields and ranking information."
        ),
    )

    with st.expander("Inspect any compatible configuration", expanded=False):
        inspect_options = all_configs["Configuration ID"].tolist() if "Configuration ID" in all_configs.columns else []
        if inspect_options:
            selected_id = st.selectbox(
                "Choose a compatible configuration",
                inspect_options,
                index=0,
                key="inspect_compatible_configuration",
            )
            selected_rows = ranked[
                ranked["canonical_configuration_id"].astype(str).eq(str(selected_id))
            ]
            if not selected_rows.empty:
                selected_row = selected_rows.iloc[0]
                st.markdown(
                    f"**Rank #{int(selected_row['rank'])} · "
                    f"{package_label(selected_row.get('package_context', package))}**"
                )
                st.dataframe(
                    backend_configuration_summary(selected_row),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    "The inspected configuration is part of the complete technically "
                    "compatible set; Rank #1 is a ranking outcome, not a technical-validity requirement."
                )

    with st.expander("Show complete technical columns for all compatible configurations", expanded=False):
        st.caption(
            "These columns preserve the backend catalogue values. Internal 731 codes are "
            "not given invented meanings when the project source does not establish one."
        )
        st.dataframe(
            all_configs,
            use_container_width=True,
            hide_index=True,
            height=650,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", format="%d"),
                "Ranking score": st.column_config.NumberColumn("Ranking score", format="%.6f"),
            },
        )

    with st.expander("Technical configuration details", expanded=False):
        st.caption("These values come directly from the 731 catalogue and are retained for expert review. The PoC does not invent meanings for internal codes.")
        st.dataframe(nonempty_specs(top), use_container_width=True, hide_index=True, column_config={
            "Requirement": st.column_config.TextColumn("Technical area"),
            "Selected option": st.column_config.TextColumn("Catalogue value"),
        })

    with st.expander("Research & audit information — internal"):
        research_text = f'''
**Internal configuration identifier:** `{top["canonical_configuration_id"]}`

**Model ranking score:** `{float(top["ml_score"]):.4f}`
**Top-1 vs Top-2 ranking gap:** `{("N/A" if margin is None else f"{margin:.4f}")}`
**Ranking separation label:** `{margin_label}`

**Recommendation review state:** `{review_label}`

**Product:** {product}

**Configuration family (internal catalogue code):** `{result["package"]}`

**Initial package candidate pool:** {result["initial_count"]:,}

**Mandatory-valid candidate pool:** {result["valid_count"]:,}

**Knowledge Graph validation:** {"PASS" if result.get("kg_gate") else "BLOCKED"}

**Knowledge Graph candidate count:** {result.get("kg_candidate_count", result.get("valid_count", 0)):,}

**Formal result-integrity audit:** {"PASS" if audit_pass else "FAIL"}

**Ranking model:** {result["model"]}

**Feature regime:** realistic baseline

**Training source:** V4.1 ML candidate + V4.1 pairwise datasets

**Ground-truth engineer choice:** not used by the UI
'''
        st.markdown(research_text)
        st.markdown(
            '<div class="research-note">'
            'The model score is a ranking score used to order candidates; '
            'it is not a probability that the configuration is correct. '
            'This is a research proof of concept using synthetic engineer-preference '
            'labels, not historical engineer decisions.'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)



if __name__ == "__main__":
    main()
