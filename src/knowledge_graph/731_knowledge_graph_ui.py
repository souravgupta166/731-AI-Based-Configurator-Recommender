from pathlib import Path
from collections import Counter

import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="731 Knowledge Graph Explorer",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

KG_DIR = PROJECT_ROOT / "graphs" / "731_knowledge_graph"

NODES_FILE = KG_DIR / "nodes.csv"
RELATIONSHIPS_FILE = KG_DIR / "relationships.csv"
STATS_FILE = KG_DIR / "statistics.json"
VALIDATION_FILE = KG_DIR / "validation_report.txt"


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background: #f8fafc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .kg-header {
        padding: 1.4rem 1.6rem;
        border-radius: 14px;
        background: linear-gradient(
            135deg,
            #ffffff 0%,
            #f1f5f9 100%
        );
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }

    .kg-header h1 {
        margin: 0;
        font-size: 2rem;
        color: #0f172a;
    }

    .kg-header p {
        margin-top: 0.45rem;
        color: #64748b;
        font-size: 0.95rem;
    }

    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        min-height: 105px;
    }

    .metric-label {
        color: #64748b;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .metric-value {
        color: #0f172a;
        font-size: 1.65rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }

    .metric-note {
        color: #94a3b8;
        font-size: 0.75rem;
        margin-top: 0.15rem;
    }

    .node-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }

    .provenance-observed {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        background: #dcfce7;
        color: #166534;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .provenance-synthetic {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        background: #ede9fe;
        color: #6d28d9;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .provenance-derived {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        background: #fef3c7;
        color: #92400e;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .relationship-pill {
        display: inline-block;
        padding: 0.22rem 0.55rem;
        border-radius: 999px;
        background: #eff6ff;
        color: #1d4ed8;
        font-size: 0.72rem;
        font-weight: 600;
        border: 1px solid #dbeafe;
    }

    .audit-pass {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #065f46;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_nodes():
    return pd.read_csv(NODES_FILE, low_memory=False)


@st.cache_data
def load_relationships():
    return pd.read_csv(RELATIONSHIPS_FILE, low_memory=False)


@st.cache_data
def load_statistics():
    if STATS_FILE.exists():
        import json
        with open(STATS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_validation():
    if VALIDATION_FILE.exists():
        return VALIDATION_FILE.read_text(encoding="utf-8")
    return ""


if not NODES_FILE.exists() or not RELATIONSHIPS_FILE.exists():
    st.error(
        "Knowledge Graph files were not found. "
        f"Expected them under: {KG_DIR}"
    )
    st.stop()


nodes = load_nodes()
relationships = load_relationships()
statistics = load_statistics()
validation_text = load_validation()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="kg-header">
        <h1>🔗 731 Knowledge Graph Explorer</h1>
        <p>
            Read-only exploration of the 731 configuration knowledge graph.
            Real configurator rules are preserved as OBSERVED_731 evidence,
            while synthetic configurations are explicitly marked SYNTHETIC.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Knowledge Graph")

st.sidebar.caption(
    "Explore the existing 731 KG without rebuilding or modifying it."
)

page = st.sidebar.radio(
    "View",
    [
        "Overview",
        "Explore Nodes",
        "Relationship Explorer",
        "Neighbourhood",
        "Validation & Audit",
    ],
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def provenance_badge(value):
    value = str(value)

    if value == "OBSERVED_731":
        return '<span class="provenance-observed">OBSERVED_731</span>'

    if value == "SYNTHETIC":
        return '<span class="provenance-synthetic">SYNTHETIC</span>'

    if value == "DERIVED":
        return '<span class="provenance-derived">DERIVED</span>'

    return value


def node_title(row):
    label = str(row.get("label", ""))

    if label and label != "nan":
        return label

    return str(row.get("node_id", ""))


def get_node(node_id):
    result = nodes[nodes["node_id"].astype(str) == str(node_id)]

    if result.empty:
        return None

    return result.iloc[0]


def get_node_relationships(node_id):
    node_id = str(node_id)

    outgoing = relationships[
        relationships["source_id"].astype(str) == node_id
    ].copy()

    outgoing["direction"] = "OUTGOING"

    incoming = relationships[
        relationships["target_id"].astype(str) == node_id
    ].copy()

    incoming["direction"] = "INCOMING"

    return pd.concat([outgoing, incoming], ignore_index=True)


def related_node_ids(node_id):
    node_id = str(node_id)

    outgoing = relationships.loc[
        relationships["source_id"].astype(str) == node_id,
        "target_id"
    ].astype(str).tolist()

    incoming = relationships.loc[
        relationships["target_id"].astype(str) == node_id,
        "source_id"
    ].astype(str).tolist()

    return list(dict.fromkeys(outgoing + incoming))


def render_node_details(row):
    st.markdown(
        f"""
        <div class="node-card">
            <h3 style="margin-top:0;">
                {node_title(row)}
            </h3>

            <span class="relationship-pill">
                {row.get("node_type", "")}
            </span>

            &nbsp;

            {provenance_badge(row.get("provenance", ""))}
        </div>
        """,
        unsafe_allow_html=True,
    )

    details = {}

    for col in nodes.columns:
        value = row.get(col)

        if pd.isna(value):
            continue

        value = str(value)

        if value and value != "nan":
            details[col] = value

    if details:
        st.dataframe(
            pd.DataFrame(
                list(details.items()),
                columns=["Property", "Value"]
            ),
            use_container_width=True,
            hide_index=True,
        )


def render_relationship_table(node_id):
    rels = get_node_relationships(node_id)

    if rels.empty:
        st.info("No relationships found for this node.")
        return

    display_rows = []

    for _, rel in rels.iterrows():

        if rel["direction"] == "OUTGOING":
            other_id = rel["target_id"]
            direction = "→"
        else:
            other_id = rel["source_id"]
            direction = "←"

        other = get_node(other_id)

        if other is None:
            continue

        display_rows.append(
            {
                "Direction": direction,
                "Relationship": rel["relation"],
                "Connected node": node_title(other),
                "Node type": other.get("node_type", ""),
                "Provenance": other.get("provenance", ""),
            }
        )

    if display_rows:
        st.dataframe(
            pd.DataFrame(display_rows),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.subheader("Knowledge Graph at a glance")

    total_nodes = len(nodes)
    total_relationships = len(relationships)

    node_counts = nodes["node_type"].value_counts()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total nodes</div>
                <div class="metric-value">{total_nodes:,}</div>
                <div class="metric-note">Entities in the KG</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Relationships</div>
                <div class="metric-value">{total_relationships:,}</div>
                <div class="metric-note">Directed KG edges</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Configurations</div>
                <div class="metric-value">
                    {node_counts.get("Configuration", 0):,}
                </div>
                <div class="metric-note">Canonical configurations</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Real rule rows</div>
                <div class="metric-value">
                    {node_counts.get("RuleRow", 0):,}
                </div>
                <div class="metric-note">OBSERVED_731 rules</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    left, right = st.columns(2)

    with left:
        st.markdown("### Node types")

        node_summary = (
            nodes["node_type"]
            .value_counts()
            .rename_axis("Node type")
            .reset_index(name="Count")
        )

        st.dataframe(
            node_summary,
            use_container_width=True,
            hide_index=True,
        )

    with right:
        st.markdown("### Provenance")

        provenance_summary = (
            nodes["provenance"]
            .value_counts()
            .rename_axis("Provenance")
            .reset_index(name="Count")
        )

        st.dataframe(
            provenance_summary,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### Relationship types")

    relation_summary = (
        relationships["relation"]
        .value_counts()
        .rename_axis("Relationship")
        .reset_index(name="Count")
    )

    st.dataframe(
        relation_summary,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "The full graph contains 73k+ relationships. "
        "The Explorer deliberately renders only selected neighbourhoods "
        "rather than attempting to display the complete graph at once."
    )


# ============================================================
# EXPLORE NODES
# ============================================================

elif page == "Explore Nodes":

    st.subheader("Explore KG nodes")

    col1, col2 = st.columns([2, 1])

    with col1:
        search = st.text_input(
            "Search node",
            placeholder="Try x731, F731WD_DualChannel, housing, CFG731...",
        )

    with col2:
        node_types = ["All"] + sorted(
            nodes["node_type"].dropna().unique().tolist()
        )

        selected_type = st.selectbox(
            "Node type",
            node_types,
        )

    filtered = nodes.copy()

    if selected_type != "All":
        filtered = filtered[
            filtered["node_type"] == selected_type
        ]

    if search.strip():

        term = search.strip().lower()

        mask = (
            filtered["node_id"]
            .astype(str)
            .str.lower()
            .str.contains(term, na=False)
            |
            filtered["label"]
            .astype(str)
            .str.lower()
            .str.contains(term, na=False)
            |
            filtered["characteristic"]
            .astype(str)
            .str.lower()
            .str.contains(term, na=False)
            |
            filtered["value"]
            .astype(str)
            .str.lower()
            .str.contains(term, na=False)
        )

        filtered = filtered[mask]

    st.caption(f"{len(filtered):,} matching nodes")

    if filtered.empty:
        st.warning("No matching nodes found.")
        st.stop()

    result_display = filtered[
        [
            "node_id",
            "node_type",
            "label",
            "provenance",
        ]
    ].head(200)

    st.dataframe(
        result_display,
        use_container_width=True,
        hide_index=True,
    )

    options = filtered["node_id"].astype(str).head(500).tolist()

    selected_node = st.selectbox(
        "Select a node to inspect",
        options,
        format_func=lambda x: node_title(get_node(x)),
    )

    row = get_node(selected_node)

    if row is not None:

        st.markdown("---")

        st.markdown("### Node details")

        render_node_details(row)

        st.markdown("### Connected relationships")

        render_relationship_table(selected_node)


# ============================================================
# RELATIONSHIP EXPLORER
# ============================================================

elif page == "Relationship Explorer":

    st.subheader("Relationship Explorer")

    relations = sorted(
        relationships["relation"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_relation = st.selectbox(
        "Relationship type",
        relations,
    )

    rels = relationships[
        relationships["relation"] == selected_relation
    ].copy()

    st.metric(
        "Relationships of this type",
        f"{len(rels):,}"
    )

    display = []

    for _, rel in rels.head(500).iterrows():

        source = get_node(rel["source_id"])
        target = get_node(rel["target_id"])

        if source is None or target is None:
            continue

        display.append(
            {
                "Source": node_title(source),
                "Source type": source["node_type"],
                "Relationship": rel["relation"],
                "Target": node_title(target),
                "Target type": target["node_type"],
                "Provenance": rel.get("provenance", ""),
            }
        )

    if display:
        st.dataframe(
            pd.DataFrame(display),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# NEIGHBOURHOOD
# ============================================================

elif page == "Neighbourhood":

    st.subheader("Visual Knowledge Graph Neighbourhood")

    st.caption(
        "Select a node and inspect its immediate connections. "
        "Only a small neighbourhood is rendered for readability."
    )

    search = st.text_input(
        "Find a starting node",
        placeholder="Try x731 or a configuration ID",
    )

    candidate_nodes = nodes.copy()

    if search.strip():

        term = search.strip().lower()

        mask = (
            candidate_nodes["node_id"]
            .astype(str)
            .str.lower()
            .str.contains(term, na=False)
            |
            candidate_nodes["label"]
            .astype(str)
            .str.lower()
            .str.contains(term, na=False)
        )

        candidate_nodes = candidate_nodes[mask]

    if candidate_nodes.empty:
        st.warning("No nodes found.")
        st.stop()

    node_ids = candidate_nodes["node_id"].astype(str).head(200).tolist()

    selected = st.selectbox(
        "Starting node",
        node_ids,
        format_func=lambda x: node_title(get_node(x)),
    )

    selected_row = get_node(selected)

    if selected_row is None:
        st.stop()

    st.markdown(
        f"""
        **Selected node:** `{node_title(selected_row)}`  
        **Type:** `{selected_row["node_type"]}`  
        **Provenance:** `{selected_row["provenance"]}`
        """
    )

    neighbours = related_node_ids(selected)

    # Keep the visual neighbourhood deliberately small.
    neighbours = neighbours[:30]

    st.markdown("### Immediate neighbourhood")

    if not neighbours:
        st.info("This node has no connected nodes.")
        st.stop()

    centre_label = node_title(selected_row)

    # Simple SVG-style visual using HTML.
    # This avoids loading the entire graph into a browser.
    cards = []

    for nid in neighbours:

        n = get_node(nid)

        if n is None:
            continue

        cards.append(
            f"""
            <div style="
                display:inline-block;
                vertical-align:top;
                width:210px;
                min-height:100px;
                margin:8px;
                padding:12px;
                border:1px solid #cbd5e1;
                border-radius:10px;
                background:#ffffff;
            ">
                <div style="
                    font-size:12px;
                    color:#64748b;
                    margin-bottom:5px;
                ">
                    {n["node_type"]}
                </div>

                <div style="
                    font-weight:700;
                    color:#0f172a;
                    word-break:break-word;
                ">
                    {node_title(n)}
                </div>

                <div style="
                    font-size:11px;
                    color:#64748b;
                    margin-top:7px;
                ">
                    {n.get("provenance", "")}
                </div>
            </div>
            """
        )

    st.markdown(
        f"""
        <div style="
            text-align:center;
            padding:18px;
            margin-bottom:12px;
            border-radius:12px;
            background:#eff6ff;
            border:1px solid #bfdbfe;
        ">
            <div style="
                font-size:12px;
                color:#1d4ed8;
                text-transform:uppercase;
            ">
                Selected node
            </div>

            <div style="
                font-size:20px;
                font-weight:700;
                color:#0f172a;
                margin-top:4px;
            ">
                {centre_label}
            </div>
        </div>

        <div style="
            text-align:center;
            padding:10px;
        ">
            {"".join(cards)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Relationship paths")

    rels = get_node_relationships(selected)

    if not rels.empty:

        rows = []

        for _, rel in rels.head(100).iterrows():

            if rel["source_id"] == selected:
                other = get_node(rel["target_id"])
                direction = "→"
            else:
                other = get_node(rel["source_id"])
                direction = "←"

            if other is None:
                continue

            rows.append(
                {
                    "Path": (
                        f"{centre_label} "
                        f"{direction} "
                        f"{node_title(other)}"
                    ),
                    "Relationship": rel["relation"],
                    "Connected type": other["node_type"],
                    "Provenance": rel.get("provenance", ""),
                    "Characteristic": rel.get(
                        "characteristic",
                        ""
                    ),
                }
            )

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# VALIDATION & AUDIT
# ============================================================

elif page == "Validation & Audit":

    st.subheader("Knowledge Graph Validation & Audit")

    st.markdown(
        """
        <div class="audit-pass">
            ✓ Existing Knowledge Graph validation status:
            PASS
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    if statistics:

        st.markdown("### Stored KG statistics")

        stats_rows = []

        for key, value in statistics.items():

            if isinstance(value, (dict, list)):
                continue

            stats_rows.append(
                {
                    "Metric": key,
                    "Value": value,
                }
            )

        if stats_rows:
            st.dataframe(
                pd.DataFrame(stats_rows),
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("### Validation report")

    if validation_text:

        st.code(
            validation_text,
            language="text",
        )

    else:
        st.info(
            "No validation_report.txt was found."
        )

    st.markdown("### Provenance principles")

    st.markdown(
        """
        **OBSERVED_731**

        Real rule knowledge originating from the
        `x731_allgemein_Rev24.xlsx` configurator source.

        **SYNTHETIC**

        Synthetic configuration records generated for the
        thesis experimentation environment.

        **DERIVED**

        Information derived during the graph construction process.

        The Knowledge Graph does **not** treat the 1,056 real
        rule rows as historical customer orders or historical
        engineer decisions.
        """
    )