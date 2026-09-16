#!/usr/bin/env python3

"""
731 Knowledge Graph Explorer

Read-only explorer for the already-built 731 Knowledge Graph.

The application does NOT rebuild or modify the graph.

Data sources:
    graphs/731_knowledge_graph/nodes.csv
    graphs/731_knowledge_graph/relationships.csv
    graphs/731_knowledge_graph/statistics.json
    graphs/731_knowledge_graph/validation_report.txt
"""

from pathlib import Path
import html
import json
import math

import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRAPH_DIR = PROJECT_ROOT / "graphs" / "731_knowledge_graph"

NODES_FILE = GRAPH_DIR / "nodes.csv"
EDGES_FILE = GRAPH_DIR / "relationships.csv"
STATS_FILE = GRAPH_DIR / "statistics.json"
VALIDATION_FILE = GRAPH_DIR / "validation_report.txt"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="731 Knowledge Graph",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# THEME
# ============================================================

st.markdown(
    """
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* Main hero */

.kg-hero {
    background: #f7faff;
    border: 1px solid #dbe5f0;
    border-radius: 18px;
    padding: 1.5rem 1.7rem;
    margin-bottom: 1.3rem;
}

.kg-eyebrow {
    color: #58708f;
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}

.kg-title {
    color: #173b70;
    font-size: 2rem;
    font-weight: 800;
    margin: 0;
}

.kg-subtitle {
    color: #627286;
    margin-top: 0.45rem;
    font-size: 0.98rem;
}

/* Information cards */

.info-card {
    border: 1px solid #dbe5f0;
    border-radius: 15px;
    padding: 1.05rem 1.15rem;
    background: #ffffff;
    height: 100%;
}

.info-label {
    color: #6c7c90;
    font-size: 0.74rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

.info-value {
    color: #18283d;
    font-size: 1.45rem;
    font-weight: 750;
    margin-top: 0.25rem;
}

/* Selected entity */

.entity-card {
    border: 1px solid #d8e2ed;
    border-radius: 16px;
    background: #ffffff;
    padding: 1.2rem 1.3rem;
}

.entity-type {
    color: #5c718b;
    font-size: 0.74rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.entity-title {
    color: #173b70;
    font-size: 1.45rem;
    font-weight: 800;
    margin-top: 0.25rem;
}

.entity-meta {
    color: #65758a;
    font-size: 0.86rem;
    margin-top: 0.45rem;
}

/* Explanation */

.explanation-card {
    border-left: 4px solid #4f7fbf;
    background: #f7faff;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    color: #41536a;
    font-size: 0.92rem;
}

/* Section spacing */

.section-title {
    color: #1d2f45;
    font-size: 1.25rem;
    font-weight: 750;
    margin-top: 1.5rem;
    margin-bottom: 0.2rem;
}

.section-caption {
    color: #708096;
    font-size: 0.88rem;
    margin-bottom: 0.8rem;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    border-right: 1px solid #e2e8f0;
}

/* Buttons */

.stButton > button {
    border-radius: 9px;
}

/* Dataframe */

[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# NODE TYPE LABELS
# ============================================================

NODE_TYPE_LABELS = {
    "Product": "Product",
    "ConfigurationFamily": "Configuration Family",
    "Configuration": "Configuration",
    "Characteristic": "Technical Characteristic",
    "CharacteristicValue": "Characteristic Value",
    "RuleRow": "Configurator Rule",
}


NODE_TYPE_HELP = {
    "Product":
        "The 731 product represented in the knowledge graph.",

    "ConfigurationFamily":
        "A configuration family represented in the source data.",

    "Configuration":
        "A concrete configuration record represented in the synthetic configuration catalogue.",

    "Characteristic":
        "A technical field or attribute used by the configurator.",

    "CharacteristicValue":
        "A value associated with a technical characteristic.",

    "RuleRow":
        "A row from the real 731 configurator workbook represented as product knowledge.",
}


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_graph():

    if not NODES_FILE.exists():
        raise FileNotFoundError(
            f"Could not find:\n{NODES_FILE}"
        )

    if not EDGES_FILE.exists():
        raise FileNotFoundError(
            f"Could not find:\n{EDGES_FILE}"
        )

    nodes = pd.read_csv(
        NODES_FILE,
        low_memory=False
    ).fillna("")

    edges = pd.read_csv(
        EDGES_FILE,
        low_memory=False
    ).fillna("")

    stats = {}

    if STATS_FILE.exists():
        with open(
            STATS_FILE,
            "r",
            encoding="utf-8"
        ) as f:
            stats = json.load(f)

    validation = ""

    if VALIDATION_FILE.exists():
        validation = VALIDATION_FILE.read_text(
            encoding="utf-8"
        )

    return nodes, edges, stats, validation


# ============================================================
# LOAD
# ============================================================

try:

    nodes, edges, stats, validation = load_graph()

except Exception as exc:

    st.error(
        f"Knowledge Graph could not be loaded.\n\n{exc}"
    )

    st.stop()


# ============================================================
# HELPERS
# ============================================================

def friendly_node_type(node_type):

    return NODE_TYPE_LABELS.get(
        str(node_type),
        str(node_type)
    )


def node_type_description(node_type):

    return NODE_TYPE_HELP.get(
        str(node_type),
        "Entity represented in the knowledge graph."
    )


def get_node(node_id):

    result = nodes[
        nodes["node_id"].astype(str)
        == str(node_id)
    ]

    if result.empty:
        return None

    return result.iloc[0]


def get_neighbourhood(root_id, depth=1):

    root_id = str(root_id)

    visited = {root_id}
    frontier = {root_id}

    for _ in range(depth):

        if not frontier:
            break

        mask = (
            edges["source_id"]
            .astype(str)
            .isin(frontier)
        ) | (
            edges["target_id"]
            .astype(str)
            .isin(frontier)
        )

        connected = edges.loc[mask]

        next_nodes = set(
            connected["source_id"]
            .astype(str)
        )

        next_nodes.update(
            connected["target_id"]
            .astype(str)
        )

        next_nodes -= visited

        visited.update(next_nodes)

        frontier = next_nodes

    return visited


# ============================================================
# LIGHTWEIGHT SVG GRAPH
# ============================================================

def render_graph(
    graph_nodes,
    graph_edges,
    root_id
):

    root_id = str(root_id)

    node_ids = (
        graph_nodes["node_id"]
        .astype(str)
        .tolist()
    )

    if root_id not in node_ids:
        return

    # Keep root first
    ordered_ids = [
        root_id
    ] + [
        x for x in node_ids
        if x != root_id
    ]

    width = 1100
    height = 620

    visible_ids = ordered_ids[:80]

    center_x = width / 2
    center_y = height / 2

    positions = {
        root_id: (
            center_x,
            center_y
        )
    }

    others = [
        x for x in visible_ids
        if x != root_id
    ]

    radius = min(
        250,
        max(
            160,
            32 * math.sqrt(
                max(len(others), 1)
            )
        )
    )

    for i, node_id in enumerate(others):

        angle = (
            2
            * math.pi
            * i
            / max(len(others), 1)
        ) - math.pi / 2

        positions[node_id] = (
            center_x
            + radius * math.cos(angle),

            center_y
            + radius * math.sin(angle)
        )

    palette = {
        "Product": "#173b70",
        "ConfigurationFamily": "#7c5ac7",
        "Configuration": "#d58a20",
        "Characteristic": "#3d7ec1",
        "CharacteristicValue": "#3b9b76",
        "RuleRow": "#697b91",
    }

    edge_svg = []

    for _, edge in graph_edges.iterrows():

        source = str(edge["source_id"])
        target = str(edge["target_id"])

        if (
            source not in positions
            or target not in positions
        ):
            continue

        x1, y1 = positions[source]
        x2, y2 = positions[target]

        relation = html.escape(
            str(edge["relation"])
        )

        edge_svg.append(
            f"""
<line
x1="{x1:.1f}"
y1="{y1:.1f}"
x2="{x2:.1f}"
y2="{y2:.1f}"
stroke="#c8d3df"
stroke-width="1.4">
<title>{relation}</title>
</line>
"""
        )

    node_svg = []

    for _, row in graph_nodes.iterrows():

        node_id = str(
            row["node_id"]
        )

        if node_id not in positions:
            continue

        x, y = positions[node_id]

        node_type = str(
            row["node_type"]
        )

        label = html.escape(
            str(row["label"])
        )

        fill = palette.get(
            node_type,
            "#697b91"
        )

        radius_value = (
            25
            if node_id == root_id
            else 18
        )

        node_svg.append(
            f"""
<circle
cx="{x:.1f}"
cy="{y:.1f}"
r="{radius_value}"
fill="{fill}"
stroke="#ffffff"
stroke-width="3">
<title>
{label}
|
{friendly_node_type(node_type)}
</title>
</circle>

<text
x="{x:.1f}"
y="{y + radius_value + 16:.1f}"
text-anchor="middle"
font-size="11"
fill="#34465b">
{label[:30]}
</text>
"""
        )

    svg = f"""
<div style="
border:1px solid #dbe5f0;
border-radius:16px;
background:#ffffff;
padding:10px;
overflow:auto;
">

<svg
viewBox="0 0 {width} {height}"
width="100%"
height="620"
xmlns="http://www.w3.org/2000/svg">

<rect
x="0"
y="0"
width="{width}"
height="{height}"
fill="#ffffff"/>

{''.join(edge_svg)}

{''.join(node_svg)}

</svg>

</div>
"""

    st.components.v1.html(
        svg,
        height=650,
        scrolling=True
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="kg-hero">
<div class="kg-eyebrow">731 Series · Product Knowledge</div>
<div class="kg-title">Knowledge Graph Explorer</div>
<div class="kg-subtitle">
Explore how products, configuration families, technical characteristics,
values and configurator rules are connected.
</div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# GRAPH STATUS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        '<div class="info-card">'
        '<div class="info-label">Knowledge Nodes</div>'
        f'<div class="info-value">{len(nodes):,}</div>'
        '</div>',
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        '<div class="info-card">'
        '<div class="info-label">Relationships</div>'
        f'<div class="info-value">{len(edges):,}</div>'
        '</div>',
        unsafe_allow_html=True
    )

with col3:
    config_count = (
        nodes["node_type"]
        == "Configuration"
    ).sum()

    st.markdown(
        '<div class="info-card">'
        '<div class="info-label">Configurations</div>'
        f'<div class="info-value">{config_count:,}</div>'
        '</div>',
        unsafe_allow_html=True
    )

with col4:
    rule_count = (
        nodes["node_type"]
        == "RuleRow"
    ).sum()

    st.markdown(
        '<div class="info-card">'
        '<div class="info-label">Configurator Rules</div>'
        f'<div class="info-value">{rule_count:,}</div>'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# VALIDATION
# ============================================================

if "PASS" in validation:

    st.success(
        "Knowledge Graph validated successfully. "
        "No invalid relationship references or orphan nodes "
        "were reported."
    )

else:

    st.warning(
        "Validation information should be reviewed "
        "before using the graph."
    )


# ============================================================
# SIDEBAR — FIND SOMETHING
# ============================================================

st.sidebar.header(
    "Find product knowledge"
)

st.sidebar.caption(
    "Search the graph and select an entity to explore "
    "its connected information."
)


node_type_options = [
    "All"
] + sorted(
    nodes["node_type"]
    .astype(str)
    .unique()
    .tolist()
)


selected_type = st.sidebar.selectbox(
    "What do you want to explore?",
    node_type_options,
    format_func=lambda x:
        "All entities"
        if x == "All"
        else friendly_node_type(x)
)


search_text = st.sidebar.text_input(
    "Search",
    placeholder="Try: x731, housing, ST, 731..."
)


if selected_type == "All":

    filtered = nodes.copy()

else:

    filtered = nodes[
        nodes["node_type"]
        .astype(str)
        == selected_type
    ].copy()


if search_text.strip():

    query = search_text.strip().lower()

    mask = (

        filtered["label"]
        .astype(str)
        .str.lower()
        .str.contains(
            query,
            regex=False
        )

        |

        filtered["node_id"]
        .astype(str)
        .str.lower()
        .str.contains(
            query,
            regex=False
        )

        |

        filtered["provenance"]
        .astype(str)
        .str.lower()
        .str.contains(
            query,
            regex=False
        )
    )

    filtered = filtered.loc[mask]


st.sidebar.caption(
    f"{len(filtered):,} matching entities"
)


if filtered.empty:

    st.warning(
        "No matching entities were found."
    )

    st.stop()


# ============================================================
# NODE SELECTOR
# ============================================================

selection_labels = []

for _, row in filtered.head(150).iterrows():

    selection_labels.append(
        f'{friendly_node_type(row["node_type"])}'
        f' · {row["label"]}'
    )


selected_label = st.sidebar.selectbox(
    "Select an entity",
    selection_labels
)


selected_index = selection_labels.index(
    selected_label
)

selected = filtered.head(150).iloc[
    selected_index
]

root_id = str(
    selected["node_id"]
)


# ============================================================
# GRAPH SETTINGS
# ============================================================

st.sidebar.divider()

st.sidebar.subheader(
    "Graph view"
)

depth = st.sidebar.radio(
    "How much should I show?",
    [1, 2],
    format_func=lambda x:
        "Direct connections"
        if x == 1
        else "Connections + one more level"
)


max_visible = st.sidebar.slider(
    "Maximum visible entities",
    min_value=10,
    max_value=80,
    value=40,
    step=5
)


# ============================================================
# SELECTED ENTITY
# ============================================================

st.markdown(
    '<div class="section-title">Selected entity</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-caption">'
    'This is the entity you selected from the knowledge graph.'
    '</div>',
    unsafe_allow_html=True
)


entity_col1, entity_col2 = st.columns(
    [2.5, 1]
)


with entity_col1:

    st.markdown(
        f"""
<div class="entity-card">

<div class="entity-type">
{html.escape(
    friendly_node_type(
        selected["node_type"]
    )
)}
</div>

<div class="entity-title">
{html.escape(
    str(selected["label"])
)}
</div>

<div class="entity-meta">
{html.escape(
    node_type_description(
        selected["node_type"]
    )
)}
</div>

</div>
""",
        unsafe_allow_html=True
    )


with entity_col2:

    provenance = str(
        selected["provenance"]
    )

    st.metric(
        "Provenance",
        provenance
    )


# ============================================================
# EXPLANATION
# ============================================================

st.markdown(
    """
<div class="explanation-card">

<b>How to read this:</b>
The graph shows relationships that already exist in the generated
731 Knowledge Graph. Real workbook rows are represented as
configurator-rule knowledge, while synthetic configuration records
retain their synthetic provenance.

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# NEIGHBOURHOOD
# ============================================================

neighbour_ids = get_neighbourhood(
    root_id,
    depth
)


sub_nodes = nodes[
    nodes["node_id"]
    .astype(str)
    .isin(neighbour_ids)
].copy()


# Limit visible graph

if len(sub_nodes) > max_visible:

    direct_edges = edges[
        (
            edges["source_id"]
            .astype(str)
            == root_id
        )
        |
        (
            edges["target_id"]
            .astype(str)
            == root_id
        )
    ]

    direct_ids = set(
        direct_edges["source_id"]
        .astype(str)
    )

    direct_ids.update(
        direct_edges["target_id"]
        .astype(str)
    )

    keep = {
        root_id
    } | (
        direct_ids
        & neighbour_ids
    )

    remaining = sorted(
        neighbour_ids
        - keep
    )

    keep.update(
        remaining[
            :max(
                max_visible - len(keep),
                0
            )
        ]
    )

    sub_nodes = nodes[
        nodes["node_id"]
        .astype(str)
        .isin(keep)
    ].copy()


sub_ids = set(
    sub_nodes["node_id"]
    .astype(str)
)


sub_edges = edges[
    edges["source_id"]
    .astype(str)
    .isin(sub_ids)
    &
    edges["target_id"]
    .astype(str)
    .isin(sub_ids)
].copy()


# ============================================================
# GRAPH VIEW
# ============================================================

st.markdown(
    '<div class="section-title">Relationship map</div>',
    unsafe_allow_html=True
)

st.markdown(
    f"""
<div class="section-caption">
Showing {len(sub_nodes):,} connected entities and
{sub_edges.shape[0]:,} relationships around the selected entity.
</div>
""",
    unsafe_allow_html=True
)


render_graph(
    sub_nodes,
    sub_edges,
    root_id
)


# ============================================================
# RELATIONSHIP TABLE
# ============================================================

st.markdown(
    '<div class="section-title">Connected information</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-caption">'
    'These are the actual relationships stored in the graph.'
    '</div>',
    unsafe_allow_html=True
)


connected = edges[
    (
        edges["source_id"]
        .astype(str)
        == root_id
    )
    |
    (
        edges["target_id"]
        .astype(str)
        == root_id
    )
].copy()


if connected.empty:

    st.info(
        "No direct relationships were found for this entity."
    )

else:

    rows = []

    for _, edge in connected.iterrows():

        source = str(
            edge["source_id"]
        )

        target = str(
            edge["target_id"]
        )

        if source == root_id:

            other_id = target
            direction = "→"

        else:

            other_id = source
            direction = "←"


        other = get_node(
            other_id
        )


        rows.append(
            {
                "Direction":
                    direction,

                "Relationship":
                    str(edge["relation"]),

                "Connected entity":
                    (
                        str(other["label"])
                        if other is not None
                        else other_id
                    ),

                "Entity type":
                    (
                        friendly_node_type(
                            other["node_type"]
                        )
                        if other is not None
                        else ""
                    ),

                "Provenance":
                    str(edge["provenance"]),
            }
        )


    relationship_df = pd.DataFrame(
        rows
    )


    st.dataframe(
        relationship_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# SOURCE & PROVENANCE
# ============================================================

with st.expander(
    "Source and provenance"
):

    st.write(
        "The Knowledge Graph combines product knowledge from "
        "the real 731 configurator workbook with synthetic "
        "configuration records."
    )

    st.write(
        "**Selected entity source:**"
    )

    st.code(
        str(selected["source"])
        if selected["source"]
        else "Not specified"
    )

    st.write(
        "**Provenance distribution**"
    )

    provenance_table = (
        nodes["provenance"]
        .value_counts()
        .rename_axis("Provenance")
        .reset_index(name="Nodes")
    )

    st.dataframe(
        provenance_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DATA INSPECTION
# ============================================================

with st.expander(
    "Technical data inspection"
):

    tab1, tab2, tab3 = st.tabs(
        [
            "Nodes",
            "Relationships",
            "Graph statistics"
        ]
    )

    with tab1:

        st.caption(
            "Raw node records from the generated graph."
        )

        st.dataframe(
            nodes.head(500),
            use_container_width=True,
            hide_index=True
        )

    with tab2:

        st.caption(
            "Raw relationship records from the generated graph."
        )

        st.dataframe(
            edges.head(500),
            use_container_width=True,
            hide_index=True
        )

    with tab3:

        st.json(
            stats
        )


# ============================================================
# VALIDATION
# ============================================================

with st.expander(
    "Knowledge Graph validation"
):

    st.code(
        validation
        if validation
        else "Validation report not available.",
        language="text"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div style="
margin-top:2rem;
padding-top:1rem;
border-top:1px solid #e1e7ee;
color:#78879a;
font-size:0.78rem;
">
731 Knowledge Graph Explorer · Read-only view ·
Source-grounded product knowledge · Synthetic records retain provenance
</div>
""",
    unsafe_allow_html=True
)