#!/usr/bin/env python3

"""
731 Knowledge Graph V2 Explorer

Read-only visual explorer for the V2 731 Knowledge Graph.

Reads:
    graphs/731_knowledge_graph_v2/nodes.csv
    graphs/731_knowledge_graph_v2/relationships.csv
    graphs/731_knowledge_graph_v2/statistics.json
    graphs/731_knowledge_graph_v2/validation_report.txt

Does not modify or rebuild the graph.
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

GRAPH_DIR = (
    PROJECT_ROOT
    / "graphs"
    / "731_knowledge_graph_v2"
)

NODES_FILE = GRAPH_DIR / "nodes.csv"
EDGES_FILE = GRAPH_DIR / "relationships.csv"
STATS_FILE = GRAPH_DIR / "statistics.json"
VALIDATION_FILE = GRAPH_DIR / "validation_report.txt"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="731 Knowledge Graph V2",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# VISUAL THEME
# ============================================================

st.markdown(
    """
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}


/* ============================================================
   HERO
   ============================================================ */

.kg-hero {
    background: #f7faff;
    border: 1px solid #dbe5f0;
    border-radius: 18px;
    padding: 1.55rem 1.75rem;
    margin-bottom: 1.25rem;
}

.kg-eyebrow {
    color: #58708f;
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.kg-title {
    color: #173b70;
    font-size: 2rem;
    font-weight: 800;
    margin-top: 0.2rem;
}

.kg-subtitle {
    color: #627286;
    font-size: 0.98rem;
    margin-top: 0.4rem;
    max-width: 950px;
    line-height: 1.55;
}


/* ============================================================
   METRIC CARDS
   ============================================================ */

.metric-card {
    background: #ffffff;
    border: 1px solid #dbe5f0;
    border-radius: 15px;
    padding: 1rem 1.15rem;
    min-height: 105px;
}

.metric-label {
    color: #6c7c90;
    font-size: 0.72rem;
    font-weight: 750;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

.metric-value {
    color: #173b70;
    font-size: 1.55rem;
    font-weight: 800;
    margin-top: 0.25rem;
}


/* ============================================================
   SECTION HEADINGS
   ============================================================ */

.section-title {
    color: #1d2f45;
    font-size: 1.25rem;
    font-weight: 750;
    margin-top: 1.45rem;
    margin-bottom: 0.2rem;
}

.section-caption {
    color: #708096;
    font-size: 0.87rem;
    margin-bottom: 0.8rem;
}


/* ============================================================
   ENTITY CARD
   ============================================================ */

.entity-card {
    background: #ffffff;
    border: 1px solid #d8e2ed;
    border-radius: 16px;
    padding: 1.25rem 1.35rem;
}

.entity-type {
    color: #5c718b;
    font-size: 0.72rem;
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

.entity-description {
    color: #65758a;
    font-size: 0.87rem;
    margin-top: 0.45rem;
    line-height: 1.5;
}


/* ============================================================
   INFORMATION BOX
   ============================================================ */

.info-box {
    background: #f7faff;
    border-left: 4px solid #4f7fbf;
    border-radius: 9px;
    padding: 0.85rem 1rem;
    color: #465a72;
    font-size: 0.9rem;
    line-height: 1.5;
}


/* ============================================================
   PROVENANCE
   ============================================================ */

.provenance-observed {
    display: inline-block;
    background: #eef5ff;
    color: #285c91;
    border: 1px solid #cfe0f4;
    padding: 0.25rem 0.55rem;
    border-radius: 7px;
    font-size: 0.76rem;
    font-weight: 750;
}

.provenance-synthetic {
    display: inline-block;
    background: #f4f0ff;
    color: #6746a5;
    border: 1px solid #ded2f5;
    padding: 0.25rem 0.55rem;
    border-radius: 7px;
    font-size: 0.76rem;
    font-weight: 750;
}


/* ============================================================
   SIDEBAR
   ============================================================ */

section[data-testid="stSidebar"] {
    border-right: 1px solid #e2e8f0;
}


/* ============================================================
   FOOTER
   ============================================================ */

.kg-footer {
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #e1e7ee;
    color: #78879a;
    font-size: 0.76rem;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HUMAN-READABLE ENTITY NAMES
# ============================================================

TYPE_LABELS = {

    "Product":
        "Product",

    "ConfigurationFamily":
        "Configuration Family",

    "FamilyEvidence":
        "Family Evidence",

    "Characteristic":
        "Technical Characteristic",

    "CharacteristicValue":
        "Characteristic Value",

    "RuleRow":
        "Configurator Rule",

    "CanonicalConfiguration":
        "Canonical Configuration",

    "ConfigurationRecord":
        "Configuration Record",
}


TYPE_DESCRIPTIONS = {

    "Product":
        "The 731 product represented by the source configurator data.",

    "ConfigurationFamily":
        "A canonical configuration-family entity used to group related configurations.",

    "FamilyEvidence":
        "Source evidence describing how a configuration family is represented in the underlying data.",

    "Characteristic":
        "A technical field or attribute used by the 731 configurator.",

    "CharacteristicValue":
        "A value associated with a technical characteristic.",

    "RuleRow":
        "A row from the real 731 configurator workbook represented as product knowledge.",

    "CanonicalConfiguration":
        "A canonical configuration identity from the synthetic configuration catalogue.",

    "ConfigurationRecord":
        "An individual synthetic configuration record preserved with its source-row provenance.",
}


RELATIONSHIP_LABELS = {

    "HAS_CHARACTERISTIC":
        "has technical characteristic",

    "HAS_ALLOWED_VALUE":
        "has allowed value",

    "HAS_RULE_ROW":
        "contains configurator rule",

    "RULE_APPLIES_TO_FAMILY":
        "rule applies to family",

    "RULE_SPECIFIES":
        "rule specifies value",

    "HAS_OBSERVED_EXPRESSION":
        "has observed expression",

    "HAS_SYNTHETIC_PACKAGE_CONTEXT":
        "has synthetic package context",

    "HAS_CANONICAL_CONFIGURATION":
        "has canonical configuration",

    "REPRESENTS_CANONICAL_CONFIGURATION":
        "represents canonical configuration",

    "HAS_PACKAGE_CONTEXT":
        "has package context",

    "HAS_TECHNICAL_VALUE":
        "has technical value",

    "BELONGS_TO_FAMILY":
        "belongs to family",
}


# ============================================================
# LOAD GRAPH
# ============================================================

@st.cache_data
def load_graph():

    if not NODES_FILE.exists():

        raise FileNotFoundError(
            f"Nodes file not found:\n{NODES_FILE}"
        )

    if not EDGES_FILE.exists():

        raise FileNotFoundError(
            f"Relationships file not found:\n{EDGES_FILE}"
        )

    nodes_df = pd.read_csv(
        NODES_FILE,
        low_memory=False
    ).fillna("")

    edges_df = pd.read_csv(
        EDGES_FILE,
        low_memory=False
    ).fillna("")

    graph_stats = {}

    if STATS_FILE.exists():

        graph_stats = json.loads(
            STATS_FILE.read_text(
                encoding="utf-8"
            )
        )

    validation_text = ""

    if VALIDATION_FILE.exists():

        validation_text = (
            VALIDATION_FILE.read_text(
                encoding="utf-8"
            )
        )

    return (
        nodes_df,
        edges_df,
        graph_stats,
        validation_text
    )


# ============================================================
# LOAD
# ============================================================

try:

    nodes, edges, stats, validation = load_graph()

except Exception as exc:

    st.error(
        "Knowledge Graph could not be loaded."
    )

    st.exception(exc)

    st.stop()


# ============================================================
# BASIC DATA VALIDATION
# ============================================================

required_node_columns = {
    "node_id",
    "node_type",
    "label",
    "provenance",
}

required_edge_columns = {
    "source_id",
    "target_id",
    "relation",
    "provenance",
}


missing_node_columns = (
    required_node_columns
    - set(nodes.columns)
)


missing_edge_columns = (
    required_edge_columns
    - set(edges.columns)
)


if missing_node_columns:

    st.error(
        "The V2 nodes file is missing columns: "
        + ", ".join(
            sorted(missing_node_columns)
        )
    )

    st.stop()


if missing_edge_columns:

    st.error(
        "The V2 relationships file is missing columns: "
        + ", ".join(
            sorted(missing_edge_columns)
        )
    )

    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def friendly_type(value):

    return TYPE_LABELS.get(
        str(value),
        str(value)
    )


def friendly_relation(value):

    return RELATIONSHIP_LABELS.get(
        str(value),
        str(value)
    )


def description_for_type(value):

    return TYPE_DESCRIPTIONS.get(
        str(value),
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


def get_neighbourhood(
    root_id,
    depth
):

    root_id = str(root_id)

    visited = {
        root_id
    }

    frontier = {
        root_id
    }

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

        connected = edges.loc[
            mask
        ]

        next_ids = set(
            connected[
                "source_id"
            ].astype(str)
        )

        next_ids.update(
            connected[
                "target_id"
            ].astype(str)
        )

        next_ids -= visited

        visited.update(
            next_ids
        )

        frontier = next_ids

    return visited


# ============================================================
# LIGHTWEIGHT GRAPH VISUALIZATION
# ============================================================

def draw_graph(
    graph_nodes,
    graph_edges,
    root_id,
    max_nodes
):

    root_id = str(root_id)

    all_ids = (
        graph_nodes["node_id"]
        .astype(str)
        .tolist()
    )

    if root_id not in all_ids:
        return

    ordered_ids = [
        root_id
    ] + [
        node_id
        for node_id in all_ids
        if node_id != root_id
    ]

    ordered_ids = ordered_ids[
        :max_nodes
    ]

    width = 1100
    height = 620

    center_x = width / 2
    center_y = height / 2

    positions = {
        root_id: (
            center_x,
            center_y
        )
    }

    others = [
        node_id
        for node_id in ordered_ids
        if node_id != root_id
    ]

    radius = min(
        255,
        max(
            150,
            34 * math.sqrt(
                max(
                    len(others),
                    1
                )
            )
        )
    )

    for i, node_id in enumerate(
        others
    ):

        angle = (
            2
            * math.pi
            * i
            / max(
                len(others),
                1
            )
        ) - math.pi / 2

        positions[node_id] = (

            center_x
            + radius
            * math.cos(angle),

            center_y
            + radius
            * math.sin(angle),
        )


    colors = {

        "Product":
            "#173b70",

        "ConfigurationFamily":
            "#7355b5",

        "FamilyEvidence":
            "#9b83d1",

        "Characteristic":
            "#3d7ec1",

        "CharacteristicValue":
            "#3b9b76",

        "RuleRow":
            "#697b91",

        "CanonicalConfiguration":
            "#d58a20",

        "ConfigurationRecord":
            "#c87535",
    }


    visible_ids = set(
        ordered_ids
    )


    # --------------------------------------------------------
    # EDGES
    # --------------------------------------------------------

    edge_svg = []

    for _, edge in graph_edges.iterrows():

        source = str(
            edge["source_id"]
        )

        target = str(
            edge["target_id"]
        )

        if (
            source not in visible_ids
            or target not in visible_ids
        ):
            continue

        x1, y1 = positions[
            source
        ]

        x2, y2 = positions[
            target
        ]

        relation = html.escape(
            friendly_relation(
                edge["relation"]
            )
        )

        edge_svg.append(
            f"""
<line
x1="{x1:.1f}"
y1="{y1:.1f}"
x2="{x2:.1f}"
y2="{y2:.1f}"
stroke="#c7d2df"
stroke-width="1.4">

<title>{relation}</title>

</line>
"""
        )


    # --------------------------------------------------------
    # NODES
    # --------------------------------------------------------

    node_svg = []

    for _, row in graph_nodes.iterrows():

        node_id = str(
            row["node_id"]
        )

        if node_id not in positions:
            continue

        x, y = positions[
            node_id
        ]

        node_type = str(
            row["node_type"]
        )

        label = html.escape(
            str(row["label"])
        )

        fill = colors.get(
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
{friendly_type(node_type)}
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


    # --------------------------------------------------------
    # SVG
    # --------------------------------------------------------

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

<div class="kg-eyebrow">
731 Series · Product Knowledge · V2
</div>

<div class="kg-title">
Knowledge Graph Explorer
</div>

<div class="kg-subtitle">
Explore the relationships between the real 731 configurator
knowledge, configuration families, technical characteristics,
configuration identities and synthetic configuration records.
</div>

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# METRICS
# ============================================================

canonical_count = (
    nodes["node_type"]
    == "CanonicalConfiguration"
).sum()


record_count = (
    nodes["node_type"]
    == "ConfigurationRecord"
).sum()


family_count = (
    nodes["node_type"]
    == "ConfigurationFamily"
).sum()


rule_count = (
    nodes["node_type"]
    == "RuleRow"
).sum()


metric_values = [

    (
        "Knowledge Nodes",
        len(nodes)
    ),

    (
        "Relationships",
        len(edges)
    ),

    (
        "Configuration Records",
        record_count
    ),

    (
        "Canonical Configurations",
        canonical_count
    ),

    (
        "Configurator Rules",
        rule_count
    ),
]


metric_columns = st.columns(
    5
)


for column, (
    label,
    value
) in zip(
    metric_columns,
    metric_values
):

    with column:

        st.markdown(
            f"""
<div class="metric-card">

<div class="metric-label">
{label}
</div>

<div class="metric-value">
{value:,}
</div>

</div>
""",
            unsafe_allow_html=True
        )


# ============================================================
# VALIDATION STATUS
# ============================================================

valid_node_ids = set(
    nodes["node_id"]
    .astype(str)
)


invalid_source_count = (
    ~edges["source_id"]
    .astype(str)
    .isin(valid_node_ids)
).sum()


invalid_target_count = (
    ~edges["target_id"]
    .astype(str)
    .isin(valid_node_ids)
).sum()


# Read orphan status from the validation report.

validation_pass = (
    "Orphan nodes: 0"
    in validation
)


if (
    invalid_source_count == 0
    and invalid_target_count == 0
    and validation_pass
):

    st.success(
        "Knowledge Graph V2 validated successfully. "
        "All relationship references are valid and "
        "no orphan nodes were reported."
    )

else:

    st.warning(
        "Review the V2 validation report before using the graph."
    )


# ============================================================
# HOW THE GRAPH WORKS
# ============================================================

st.markdown(
    '<div class="section-title">'
    'How the V2 graph is organised'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-caption">'
    'The graph separates real configurator knowledge from '
    'synthetic configuration evidence.'
    '</div>',
    unsafe_allow_html=True
)


st.markdown(
    """
<div class="info-box">

<b>Real 731 configurator knowledge</b><br>
Product → Technical Characteristics → Values → Configurator Rules

<br><br>

<b>Configuration knowledge</b><br>
Configuration Records → Canonical Configurations → Configuration Families

<br><br>

<b>Provenance</b><br>
Every entity retains its source/provenance information so that
observed 731 knowledge is not confused with synthetic configuration data.

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Explore product knowledge"
)

st.sidebar.caption(
    "Choose an entity type, search for an entity, "
    "and inspect its relationships."
)


type_options = [
    "All"
] + sorted(
    nodes["node_type"]
    .astype(str)
    .unique()
    .tolist()
)


selected_type = st.sidebar.selectbox(
    "Explore by",
    type_options,
    format_func=lambda value:
        (
            "All entities"
            if value == "All"
            else friendly_type(value)
        )
)


search_text = st.sidebar.text_input(
    "Search",
    placeholder=(
        "Try: x731, housing, ST, CFG731..."
    )
)


# ============================================================
# FILTER NODES
# ============================================================

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

    label_match = (
        filtered["label"]
        .astype(str)
        .str.lower()
        .str.contains(
            query,
            regex=False
        )
    )

    id_match = (
        filtered["node_id"]
        .astype(str)
        .str.lower()
        .str.contains(
            query,
            regex=False
        )
    )

    provenance_match = (
        filtered["provenance"]
        .astype(str)
        .str.lower()
        .str.contains(
            query,
            regex=False
        )
    )

    filtered = filtered[
        label_match
        | id_match
        | provenance_match
    ]


filtered = filtered.head(
    150
)


st.sidebar.caption(
    f"{len(filtered):,} matching entities"
)


if filtered.empty:

    st.warning(
        "No matching entities were found. "
        "Try a broader search."
    )

    st.stop()


# ============================================================
# SELECT ENTITY
# ============================================================

selection_labels = []

for _, row in filtered.iterrows():

    selection_labels.append(
        f'{friendly_type(row["node_type"])}'
        f' · {row["label"]}'
    )


selected_label = st.sidebar.selectbox(
    "Select entity",
    selection_labels
)


selected_index = selection_labels.index(
    selected_label
)


selected = filtered.iloc[
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
    "Relationship view"
)


depth = st.sidebar.radio(
    "View depth",
    [1, 2],
    format_func=lambda value:
        (
            "Direct relationships"
            if value == 1
            else "Two relationship levels"
        )
)


max_nodes = st.sidebar.slider(
    "Visible entities",
    min_value=10,
    max_value=80,
    value=40,
    step=5
)


# ============================================================
# SELECTED ENTITY
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Selected entity'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-caption">'
    'The entity selected from the V2 knowledge graph.'
    '</div>',
    unsafe_allow_html=True
)


entity_left, entity_right = st.columns(
    [2.7, 1]
)


with entity_left:

    st.markdown(
        f"""
<div class="entity-card">

<div class="entity-type">
{html.escape(
    friendly_type(
        selected["node_type"]
    )
)}
</div>

<div class="entity-title">
{html.escape(
    str(selected["label"])
)}
</div>

<div class="entity-description">
{html.escape(
    description_for_type(
        selected["node_type"]
    )
)}
</div>

</div>
""",
        unsafe_allow_html=True
    )


with entity_right:

    st.metric(
        "Provenance",
        str(
            selected["provenance"]
        )
    )


# ============================================================
# ENTITY DETAILS
# ============================================================

detail_fields = [

    (
        "Source",
        "source"
    ),

    (
        "Source row",
        "source_row"
    ),

    (
        "Package context",
        "package_context"
    ),

    (
        "Canonical configuration",
        "canonical_configuration_id"
    ),

    (
        "Record type",
        "record_type"
    ),

    (
        "Generation method",
        "generation_method"
    ),
]


details = []


for display_name, field in detail_fields:

    if field not in selected.index:
        continue

    value = str(
        selected[field]
    )

    if not value:
        continue

    details.append(
        {
            "Field":
                display_name,

            "Value":
                value,
        }
    )


if details:

    with st.expander(
        "Entity details"
    ):

        st.dataframe(
            pd.DataFrame(details),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# FIND NEIGHBOURHOOD
# ============================================================

visible_ids = get_neighbourhood(
    root_id,
    depth
)


sub_nodes = nodes[
    nodes["node_id"]
    .astype(str)
    .isin(visible_ids)
].copy()


# ------------------------------------------------------------
# LIMIT GRAPH SIZE
# ------------------------------------------------------------

if len(sub_nodes) > max_nodes:

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
        direct_edges[
            "source_id"
        ].astype(str)
    )

    direct_ids.update(
        direct_edges[
            "target_id"
        ].astype(str)
    )


    keep = {
        root_id
    } | (
        direct_ids
        & visible_ids
    )


    remaining = sorted(
        visible_ids
        - keep
    )


    keep.update(
        remaining[
            :max(
                max_nodes
                - len(keep),
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
# RELATIONSHIP MAP
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Relationship map'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    f"""
<div class="section-caption">
Showing {len(sub_nodes):,} connected entities and
{len(sub_edges):,} relationships around the selected entity.
</div>
""",
    unsafe_allow_html=True
)


draw_graph(
    sub_nodes,
    sub_edges,
    root_id,
    max_nodes
)


# ============================================================
# CONNECTED INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Connected information'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-caption">'
    'These are the relationships actually stored in the V2 graph.'
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
        "No direct relationships were found."
    )

else:

    relationship_rows = []

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


        relationship_rows.append(
            {
                "Direction":
                    direction,

                "Relationship":
                    friendly_relation(
                        edge["relation"]
                    ),

                "Connected entity":
                    (
                        str(
                            other["label"]
                        )
                        if other is not None
                        else other_id
                    ),

                "Entity type":
                    (
                        friendly_type(
                            other["node_type"]
                        )
                        if other is not None
                        else ""
                    ),

                "Provenance":
                    str(
                        edge["provenance"]
                    ),
            }
        )


    relationship_df = pd.DataFrame(
        relationship_rows
    )


    st.dataframe(
        relationship_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# CONFIGURATION RECORD SPECIAL VIEW
# ============================================================

if (
    selected["node_type"]
    == "ConfigurationRecord"
):

    st.markdown(
        '<div class="section-title">'
        'Configuration identity'
        '</div>',
        unsafe_allow_html=True
    )


    record_edges = edges[
        (
            edges["source_id"]
            .astype(str)
            == root_id
        )
    ]


    canonical_links = record_edges[
        record_edges["relation"]
        == "REPRESENTS_CANONICAL_CONFIGURATION"
    ]


    if not canonical_links.empty:

        canonical_id = str(
            canonical_links.iloc[0][
                "target_id"
            ]
        )


        canonical = get_node(
            canonical_id
        )


        if canonical is not None:

            st.info(
                "This synthetic configuration record "
                "represents the canonical configuration: "
                f"{canonical['label']}"
            )


# ============================================================
# CANONICAL CONFIGURATION SPECIAL VIEW
# ============================================================

if (
    selected["node_type"]
    == "CanonicalConfiguration"
):

    st.markdown(
        '<div class="section-title">'
        'Configuration records'
        '</div>',
        unsafe_allow_html=True
    )


    record_links = edges[
        (
            edges["target_id"]
            .astype(str)
            == root_id
        )
        &
        (
            edges["relation"]
            == "REPRESENTS_CANONICAL_CONFIGURATION"
        )
    ]


    if record_links.empty:

        st.info(
            "No configuration records are linked "
            "to this canonical configuration."
        )

    else:

        record_rows = []


        for _, edge in record_links.iterrows():

            record = get_node(
                edge["source_id"]
            )


            if record is None:
                continue


            record_rows.append(
                {
                    "Record":
                        record["label"],

                    "Source row":
                        record["source_row"],

                    "Package context":
                        record["package_context"],

                    "Record type":
                        record["record_type"],

                    "Generation method":
                        record["generation_method"],

                    "Provenance":
                        record["provenance"],
                }
            )


        st.dataframe(
            pd.DataFrame(
                record_rows
            ),
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
        "V2 separates observed 731 configurator knowledge "
        "from synthetic configuration evidence."
    )


    provenance_table = (
        nodes[
            [
                "node_type",
                "provenance"
            ]
        ]
        .value_counts()
        .reset_index(
            name="Count"
        )
    )


    provenance_table[
        "node_type"
    ] = provenance_table[
        "node_type"
    ].map(
        friendly_type
    )


    st.dataframe(
        provenance_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TECHNICAL GRAPH DATA
# ============================================================

with st.expander(
    "Technical graph data"
):

    tab1, tab2, tab3 = st.tabs(
        [
            "Nodes",
            "Relationships",
            "Statistics"
        ]
    )


    with tab1:

        st.caption(
            "Raw node records from the generated V2 graph."
        )

        st.dataframe(
            nodes.head(500),
            use_container_width=True,
            hide_index=True
        )


    with tab2:

        st.caption(
            "Raw relationship records from the generated V2 graph."
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
# VALIDATION REPORT
# ============================================================

with st.expander(
    "Knowledge Graph validation"
):

    st.code(
        validation
        if validation
        else "Validation report unavailable.",
        language="text"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="kg-footer">

731 Knowledge Graph V2 · Read-only explorer ·
Real configurator knowledge remains distinguished from
synthetic configuration evidence ·
Undocumented technical-code meanings are not inferred.

</div>
""",
    unsafe_allow_html=True
)