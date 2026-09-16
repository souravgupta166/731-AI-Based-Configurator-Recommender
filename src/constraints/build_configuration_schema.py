import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCHEMA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "731_schema_initial.csv"
)

VALUE_PROFILE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "731_configuration_value_profile.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


# ============================================================
# LOAD
# ============================================================

schema = pd.read_csv(
    SCHEMA_FILE
)

value_profile = pd.read_csv(
    VALUE_PROFILE_FILE
)


# ============================================================
# CORE CONFIGURATION CHARACTERISTICS
# ============================================================

CORE_CHARACTERISTICS = [

    "packageFixed_dev",

    "devCategory",

    "characteristic",

    "housing",

    "powerSupply",

    "numberOfChannels",

    "explosionApproval",

    "protectionArea",

    "certification",

    "dataInterface",

    "stromSchaltbar",

    "stromEingaenge",

    "temperaturEingaenge",
]


# ============================================================
# CONDITIONAL / OPTIONAL FEATURES
# ============================================================

OPTIONAL_CHARACTERISTICS = [

    "binaerDigitalOpenColl_MN",

    "binaerOpenColl_MP",

    "steamApplication",

    "waveInjector",

    "dev_advMeterVerification",

    "dynamicGasMaster",

    "customUserFluid",
]


# ============================================================
# CONTEXT / ADMINISTRATIVE
# ============================================================

CONTEXT_CHARACTERISTICS = [

    "OrgUnitAreaOfDocument",
]


# ============================================================
# CREATE CLASSIFICATION
# ============================================================

def classify_characteristic(column):

    if column in CORE_CHARACTERISTICS:

        return "core_configuration"

    if column in OPTIONAL_CHARACTERISTICS:

        return "conditional_feature"

    if column in CONTEXT_CHARACTERISTICS:

        return "context_or_administrative"

    return "unclassified"


schema[
    "configuration_role"
] = schema[
    "column_name"
].apply(
    classify_characteristic
)


# ============================================================
# ADD RULE INFORMATION
# ============================================================

profile_summary = (
    value_profile
    .groupby("characteristic")
    .agg(
        observed_rule_states=(
            "raw_value",
            "nunique"
        ),

        total_occurrences=(
            "occurrences",
            "sum"
        ),

        maximum_rule_frequency=(
            "percentage",
            "max"
        )
    )
    .reset_index()
)


schema = schema.merge(
    profile_summary,
    left_on="column_name",
    right_on="characteristic",
    how="left"
)


schema = schema.drop(
    columns=["characteristic"],
    errors="ignore"
)


# ============================================================
# SORT
# ============================================================

role_order = {

    "core_configuration": 1,

    "conditional_feature": 2,

    "context_or_administrative": 3,

    "unclassified": 4,
}


schema[
    "role_order"
] = schema[
    "configuration_role"
].map(
    role_order
)


schema = schema.sort_values(
    [
        "role_order",
        "column_name"
    ]
)


schema = schema.drop(
    columns=["role_order"]
)


# ============================================================
# DISPLAY
# ============================================================

print("=" * 90)
print("731 CONFIGURATION CHARACTERISTIC MODEL")
print("=" * 90)


for role in [

    "core_configuration",

    "conditional_feature",

    "context_or_administrative",

    "unclassified",
]:

    subset = schema[
        schema["configuration_role"]
        == role
    ]

    print("\n" + "-" * 90)
    print(role.upper())
    print("-" * 90)

    if len(subset) == 0:

        print("None")

        continue

    print(
        subset[
            [
                "column_name",
                "unique_values",
                "novalue_percentage",
                "observed_rule_states"
            ]
        ]
        .to_string(
            index=False
        )
    )


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

output_file = (
    OUTPUT_DIR
    / "731_configuration_schema.csv"
)

schema.to_csv(
    output_file,
    index=False
)


print("\n" + "=" * 90)
print("SAVED")
print("=" * 90)

print(output_file)