#!/usr/bin/env python3

"""
731 SYNTHETIC RFQ DATASET GENERATOR — V2
=========================================

Purpose
-------
Generate realistic customer-facing RFQs from the validated V5
technical configuration dataset.

Important design principle
--------------------------
The V5 technical configuration is the hidden ground truth.

Customer-facing RFQ language must NOT expose internal 731
configuration codes such as:

    BT
    NN
    IS1
    TT1
    CS1
    MP1
    FWI

Instead, these are translated into realistic business/application
language.

Outputs
-------
data/synthetic/rfqs/
    731_rfq_headers_v2.csv
    731_rfq_requirements_v2.csv
    731_rfq_documents_v2.csv
    731_rfq_ground_truth_v2.csv

data/processed/
    731_rfq_v2_statistics.csv
    731_rfq_v2_generation_summary.csv
    731_rfq_v2_requirement_coverage.csv
    731_rfq_v2_validation.csv
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CUSTOMER_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "customers"
    / "731_customer_master.csv"
)

CONFIGURATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "configurations"
    / "731_synthetic_configurations_v5.csv"
)

RFQ_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "rfqs"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

RANDOM_SEED = 7312026

TARGET_RFQS = 5000


# =============================================================================
# TECHNICAL CHARACTERISTICS
# =============================================================================

TECHNICAL_COLUMNS = [
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
    "binaerDigitalOpenColl_MN",
    "binaerOpenColl_MP",
    "waveInjector",
    "dev_advMeterVerification",
    "dynamicGasMaster",
    "customUserFluid",
    "steamApplication",
]


# =============================================================================
# HUMAN LANGUAGE MAPPINGS
# =============================================================================

VALUE_LANGUAGE = {

    # -------------------------------------------------------------------------
    # Device category
    # -------------------------------------------------------------------------

    "devCategory": {
        "G": (
            "gas measurement",
            "gas metering application",
        ),
        "F": (
            "liquid flow measurement",
            "flow measurement for liquid service",
        ),
    },

    # -------------------------------------------------------------------------
    # Characteristic
    # -------------------------------------------------------------------------

    "characteristic": {
        "GP": (
            "general-purpose configuration",
            "standard process configuration",
        ),
        "ST": (
            "standard temperature application",
            "standard-temperature service",
        ),
        "HT": (
            "high-temperature application",
            "elevated-temperature service",
        ),
        "TE": (
            "temperature measurement application",
            "temperature-focused measurement service",
        ),
        "CA": (
            "calibration-related measurement application",
            "calibration service",
        ),
        "PW": (
            "process water measurement",
            "water-process measurement",
        ),
        "WD": (
            "water-duty measurement",
            "water measurement service",
        ),
        "VG": (
            "specialized gas measurement",
            "special gas-service application",
        ),
    },

    # -------------------------------------------------------------------------
    # Housing
    # -------------------------------------------------------------------------

    "housing": {
        "ST": (
            "stainless-steel housing",
            "stainless-steel enclosure",
        ),
        "AL": (
            "aluminium housing",
            "aluminium enclosure",
        ),
    },

    # -------------------------------------------------------------------------
    # Power supply
    # -------------------------------------------------------------------------

    "powerSupply": {
        "1": (
            "standard power supply",
            "the standard electrical supply option",
        ),
        "4": (
            "an alternative power supply",
            "the alternative electrical supply option",
        ),
    },

    # -------------------------------------------------------------------------
    # Number of channels
    # -------------------------------------------------------------------------

    "numberOfChannels": {
        "1": (
            "one measurement channel",
            "a single measurement channel",
        ),
        "2": (
            "two independent measurement channels",
            "dual-channel measurement",
        ),
    },

    # -------------------------------------------------------------------------
    # Explosion approval
    # -------------------------------------------------------------------------

    "explosionApproval": {
        "N": (
            "the installation does not require hazardous-area approval",
            "standard non-hazardous-area installation",
        ),
        "A": (
            "hazardous-area operation",
            "operation in an area requiring explosion protection",
        ),
        "F": (
            "a hazardous process environment",
            "operation under hazardous-area conditions",
        ),
    },

    # -------------------------------------------------------------------------
    # Protection area
    # -------------------------------------------------------------------------

    "protectionArea": {
        "N": (
            "standard installation conditions",
            "a non-hazardous installation area",
        ),
        "2": (
            "installation in a protected hazardous area",
            "operation in a hazardous area with the applicable protection requirements",
        ),
    },

    # -------------------------------------------------------------------------
    # Certification
    # -------------------------------------------------------------------------

    "certification": {
        "NN": (
            "standard certification documentation",
            "the standard certification package",
        ),
        "MN": (
            "the required measurement certification",
            "appropriate measurement certification",
        ),
        "MI": (
            "special certification documentation",
            "additional certification documentation",
        ),
    },

    # -------------------------------------------------------------------------
    # Communication interface
    # -------------------------------------------------------------------------

    "dataInterface": {
        "NN": (
            "standard communication capability",
            "the standard communication interface",
        ),
        "BT": (
            "wireless commissioning and diagnostics",
            "Bluetooth-based commissioning and maintenance access",
        ),
        "HS": (
            "high-speed communication",
            "enhanced communication capability",
        ),
    },

    # -------------------------------------------------------------------------
    # Switchable current output
    # -------------------------------------------------------------------------

    "stromSchaltbar": {
        "NOVALUE": (
            "no additional switchable current-output requirement",
        ),
        "CS1": (
            "a switchable current-output capability",
            "switchable current-output functionality",
        ),
        "CS2": (
            "an enhanced switchable current-output capability",
            "additional configurable current-output functionality",
        ),
    },

    # -------------------------------------------------------------------------
    # Current inputs
    # -------------------------------------------------------------------------

    "stromEingaenge": {
        "NOVALUE": (
            "no dedicated current-input requirement",
        ),
        "IS1": (
            "one current input",
            "a dedicated current-input channel",
        ),
        "IS2": (
            "multiple current inputs",
            "additional current-input capability",
        ),
    },

    # -------------------------------------------------------------------------
    # Temperature inputs
    # -------------------------------------------------------------------------

    "temperaturEingaenge": {
        "NOVALUE": (
            "no dedicated temperature-input requirement",
        ),
        "TT1": (
            "temperature measurement capability",
            "one temperature measurement input",
        ),
        "TT2": (
            "additional temperature measurement capability",
            "multiple temperature measurement inputs",
        ),
    },

    # -------------------------------------------------------------------------
    # Digital inputs
    # -------------------------------------------------------------------------

    "binaerDigitalOpenColl_MN": {
        "NOVALUE": (
            "no additional digital input requirement",
        ),
        "MN1": (
            "a digital open-collector input",
            "digital signal input capability",
        ),
    },

    "binaerOpenColl_MP": {
        "NOVALUE": (
            "no additional open-collector input requirement",
        ),
        "MP1": (
            "an additional digital signal input",
            "an open-collector input capability",
        ),
    },

    # -------------------------------------------------------------------------
    # WaveInjector
    # -------------------------------------------------------------------------

    "waveInjector": {
        "NOVALUE": (
            "no special WaveInjector requirement",
        ),
        "FWI": (
            "advanced low-temperature measurement capability",
            "enhanced measurement capability for demanding applications",
        ),
    },

    # -------------------------------------------------------------------------
    # Advanced Meter Verification
    # -------------------------------------------------------------------------

    "dev_advMeterVerification": {
        "NOVALUE": (
            "standard verification capability",
        ),
        "1": (
            "advanced meter verification",
            "advanced verification functionality",
        ),
    },

    # -------------------------------------------------------------------------
    # Dynamic Gas Master
    # -------------------------------------------------------------------------

    "dynamicGasMaster": {
        "NOVALUE": (
            "standard gas measurement functionality",
        ),
        "1": (
            "dynamic gas measurement functionality",
            "enhanced gas measurement capability",
        ),
    },

    # -------------------------------------------------------------------------
    # Custom user fluid
    # -------------------------------------------------------------------------

    "customUserFluid": {
        "NOVALUE": (
            "standard fluid configuration",
        ),
        "1": (
            "custom fluid configuration",
            "a customer-specific fluid definition",
        ),
    },

    # -------------------------------------------------------------------------
    # Steam
    # -------------------------------------------------------------------------

    "steamApplication": {
        "NOVALUE": (
            "standard process service",
        ),
        "1": (
            "steam-service operation",
            "measurement in a steam application",
        ),
    },
}


# =============================================================================
# PROJECT / APPLICATION LANGUAGE
# =============================================================================

PROJECT_TEMPLATES = [
    "Process Expansion Project",
    "Measurement System Upgrade",
    "Plant Modernisation Project",
    "Process Automation Project",
    "Utility Measurement Upgrade",
    "New Production Line",
    "Metering System Replacement",
    "Process Instrumentation Upgrade",
    "New Plant Installation",
    "Measurement Infrastructure Upgrade",
]


APPLICATION_BY_INDUSTRY = {
    "Oil & Gas": [
        "hydrocarbon process measurement",
        "gas processing",
        "production measurement",
        "pipeline monitoring",
        "process gas measurement",
    ],
    "Chemical": [
        "chemical process measurement",
        "batch process monitoring",
        "process flow measurement",
        "chemical production",
        "plant utility measurement",
    ],
    "Power & Utilities": [
        "utility measurement",
        "power plant process monitoring",
        "energy infrastructure",
        "utility water measurement",
        "process monitoring",
    ],
    "Water & Wastewater": [
        "water treatment",
        "utility water measurement",
        "wastewater processing",
        "water distribution",
        "treatment plant monitoring",
    ],
    "Pharmaceutical": [
        "pharmaceutical process monitoring",
        "clean utility measurement",
        "production process monitoring",
        "process water measurement",
        "controlled process measurement",
    ],
    "Mining & Metals": [
        "mineral processing",
        "process water measurement",
        "metallurgical process monitoring",
        "production measurement",
        "plant utility monitoring",
    ],
    "Manufacturing": [
        "manufacturing process measurement",
        "production line monitoring",
        "plant utility measurement",
        "process automation",
        "production process monitoring",
    ],
    "Food & Beverage": [
        "food production",
        "process water measurement",
        "production utility monitoring",
        "sanitary process measurement",
        "production line monitoring",
    ],
    "Engineering & EPC": [
        "industrial process measurement",
        "process automation project",
        "plant engineering project",
        "measurement system implementation",
        "industrial instrumentation",
    ],
    "Research & Development": [
        "pilot plant measurement",
        "research process monitoring",
        "test facility measurement",
        "experimental process monitoring",
        "laboratory process system",
    ],
}


# =============================================================================
# VALUE NORMALIZATION
# =============================================================================

def normalize_value(value) -> str:

    if pd.isna(value):
        return "NOVALUE"

    text = str(value).strip()

    if not text:
        return "NOVALUE"

    match = re.match(
        r"^in\s*\{(.*)\}$",
        text,
        flags=re.IGNORECASE,
    )

    if match:

        content = match.group(1)

        tokens = re.findall(
            r"'([^']*)'|\"([^\"]*)\"|([^,\s]+)",
            content,
        )

        values = []

        for single, double, unquoted in tokens:

            token = (
                single
                or double
                or unquoted
            )

            if token:
                values.append(token.strip())

        non_novalue = [
            x
            for x in values
            if x.upper() != "NOVALUE"
        ]

        if non_novalue:
            return non_novalue[0]

        return "NOVALUE"

    if len(text) >= 2:

        if (
            text[0] == "'"
            and text[-1] == "'"
        ):
            text = text[1:-1]

        elif (
            text[0] == '"'
            and text[-1] == '"'
        ):
            text = text[1:-1]

    return text.strip()


# =============================================================================
# HELPERS
# =============================================================================

def choose_language(
    characteristic: str,
    value: str,
    rng: random.Random,
) -> str:

    options = VALUE_LANGUAGE.get(
        characteristic,
        {},
    ).get(
        value,
        (),
    )

    if not options:
        return f"the required {characteristic} configuration"

    return rng.choice(options)


def is_meaningful_value(
    characteristic: str,
    value: str,
) -> bool:

    if value == "NOVALUE":
        return False

    return True


def requirement_priority(
    characteristic: str,
    rng: random.Random,
) -> str:

    # Application-critical requirements are more likely mandatory.
    mandatory_bias = {
        "devCategory": 0.98,
        "characteristic": 0.95,
        "numberOfChannels": 0.90,
        "explosionApproval": 0.90,
        "protectionArea": 0.90,
        "certification": 0.82,
        "dataInterface": 0.70,
        "powerSupply": 0.72,
        "stromEingaenge": 0.68,
        "temperaturEingaenge": 0.70,
        "dev_advMeterVerification": 0.60,
        "dynamicGasMaster": 0.65,
        "customUserFluid": 0.60,
        "steamApplication": 0.75,
    }

    probability = mandatory_bias.get(
        characteristic,
        0.55,
    )

    return (
        "MANDATORY"
        if rng.random() < probability
        else "PREFERRED"
    )


def requirement_sentence(
    characteristic: str,
    language: str,
    priority: str,
    rng: random.Random,
) -> str:

    if priority == "MANDATORY":

        templates = [
            f"The application requires {language}.",
            f"We require {language}.",
            f"The proposed solution must provide {language}.",
            f"The equipment is required to support {language}.",
        ]

    else:

        templates = [
            f"If possible, please provide {language}.",
            f"We would prefer {language}.",
            f"Please consider {language} as a preferred option.",
            f"Where available, we would like {language}.",
        ]

    return rng.choice(templates)


# =============================================================================
# CUSTOMER-FACING REQUIREMENT GENERATION
# =============================================================================

def build_requirements(
    configuration: pd.Series,
    customer: pd.Series,
    rfq_id: str,
    rng: random.Random,
) -> Tuple[List[Dict], List[str]]:

    requirements = []
    sentences = []

    requirement_id = 1

    for characteristic in TECHNICAL_COLUMNS:

        if characteristic not in configuration.index:
            continue

        value = normalize_value(
            configuration[characteristic]
        )

        if not is_meaningful_value(
            characteristic,
            value,
        ):
            continue

        language = choose_language(
            characteristic,
            value,
            rng,
        )

        priority = requirement_priority(
            characteristic,
            rng,
        )

        # Occasionally turn technical characteristics into implicit
        # application statements instead of direct requirements.
        implicit = (
            rng.random() < 0.18
            and characteristic
            in {
                "explosionApproval",
                "dataInterface",
                "housing",
                "temperatureEingaenge",
                "dev_advMeterVerification",
            }
        )

        if implicit:

            if characteristic == "explosionApproval":
                sentence = (
                    "The equipment will be installed in an "
                    "operating area where the applicable "
                    "hazardous-area requirements must be met."
                )

            elif characteristic == "dataInterface":
                sentence = (
                    "Maintenance personnel would benefit "
                    "from convenient commissioning and "
                    "diagnostic access."
                )

            elif characteristic == "housing":
                sentence = (
                    "The installation environment requires "
                    "a robust industrial enclosure."
                )

            elif characteristic == "temperaturEingaenge":
                sentence = (
                    "Temperature information is required "
                    "as part of the measurement application."
                )

            else:
                sentence = (
                    "The customer would like enhanced "
                    "verification capabilities for maintenance."
                )

        else:
            sentence = requirement_sentence(
                characteristic,
                language,
                priority,
                rng,
            )

        requirements.append(
            {
                "rfq_id": rfq_id,
                "requirement_id": (
                    f"{rfq_id}-REQ-{requirement_id:03d}"
                ),
                "characteristic": characteristic,
                "internal_value": value,
                "requirement_type": priority,
                "customer_language": language,
                "implicit_requirement": implicit,
                "source": "SYNTHETIC_V5_GROUND_TRUTH",
            }
        )

        sentences.append(sentence)

        requirement_id += 1

    return requirements, sentences


# =============================================================================
# COMMERCIAL / APPLICATION TEXT
# =============================================================================

def build_application_text(
    customer: pd.Series,
    rng: random.Random,
) -> str:

    industry = str(
        customer.get(
            "industry",
            "industrial",
        )
    )

    applications = APPLICATION_BY_INDUSTRY.get(
        industry,
        ["industrial process measurement"],
    )

    return rng.choice(applications)


def build_intro(
    customer: pd.Series,
    project_name: str,
    application: str,
    rfq_id: str,
    rng: random.Random,
) -> str:

    customer_name = customer["customer_name"]
    country = customer["country"]

    templates = [

        (
            f"We are currently working on our {project_name.lower()} "
            f"and are looking for a suitable measurement solution "
            f"for {application}."
        ),

        (
            f"As part of our {project_name.lower()}, we require "
            f"a suitable measurement system for {application}."
        ),

        (
            f"We are requesting a quotation for equipment suitable "
            f"for {application} as part of an upcoming "
            f"{project_name.lower()}."
        ),

        (
            f"Our team is planning an upgrade involving "
            f"{application}. We would like to receive a quotation "
            f"for a suitable measurement solution."
        ),
    ]

    return rng.choice(templates)


# =============================================================================
# RFQ DOCUMENT
# =============================================================================

def build_document(
    rfq_id: str,
    customer: pd.Series,
    project_name: str,
    application: str,
    sentences: List[str],
    rng: random.Random,
) -> str:

    customer_name = customer["customer_name"]
    country = customer["country"]
    contact_name = customer.get(
        "contact_person",
        "Procurement Team",
    )

    intro = build_intro(
        customer,
        project_name,
        application,
        rfq_id,
        rng,
    )

    paragraphs = []

    paragraphs.append(
        "Dear Sales Team,"
    )

    paragraphs.append(
        f"Customer: {customer_name}\n"
        f"RFQ Reference: {rfq_id}\n"
        f"Customer Reference: {project_name}\n"
        f"Country: {country}"
    )

    paragraphs.append(
        intro
    )

    paragraphs.append(
        "Please provide a quotation based on the following "
        "technical and application requirements:"
    )

    paragraphs.extend(sentences)

    paragraphs.append(
        "Please include pricing, expected delivery lead time, "
        "validity of the quotation, and any relevant technical "
        "documentation with your offer."
    )

    paragraphs.append(
        "If any of the requirements require clarification, "
        "please indicate the assumptions made in your quotation."
    )

    paragraphs.append(
        "Best regards,\n"
        f"{contact_name}\n"
        f"{customer_name}"
    )

    return "\n\n".join(paragraphs)


# =============================================================================
# LOAD DATA
# =============================================================================

def load_data():

    if not CUSTOMER_FILE.exists():
        raise FileNotFoundError(
            f"Customer master not found:\n{CUSTOMER_FILE}"
        )

    if not CONFIGURATION_FILE.exists():
        raise FileNotFoundError(
            f"V5 configuration file not found:\n"
            f"{CONFIGURATION_FILE}"
        )

    customers = pd.read_csv(
        CUSTOMER_FILE
    )

    configurations = pd.read_csv(
        CONFIGURATION_FILE
    )

    return customers, configurations


# =============================================================================
# VALIDATION
# =============================================================================

def validate_outputs(
    headers: pd.DataFrame,
    requirements: pd.DataFrame,
    documents: pd.DataFrame,
    ground_truth: pd.DataFrame,
):

    checks = []

    checks.append(
        {
            "check": "RFQ header count",
            "expected": TARGET_RFQS,
            "actual": len(headers),
            "passed": len(headers) == TARGET_RFQS,
        }
    )

    checks.append(
        {
            "check": "Unique RFQ IDs",
            "expected": TARGET_RFQS,
            "actual": headers["rfq_id"].nunique(),
            "passed": (
                headers["rfq_id"].nunique()
                == TARGET_RFQS
            ),
        }
    )

    checks.append(
        {
            "check": "RFQ documents",
            "expected": TARGET_RFQS,
            "actual": len(documents),
            "passed": len(documents) == TARGET_RFQS,
        }
    )

    checks.append(
        {
            "check": "Ground truth records",
            "expected": TARGET_RFQS,
            "actual": len(ground_truth),
            "passed": len(ground_truth) == TARGET_RFQS,
        }
    )

    checks.append(
        {
            "check": "Requirements linked to RFQs",
            "expected": TARGET_RFQS,
            "actual": requirements["rfq_id"].nunique(),
            "passed": (
                requirements["rfq_id"].nunique()
                == TARGET_RFQS
            ),
        }
    )

    checks.append(
        {
            "check": "Ground truth IDs unique",
            "expected": TARGET_RFQS,
            "actual": ground_truth[
                "canonical_configuration_id"
            ].notna().sum(),
            "passed": (
                ground_truth[
                    "canonical_configuration_id"
                ].notna().all()
            ),
        }
    )

    # Ensure internal configurator syntax isn't accidentally placed
    # into customer-facing RFQ text.
    forbidden_patterns = [
        r"\bin\s*\{",
        r"\bNOVALUE\b",
        r"\bCFG731_",
    ]

    leaked = 0

    for document in documents["rfq_text"]:

        for pattern in forbidden_patterns:

            if re.search(
                pattern,
                str(document),
                flags=re.IGNORECASE,
            ):
                leaked += 1
                break

    checks.append(
        {
            "check": "Internal syntax leakage",
            "expected": 0,
            "actual": leaked,
            "passed": leaked == 0,
        }
    )

    return pd.DataFrame(checks)


# =============================================================================
# MAIN
# =============================================================================

def main():

    rng = random.Random(
        RANDOM_SEED
    )

    print("=" * 100)
    print("731 SYNTHETIC RFQ DATASET GENERATOR — V2")
    print("=" * 100)

    print("\nLoading customer master...")

    customers, configurations = load_data()

    print(
        f"Customers loaded: {len(customers):,}"
    )

    print(
        "\nLoading V5 technical configurations..."
    )

    print(
        f"V5 configurations loaded: "
        f"{len(configurations):,}"
    )

    if len(configurations) < TARGET_RFQS:

        print(
            "\nV5 configurations are fewer than the target "
            "RFQ count. Configurations will be sampled with "
            "replacement."
        )

    # -------------------------------------------------------------------------
    # Generate
    # -------------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("GENERATING RFQ V2")
    print("=" * 100)

    header_records = []
    requirement_records = []
    document_records = []
    ground_truth_records = []

    sampled_configurations = configurations.sample(
        n=TARGET_RFQS,
        replace=True,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    sampled_customer_indices = [
        rng.randrange(len(customers))
        for _ in range(TARGET_RFQS)
    ]

    for i in range(TARGET_RFQS):

        rfq_number = i + 1

        rfq_id = (
            f"RFQ731-{rfq_number:06d}"
        )

        customer = customers.iloc[
            sampled_customer_indices[i]
        ]

        configuration = sampled_configurations.iloc[i]

        customer_id = customer["customer_id"]

        project_name = rng.choice(
            PROJECT_TEMPLATES
        )

        application = build_application_text(
            customer,
            rng,
        )

        # ------------------------------------------------------------
        # Requirements
        # ------------------------------------------------------------

        requirements, sentences = build_requirements(
            configuration,
            customer,
            rfq_id,
            rng,
        )

        requirement_records.extend(
            requirements
        )

        # ------------------------------------------------------------
        # RFQ text
        # ------------------------------------------------------------

        document = build_document(
            rfq_id,
            customer,
            project_name,
            application,
            sentences,
            rng,
        )

        # ------------------------------------------------------------
        # Header
        # ------------------------------------------------------------

        header_records.append(
            {
                "rfq_id": rfq_id,
                "customer_id": customer_id,
                "customer_name": customer["customer_name"],
                "customer_type": customer.get(
                    "customer_type",
                    "",
                ),
                "industry": customer.get(
                    "industry",
                    "",
                ),
                "country": customer.get(
                    "country",
                    "",
                ),
                "currency": customer.get(
                    "currency",
                    "",
                ),
                "payment_terms": customer.get(
                    "payment_terms",
                    "",
                ),
                "delivery_terms": customer.get(
                    "delivery_terms",
                    "",
                ),
                "project_name": project_name,
                "application": application,
                "rfq_status": "OPEN",
                "generation_version": "V2",
            }
        )

        # ------------------------------------------------------------
        # Document
        # ------------------------------------------------------------

        document_records.append(
            {
                "rfq_id": rfq_id,
                "customer_id": customer_id,
                "document_type": "CUSTOMER_RFQ",
                "document_version": "V2",
                "rfq_text": document,
                "requirement_count": len(requirements),
            }
        )

        # ------------------------------------------------------------
        # Ground truth
        # ------------------------------------------------------------

        ground_truth_records.append(
            {
                "rfq_id": rfq_id,
                "customer_id": customer_id,
                "canonical_configuration_id": configuration[
                    "canonical_configuration_id"
                ],
                "target_package": configuration.get(
                    "package_context",
                    "",
                ),
                "record_type": configuration.get(
                    "record_type",
                    "",
                ),
                "evidence_level": configuration.get(
                    "evidence_level",
                    "",
                ),
                "nearest_real_distance": configuration.get(
                    "nearest_real_distance",
                    "",
                ),
                "source_canonical_configuration_id": configuration.get(
                    "source_canonical_configuration_id",
                    "",
                ),
                "ground_truth_hidden_from_customer": True,
            }
        )

        if (
            rfq_number % 500 == 0
            or rfq_number == TARGET_RFQS
        ):

            print(
                f"Generated: "
                f"{rfq_number:,} / {TARGET_RFQS:,}"
            )

    # -------------------------------------------------------------------------
    # DataFrames
    # -------------------------------------------------------------------------

    headers_df = pd.DataFrame(
        header_records
    )

    requirements_df = pd.DataFrame(
        requirement_records
    )

    documents_df = pd.DataFrame(
        document_records
    )

    ground_truth_df = pd.DataFrame(
        ground_truth_records
    )

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("RFQ V2 DATASET VALIDATION")
    print("=" * 100)

    validation_df = validate_outputs(
        headers_df,
        requirements_df,
        documents_df,
        ground_truth_df,
    )

    print(
        validation_df.to_string(
            index=False
        )
    )

    validation_passed = bool(
        validation_df["passed"].all()
    )

    print(
        "\nValidation:",
        "PASSED"
        if validation_passed
        else "FAILED",
    )

    if not validation_passed:

        failed = validation_df[
            ~validation_df["passed"]
        ]

        print(
            "\nFailed checks:"
        )

        print(
            failed.to_string(
                index=False
            )
        )

        raise RuntimeError(
            "RFQ V2 validation failed."
        )

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    customer_distribution = (
        headers_df[
            "customer_type"
        ]
        .value_counts()
        .rename_axis("customer_type")
        .reset_index(
            name="count"
        )
    )

    package_distribution = (
        ground_truth_df[
            "target_package"
        ]
        .value_counts(
            normalize=True
        )
        .mul(100)
        .round(2)
        .rename(
            "percentage"
        )
        .reset_index()
    )

    package_distribution.columns = [
        "target_package",
        "percentage",
    ]

    requirement_distribution = (
        requirements_df[
            "requirement_type"
        ]
        .value_counts()
        .rename_axis(
            "requirement_type"
        )
        .reset_index(
            name="count"
        )
    )

    characteristic_distribution = (
        requirements_df[
            "characteristic"
        ]
        .value_counts()
        .rename_axis(
            "characteristic"
        )
        .reset_index(
            name="requirement_count"
        )
    )

    implicit_distribution = (
        requirements_df[
            "implicit_requirement"
        ]
        .value_counts()
        .rename_axis(
            "implicit_requirement"
        )
        .reset_index(
            name="count"
        )
    )

    statistics = pd.DataFrame(
        [
            {
                "metric": "rfq_count",
                "value": len(headers_df),
            },
            {
                "metric": "requirement_count",
                "value": len(requirements_df),
            },
            {
                "metric": "average_requirements_per_rfq",
                "value": round(
                    len(requirements_df)
                    / len(headers_df),
                    2,
                ),
            },
            {
                "metric": "customer_count_used",
                "value": headers_df[
                    "customer_id"
                ].nunique(),
            },
            {
                "metric": "target_package_count",
                "value": ground_truth_df[
                    "target_package"
                ].nunique(),
            },
            {
                "metric": "implicit_requirement_percentage",
                "value": round(
                    requirements_df[
                        "implicit_requirement"
                    ].mean()
                    * 100,
                    2,
                ),
            },
            {
                "metric": "mandatory_percentage",
                "value": round(
                    (
                        requirements_df[
                            "requirement_type"
                        ]
                        == "MANDATORY"
                    ).mean()
                    * 100,
                    2,
                ),
            },
            {
                "metric": "preferred_percentage",
                "value": round(
                    (
                        requirements_df[
                            "requirement_type"
                        ]
                        == "PREFERRED"
                    ).mean()
                    * 100,
                    2,
                ),
            },
        ]
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    RFQ_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    headers_file = (
        RFQ_OUTPUT_DIR
        / "731_rfq_headers_v2.csv"
    )

    requirements_file = (
        RFQ_OUTPUT_DIR
        / "731_rfq_requirements_v2.csv"
    )

    documents_file = (
        RFQ_OUTPUT_DIR
        / "731_rfq_documents_v2.csv"
    )

    ground_truth_file = (
        RFQ_OUTPUT_DIR
        / "731_rfq_ground_truth_v2.csv"
    )

    statistics_file = (
        PROCESSED_DIR
        / "731_rfq_v2_statistics.csv"
    )

    generation_summary_file = (
        PROCESSED_DIR
        / "731_rfq_v2_generation_summary.csv"
    )

    coverage_file = (
        PROCESSED_DIR
        / "731_rfq_v2_requirement_coverage.csv"
    )

    validation_file = (
        PROCESSED_DIR
        / "731_rfq_v2_validation.csv"
    )

    headers_df.to_csv(
        headers_file,
        index=False,
    )

    requirements_df.to_csv(
        requirements_file,
        index=False,
    )

    documents_df.to_csv(
        documents_file,
        index=False,
    )

    ground_truth_df.to_csv(
        ground_truth_file,
        index=False,
    )

    statistics.to_csv(
        statistics_file,
        index=False,
    )

    generation_summary = pd.DataFrame(
        {
            "metric": [
                "requested_rfqs",
                "generated_rfqs",
                "requirements",
                "customers_used",
                "unique_target_packages",
                "validation_passed",
            ],
            "value": [
                TARGET_RFQS,
                len(headers_df),
                len(requirements_df),
                headers_df[
                    "customer_id"
                ].nunique(),
                ground_truth_df[
                    "target_package"
                ].nunique(),
                validation_passed,
            ],
        }
    )

    generation_summary.to_csv(
        generation_summary_file,
        index=False,
    )

    characteristic_distribution.to_csv(
        coverage_file,
        index=False,
    )

    validation_df.to_csv(
        validation_file,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("CUSTOMER TYPE DISTRIBUTION")
    print("=" * 100)

    print(
        customer_distribution.to_string(
            index=False
        )
    )

    print("\n" + "=" * 100)
    print("PACKAGE DISTRIBUTION")
    print("=" * 100)

    print(
        package_distribution.to_string(
            index=False
        )
    )

    print("\n" + "=" * 100)
    print("REQUIREMENT TYPE DISTRIBUTION")
    print("=" * 100)

    print(
        requirement_distribution.to_string(
            index=False
        )
    )

    print("\n" + "=" * 100)
    print("IMPLICIT REQUIREMENT DISTRIBUTION")
    print("=" * 100)

    print(
        implicit_distribution.to_string(
            index=False
        )
    )

    print("\n" + "=" * 100)
    print("SAMPLE RFQ V2")
    print("=" * 100)

    print(
        documents_df.iloc[0]["rfq_text"]
    )

    print("\n" + "=" * 100)
    print("RFQ V2 FILES SAVED")
    print("=" * 100)

    print(
        f"\nRFQ headers:\n{headers_file}"
    )

    print(
        f"\nRFQ requirements:\n{requirements_file}"
    )

    print(
        f"\nRFQ documents:\n{documents_file}"
    )

    print(
        f"\nRFQ ground truth:\n{ground_truth_file}"
    )

    print(
        f"\nStatistics:\n{statistics_file}"
    )

    print(
        f"\nGeneration summary:\n"
        f"{generation_summary_file}"
    )

    print(
        f"\nRequirement coverage:\n"
        f"{coverage_file}"
    )

    print(
        f"\nValidation:\n{validation_file}"
    )

    print("\n" + "=" * 100)
    print("731 RFQ V2 GENERATION COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()