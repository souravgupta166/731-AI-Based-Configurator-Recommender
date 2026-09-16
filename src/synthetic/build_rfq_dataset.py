#!/usr/bin/env python3

"""
731 SYNTHETIC RFQ DATASET GENERATOR — V1
========================================

Purpose
-------
Generate realistic synthetic RFQs connected to:

    1. Synthetic customer master
    2. V5 technical configurations
    3. Real 731 configurator terminology
    4. Real PO / order-document structure

The generator creates:

    RFQ HEADER
    RFQ REQUIREMENTS
    RFQ DOCUMENT / NATURAL LANGUAGE

Important
---------
The technical target configuration is ALWAYS taken from the
validated V5 configuration pool.

The RFQ text is generated from the target configuration.

Therefore:

    RFQ text
       |
       v
requirements
       |
       v
target technical configuration

This provides ground truth for later NLP evaluation.

Outputs
-------
data/synthetic/rfqs/731_rfq_headers_v1.csv
data/synthetic/rfqs/731_rfq_requirements_v1.csv
data/synthetic/rfqs/731_rfq_documents_v1.csv

data/processed/731_rfq_generation_summary_v1.csv
data/processed/731_rfq_statistics_v1.csv
"""

from __future__ import annotations

import random
import re
import uuid
from datetime import timedelta
from pathlib import Path

import pandas as pd


# =============================================================================
# PATHS
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

OUTPUT_DIR = (
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

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_SEED = 7312026

NUMBER_OF_RFQS = 5000


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
# HUMAN-READABLE TECHNICAL TERMINOLOGY
# =============================================================================

CHARACTERISTIC_LABELS = {

    "devCategory":
        "device category",

    "characteristic":
        "device characteristic",

    "housing":
        "housing",

    "powerSupply":
        "power supply",

    "numberOfChannels":
        "number of measurement channels",

    "explosionApproval":
        "explosion protection",

    "protectionArea":
        "protection area",

    "certification":
        "certification",

    "dataInterface":
        "data interface",

    "stromSchaltbar":
        "switchable current output",

    "stromEingaenge":
        "current input",

    "temperaturEingaenge":
        "temperature input",

    "binaerDigitalOpenColl_MN":
        "digital open-collector input",

    "binaerOpenColl_MP":
        "open-collector output",

    "waveInjector":
        "WaveInjector option",

    "dev_advMeterVerification":
        "Advanced Meter Verification",

    "dynamicGasMaster":
        "Dynamic Gas Master",

    "customUserFluid":
        "custom user fluid",

    "steamApplication":
        "steam application",
}


# =============================================================================
# VALUE TRANSLATIONS
# =============================================================================

VALUE_DESCRIPTIONS = {

    # -------------------------------------------------------------------------
    # Device categories
    # -------------------------------------------------------------------------

    ("devCategory", "G"):
        "gas measurement",

    ("devCategory", "F"):
        "flow measurement",

    # -------------------------------------------------------------------------
    # Characteristics
    # -------------------------------------------------------------------------

    ("characteristic", "GP"):
        "general-purpose configuration",

    ("characteristic", "ST"):
        "standard temperature application",

    ("characteristic", "TE"):
        "temperature measurement application",

    ("characteristic", "CA"):
        "calibration application",

    ("characteristic", "PW"):
        "process-water application",

    ("characteristic", "WD"):
        "dual/single channel measurement configuration",

    ("characteristic", "VG"):
        "specialized gas measurement",

    # -------------------------------------------------------------------------
    # Housing
    # -------------------------------------------------------------------------

    ("housing", "ST"):
        "stainless-steel housing",

    ("housing", "AL"):
        "aluminium housing",

    # -------------------------------------------------------------------------
    # Power
    # -------------------------------------------------------------------------

    ("powerSupply", "1"):
        "standard power supply",

    ("powerSupply", "4"):
        "alternative power supply",

    # -------------------------------------------------------------------------
    # Explosion
    # -------------------------------------------------------------------------

    ("explosionApproval", "N"):
        "non-hazardous-area application",

    ("explosionApproval", "F"):
        "hazardous-area application",

    # -------------------------------------------------------------------------
    # Certification
    # -------------------------------------------------------------------------

    ("certification", "NN"):
        "standard certification",

    # -------------------------------------------------------------------------
    # Interfaces
    # -------------------------------------------------------------------------

    ("dataInterface", "BT"):
        "Bluetooth communication",

    ("dataInterface", "HS"):
        "high-speed communication",

    # -------------------------------------------------------------------------
    # Special features
    # -------------------------------------------------------------------------

    ("dev_advMeterVerification", "FVA"):
        "Advanced Meter Verification",

    ("waveInjector", "FWI"):
        "WaveInjector cryogenic option",

    ("dynamicGasMaster", "DGM"):
        "Dynamic Gas Master",

    ("customUserFluid", "CUF"):
        "custom user fluid",

    ("steamApplication", "STEAM"):
        "steam measurement",

}


# =============================================================================
# VALUE NORMALIZATION
# =============================================================================

def normalize_value(value):

    if pd.isna(value):

        return "NOVALUE"

    text = str(value).strip()

    if not text:

        return "NOVALUE"

    return text


def clean_text(value):

    value = normalize_value(
        value
    )

    if value.upper() == "NOVALUE":

        return None

    return value


# =============================================================================
# LOAD DATA
# =============================================================================

def load_customers():

    if not CUSTOMER_FILE.exists():

        raise FileNotFoundError(
            f"Customer master not found:\n"
            f"{CUSTOMER_FILE}"
        )

    df = pd.read_csv(
        CUSTOMER_FILE
    )

    print(
        f"Customers loaded: {len(df):,}"
    )

    return df


def load_configurations():

    if not CONFIGURATION_FILE.exists():

        raise FileNotFoundError(
            f"V5 configuration file not found:\n"
            f"{CONFIGURATION_FILE}"
        )

    df = pd.read_csv(
        CONFIGURATION_FILE
    )

    print(
        f"V5 configurations loaded: {len(df):,}"
    )

    return df


# =============================================================================
# RFQ IDENTIFIERS
# =============================================================================

def generate_rfq_id(index):

    return (
        f"RFQ731-{index:06d}"
    )


def generate_customer_reference(
    rng,
    index,
):

    prefixes = [
        "RFQ",
        "REQ",
        "ENQ",
        "PROJECT",
        "INST",
    ]

    return (
        f"{rng.choice(prefixes)}-"
        f"{rng.randint(100000, 999999)}"
    )


def generate_project_reference(
    rng,
    customer,
):

    if customer["is_project_customer"] == "Y":

        return customer[
            "project_name"
        ]

    return (
        rng.choice(
            [
                "Process Measurement Upgrade",
                "Flow Measurement Replacement",
                "Plant Instrumentation",
                "Process Monitoring",
                "Flow Metering Requirement",
                "Measurement System Upgrade",
            ]
        )
    )


# =============================================================================
# TECHNICAL INTERPRETATION
# =============================================================================

def technical_description(
    characteristic,
    value,
):

    value = clean_text(
        value
    )

    if value is None:

        return None

    specific = VALUE_DESCRIPTIONS.get(
        (
            characteristic,
            value,
        )
    )

    if specific:

        return specific

    label = CHARACTERISTIC_LABELS.get(
        characteristic,
        characteristic,
    )

    return (
        f"{label} {value}"
    )


# =============================================================================
# REQUIREMENT EXTRACTION FROM CONFIGURATION
# =============================================================================

def extract_requirements(
    configuration,
    rng,
):

    requirements = []

    for characteristic in TECHNICAL_COLUMNS:

        if characteristic not in configuration:

            continue

        value = clean_text(
            configuration[
                characteristic
            ]
        )

        if value is None:

            continue

        description = (
            technical_description(
                characteristic,
                value,
            )
        )

        # -------------------------------------------------------------
        # Determine whether this is a hard requirement.
        # -------------------------------------------------------------

        if characteristic in {
            "devCategory",
            "characteristic",
            "powerSupply",
            "numberOfChannels",
            "explosionApproval",
            "dataInterface",
        }:

            requirement_type = (
                "MANDATORY"
            )

        elif characteristic in {
            "housing",
            "certification",
            "protectionArea",
            "stromSchaltbar",
            "stromEingaenge",
            "temperaturEingaenge",
        }:

            requirement_type = (
                "MANDATORY"
                if rng.random() < 0.75
                else "PREFERRED"
            )

        else:

            requirement_type = (
                "MANDATORY"
                if rng.random() < 0.60
                else "PREFERRED"
            )

        requirements.append(
            {
                "characteristic":
                    characteristic,

                "required_value":
                    value,

                "requirement_description":
                    description,

                "requirement_type":
                    requirement_type,

                "source":
                    "SYNTHETIC_CONFIGURATION_GROUND_TRUTH",
            }
        )

    return requirements


# =============================================================================
# APPLICATION LANGUAGE
# =============================================================================

def build_application_context(
    customer,
    rng,
):

    industry = customer[
        "industry"
    ]

    application = customer[
        "primary_application"
    ]

    templates = [

        (
            f"We are looking for a flow "
            f"measurement solution for our "
            f"{industry.lower()} application."
        ),

        (
            f"The equipment will be used for "
            f"{application.lower()} in our "
            f"{industry.lower()} facility."
        ),

        (
            f"Please provide a suitable "
            f"measurement system for "
            f"{application.lower()}."
        ),

        (
            f"We require an ultrasonic flow "
            f"measurement solution for an "
            f"industrial {industry.lower()} process."
        ),

    ]

    return rng.choice(
        templates
    )


# =============================================================================
# NATURAL LANGUAGE RFQ GENERATION
# =============================================================================

def build_rfq_text(
    customer,
    configuration,
    requirements,
    rfq_id,
    project_reference,
    rng,
):

    customer_name = customer[
        "customer_name"
    ]

    country = customer[
        "country"
    ]

    application_context = (
        build_application_context(
            customer,
            rng,
        )
    )

    # -------------------------------------------------------------------------
    # Opening
    # -------------------------------------------------------------------------

    openings = [

        "Dear Sales Team,",

        "Dear Sir or Madam,",

        "Hello Sales Team,",

        "Dear FLEXIM Sales Team,",

    ]

    opening = rng.choice(
        openings
    )

    # -------------------------------------------------------------------------
    # Requirement sentences
    # -------------------------------------------------------------------------

    sentences = []

    mandatory = [
        r
        for r in requirements
        if r[
            "requirement_type"
        ]
        == "MANDATORY"
    ]

    preferred = [
        r
        for r in requirements
        if r[
            "requirement_type"
        ]
        == "PREFERRED"
    ]

    # -------------------------------------------------------------
    # Group some requirements into natural sentences.
    # -------------------------------------------------------------

    for requirement in mandatory:

        characteristic = requirement[
            "characteristic"
        ]

        value = requirement[
            "required_value"
        ]

        description = requirement[
            "requirement_description"
        ]

        if characteristic == "devCategory":

            sentence = (
                f"The application requires "
                f"{description}."
            )

        elif characteristic == "characteristic":

            sentence = (
                f"The required device "
                f"configuration is "
                f"{description}."
            )

        elif characteristic == "powerSupply":

            sentence = (
                f"The unit should support "
                f"{description}."
            )

        elif characteristic == "numberOfChannels":

            sentence = (
                f"We require "
                f"{description}."
            )

        elif characteristic == "explosionApproval":

            sentence = (
                f"The measurement point requires "
                f"{description}."
            )

        elif characteristic == "dataInterface":

            sentence = (
                f"The required communication "
                f"interface is "
                f"{description}."
            )

        elif characteristic == "stromSchaltbar":

            sentence = (
                f"A "
                f"{description} "
                f"is required."
            )

        elif characteristic == "stromEingaenge":

            sentence = (
                f"The system should provide "
                f"{description}."
            )

        elif characteristic == "temperaturEingaenge":

            sentence = (
                f"Temperature measurement "
                f"capability is required "
                f"({value})."
            )

        elif characteristic == "certification":

            sentence = (
                f"The required certification "
                f"is {value}."
            )

        elif characteristic == "housing":

            sentence = (
                f"The preferred housing "
                f"material is {description}."
            )

        else:

            sentence = (
                f"The system should include "
                f"{description}."
            )

        sentences.append(
            sentence
        )

    # -------------------------------------------------------------------------
    # Preferred requirements
    # -------------------------------------------------------------------------

    for requirement in preferred:

        description = requirement[
            "requirement_description"
        ]

        sentences.append(
            f"If possible, please include "
            f"{description}."
        )

    # -------------------------------------------------------------------------
    # Commercial request
    # -------------------------------------------------------------------------

    commercial = [

        (
            f"Please provide your quotation "
            f"including pricing, expected "
            f"delivery time and applicable "
            f"commercial terms."
        ),

        (
            f"Please quote the complete "
            f"configuration and advise the "
            f"earliest possible delivery date."
        ),

        (
            f"Please provide your best "
            f"commercial offer including "
            f"delivery lead time."
        ),

    ]

    commercial_sentence = rng.choice(
        commercial
    )

    # -------------------------------------------------------------------------
    # Closing
    # -------------------------------------------------------------------------

    closings = [

        "Please let us know if additional technical information is required.",

        "Please contact us if any clarification is required.",

        "We look forward to receiving your quotation.",

        "Please confirm whether the requested configuration can be supplied.",

    ]

    closing = rng.choice(
        closings
    )

    # -------------------------------------------------------------------------
    # Construct document
    # -------------------------------------------------------------------------

    lines = [

        opening,

        "",

        f"Customer: {customer_name}",

        f"RFQ Reference: {rfq_id}",

        f"Customer Reference: {project_reference}",

        f"Country: {country}",

        "",

        application_context,

        "",

        (
            "We would like to request a quotation "
            "for the following measurement solution:"
        ),

        "",

    ]

    lines.extend(
        sentences
    )

    lines.extend(
        [
            "",
            commercial_sentence,
            "",
            closing,
            "",
            "Best regards,",
            customer[
                "contact_full_name"
            ],
            customer_name,
        ]
    )

    return "\n".join(
        lines
    )


# =============================================================================
# RFQ GENERATION
# =============================================================================

def generate_rfq_dataset(
    customers,
    configurations,
    rng,
):

    headers = []

    requirement_records = []

    document_records = []

    # -------------------------------------------------------------------------
    # Ensure broad customer coverage
    # -------------------------------------------------------------------------

    customers_pool = (
        customers
        .sample(
            n=len(customers),
            random_state=RANDOM_SEED,
            replace=False,
        )
        .reset_index(
            drop=True
        )
    )

    # -------------------------------------------------------------------------
    # Generate RFQs
    # -------------------------------------------------------------------------

    for index in range(
        1,
        NUMBER_OF_RFQS + 1,
    ):

        rfq_id = generate_rfq_id(
            index
        )

        # -------------------------------------------------------------
        # Customer
        # -------------------------------------------------------------

        customer = (
            customers_pool.iloc[
                (index - 1)
                % len(customers_pool)
            ]
        )

        # -------------------------------------------------------------
        # Technical configuration
        # -------------------------------------------------------------

        configuration = (
            configurations.iloc[
                rng.randrange(
                    len(configurations)
                )
            ]
            .to_dict()
        )

        # -------------------------------------------------------------
        # Project reference
        # -------------------------------------------------------------

        project_reference = (
            generate_project_reference(
                rng,
                customer,
            )
        )

        customer_reference = (
            generate_customer_reference(
                rng,
                index,
            )
        )

        # -------------------------------------------------------------
        # Dates
        # -------------------------------------------------------------

        rfq_date = (
            pd.Timestamp(
                "2026-01-01"
            )
            + pd.Timedelta(
                days=rng.randint(
                    0,
                    210,
                )
            )
        )

        requested_delivery_date = (
            rfq_date
            + pd.Timedelta(
                days=rng.randint(
                    30,
                    180,
                )
            )
        )

        # -------------------------------------------------------------
        # Quantity
        # -------------------------------------------------------------

        quantity = rng.choices(
            [
                1,
                2,
                3,
                4,
                5,
                10,
                20,
            ],
            weights=[
                40,
                25,
                12,
                8,
                6,
                6,
                3,
            ],
            k=1,
        )[0]

        # -------------------------------------------------------------
        # Requirements
        # -------------------------------------------------------------

        requirements = (
            extract_requirements(
                configuration,
                rng,
            )
        )

        # -------------------------------------------------------------
        # Natural-language RFQ
        # -------------------------------------------------------------

        rfq_text = build_rfq_text(
            customer=customer,
            configuration=configuration,
            requirements=requirements,
            rfq_id=rfq_id,
            project_reference=project_reference,
            rng=rng,
        )

        # -------------------------------------------------------------
        # Header
        # -------------------------------------------------------------

        header = {

            "rfq_id":
                rfq_id,

            "customer_id":
                customer[
                    "customer_id"
                ],

            "customer_name":
                customer[
                    "customer_name"
                ],

            "customer_reference":
                customer_reference,

            "project_reference":
                project_reference,

            "rfq_date":
                rfq_date.strftime(
                    "%Y-%m-%d"
                ),

            "requested_delivery_date":
                requested_delivery_date.strftime(
                    "%Y-%m-%d"
                ),

            "quantity":
                quantity,

            "currency":
                customer[
                    "currency"
                ],

            "payment_terms":
                customer[
                    "payment_terms"
                ],

            "delivery_terms":
                customer[
                    "delivery_terms"
                ],

            "customer_country":
                customer[
                    "country"
                ],

            "industry":
                customer[
                    "industry"
                ],

            "primary_application":
                customer[
                    "primary_application"
                ],

            "customer_type":
                customer[
                    "customer_type"
                ],

            # Ground truth
            "target_configuration_id":
                configuration[
                    "canonical_configuration_id"
                ],

            "target_package":
                configuration[
                    "package_context"
                ],

            "target_generation_method":
                configuration.get(
                    "generation_method",
                    "REAL_OBSERVED",
                ),

            "target_evidence_level":
                configuration.get(
                    "evidence_level",
                    "OBSERVED",
                ),

            "source_type":
                "SYNTHETIC",

            "synthetic_version":
                "731_RFQ_V1",
        }

        headers.append(
            header
        )

        # -------------------------------------------------------------
        # Requirement records
        # -------------------------------------------------------------

        for sequence, requirement in enumerate(
            requirements,
            start=1,
        ):

            requirement_records.append(
                {

                    "rfq_id":
                        rfq_id,

                    "requirement_id":
                        f"{rfq_id}-REQ-{sequence:03d}",

                    "requirement_sequence":
                        sequence,

                    "characteristic":
                        requirement[
                            "characteristic"
                        ],

                    "required_value":
                        requirement[
                            "required_value"
                        ],

                    "requirement_description":
                        requirement[
                            "requirement_description"
                        ],

                    "requirement_type":
                        requirement[
                            "requirement_type"
                        ],

                    "ground_truth":
                        True,

                    "source":
                        requirement[
                            "source"
                        ],
                }
            )

        # -------------------------------------------------------------
        # Document
        # -------------------------------------------------------------

        document_records.append(
            {

                "rfq_id":
                    rfq_id,

                "document_type":
                    "RFQ",

                "document_language":
                    customer[
                        "language"
                    ],

                "subject":
                    (
                        f"RFQ - "
                        f"{project_reference}"
                    ),

                "rfq_text":
                    rfq_text,

                "text_length":
                    len(rfq_text),

                "target_configuration_id":
                    configuration[
                        "canonical_configuration_id"
                    ],

                "target_package":
                    configuration[
                        "package_context"
                    ],

                "source_type":
                    "SYNTHETIC",

                "synthetic_version":
                    "731_RFQ_V1",
            }
        )

    return (
        pd.DataFrame(headers),
        pd.DataFrame(requirement_records),
        pd.DataFrame(document_records),
    )


# =============================================================================
# VALIDATION
# =============================================================================

def validate_dataset(
    headers,
    requirements,
    documents,
):

    print(
        "\n" + "=" * 100
    )

    print(
        "RFQ DATASET VALIDATION"
    )

    print(
        "=" * 100
    )

    # -------------------------------------------------------------------------
    # Header validation
    # -------------------------------------------------------------------------

    assert (
        len(headers)
        == NUMBER_OF_RFQS
    )

    assert (
        headers["rfq_id"]
        .is_unique
    )

    assert (
        headers["target_configuration_id"]
        .notna()
        .all()
    )

    # -------------------------------------------------------------------------
    # Requirement validation
    # -------------------------------------------------------------------------

    assert (
        requirements["rfq_id"]
        .isin(
            headers["rfq_id"]
        )
        .all()
    )

    assert (
        requirements["required_value"]
        .notna()
        .all()
    )

    # -------------------------------------------------------------------------
    # Document validation
    # -------------------------------------------------------------------------

    assert (
        documents["rfq_id"]
        .is_unique
    )

    assert (
        documents["rfq_id"]
        .isin(
            headers["rfq_id"]
        )
        .all()
    )

    assert (
        documents["rfq_text"]
        .str.len()
        .gt(100)
        .all()
    )

    print(
        f"\nRFQ headers: "
        f"{len(headers):,}"
    )

    print(
        f"Requirement records: "
        f"{len(requirements):,}"
    )

    print(
        f"RFQ documents: "
        f"{len(documents):,}"
    )

    print(
        "\nValidation: PASSED"
    )


# =============================================================================
# SUMMARY
# =============================================================================

def build_statistics(
    headers,
    requirements,
    documents,
):

    records = []

    records.append(
        {
            "metric":
                "rfq_count",

            "value":
                len(headers),
        }
    )

    records.append(
        {
            "metric":
                "unique_customers",

            "value":
                headers[
                    "customer_id"
                ].nunique(),
        }
    )

    records.append(
        {
            "metric":
                "unique_target_configurations",

            "value":
                headers[
                    "target_configuration_id"
                ].nunique(),
        }
    )

    records.append(
        {
            "metric":
                "total_requirements",

            "value":
                len(requirements),
        }
    )

    records.append(
        {
            "metric":
                "average_requirements_per_rfq",

            "value":
                round(
                    len(requirements)
                    / len(headers),
                    2,
                ),
        }
    )

    records.append(
        {
            "metric":
                "mandatory_requirements",

            "value":
                (
                    requirements[
                        "requirement_type"
                    ]
                    == "MANDATORY"
                ).sum(),
        }
    )

    records.append(
        {
            "metric":
                "preferred_requirements",

            "value":
                (
                    requirements[
                        "requirement_type"
                    ]
                    == "PREFERRED"
                ).sum(),
        }
    )

    records.append(
        {
            "metric":
                "average_rfq_text_length",

            "value":
                round(
                    documents[
                        "text_length"
                    ].mean(),
                    2,
                ),
        }
    )

    return pd.DataFrame(
        records
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 100)

    print(
        "731 SYNTHETIC RFQ DATASET GENERATOR — V1"
    )

    print("=" * 100)

    print(
        "\nLoading customer master..."
    )

    customers = load_customers()

    print(
        "\nLoading V5 technical configurations..."
    )

    configurations = load_configurations()

    print(
        "\nCustomer records:"
        f" {len(customers):,}"
    )

    print(
        "V5 configurations:"
        f" {len(configurations):,}"
    )

    # -------------------------------------------------------------------------
    # Random generator
    # -------------------------------------------------------------------------

    rng = random.Random(
        RANDOM_SEED
    )

    print(
        "\n" + "=" * 100
    )

    print(
        "GENERATING RFQs"
    )

    print(
        "=" * 100
    )

    print(
        f"\nTarget RFQs: "
        f"{NUMBER_OF_RFQS:,}"
    )

    (
        headers,
        requirements,
        documents,
    ) = generate_rfq_dataset(
        customers,
        configurations,
        rng,
    )

    validate_dataset(
        headers,
        requirements,
        documents,
    )

    # -------------------------------------------------------------------------
    # Print distributions
    # -------------------------------------------------------------------------

    print(
        "\n" + "=" * 100
    )

    print(
        "RFQ CUSTOMER DISTRIBUTION"
    )

    print(
        "=" * 100
    )

    print(
        headers[
            "customer_type"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\n" + "=" * 100
    )

    print(
        "RFQ PACKAGE DISTRIBUTION"
    )

    print(
        "=" * 100
    )

    package_distribution = (
        headers[
            "target_package"
        ]
        .value_counts(
            normalize=True
        )
        .mul(100)
        .round(2)
    )

    print(
        package_distribution
        .to_string()
    )

    print(
        "\n" + "=" * 100
    )

    print(
        "REQUIREMENT TYPE DISTRIBUTION"
    )

    print(
        "=" * 100
    )

    print(
        requirements[
            "requirement_type"
        ]
        .value_counts()
        .to_string()
    )

    # -------------------------------------------------------------------------
    # Sample RFQ
    # -------------------------------------------------------------------------

    print(
        "\n" + "=" * 100
    )

    print(
        "SAMPLE RFQ"
    )

    print(
        "=" * 100
    )

    sample = documents.iloc[0]

    print(
        f"\nRFQ ID: "
        f"{sample['rfq_id']}"
    )

    print(
        "\n" + sample["rfq_text"]
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    header_file = (
        OUTPUT_DIR
        / "731_rfq_headers_v1.csv"
    )

    requirement_file = (
        OUTPUT_DIR
        / "731_rfq_requirements_v1.csv"
    )

    document_file = (
        OUTPUT_DIR
        / "731_rfq_documents_v1.csv"
    )

    statistics_file = (
        PROCESSED_DIR
        / "731_rfq_statistics_v1.csv"
    )

    summary_file = (
        PROCESSED_DIR
        / "731_rfq_generation_summary_v1.csv"
    )

    headers.to_csv(
        header_file,
        index=False,
    )

    requirements.to_csv(
        requirement_file,
        index=False,
    )

    documents.to_csv(
        document_file,
        index=False,
    )

    statistics = build_statistics(
        headers,
        requirements,
        documents,
    )

    statistics.to_csv(
        statistics_file,
        index=False,
    )

    summary = pd.DataFrame(
        [
            {
                "dataset":
                    "731_RFQ_V1",

                "rfqs":
                    len(headers),

                "customers":
                    headers[
                        "customer_id"
                    ].nunique(),

                "target_configurations":
                    headers[
                        "target_configuration_id"
                    ].nunique(),

                "requirements":
                    len(requirements),

                "documents":
                    len(documents),

                "source_type":
                    "SYNTHETIC",
            }
        ]
    )

    summary.to_csv(
        summary_file,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Completion
    # -------------------------------------------------------------------------

    print(
        "\n" + "=" * 100
    )

    print(
        "RFQ FILES SAVED"
    )

    print(
        "=" * 100
    )

    print(
        f"\nRFQ headers:"
        f"\n{header_file}"
    )

    print(
        f"\nRFQ requirements:"
        f"\n{requirement_file}"
    )

    print(
        f"\nRFQ documents:"
        f"\n{document_file}"
    )

    print(
        f"\nStatistics:"
        f"\n{statistics_file}"
    )

    print(
        f"\nGeneration summary:"
        f"\n{summary_file}"
    )

    print(
        "\n" + "=" * 100
    )

    print(
        "731 SYNTHETIC RFQ GENERATION COMPLETE"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":

    main()