#!/usr/bin/env python3

"""
731 SYNTHETIC CUSTOMER MASTER
=============================

Purpose
-------
Build a realistic synthetic customer master for the 731 / FLEXIM
industrial flow-meter order simulation.

The structure is based on the real PO and Order Acknowledgement
documents supplied for the project.

Important
---------
All customer identities generated here are SYNTHETIC.

No real customer names, addresses, emails, phone numbers or
identifiers are copied into the synthetic dataset.

The real documents are used only to reproduce the business
structure and terminology.

Business concepts represented
-----------------------------
- Sold To
- Bill To
- Ship To
- End User
- Customer account
- Customer contact
- Industry / application
- Region / country
- Currency
- Payment terms
- Freight / delivery terms
- Project information
- End-use
- Export compliance context

Output
------
data/synthetic/customers/731_customer_master.csv
data/processed/731_customer_master_summary.csv
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "customers"
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

NUMBER_OF_CUSTOMERS = 250


# =============================================================================
# SYNTHETIC CUSTOMER NAME COMPONENTS
# =============================================================================

COMPANY_PREFIXES = [
    "Apex",
    "Atlas",
    "Nova",
    "Vertex",
    "Orion",
    "Pioneer",
    "Meridian",
    "Summit",
    "Sterling",
    "Vector",
    "Horizon",
    "Prime",
    "Nexon",
    "Global",
    "United",
    "Continental",
    "Advanced",
    "Integrated",
    "Precision",
    "Dynamic",
    "Industrial",
    "Metrix",
    "Quantum",
    "Delta",
    "Evergreen",
]

COMPANY_SUFFIXES = [
    "Process Systems",
    "Industrial Technologies",
    "Engineering Solutions",
    "Process Automation",
    "Energy Systems",
    "Instrumentation",
    "Industrial Services",
    "Process Engineering",
    "Measurement Technologies",
    "Automation Systems",
    "Petrochemical Solutions",
    "Manufacturing Technologies",
]


# =============================================================================
# COUNTRIES
# =============================================================================

COUNTRIES = [
    {
        "country": "Germany",
        "country_code": "DE",
        "currency": "EUR",
        "language": "DE",
        "region": "DACH",
        "cities": [
            ("Berlin", "10115"),
            ("Hamburg", "20095"),
            ("Munich", "80331"),
            ("Frankfurt", "60311"),
            ("Cologne", "50667"),
            ("Stuttgart", "70173"),
            ("Leipzig", "04109"),
        ],
    },
    {
        "country": "Türkiye",
        "country_code": "TR",
        "currency": "EUR",
        "language": "TR",
        "region": "EMEA",
        "cities": [
            ("Istanbul", "34752"),
            ("Ankara", "06510"),
            ("Izmir", "35210"),
            ("Bursa", "16120"),
            ("Kocaeli", "41400"),
        ],
    },
    {
        "country": "Netherlands",
        "country_code": "NL",
        "currency": "EUR",
        "language": "NL",
        "region": "EMEA",
        "cities": [
            ("Amsterdam", "1012"),
            ("Rotterdam", "3011"),
            ("Eindhoven", "5611"),
        ],
    },
    {
        "country": "France",
        "country_code": "FR",
        "currency": "EUR",
        "language": "FR",
        "region": "EMEA",
        "cities": [
            ("Paris", "75001"),
            ("Lyon", "69001"),
            ("Marseille", "13001"),
        ],
    },
    {
        "country": "Italy",
        "country_code": "IT",
        "currency": "EUR",
        "language": "IT",
        "region": "EMEA",
        "cities": [
            ("Milan", "20121"),
            ("Turin", "10121"),
            ("Rome", "00118"),
        ],
    },
    {
        "country": "Spain",
        "country_code": "ES",
        "currency": "EUR",
        "language": "ES",
        "region": "EMEA",
        "cities": [
            ("Madrid", "28001"),
            ("Barcelona", "08001"),
            ("Bilbao", "48001"),
        ],
    },
    {
        "country": "United Kingdom",
        "country_code": "GB",
        "currency": "GBP",
        "language": "EN",
        "region": "EMEA",
        "cities": [
            ("London", "SW1A"),
            ("Manchester", "M1"),
            ("Birmingham", "B1"),
        ],
    },
    {
        "country": "India",
        "country_code": "IN",
        "currency": "EUR",
        "language": "EN",
        "region": "APAC",
        "cities": [
            ("Mumbai", "400001"),
            ("Pune", "411001"),
            ("Ahmedabad", "380001"),
            ("Bengaluru", "560001"),
            ("Chennai", "600001"),
            ("Hyderabad", "500001"),
        ],
    },
    {
        "country": "United Arab Emirates",
        "country_code": "AE",
        "currency": "USD",
        "language": "EN",
        "region": "MEA",
        "cities": [
            ("Dubai", "00000"),
            ("Abu Dhabi", "00000"),
        ],
    },
    {
        "country": "Saudi Arabia",
        "country_code": "SA",
        "currency": "USD",
        "language": "EN",
        "region": "MEA",
        "cities": [
            ("Riyadh", "11564"),
            ("Jeddah", "21577"),
            ("Dammam", "31433"),
        ],
    },
]


# =============================================================================
# INDUSTRIES
# =============================================================================

INDUSTRIES = [
    {
        "industry": "Oil & Gas",
        "weight": 18,
        "applications": [
            "Pipeline measurement",
            "Process monitoring",
            "Hydrocarbon transfer",
            "Refinery process measurement",
            "Gas measurement",
        ],
    },
    {
        "industry": "Chemical",
        "weight": 15,
        "applications": [
            "Chemical process monitoring",
            "Liquid flow measurement",
            "Process control",
            "Batch process measurement",
            "Plant utilities",
        ],
    },
    {
        "industry": "Power & Utilities",
        "weight": 12,
        "applications": [
            "Cooling water measurement",
            "Steam measurement",
            "Utility monitoring",
            "Process water measurement",
            "Energy monitoring",
        ],
    },
    {
        "industry": "Water & Wastewater",
        "weight": 12,
        "applications": [
            "Water treatment",
            "Wastewater monitoring",
            "Raw water measurement",
            "Utility water measurement",
        ],
    },
    {
        "industry": "Pharmaceutical",
        "weight": 8,
        "applications": [
            "Process water",
            "Clean utility monitoring",
            "Process validation",
            "Batch process measurement",
        ],
    },
    {
        "industry": "Food & Beverage",
        "weight": 8,
        "applications": [
            "Liquid process measurement",
            "CIP monitoring",
            "Utility monitoring",
            "Production process measurement",
        ],
    },
    {
        "industry": "Mining & Metals",
        "weight": 8,
        "applications": [
            "Slurry measurement",
            "Process water",
            "Mineral processing",
            "Plant utilities",
        ],
    },
    {
        "industry": "Manufacturing",
        "weight": 10,
        "applications": [
            "Process monitoring",
            "Production utilities",
            "Cooling systems",
            "Industrial automation",
        ],
    },
    {
        "industry": "Research & Development",
        "weight": 5,
        "applications": [
            "Laboratory testing",
            "Prototype testing",
            "R&D measurement",
            "Pilot plant testing",
        ],
    },
    {
        "industry": "Engineering & EPC",
        "weight": 4,
        "applications": [
            "Engineering project",
            "Plant construction",
            "Instrumentation package",
            "Turnkey process project",
        ],
    },
]


# =============================================================================
# CUSTOMER TYPES
# =============================================================================

CUSTOMER_TYPES = [
    (
        "End User",
        55,
    ),
    (
        "EPC Contractor",
        20,
    ),
    (
        "System Integrator",
        10,
    ),
    (
        "Engineering Company",
        8,
    ),
    (
        "Distributor",
        7,
    ),
]


# =============================================================================
# PAYMENT TERMS
# =============================================================================

PAYMENT_TERMS = [
    (
        "within 30 days net",
        35,
    ),
    (
        "within 45 days net",
        15,
    ),
    (
        "within 60 days net",
        35,
    ),
    (
        "within 90 days net",
        10,
    ),
    (
        "advance payment",
        5,
    ),
]


# =============================================================================
# DELIVERY TERMS
# =============================================================================

DELIVERY_TERMS = [
    (
        "CPT",
        35,
    ),
    (
        "DAP",
        25,
    ),
    (
        "FCA",
        20,
    ),
    (
        "EXW",
        10,
    ),
    (
        "DDP",
        10,
    ),
]


# =============================================================================
# CUSTOMER SEGMENTS
# =============================================================================

CUSTOMER_SEGMENTS = [
    (
        "Strategic",
        15,
    ),
    (
        "Key Account",
        25,
    ),
    (
        "Standard",
        50,
    ),
    (
        "Project",
        10,
    ),
]


# =============================================================================
# CONTACT NAMES
# =============================================================================

FIRST_NAMES = [
    "Alexander",
    "Daniel",
    "Michael",
    "Thomas",
    "Martin",
    "Stefan",
    "David",
    "Markus",
    "Robert",
    "James",
    "Oliver",
    "Andreas",
    "Laura",
    "Anna",
    "Sarah",
    "Julia",
    "Maria",
    "Elena",
    "Sophie",
    "Emma",
]

LAST_NAMES = [
    "Meyer",
    "Schmidt",
    "Fischer",
    "Weber",
    "Wagner",
    "Keller",
    "Hoffmann",
    "Bauer",
    "Richter",
    "Klein",
    "Hartmann",
    "Koch",
    "Schneider",
    "Zimmermann",
    "Parker",
    "Wilson",
    "Brown",
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def weighted_choice(
    rng,
    values,
):

    choices = [
        item[0]
        for item in values
    ]

    weights = [
        item[1]
        for item in values
    ]

    return rng.choices(
        choices,
        weights=weights,
        k=1,
    )[0]


def generate_customer_name(
    rng,
    used_names,
):

    attempts = 0

    while attempts < 1000:

        attempts += 1

        name = (
            f"{rng.choice(COMPANY_PREFIXES)} "
            f"{rng.choice(COMPANY_SUFFIXES)} "
            f"{rng.choice(['GmbH', 'AG', 'Ltd.', 'S.A.', 'B.V.', 'FZE'])}"
        )

        if name not in used_names:

            used_names.add(
                name
            )

            return name

    raise RuntimeError(
        "Could not generate a unique customer name."
    )


def generate_customer_code(
    index
):

    return (
        f"CUST731{index:05d}"
    )


def generate_account_number(
    index
):

    return (
        f"73{index:08d}"
    )


def generate_tax_id(
    country_code,
    index,
):

    # Synthetic placeholder.
    #
    # This deliberately does NOT attempt to reproduce real tax numbers.

    digest = hashlib.sha256(
        f"{country_code}-{index}-731"
        .encode("utf-8")
    ).hexdigest()

    return (
        f"SYN-{country_code}-"
        f"{digest[:10].upper()}"
    )


def generate_contact(
    rng
):

    first = rng.choice(
        FIRST_NAMES
    )

    last = rng.choice(
        LAST_NAMES
    )

    return {
        "contact_first_name":
            first,

        "contact_last_name":
            last,

        "contact_full_name":
            f"{first} {last}",
    }


def generate_email(
    first,
    last,
    company_name,
):

    domain = (
        company_name
        .lower()
        .replace(
            " ",
            "",
        )
        .replace(
            "&",
            "",
        )
        .replace(
            ".",
            "",
        )
        .replace(
            ",",
            "",
        )
    )

    # Remove legal suffixes.

    for suffix in [
        "gmbh",
        "ag",
        "ltd",
        "sa",
        "bv",
        "fze",
    ]:

        domain = domain.replace(
            suffix,
            "",
        )

    return (
        f"{first.lower()}.{last.lower()}"
        f"@{domain[:25]}.example.com"
    )


# =============================================================================
# ADDRESS
# =============================================================================

def build_address(
    rng,
    country,
):

    city, postal_code = rng.choice(
        country["cities"]
    )

    street_number = rng.randint(
        1,
        180,
    )

    street_names = [
        "Industriestrasse",
        "Werkstrasse",
        "Gewerbestrasse",
        "Process Road",
        "Technology Park",
        "Engineering Avenue",
        "Industrial Park",
        "Automation Street",
    ]

    street = rng.choice(
        street_names
    )

    return {
        "street_address":
            f"{street} {street_number}",

        "city":
            city,

        "postal_code":
            postal_code,

        "country":
            country["country"],

        "country_code":
            country["country_code"],
    }


# =============================================================================
# MAIN CUSTOMER GENERATION
# =============================================================================

def build_customer_master():

    rng = random.Random(
        RANDOM_SEED
    )

    used_names = set()

    records = []

    print("=" * 100)
    print("731 SYNTHETIC CUSTOMER MASTER")
    print("=" * 100)

    print(
        "\nGenerating "
        f"{NUMBER_OF_CUSTOMERS} synthetic industrial customers..."
    )

    for index in range(
        1,
        NUMBER_OF_CUSTOMERS + 1,
    ):

        country = rng.choices(
            COUNTRIES,
            weights=[
                22,
                16,
                10,
                9,
                8,
                7,
                7,
                10,
                6,
                5,
            ],
            k=1,
        )[0]

        customer_name = (
            generate_customer_name(
                rng,
                used_names,
            )
        )

        customer_type = weighted_choice(
            rng,
            CUSTOMER_TYPES,
        )

        industry_record = weighted_choice(
            rng,
            [
                (
                    item,
                    item["weight"],
                )
                for item in INDUSTRIES
            ],
        )

        industry = (
            industry_record[
                "industry"
            ]
        )

        application = rng.choice(
            industry_record[
                "applications"
            ]
        )

        customer_segment = (
            weighted_choice(
                rng,
                CUSTOMER_SEGMENTS,
            )
        )

        payment_terms = (
            weighted_choice(
                rng,
                PAYMENT_TERMS,
            )
        )

        delivery_terms = (
            weighted_choice(
                rng,
                DELIVERY_TERMS,
            )
        )

        # ---------------------------------------------------------------------
        # Contact
        # ---------------------------------------------------------------------

        contact = generate_contact(
            rng
        )

        email = generate_email(
            contact[
                "contact_first_name"
            ],
            contact[
                "contact_last_name"
            ],
            customer_name,
        )

        phone = (
            f"+49 30 "
            f"{rng.randint(10000000, 99999999)}"
        )

        # ---------------------------------------------------------------------
        # Addresses
        # ---------------------------------------------------------------------

        sold_to = build_address(
            rng,
            country,
        )

        # Bill-to can be same or different.
        if rng.random() < 0.75:

            bill_to = dict(
                sold_to
            )

        else:

            bill_to = build_address(
                rng,
                country,
            )

        # Ship-to can differ from sold-to.
        if rng.random() < 0.55:

            ship_to = build_address(
                rng,
                country,
            )

        else:

            ship_to = dict(
                sold_to
            )

        # ---------------------------------------------------------------------
        # End user
        # ---------------------------------------------------------------------

        has_separate_end_user = (
            rng.random() < 0.35
        )

        if has_separate_end_user:

            end_user_name = (
                generate_customer_name(
                    rng,
                    used_names,
                )
            )

            end_user_address = (
                build_address(
                    rng,
                    country,
                )
            )

        else:

            end_user_name = (
                customer_name
            )

            end_user_address = dict(
                ship_to
            )

        # ---------------------------------------------------------------------
        # Project characteristics
        # ---------------------------------------------------------------------

        is_project_customer = (
            customer_type
            in {
                "EPC Contractor",
                "Engineering Company",
            }
            or
            customer_segment
            == "Project"
        )

        if is_project_customer:

            project_names = [
                "Process Expansion Project",
                "Plant Modernization Project",
                "Measurement Upgrade Project",
                "Pipeline Monitoring Project",
                "Process Automation Project",
                "Utility Measurement Project",
                "Instrumentation Package Project",
                "New Production Line Project",
            ]

            project_name = rng.choice(
                project_names
            )

            contract_number = (
                f"CT-{rng.randint(100000, 999999)}"
            )

        else:

            project_name = (
                "Standard Process Requirement"
            )

            contract_number = ""

        # ---------------------------------------------------------------------
        # End use
        # ---------------------------------------------------------------------

        end_use_options = [
            "Industrial process measurement",
            "Plant process monitoring",
            "Process control and automation",
            "R&D testing",
            "Production monitoring",
            "Utility measurement",
            "Pipeline measurement",
            "Equipment performance monitoring",
        ]

        end_use = rng.choice(
            end_use_options
        )

        # ---------------------------------------------------------------------
        # Export / compliance
        # ---------------------------------------------------------------------

        suspicious_probability = (
            0.015
        )

        suspicious_order_default = (
            "N"
        )

        if rng.random() < suspicious_probability:

            suspicious_order_default = "Y"

            suspicious_reason = (
                "Additional export compliance review required"
            )

        else:

            suspicious_reason = ""

        # ---------------------------------------------------------------------
        # Customer account
        # ---------------------------------------------------------------------

        customer_id = (
            generate_customer_code(
                index
            )
        )

        account_number = (
            generate_account_number(
                index
            )
        )

        # ---------------------------------------------------------------------
        # Record
        # ---------------------------------------------------------------------

        record = {

            # =============================================================
            # Customer identity
            # =============================================================

            "customer_id":
                customer_id,

            "customer_account_number":
                account_number,

            "customer_name":
                customer_name,

            "customer_type":
                customer_type,

            "customer_segment":
                customer_segment,

            "industry":
                industry,

            "primary_application":
                application,

            # =============================================================
            # Region
            # =============================================================

            "country":
                country["country"],

            "country_code":
                country["country_code"],

            "sales_region":
                country["region"],

            "language":
                country["language"],

            "currency":
                country["currency"],

            # =============================================================
            # Contact
            # =============================================================

            **contact,

            "contact_email":
                email,

            "contact_phone":
                phone,

            # =============================================================
            # SOLD TO
            # =============================================================

            "sold_to_name":
                customer_name,

            "sold_to_street":
                sold_to[
                    "street_address"
                ],

            "sold_to_city":
                sold_to[
                    "city"
                ],

            "sold_to_postal_code":
                sold_to[
                    "postal_code"
                ],

            "sold_to_country":
                sold_to[
                    "country"
                ],

            # =============================================================
            # BILL TO
            # =============================================================

            "bill_to_name":
                customer_name,

            "bill_to_street":
                bill_to[
                    "street_address"
                ],

            "bill_to_city":
                bill_to[
                    "city"
                ],

            "bill_to_postal_code":
                bill_to[
                    "postal_code"
                ],

            "bill_to_country":
                bill_to[
                    "country"
                ],

            # =============================================================
            # SHIP TO
            # =============================================================

            "ship_to_name":
                customer_name,

            "ship_to_street":
                ship_to[
                    "street_address"
                ],

            "ship_to_city":
                ship_to[
                    "city"
                ],

            "ship_to_postal_code":
                ship_to[
                    "postal_code"
                ],

            "ship_to_country":
                ship_to[
                    "country"
                ],

            # =============================================================
            # END USER
            # =============================================================

            "end_user_name":
                end_user_name,

            "end_user_street":
                end_user_address[
                    "street_address"
                ],

            "end_user_city":
                end_user_address[
                    "city"
                ],

            "end_user_postal_code":
                end_user_address[
                    "postal_code"
                ],

            "end_user_country":
                end_user_address[
                    "country"
                ],

            # =============================================================
            # COMMERCIAL TERMS
            # =============================================================

            "payment_terms":
                payment_terms,

            "delivery_terms":
                delivery_terms,

            "freight_term":
                delivery_terms,

            "default_delivery_country":
                ship_to[
                    "country"
                ],

            # =============================================================
            # PROJECT
            # =============================================================

            "is_project_customer":
                "Y"
                if is_project_customer
                else "N",

            "project_name":
                project_name,

            "contract_number":
                contract_number,

            # =============================================================
            # END USE / COMPLIANCE
            # =============================================================

            "end_use":
                end_use,

            "suspicious_order_default":
                suspicious_order_default,

            "suspicious_order_reason":
                suspicious_reason,

            # =============================================================
            # TAX / ERP
            # =============================================================

            "synthetic_tax_identifier":
                generate_tax_id(
                    country["country_code"],
                    index,
                ),

            "customer_status":
                "ACTIVE",

            "source_type":
                "SYNTHETIC",

            "synthetic_version":
                "731_CUSTOMER_V1",
        }

        records.append(
            record
        )

    return pd.DataFrame(
        records
    )


# =============================================================================
# VALIDATION
# =============================================================================

def validate_customer_master(
    df
):

    print(
        "\n" + "=" * 100
    )

    print(
        "CUSTOMER MASTER VALIDATION"
    )

    print(
        "=" * 100
    )

    print(
        f"\nCustomers: "
        f"{len(df):,}"
    )

    print(
        f"Columns: "
        f"{len(df.columns):,}"
    )

    assert (
        df["customer_id"]
        .is_unique
    ), "Duplicate customer IDs."

    assert (
        df["customer_account_number"]
        .is_unique
    ), "Duplicate account numbers."

    assert (
        df["customer_name"]
        .is_unique
    ), "Duplicate customer names."

    assert (
        df["source_type"]
        == "SYNTHETIC"
    ).all()

    assert (
        df["customer_status"]
        == "ACTIVE"
    ).all()

    required_columns = [
        "customer_id",
        "customer_name",
        "customer_type",
        "industry",
        "country",
        "currency",
        "payment_terms",
        "delivery_terms",
        "sold_to_name",
        "bill_to_name",
        "ship_to_name",
        "end_user_name",
        "end_use",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    print(
        "\nValidation: PASSED"
    )


# =============================================================================
# SUMMARY
# =============================================================================

def build_summary(
    df
):

    records = []

    records.append(
        {
            "metric":
                "customer_count",

            "value":
                len(df),
        }
    )

    records.append(
        {
            "metric":
                "country_count",

            "value":
                df[
                    "country"
                ].nunique(),
        }
    )

    records.append(
        {
            "metric":
                "industry_count",

            "value":
                df[
                    "industry"
                ].nunique(),
        }
    )

    records.append(
        {
            "metric":
                "project_customer_count",

            "value":
                (
                    df[
                        "is_project_customer"
                    ]
                    == "Y"
                ).sum(),
        }
    )

    records.append(
        {
            "metric":
                "separate_end_user_count",

            "value":
                (
                    df[
                        "end_user_name"
                    ]
                    !=
                    df[
                        "customer_name"
                    ]
                ).sum(),
        }
    )

    records.append(
        {
            "metric":
                "suspicious_order_default_count",

            "value":
                (
                    df[
                        "suspicious_order_default"
                    ]
                    == "Y"
                ).sum(),
        }
    )

    return pd.DataFrame(
        records
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    df = build_customer_master()

    validate_customer_master(
        df
    )

    # -------------------------------------------------------------------------
    # Display distributions
    # -------------------------------------------------------------------------

    print(
        "\n" + "=" * 100
    )

    print(
        "COUNTRY DISTRIBUTION"
    )

    print(
        "=" * 100
    )

    print(
        df[
            "country"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\n" + "=" * 100
    )

    print(
        "INDUSTRY DISTRIBUTION"
    )

    print(
        "=" * 100
    )

    print(
        df[
            "industry"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\n" + "=" * 100
    )

    print(
        "CUSTOMER TYPE DISTRIBUTION"
    )

    print(
        "=" * 100
    )

    print(
        df[
            "customer_type"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\n" + "=" * 100
    )

    print(
        "CUSTOMER MASTER SAMPLE"
    )

    print(
        "=" * 100
    )

    display_columns = [
        "customer_id",
        "customer_name",
        "customer_type",
        "industry",
        "country",
        "currency",
        "payment_terms",
        "delivery_terms",
        "project_name",
        "end_user_name",
    ]

    print(
        df[
            display_columns
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    output_file = (
        OUTPUT_DIR
        / "731_customer_master.csv"
    )

    df.to_csv(
        output_file,
        index=False,
    )

    summary = build_summary(
        df
    )

    summary_file = (
        PROCESSED_DIR
        / "731_customer_master_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False,
    )

    print(
        "\n" + "=" * 100
    )

    print(
        "CUSTOMER MASTER SAVED"
    )

    print(
        "=" * 100
    )

    print(
        f"\nCustomer master:"
        f"\n{output_file}"
    )

    print(
        f"\nSummary:"
        f"\n{summary_file}"
    )

    print(
        "\n" + "=" * 100
    )

    print(
        "731 CUSTOMER MASTER COMPLETE"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()