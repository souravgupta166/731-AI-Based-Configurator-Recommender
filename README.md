
# 731 AI-Based Configurator Recommender (https://souravgupta166-731-ai-base-srcui731-poc-configurator-app-gitsyl.streamlit.app/)

An AI-assisted configuration recommendation system developed as part of a Master's thesis in Data Science, AI, and Digital Business.

The project combines industrial configurator knowledge, synthetic customer requirements, deterministic technical validation, Knowledge Graph verification, and machine-learning-based configuration ranking.

---

## Project Overview

The system supports a complete configuration recommendation workflow:

1. Capture customer and application requirements.
2. Identify technically compatible configurations.
3. Validate configurations using a Knowledge Graph.
4. Rank compatible configurations using pairwise machine learning.
5. Display the recommended configuration and alternatives.
6. Provide technical differences, validation results, and audit information.

The project focuses on the Emerson/FLEXIM 731-series configurator domain.

> Important: The machine-learning labels and engineer preferences used in this project are synthetic. Model results should not be interpreted as historical industrial engineer decisions or real-world configuration accuracy.

---

## Key Features

- Customer requirement input through a Streamlit interface
- Deterministic mandatory-requirement filtering
- Knowledge Graph-based configuration validation
- Pairwise Learning-to-Rank recommendation
- Complete compatible configuration explorer
- Alternative configuration comparison
- Technical difference explanations
- Ranking-strength and engineer-review indicators
- Formal result-integrity audit
- CSV export of compatible configurations
- Reproducible validation scripts

---

## System Architecture

```text
Customer Requirements
        |
        v
Streamlit User Interface
        |
        v
Requirement Processing
        |
        v
Deterministic Technical Filtering
        |
        v
Knowledge Graph Validation
        |
        v
Compatible Configuration Set
        |
        v
Pairwise ML Ranking Model
        |
        v
Ranked Configurations
        |
        v
Recommendation + Alternatives
        |
        v
Integrity Audit and Engineer Review
```

---

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Knowledge Graph representation
- GraphML and CSV
- Logistic Regression
- Random Forest
- HistGradientBoosting
- Pairwise Learning-to-Rank

---

## Repository Structure

```text
731-AI-Based-Configurator-Recommender/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── graphs/
│   └── 731_knowledge_graph/
│
├── src/
│   ├── knowledge_graph/
│   ├── synthetic/
│   └── ui/
│
├── requirements.txt
├── README.md
└── MENTOR_TEST_GUIDE.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/souravgupta166/731-AI-Based-Configurator-Recommender.git
cd 731-AI-Based-Configurator-Recommender
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

Activate the environment:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If a requirements file is not available, install the main dependencies:

```bash
pip install pandas numpy scikit-learn streamlit openpyxl
```

---

## Running the Streamlit Application

Run the frozen configurator application:

```bash
streamlit run src/ui/731_poc_configurator_app_V16.py
```

After starting the application, open the local Streamlit URL shown in the terminal.

The exact UI filename should be confirmed against the repository's current `src/ui/` directory.

---

## Knowledge Graph

The Knowledge Graph integrates:

- Real 731-series configurator rule records
- Product characteristics
- Characteristic values
- Configuration families
- Synthetic configuration records
- Rule-to-family relationships
- Configuration-to-value relationships

### Build the Knowledge Graph

```bash
python src/knowledge_graph/731_build_knowledge_graph.py
```

### Run Knowledge Graph Validation

```bash
python src/knowledge_graph/731_kg_validation_test_suite.py
```

The Knowledge Graph is used as a validation layer. It does not replace the deterministic technical compatibility checks.

The application includes a safety gate that checks consistency between the Knowledge Graph validation results and the deterministic candidate set.

---

## Data Provenance

The project uses multiple provenance categories:

| Provenance | Meaning |
|---|---|
| OBSERVED_731 | Information derived from the real 731 configurator workbook |
| OBSERVED_DOCUMENT | Information extracted from source documents |
| DERIVED | Information generated through deterministic processing or transformation |
| SYNTHETIC | Artificially generated data used for experimentation |

The real 731 workbook represents configurator rules and product knowledge. It is not treated as a historical dataset of customer orders or engineer decisions.

---

## Machine Learning Evaluation

The project evaluates pairwise Learning-to-Rank models under different feature regimes:

- Realistic baseline
- Provenance-aware
- Mechanistic

The primary thesis evaluation uses the realistic feature regime with Logistic Regression.

The evaluation measures include:

- RFQ-level Top-1 recovery
- Top-3 recovery
- Top-5 recovery
- Top-10 recovery
- Mean Reciprocal Rank
- NDCG
- Bootstrap confidence intervals
- Paired statistical comparisons

### Important Interpretation

The reported results measure recovery of synthetic engineer-preference labels generated by the experimental pipeline.

They do not represent:

- Historical engineer decision accuracy
- Real customer conversion performance
- Production deployment accuracy
- Guaranteed industrial configuration correctness

---

## Technical Validation

The system separates the following concepts:

### Technical Compatibility

Whether a configuration satisfies the defined technical requirements.

### Knowledge Graph Validation

Whether the candidate configuration and its required values can be verified through the Knowledge Graph.

### Machine Learning Ranking

The relative ordering of configurations that have already passed technical validation.

### Engineer Review

A review step recommended when ranking scores are close, requirements are incomplete, or multiple technically valid alternatives exist.

Machine learning ranking must not override mandatory technical incompatibilities.

---

## Testing

The project includes validation and end-to-end testing for:

- No technical requirements
- Valid package and channel combinations
- Mandatory requirement filtering
- Conflicting technical requirements
- Knowledge Graph consistency
- Compatible configuration exploration
- Alternative comparison
- Ranking strength
- Formal result-integrity audit
- CSV export
- Cross-family validation

Refer to the mentor guide:

```text
MENTOR_TEST_GUIDE.md
```

---

## Example Conflict Scenario

The validation suite includes conflict scenarios involving:

```text
Package: x731
Number of channels: 2
```

and:

```text
Package: x731
Housing: ST
Number of channels: 2
```

These scenarios are expected to produce no compatible configurations under the implemented backend rules.

The purpose is to verify that the application does not recommend technically invalid configurations.

---

## Limitations

1. The configuration catalogue is synthetic and does not represent historical customer orders.
2. Engineer preference labels are synthetically generated.
3. Model performance reflects recovery of the synthetic preference-generation mechanism.
4. Internal technical abbreviations may not have a verified public expansion.
5. The Knowledge Graph is based on available configurator rules and generated configuration data.
6. The system is a research proof of concept and has not been validated for production deployment.
7. Industry and application fields in the demonstration interface should not be interpreted as independent sources of technical configuration truth unless explicitly connected to validated backend rules.

---

## Reproducibility

The project preserves the following workflow:

```text
Real Configurator Rules
        |
        v
Rule and Product Knowledge
        |
        v
Synthetic Configurations
        |
        v
Synthetic Customer Requirements
        |
        v
Technical Compatibility Filtering
        |
        v
Knowledge Graph Validation
        |
        v
Synthetic Engineer Preferences
        |
        v
Pairwise Learning-to-Rank
        |
        v
Evaluation and Streamlit Demonstration
```

Generated outputs should be interpreted according to their provenance and generation method.

---

## Research Contribution

The project demonstrates an experimental architecture for combining:

- Rule-based technical validation
- Knowledge Graph verification
- Synthetic customer requirement modelling
- Pairwise machine-learning ranking
- Transparent alternative comparison
- Result-integrity auditing

The architecture is designed to keep technical compatibility separate from machine-learning preference ranking.

---

## Author

**Sourav Gupta**

Master's in Data Science, AI, and Digital Business

GitHub:

https://github.com/souravgupta166

---

## Project Status

**Status: Frozen thesis proof of concept**

The implementation has been frozen for thesis documentation and mentor evaluation. Further changes should be documented and version-controlled separately.
