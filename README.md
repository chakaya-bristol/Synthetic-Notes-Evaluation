# Beyond Batch Metrics: Explainable Evaluation of Synthetic Clinical Notes

<p align="center">
An explainable AI framework for evaluating where and how synthetic clinical notes differ from real clinical documentation.
</p>

<p align="center">
<strong>University of Bristol · Practice-Oriented AI CDT</strong>
</p>

<p align="center">
<a href="https://github.com/chakaya-bristol/Synthetic-Notes-Evaluation"><strong>Explore the code »</strong></a>
</p>

## About the Project

Synthetic clinical data has the potential to support healthcare research while reducing reliance on sensitive patient data. However, evaluating synthetic clinical text remains challenging.

Traditional evaluation approaches typically rely on aggregate measures of similarity, utility, or detectability. While these metrics can indicate whether a synthetic dataset resembles real data overall, they provide limited information about **which records differ from real clinical notes and why**.

This project investigates whether **Explainable Artificial Intelligence (XAI)** can provide a more informative approach to synthetic clinical-text evaluation.

Synthetic discharge summaries are generated from real **MIMIC-IV discharge summaries** using multiple language models and generation strategies. Classifiers are then trained to distinguish real from synthetic notes, before XAI methods are applied to investigate the features, linguistic patterns, and record-level characteristics driving those predictions.

Rather than asking only:

> **Can synthetic clinical notes be distinguished from real ones?**

this project focuses on:

> **Where, how, and why do synthetic clinical notes differ from real clinical documentation?**

The project consists of four main stages:

1. **Clinical-note preprocessing**
2. **Synthetic-note generation**
3. **Real vs synthetic classification**
4. **Explainable evaluation using XAI**

---

<details>
<summary><strong>Table of Contents</strong></summary>

1. [About the Project](#about-the-project)
2. [Project Pipeline](#project-pipeline)
3. [Repository Structure](#repository-structure)
4. [Data Preprocessing](#data-preprocessing)
5. [Synthetic Note Generation](#synthetic-note-generation)
6. [Classification](#classification)
7. [Explainable AI Evaluation](#explainable-ai-evaluation)
8. [Getting Started](#getting-started)
9. [Usage](#usage)
10. [Data Access](#data-access)
11. [Generated Outputs](#generated-outputs)
12. [Acknowledgments](#acknowledgments)

</details>

---

## Project Pipeline

The project follows the pipeline below:

```text
                        MIMIC-IV
                           │
                           ▼
                Real Discharge Summaries
                           │
                           ▼
                     Preprocessing
                           │
                           ▼
               Synthetic Note Generation
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
     DistilGPT-2           Qwen            Llama
       Baseline             │                │
                       ┌────┴────┐      ┌────┴────┐
                       ▼         ▼      ▼         ▼
                     Free      Rewrite Free      Rewrite
                  Generation          Generation
                       │                 │
                       └────────┬────────┘
                                ▼
                    Real vs Synthetic Data
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
                 XGBoost              ClinicalBERT
                    │                       │
                    └───────────┬───────────┘
                                ▼
                         XAI Evaluation
                                │
       ┌────────────┬────────────┬────────────┬──────────────┐
       ▼            ▼            ▼            ▼
     Global       Feature      Local      Counterfactual
   Importance     Effects   Explanations     Analysis
      (Q1)         (Q2)        (Q3)           (Q4)

```

---

## Repository Structure

The repository is organised around the four main stages of the evaluation pipeline:

```text
Synthetic-Notes-Evaluation/
│
├── README.md
├── .gitignore
│
├── preprocessing/
│   ├── heuristic_tokenize.py
│   ├── preprocess_isambard.py
│   ├── preprocess_mimic.ipynb
│   └── split_clean_dataset.ipynb
│
├── note_generation/
│   │
│   ├── distilgpt2/
│   │   ├── fine_tune_distilgpt2.ipynb
│   │   ├── generate_notes_distilgpt2.ipynb
│   │   └── generate_notes_isambard.py
│   │
│   ├── qwen/
│   │   ├── download_qwen.slurm
│   │   ├── generate_notes_qwen.py
│   │   ├── generate_qwen.slurm
│   │   ├── qwen_rewrite.py
│   │   └── qwen_rewrite.slurm
│   │
│   └── llama/
│       ├── download_llama.slurm
│       ├── generate_llama.slurm
│       ├── generate_notes_llama.py
│       ├── inspect-qwen.ipynb
│       ├── llama_rewrite.py
│       └── llama_rewrite.slurm
│
├── classification/
│   │
│   ├── clinicalbert/
│   │   ├── clinical_bert.py
│   │   ├── clinical_bert.slurm
│   │   ├── clinicalbert_aug.py
│   │   ├── clinicalbert_aug.slurm
│   │   ├── clinicalbert_rewrite.py
│   │   └── clinicalbert_rewrite.slurm
│   │
│   └── xgboost/
│       ├── xgboost_classifier.ipynb
│       ├── xgboost_full.ipynb
│       └── xgboost_rewrite.ipynb
│
└── xai/
    │
    ├── clinicalbert/
    │   ├── clinicalbert_q1_global_tokens.py
    │   ├── clinicalbert_q3_local_tokens.py
    │   └── clinicalbert_xai.slurm
    │
    └── xgboost/
        ├── Q1_global_importance.ipynb
        ├── Q2_feature_effects.ipynb
        ├── Q3_local_explanations.ipynb
        ├── Q4_counterfactuals.ipynb
        │
        └── figures/
            ├── q1_shap_*.png
            ├── q2_pdp_*.png
            └── q3_waterfall_*.png
```

---
---

## Data Preprocessing

The `preprocessing/` directory contains the code used to prepare MIMIC-IV discharge summaries for generation and classification.

### `preprocess_mimic.ipynb`

Initial exploration and preprocessing of the MIMIC-IV discharge-summary data.

### `preprocess_isambard.py`

Preprocessing pipeline adapted for execution on the Isambard high-performance computing environment.

### `heuristic_tokenize.py`

Provides heuristic tokenisation functionality used during preprocessing.

### `split_clean_dataset.ipynb`

Cleans and partitions the data into the datasets required for downstream training and generation experiments.

Preprocessing also removes or normalises structural information that could provide trivial indicators of whether a record is real or synthetic, allowing subsequent classifiers to focus more strongly on linguistic and stylistic differences.

---

## Synthetic Note Generation

Synthetic discharge summaries are generated using three language-model approaches.

### DistilGPT-2

The `note_generation/distilgpt2/` directory contains the initial baseline experiments.

DistilGPT-2 is used as a **weak baseline generator** to provide a comparison with stronger instruction-tuned models.

---

### Qwen

Qwen is evaluated using two generation approaches.

#### Free generation

The model generates a new discharge summary from few-shot examples without being given a specific source note to reproduce.

#### Rewriting

The model is given a real discharge summary and instructed to rewrite it while preserving its underlying clinical content.

---

### Llama

Llama is evaluated using the same two generation strategies.

#### Free generation

#### Rewriting


---

## Classification

Synthetic-note evaluation is formulated as a binary classification problem:

```text
Real clinical note      → 0
Synthetic clinical note → 1
```

Two classifiers are used to investigate whether synthetic notes contain systematic signals that distinguish them from real clinical documentation.

---

### XGBoost

The XGBoost classifier combines interpretable linguistic features with lexical representations of the clinical notes.

The classifier uses features including:

- word count;
- sentence count;
- average sentence length;
- sentence-length variation;
- vocabulary richness;
- uppercase-word ratio;
- punctuation-based features; and
- TF-IDF representations.

The XGBoost experiments are separated according to the synthetic datasets being evaluated.

The feature-based representation also provides the basis for the majority of the project's record-level XAI analysis.

---

### ClinicalBERT

ClinicalBERT provides a transformer-based classifier operating directly on clinical text.

The model is used as a comparison with the more interpretable XGBoost feature-based classifier.

The ClinicalBERT experiments include:

- individual synthetic-generator datasets;
- rewriting datasets; and
- an augmented dataset combining multiple synthetic sources.

---
## Explainable AI Evaluation

Classification accuracy alone does not explain **why** a synthetic record is distinguishable from a real clinical note.

The project therefore organises the explainability analysis around four questions.

| Question | Aim | Technique |
|---|---|---|
| **Q1** | Which features or tokens most strongly distinguish real from synthetic notes? | SHAP / token attribution |
| **Q2** | How does classifier behaviour change across different values of important features? | Partial Dependence and ICE |
| **Q3** | Why is an individual record classified as real or synthetic? | Local explanations |
| **Q4** | What minimal feature changes could make a synthetic record appear real? | Counterfactual explanations |

---

### XGBoost XAI

The complete four-question XAI framework is implemented for the XGBoost classifier.

#### Q1 – Global importance

SHAP is used to identify the features that contribute most strongly to distinguishing real and synthetic clinical notes.

---

#### Q2 – Feature effects

Partial Dependence Plots and Individual Conditional Expectation plots are used to investigate how classifier predictions vary across different regions of important features.

---

#### Q3 – Local explanations

SHAP waterfall plots provide record-level explanations showing which characteristics push an individual note towards the real or synthetic class.

---

#### Q4 – Counterfactual explanations

Counterfactual analysis investigates the minimum feature changes required for a synthetic note to cross the classifier decision boundary and appear more similar to a real note.

---

### ClinicalBERT XAI

Token-level explanations are also generated for the ClinicalBERT classifier.

These experiments investigate whether the transformer classifier relies on similar linguistic distinctions to those identified by the feature-based XGBoost model.

---
## XAI Figures

Figures generated from the XGBoost XAI experiments are stored in:

```text
xai/xgboost/figures/
```
---

## Getting Started

### Clone the repository

```bash
git clone https://github.com/chakaya-bristol/Synthetic-Notes-Evaluation.git
cd Synthetic-Notes-Evaluation
```

### Environment

The project was developed using Python.

Some model-training and generation experiments were run on the **Isambard high-performance computing environment** using SLURM jobs.

> Package versions and environment configuration may depend on the compute environment being used.

---

## Usage

The project is intended to be run broadly in the following order.

### 1. Preprocess the clinical notes

Run the preprocessing workflow in:

```text
preprocessing/
```

The resulting cleaned and split datasets are then used for generation and classification.

### 2. Generate synthetic notes

Choose one of the generators in:

```text
note_generation/
├── distilgpt2/
├── qwen/
└── llama/
```

For Qwen and Llama, experiments can be run in either:

- free-generation mode; or
- rewriting mode.

On an HPC system, submit the corresponding `.slurm` job.

### 3. Train the classifiers

Run either the XGBoost notebooks under:

```text
classification/xgboost/
```

or the ClinicalBERT experiments under:

```text
classification/clinicalbert/
```

### 4. Run XAI analysis

XGBoost explanations can be reproduced using:

```text
xai/xgboost/
```
ClinicalBERT token-level explanation code is available under:

```text
xai/clinicalbert/
```

---

## Data Access

This project uses discharge summaries from the **MIMIC-IV** clinical database.

MIMIC-IV is a controlled-access clinical dataset and is **not distributed through this repository**.

Users wishing to reproduce the experiments must obtain appropriate access to MIMIC-IV and comply with the relevant data-use requirements.

Raw MIMIC-IV records and generated clinical datasets are intentionally excluded from this repository.

---

## Generated Outputs

Model checkpoints, classifier bundles, prediction CSV files, and intermediate synthetic datasets are not included in the repository.

This keeps the repository focused on:

- preprocessing code;
- generation scripts;
- classification experiments;
- explainability methods; and
- selected figures required to demonstrate the XAI analyses.

---

## Research Context

This project forms part of research undertaken within the **Practice-Oriented AI Centre for Doctoral Training at the University of Bristol**.

The central motivation is to move synthetic clinical-text evaluation beyond aggregate dataset-level metrics.

The proposed evaluation perspective combines:

```text
Dataset-level performance
          +
Record-level explanations
          +
Generator-specific failure analysis
```
to provide a more interpretable understanding of where synthetic clinical notes succeed and where they differ from real clinical documentation.

---

## Acknowledgments

This project uses data from the **MIMIC-IV clinical database**.

Thanks are also given to the supervisors, researchers, and members of the University of Bristol Practice-Oriented AI CDT who provided guidance and support throughout the project.

---

<p align="right">(<a href="#beyond-batch-metrics-explainable-evaluation-of-synthetic-clinical-notes">back to top</a>)</p>

