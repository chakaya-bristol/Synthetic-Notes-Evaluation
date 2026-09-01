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

