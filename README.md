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
