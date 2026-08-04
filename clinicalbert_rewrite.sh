#!/bin/bash
#SBATCH --job-name=clinicalbert_rewrite
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=/projects/b5bg/logs/clinicalbert_rewrite_%j.log

source ~/miniforge3/bin/activate
conda activate mimic_nlp

cd ~/Synthetic-Notes-Evaluation-1
python -u clinicalbert_rewrite.py