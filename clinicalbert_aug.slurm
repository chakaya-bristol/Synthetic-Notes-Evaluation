#!/bin/bash
#SBATCH --job-name=clinicalbert_aug
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=/projects/b5bg/logs/clinicalbert_aug_%j.log

source ~/miniforge3/bin/activate
conda activate mimic_nlp

cd ~/Synthetic-Notes-Evaluation-1
python -u clinicalbert_aug.py