#!/bin/bash
#SBATCH --job-name=clinical_bert
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --output=/projects/b5bg/logs/clinical_bert_%j.log

source ~/miniforge3/bin/activate
conda activate mimic_nlp

cd ~/Synthetic-Notes-Evaluation-1
python clinical_bert.py