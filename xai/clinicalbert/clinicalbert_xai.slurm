#!/bin/bash
#SBATCH --job-name=cbert_xai
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --output=/projects/b5bg/logs/cbert_xai_%j.log

source ~/miniforge3/bin/activate
conda activate mimic_nlp
pip install shap matplotlib -q

cd ~/Synthetic-Notes-Evaluation-1
#python -u clinicalbert_q1_global_tokens.py
python -u clinicalbert_q3_local_tokens.py