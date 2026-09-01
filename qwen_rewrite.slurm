#!/bin/bash
#SBATCH --job-name=qwen_rewrite
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=/projects/b5bg/logs/qwen_rewrite_%j.log

source ~/miniforge3/bin/activate
conda activate mimic_nlp

cd ~/Synthetic-Notes-Evaluation-1
python qwen_rewrite.py