#!/bin/bash
#SBATCH --job-name=generate_llama
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=/projects/b5bg/logs/generate_llama_%j.log

source ~/miniforge3/bin/activate
conda activate mimic_nlp

cd ~/Synthetic-Notes-Evaluation-1
python generate_notes_llama.py