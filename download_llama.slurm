#!/bin/bash
#SBATCH --job-name=download_llama
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/projects/b5bg/logs/download_llama_%j.log

source ~/miniforge3/bin/activate
conda activate mimic_nlp

python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    'meta-llama/Llama-3.1-8B-Instruct',
    local_dir='/projects/b5bg/models/llama-3.1-8b-instruct',
    token=os.environ['HF_TOKEN']
)
print('Download complete')
"