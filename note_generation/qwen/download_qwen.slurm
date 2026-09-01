#!/bin/bash
#SBATCH --job-name=download_qwen
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/projects/b5bg/logs/download_qwen_%j.log

source ~/miniforge3/bin/activate
conda activate mimic_nlp

pip install huggingface_hub -q

python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    'Qwen/Qwen2.5-14B-Instruct',
    local_dir='/projects/b5bg/models/qwen2.5-14b-instruct',
    token=os.environ['HF_TOKEN']
)
print('Download complete')
"