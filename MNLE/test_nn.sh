#!/bin/bash
#SBATCH --job-name=test_gpu_job      # Job name
#SBATCH --output=output_%j.log       # Save output log
#SBATCH --error=error_%j.log         # Save error log
#SBATCH --time=36:00:00              # Time limit (1h)
#SBATCH --partition=gpu              # GPU partition/queue
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --cpus-per-task=8            # 4 CPU cores
#SBATCH --mem=16G                    # 16 GB memory

# Load your environment
module load cuda/12.4
#module load gcc/13.3.0

echo "CUDA version:"
nvcc --version

module load anaconda3
source activate pytorch_env

# Run your Python script
python -u /home/mpla/wenlou/1confiProj/MNLE/test_sbi.py

