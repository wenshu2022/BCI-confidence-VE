#!/bin/bash
#SBATCH --job-name=adhoc
#SBATCH --nodes=1
#SBATCH --time=5:00:00
#SBATCH --mail-type=FAIL
#SBATCH --partition=batch
#SBATCH --mem-per-cpu=2GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --output=/home/mpla/wenlou/1confiProj/%x_%j.out
#SBATCH --error=/home/mpla/wenlou/1confiProj/%x_%j.err
# ---- Environment setup ----
module load anaconda3
source activate pytorch_env
# Project root
cd /home/mpla/wenlou/1confiProj/ || exit 1
export PYTHONPATH="$(pwd):$PYTHONPATH"
python3 adhoc.py