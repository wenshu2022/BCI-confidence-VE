#!/bin/bash
#SBATCH --job-name=MNLE
#SBATCH --nodes=1
#SBATCH --time=36:00:00
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=32GB
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/mpla/wenlou/1confiProj/MNLE/logs/%x_%j.out
#SBATCH --error=/home/mpla/wenlou/1confiProj/MNLE/logs/%x_%j.err

module load anaconda3
source activate pytorch_env

# Project root
cd /home/mpla/wenlou/1confiProj/ || exit 1
export PYTHONPATH="$(pwd):$PYTHONPATH"

python3 "/home/mpla/wenlou/1confiProj/MNLE/test_sbi.py"
