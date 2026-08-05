#!/bin/bash
#SBATCH --job-name=with_dn_m_8_bads
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=2GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/mpla/wenlou/1confiProj/fit_res/with_dn/logs/m_8/%x_%j.out
#SBATCH --error=/home/mpla/wenlou/1confiProj/fit_res/with_dn/logs/m_8/%x_%j.err

# Catch walltime kill and exit cleanly
trap 'echo "Hit walltime, exiting cleanly"; exit 0' SIGTERM

sh "/home/mpla/wenlou/1confiProj/0.shell_scripts/gpu/run_model.sh" "with_dn" "8" "25" "bads" "4" 
# Diagnostics
nvidia-smi
