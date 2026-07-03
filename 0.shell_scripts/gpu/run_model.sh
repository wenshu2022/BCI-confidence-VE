#!/bin/bash
exp_name=$1
m_id=$2
subj_setup=$3
optimizer=$4
n_runs=$5
set -e

module load cuda/12.4
#module load gcc/13.3.0

echo "CUDA version:"
nvcc --version

module load anaconda3
source activate pytorch_env

cd /home/mpla/wenlou/1confiProj/
#echo "Running task $SLURM_ARRAY_TASK_ID"
echo "Running model for expriment=$exp_name, m_id=$m_id, method=$optimizer"

python3 model_fitting.py \
        $exp_name $m_id $subj_setup $optimizer $n_runs 

