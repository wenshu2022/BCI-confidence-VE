#!/bin/bash
exp_name=$1
m_id=$2
subj_setup=$3
optimizer=$4
n_runs=$5
log_path=$6
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
mkdir -p "${log_path}"

TOTAL_GPUS=4     # your available GPUs
echo "Running jobs in batches of $TOTAL_GPUS GPUs"

# total number of runs
NUM_RUNS=20
JOBS_PER_GPU=1        # <-- two runs per GPU
BATCH_SIZE=$(( TOTAL_GPUS * JOBS_PER_GPU ))

i=0
for ((run_id=0; run_id<NUM_RUNS; run_id++)); do
    GPU_ID=$(( run_id % TOTAL_GPUS ))
    INSTANCE=$(( (run_id / TOTAL_GPUS) % JOBS_PER_GPU ))
    # query the physical GPU info
    GPU_INFO=$(nvidia-smi --id=$GPU_ID --query-gpu=index,uuid,name,pci.bus_id --format=csv,noheader)

    echo "Launching run $run_id on GPU $GPU_ID (instance $INSTANCE) on node $HOSTNAME"
    echo "    Physical GPU info: $GPU_INFO"

    CUDA_VISIBLE_DEVICES=$GPU_ID python3 model_fitting.py \
        $exp_name $m_id $subj_setup $optimizer $n_runs \
        > "${log_path}/run_${run_id}.out" 2> "${log_path}/run_${run_id}.err" &

    # wait after a full batch (all GPUs have JOBS_PER_GPU jobs running)
    if (( (i + 1) % BATCH_SIZE == 0 )); then
        wait
    fi

    i=$((i+1))
done

# wait for any remaining jobs
wait
echo "All runs completed."
