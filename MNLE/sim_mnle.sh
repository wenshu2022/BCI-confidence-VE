#!/bin/bash
#SBATCH --job-name=sim_MNLE
#SBATCH --array=0-19                  # 20 jobs total (200 sims / 10 per job)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8             # adjust if simulations are CPU-heavy
#SBATCH --mem-per-cpu=2G
#SBATCH --time=10:00:00
#SBATCH --partition=batch

# ---- Environment setup ----
module load anaconda3
source activate pytorch_env

# Project root
cd /home/mpla/wenlou/1confiProj/ || exit 1
export PYTHONPATH="$(pwd):$PYTHONPATH"

# ---- Experiment arguments ----
exp_name="MNLE-base"
m_id=3
log_path="MNLE/logs/${exp_name}_m${m_id}"
mkdir -p "${log_path}"

# ---- Simulation batching ----
BATCH_SIZE=10
SIM_START=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
SIM_END=$(( SIM_START + BATCH_SIZE - 1 ))

NUM_SIMULATIONS=200
if (( SIM_END >= NUM_SIMULATIONS )); then
    SIM_END=$(( NUM_SIMULATIONS - 1 ))
fi

# ---- Loop over simulations ----
for SIM_ID in $(seq "$SIM_START" "$SIM_END"); do
    # Skip if already done (so you can restart safely)
    if [[ -f "${log_path}/run_${SIM_ID}.out" ]]; then
        echo "Skipping simulation $SIM_ID (already done)"
        continue
    fi

    echo "Running simulation $SIM_ID"
    if ! python3 MNLE/simu_MNLE.py "$exp_name" "$m_id" "$SIM_ID" \
        > "${log_path}/run_${SIM_ID}.out" \
        2> "${log_path}/run_${SIM_ID}.err"; then
        echo "Simulation $SIM_ID failed" >&2
    fi
done
