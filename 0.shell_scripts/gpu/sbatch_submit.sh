#!/bin/bash

# Experiment and model setup
exp_name='with_dn_cons_sig'
MODEL_IDS=(2)   # list of models
opt='bads'            # COBYLA, POWELL, COBYQA, bads
subj_setup='all'      # or 'all' e.g., '2,3,5,6,11,15,20,25'
n_runs=2             # <-- define this!

# Paths
base_path="/home/mpla/wenlou/1confiProj"
job_path="${base_path}/jobs_srun"

# Clean and recreate job directory once
rm -rf "$job_path"
mkdir -p "$job_path"

PREV_JOB_ID=""

# Loop over model IDs
for m_id in "${MODEL_IDS[@]}"; do
    log_path="${base_path}/fit_res/${exp_name}/logs/m_${m_id}"
    mkdir -p "$log_path"

    JOB_NAME="${exp_name}_m_${m_id}_${opt}"
    JOB_SCRIPT="${job_path}/${JOB_NAME}.sh"

    # Write job script
    cat <<EOL > "$JOB_SCRIPT"
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=2GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=$log_path/%x_%j.out
#SBATCH --error=$log_path/%x_%j.err

# Catch walltime kill and exit cleanly
trap 'echo "Hit walltime, exiting cleanly"; exit 0' SIGTERM

sh "$base_path/0.shell_scripts/gpu/run_model.sh" "$exp_name" "$m_id" "$subj_setup" "$opt" "$n_runs" 
# Diagnostics
nvidia-smi
EOL

    # Submit job with sequential dependency
    if [ -z "$PREV_JOB_ID" ]; then
        JOB_ID=$(sbatch "$JOB_SCRIPT" | awk '{print $4}')
    else
        JOB_ID=$(sbatch --dependency=afterany:$PREV_JOB_ID "$JOB_SCRIPT" | awk '{print $4}')
    fi

    PREV_JOB_ID="$JOB_ID"
done
