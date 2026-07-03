#!/bin/bash

# Experiment and model setup
exp_name='MNLE_test'
MODEL_IDS=(1)   # list of models
opt='bads'            # COBYLA, POWELL, COBYQA, bads
subj_setup='all'      # or e.g., '2,3'
n_runs=20          

# Paths
base_path="/home/mpla/wenlou/1confiProj"
job_path="${base_path}/jobs_srun"

# Clean and recreate job directory once
rm -rf "$job_path"
mkdir -p "$job_path"

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
#SBATCH --time=10:00:00
#SBATCH --mail-type=FAIL
#SBATCH --partition=batch
#SBATCH --mem-per-cpu=2GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=$log_path/%x_%j.out
#SBATCH --error=$log_path/%x_%j.err

sh "$base_path/0.shell_scripts/run_model.sh" "$exp_name" "$m_id" "$subj_setup" "$opt" "$n_runs" 

EOL
    sbatch "$JOB_SCRIPT"
    echo "Submitted job: $JOB_NAME"

done
