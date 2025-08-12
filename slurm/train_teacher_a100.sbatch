#!/bin/bash
#SBATCH --job-name=teacher_large
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --time=07:15:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=ALL

# Section to ensure we have the "module" command defined
unalias tap >& /dev/null
if [ -f ~/.bash_profile ]; then
	source ~/.bash_profile
elif [ -f ~/.profile ]; then
	source ~/.profile
fi

export SLURM_EXPORT_ENV=ALL

# Module load section
# First clear our module list 
module purge
# and reload the standard modules
module load cuda/12.3.0

# Section to output information identifying the job, etc.
echo "Slurm job ${SLURM_JOBID} running on"
hostname
echo "To run on ${SLURM_NTASKS} CPU cores across ${SLURM_JOB_NUM_NODES} nodes"
echo "All nodes: ${SLURM_JOB_NODELIST}"
date
pwd
echo "Loaded modules are:"
module list

# Activate your environment and check to see if it worked
source /scratch/zt1/project/msml612/user/link/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/zt1/project/msml612/user/link/miniconda3/envs/drone-rl


# Set wandb to offline mode for cluster runs
export WANDB_MODE=offline
source /home/link/wandb_secrets.sh

python -c "import torch; print(torch.cuda.is_available())"

cd /scratch/zt1/project/msml612/user/link/drone_transformer_rl/src

# Launch training
python -m drone_rl.train.train \
  --config ../configs/training/teacher_large_rev2.yaml \
  --wandb

# Save the exit code from the previous command
ECODE=$?

echo "Job finished with exit code $ECODE"
date

echo "\n---"
echo "WANDB was run in offline mode. To upload your run after the job finishes, use:"
echo "  wandb sync /scratch/zt1/project/msml612/user/link/drone_transformer_rl/src/wandb/"
echo "Replace the path with your actual wandb run directory if different."
# Exit with the cached exit code
exit $ECODE