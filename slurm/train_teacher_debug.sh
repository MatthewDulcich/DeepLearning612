#!/bin/bash
#SBATCH --job-name=teacher_large
#SBATCH --partition=debug
#SBATCH --time=00:15:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

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
# module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2
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
conda activate drone-rl
conda info --envs

cd /scratch/zt1/project/msml612/user/link/drone_transformer_rl/src

# Create logs directory if it doesn't exist
mkdir -p logs

# Launch training
python -m drone_rl.train.train \
  --config ../configs/training/teacher_large_debug.yaml

# Save the exit code from the previous command
ECODE=$?

echo "Job finished with exit code $ECODE"
date

# Exit with the cached exit code
exit $ECODE