#!/bin/bash
#SBATCH --job-name=teacher_large
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH --mem=64G
#SBATCH --tmp=128G
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00
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

TMPWORKDIR="/tmp/ood-job.${SLURM_JOBID}"
mkdir $TMPWORKDIR
cd $TMPWORKDIR

# Section to output information identifying the job, etc.
echo "Slurm job ${SLURM_JOBID} running on"
hostname
echo "To run on ${SLURM_NTASKS} CPU cores across ${SLURM_JOB_NUM_NODES} nodes"
echo "All nodes: ${SLURM_JOB_NODELIST}"
date
pwd
echo "Loaded modules are:"
module list

# Unpack the conda environment
/scratch/zt1/project/msml612/user/link/conda-pack-unpacker.sh -f /scratch/zt1/project/msml612/user/link/drone-rl.tar.gz
if [ $? -ne 0 ]; then
    echo "[ERROR] Error unpackaging /scratch/zt1/project/msml612/user/link/drone-rl.tar.gz"
    exit 1
fi

# Set wandb to offline mode for cluster runs
export WANDB_MODE=offline
source /home/link/wandb_secrets.sh

# Check if CUDA is accessible
/tmp/$USER/drone-rl/bin/python -c "import torch; print(torch.cuda.is_available())"

# Copy the zip file to the temporary working directory
cp /scratch/zt1/project/msml612/user/link/drone-rl_node_deploy.zip $TMPWORKDIR

# Unzip the copied file
unzip drone-rl_node_deploy.zip

# Change to src directory
cd src

# Launch training
/tmp/$USER/drone-rl/bin/python -m drone_rl.train.train \
  --config ../configs/training/teacher_large_debug.yaml \
  --wandb

# Save the exit code from the previous command
ECODE=$?

# Copy both run and wandb directories back to the scratch space
cp -r $TMPWORKDIR/src/runs /scratch/zt1/project/msml612/user/link/
cp -r $TMPWORKDIR/src/wandb /scratch/zt1/project/msml612/user/link/

# Copy f16trace.csv
cp $TMPWORKDIR/src/f16trace.csv /scratch/zt1/project/msml612/user/link/

echo "Job finished with exit code $ECODE"
date

echo "\n---"
echo "WANDB was run in offline mode. To upload your run after the job finishes, use:"
echo "  wandb sync /scratch/zt1/project/msml612/user/link/wandb/"
echo "Replace the path with your actual wandb run directory if different."
# Exit with the cached exit code
exit $ECODE