#!/bin/bash
#SBATCH --job-name=teacher_large
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Load modules
module load cuda/12.1 gcc/11.3

# Activate Conda environment located in scratch
source "$SCRATCH/conda/envs/drone-rl/bin/activate"

# Navigate to directory where the job was submitted
cd "$SLURM_SUBMIT_DIR"

# Launch training
python -m drone_rl.train.train \
  --config configs/teacher_large.yaml \
  --wandb
