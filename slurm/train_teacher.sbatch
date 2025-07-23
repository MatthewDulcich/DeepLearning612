#!/bin/bash
#SBATCH --job-name=teacher-train         # Name for your job
#SBATCH --partition=cml-zhou           # Your dedicated partition with A100s
#SBATCH --account=cml-zhou             # Your account
#SBATCH --qos=cml-medium
#SBATCH --gres=gpu:a100:1              # Request one A100 GPU
#SBATCH --cpus-per-task=8              # Request 8 CPU cores
#SBATCH --mem=64G                      # Request 64GB of memory
#SBATCH --time=04:00:00                # Job time limit (4 hours)
#SBATCH --output=slurm-%j.out          # Log file for job output

echo "Job started on $(hostname) at $(date)"

# --- Load the required modules ---
module load gcc/15.1.0
module load cuda/12.8.1
module load Python3/3.11.2

# --- Activate your Python environment ---
cd /cmlscratch/advait25/drone_transformer_rl
source drone_venv/bin/activate

# --- Run the training command for the large teacher model ---
python -m src.drone_rl.train.train \
  --config configs/teacher_large.yaml \
  --wandb

echo "Job finished at $(date)"