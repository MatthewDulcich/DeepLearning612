#!/bin/bash
#SBATCH --job-name=baseline_lstm
#SBATCH --output=logs/baseline_lstm_%j.out
#SBATCH --error=logs/baseline_lstm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Activate your environment (adjust path as needed)
source ~/anaconda3/etc/profile.d/conda.sh
conda activate /Users/matthewdulcich/Documents/DevProjects/Reinforcement\ Learning\ Projects/DeepLearning612/drone_venv

# Create logs directory if it doesn't exist
mkdir -p logs

# Run training
python src/drone_rl/train/train.py \
  --config configs/baseline_lstm.yaml \
  --wandb
