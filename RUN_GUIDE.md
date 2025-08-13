# Drone Transformer RL – Run Guide

This guide walks you through environment setup, data generation, training, evaluation, and demo deployment **locally** and on the **zaratan@umd SLURM HPC cluster**.

---

## 0. Prerequisites

1. Conda ≥ 23.x (Mambaforge / Miniconda).  
2. NVIDIA GPU with CUDA 12 support (A100/V100) for accelerated training.  
   • If you only have CPU, set `device=cpu` in configs.  
3. Outbound internet for optional W&B logging and dataset download.

---

## 1. Clone repository & create environment

```bash
git clone <repo-url> drone_transformer_rl
cd drone_transformer_rl

# create env (≈5 min with mamba, 12 min with conda)
conda env create -f env.yml
conda activate drone-rl

# optional: install git hooks for auto-format/lint
pre-commit install
```

---

## 2. Generate training data (≥ 1 M transitions)

```bash
python -m drone_rl.data.generate_dataset \
  --out data/raw \
  --episodes 3000 \
  --terrains 5 \
  --weather 6 \
  --max-steps 500 \
  --collect-metrics \
  --seed 0
```

Outputs: sharded Parquet files in `data/raw/` and a companion `metadata.json` with counts and environment configs.

---

## 3. Local GPU training

### 3.1 Teacher (large Transformer)

```bash
python -m drone_rl.train.train \
  --config configs/teacher_large.yaml \
  --wandb
```

### 3.2 Distilled student

```bash
python -m drone_rl.train.train \
  --config configs/student_distilled.yaml \
  --teacher runs/teacher_large/final_model.zip \
  --wandb
```

### 3.3 Baseline LSTM

```bash
python -m drone_rl.train.train \
  --config configs/baseline_lstm.yaml
```

### 3.4 Perceiver + ACT (sensor-fusion)

```bash
python -m drone_rl.train.train \
  --config configs/perceiver_act.yaml \
  --wandb
```

Logs, TensorBoard files and checkpoints are created inside `runs/<run-name>/`.

---

## 4. Optuna hyper-parameter sweep (example: 32 trials)

```bash
python -m drone_rl.train.train \
  --config configs/teacher_large.yaml \
  --sweep 32 \
  --wandb
```

A `best_params.json` is written to the run directory; final retraining with the optimal parameters happens automatically.

---

## 5. Launch local Streamlit demo

```bash
streamlit run demo/app.py \
  -- \
  --checkpoint runs/student_distilled/final_model.zip
```

Then open `http://localhost:8501` in your browser.

---

## 6. Running on **zaratan@umd** (SLURM HPC)

### 6.1 Load modules & build environment

```bash
ssh <netid>@login.zaratan.umd.edu
module load cuda/12.3.0

# create env in $SCRATCH to avoid home quota limits
conda env create -f env.yml -p $SCRATCH/conda/envs/drone-rl
conda activate $SCRATCH/conda/envs/drone-rl
```

### 6.2 Submit jobs

```bash
cd $SCRATCH/drone_transformer_rl

sbatch slurm/train_teacher_a100.sbatch         # large Transformer
sbatch slurm/train_student_a100.sbatch         # distilled student
sbatch slurm/sweep_performer_a100.sbatch       # optional sweep
```

Each script requests one A100 GPU and stages logs to  
`$SCRATCH/drone_runs/<jobid>/`.  
Check progress with `squeue -u $USER` and `tensorboard --logdir <path>`.

---

## 7. Unit tests & lint

```bash
pytest -q                         # fast API smoke tests
autopep8 -d src/                  # show style differences
```

Running `pre-commit run --all-files` applies black, isort, flake8 and pytest.

---

## 8. Full reproducibility shortcut

```bash
make reproduce
```

The convenience target will:

1. Create/activate env  
2. Generate a small dataset subset  
3. Train the distilled student for 1 M steps  
4. Launch the Streamlit demo

---

Happy flying 🚁
