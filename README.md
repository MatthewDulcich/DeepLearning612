# Drone Transformer RL

Autonomous drone navigation with a **Transformer-based Reinforcement-Learning policy**.

This repository accompanies the MSML 612 group project (University of Maryland). It contains **fully-reproducible code**, a Streamlit demo, HPC SLURM job scripts, and comprehensive documentation.

---

## 1. Problem statement
Modern UAVs must avoid obstacles, regulate velocity, and remain on course in dynamic 3-D environments. Rule-based controllers struggle with unseen situations. We solve this by coupling:

1. **Sequence modelling** – a deep Transformer encoder captures long-term temporal dependencies in state trajectories.  
2. **Reinforcement learning** – PPO trains the policy through interaction with high-fidelity simulators (FlyCraft Gym, AirSim Drone Racing Lab).

---

## 2. Key contributions
- **Hierarchical Transformer Policy** – multi-scale attention, relative position encodings, windowed memory and Transformer-XL recurrence for superior long-horizon reasoning.
- **Curriculum & Domain Randomisation** – progressive task difficulty and environment variability for strong generalisation.
- **Knowledge Distillation** – large teacher → lightweight student for real-time inference on embedded hardware.
- **Optuna Hyper-parameter Sweeps** – automated search across depth, heads, dropout and RL coefficients.
- **On-policy Data Augmentation** – sensor noise injection, adversarial wind gusts.
- **Streamlit Live Demo** – run a trained student model directly in the browser and visualise flight metrics in real-time.
- **Perceiver IO + ACT** – modality-aware Perceiver cross-attention fuses position / velocity / lidar.  
  Adaptive Computation Time lets “easy” states exit early, cutting average latency while maintaining performance.  
- **Contrastive Pre-training** – optional self-supervised phase on raw flight logs before RL fine-tuning.

---

## 3. Repository structure
```
drone_transformer_rl/
├── src/                # Python packages (importable as drone_rl)
│   ├── data/           # Simulation, logging, augmentation
│   ├── models/         # Transformer architectures
│   ├── train/          # Training & evaluation loops
│   ├── utils/          # Helpers & metrics
├── configs/            # YAML/JSON config files
├── demo/               # Streamlit app
├── slurm/              # Example HPC job scripts for zaratan@umd
├── notebooks/          # Optional exploratory notebooks
├── tests/              # Unit tests / smoke tests
├── env.yml             # Conda environment spec (CUDA & CPU)
├── RUN_GUIDE.md        # End-to-end instructions
└── README.md           # You are here
```

---

## 4. Quickstart (local GPU)

```bash
# 1. Create environment (CUDA 12.1 build – adjust if your GPU uses 11.x)
conda env create -f env.yml
conda activate drone-rl
cd src

# 2. Train teacher policy (Transformer-XL-Large)
python -m drone_rl.train.train --config ../configs/teacher_large.yaml

# 3. Distil to student + evaluate
python -m drone_rl.train.train --config ../configs/student_distilled.yaml
python -m drone_rl.train.evaluate --model runs/student/best.pt

# 4. (Optional) Train Perceiver + ACT sensor-fusion policy
python -m drone_rl.train.train --config configs/perceiver_act.yaml

# 5. (Optional) Generate a dataset for Offline RL (≈ 10 min for 3 k episodes)
python -m drone_rl.data.generate_dataset --out data/raw --episodes 3000

# 6. Launch the Streamlit demo
streamlit run demo/app.py -- --checkpoint runs/student/best.pt
```


## 5. Quickstart on **zaratan@umd** (SLURM HPC)

```bash
# Login and clone
ssh <netid>@login@zaratan.umd.edu
git clone <repo-url> drone_transformer_rl
cd drone_transformer_rl

# Load CUDA module that matches available A100 GPUs
module load cuda/12.3.0

# Read HPC and Anaconda docs on how to install Miniconda

# create env in $SCRATCH, then conda-pack, and delete conda installation to avoid home and scratch quota limits
conda env create -f env.yml -p $SCRATCH/conda/envs/drone-rl
conda activate base
conda install conda-pack
conda-pack drone-rl
conda deactivate
rm -rf $SCRATCH/condainstall

# Submit a job (single-GPU)
sbatch slurm/train_teacher_a100.sbatch
```

Job scripts automatically resume from checkpoints and copy logs to `$SCRATCH/drone_logs`. See `slurm/README.md` for multi-GPU and Optuna sweep variants.

---

## 6. Evaluation metrics
The following metrics are logged each episode and summarised in `wandb`/CSV:

| Metric | Purpose |
|--------|---------|
| Time-to-Collision (↑) | Safety buffer before impact |
| Path Deviation (↓) | Route fidelity using DTW distance |
| Course Completion Time (↓) | Task efficiency |
| Success Rate on Unseen Paths (↑) | Generalisation |
| Distillation Accuracy Retention (↑) | Student vs Teacher gap |

---


## 7. License
This repository is released under the MIT License. Refer to `LICENSE` for details.

Happy flying! 🚁
