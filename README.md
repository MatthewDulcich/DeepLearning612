# Usage
1.  Zip `config` and `src` and place in scratch
2.  Add provided `conda-pack-unpacker.sh` to scratch
3.  Use `conda-pack` to compress your conda environment/installation (This will allow you to then delete the conda environment/installation as the packed conda environment is self-sufficient)
4.  Edit and use the provided sbatch script

# Reason
- Reduce space used in scratch
- Offload everything to compute node
- Copies back `runs` folder, `wandb` folder, and `f16trace.csv` into your scratch directory
- `wandb` can then be synced