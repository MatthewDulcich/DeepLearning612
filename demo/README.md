# Demo Folder: FlyCraft RL ACMI Generation

This folder contains scripts for generating and visualizing flight trajectories from trained RL models in the FlyCraft environment.

## Generating Tacview .acmi Files

The script `generate_acmi.py` runs a trained RL or PID model in the FlyCraft environment and saves the trajectory as a `.acmi` file, which can be visualized in [Tacview](https://tacview.net/).

### Usage

```bash
python generate_acmi.py --config <config_yaml> --model-path <model_zip> --save-dir <output_dir> --algo <policy_type> --save-acmi
```

- `--config`: Path to your YAML config file (e.g., `configs/baseline_lstm.yaml`)
- `--model-path`: Path to your trained model checkpoint (e.g., `runs/baseline_lstm/final_model.zip`)
- `--save-dir`: Directory to save the `.acmi` file (default: `demo/acmi_logs`)
- `--algo`: Policy type (`transformer`, `lstm`, or `pid`)
- `--save-acmi`: Flag to save the ACMI file (required for Tacview visualization)

**Example:**

```bash
python generate_acmi.py \
  --config configs/baseline_lstm.yaml \
  --model-path runs/baseline_lstm/final_model.zip \
  --save-dir demo/acmi_logs \
  --algo lstm \
  --save-acmi
```

This will create a file like `demo/acmi_logs/rollout_lstm.acmi`.

## Visualizing in Tacview

1. Download and install [Tacview](https://tacview.net/).
2. Open the generated `.acmi` file in Tacview to view the flight trajectory.

---

**Note:**
- The ACMI file uses x/y/z as placeholders for longitude/latitude/altitude. For real-world mapping, you may need to adjust the coordinate conversion.
- The script does not require any external rollout dependencies.
