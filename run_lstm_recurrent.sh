#!/bin/bash
# Run RecurrentPPO LSTM training
# Usage: ./run_lstm_recurrent.sh [--wandb] [--seed SEED]

set -e

# Default values
USE_WANDB=false
SEED=42
CONFIG="configs/baseline_lstm.yaml"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --wandb)
            USE_WANDB=true
            shift
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2" 
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--wandb] [--seed SEED] [--config CONFIG]"
            echo "  --wandb      Enable Weights & Biases logging"
            echo "  --seed SEED  Set random seed (default: 42)"
            echo "  --config CONFIG  Path to config file (default: configs/baseline_lstm.yaml)"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

echo "Starting RecurrentPPO LSTM training..."
echo "Config: $CONFIG"
echo "Seed: $SEED"
echo "Wandb: $USE_WANDB"

# Check if we should use the new dedicated training script or the main one
if [[ -f "train_lstm_recurrent.py" ]]; then
    echo "Using dedicated RecurrentPPO training script..."
    SCRIPT="train_lstm_recurrent.py"
    ARGS="--config $CONFIG"
    
    if [[ "$USE_WANDB" == "true" ]]; then
        ARGS="$ARGS --wandb"
    fi
    
    # Add seed if not using config
    if [[ "$SEED" != "42" ]]; then
        ARGS="$ARGS --seed $SEED"
    fi
else
    echo "Using main training script with RecurrentPPO support..."
    SCRIPT="src/drone_rl/train/train.py"
    ARGS="--config $CONFIG --seed $SEED"
    
    if [[ "$USE_WANDB" == "true" ]]; then
        ARGS="$ARGS --wandb"
    fi
fi

# Activate virtual environment if it exists
if [[ -d "drone_venv" ]]; then
    echo "Activating virtual environment..."
    source drone_venv/bin/activate
fi

# Install dependencies if needed
echo "Checking dependencies..."

# Check for sb3-contrib
python -c "import sb3_contrib" 2>/dev/null || {
    echo "Installing sb3-contrib..."
    pip install sb3-contrib
}

# Check for flycraft (Linux-specific)
python -c "import flycraft" 2>/dev/null || {
    echo "Warning: FlyCraft not found."
    echo "FlyCraft requires Linux environment. On macOS, training will fail."
    echo "To install FlyCraft on Linux:"
    echo "  git clone https://github.com/flycraft-team/flycraft"
    echo "  cd flycraft && pip install -e ."
    
    # Check if we're on macOS and should exit
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Detected macOS - FlyCraft training not supported."
        echo "Please run this script on a Linux system."
        exit 1
    fi
}

# Run training
echo "Running: python $SCRIPT $ARGS"
python $SCRIPT $ARGS

echo "Training completed!"
