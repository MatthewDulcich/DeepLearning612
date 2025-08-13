"""
Generate a .acmi (Tacview) flight log from a trained RL model in FlyCraft.

Usage:
    python demo/generate_acmi.py --config configs/baseline_lstm.yaml --model-path runs/baseline_lstm/final_model.zip --save-dir demo/acmi_logs --algo rl --save-acmi

This script loads a trained model, runs a rollout in FlyCraft, and saves the trajectory as a .acmi file for visualization in Tacview.
"""
import argparse
from pathlib import Path
import sys
import os
import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO

# Local imports
try:
    import flycraft  # noqa: F401
except ImportError:
    print("FlyCraft gym not installed. Install with: pip install flycraft")
    sys.exit(1)

from src.drone_rl.models.transformer_policy import TransformerActorCritic
from src.drone_rl.models.baselines import SimpleLSTMPolicy, DronePositionController
from src.drone_rl.utils.metrics import time_to_collision, path_deviation, velocity_error

# Register FlyCraft env if needed
if hasattr(gym, "register_envs"):
    gym.register_envs(flycraft)

POLICIES = {
    "transformer": TransformerActorCritic,
    "lstm": SimpleLSTMPolicy,
    "pid": DronePositionController,
}

def main():
    parser = argparse.ArgumentParser(description="Generate .acmi file from RL policy.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model .zip file.")
    parser.add_argument("--save-dir", type=str, default="demo/acmi_logs", help="Directory to save .acmi file.")
    parser.add_argument("--algo", type=str, default="lstm", choices=["transformer", "lstm", "pid"], help="Policy type.")
    parser.add_argument("--save-acmi", action="store_true", help="Save .acmi file for Tacview visualization.")
    args = parser.parse_args()

    # Load config (YAML)
    import yaml
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Create environment
    env = gym.make("FlyCraft", max_episode_steps=config.get("max_steps", 1000))

    # Load model
    if args.algo == "pid":
        model = DronePositionController()
    else:
        policy_class = POLICIES[args.algo]
        model = PPO.load(args.model_path, env=env, device="cpu", custom_objects={"policy_class": policy_class})

    # Rollout and save ACMI
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    acmi_path = save_dir / f"rollout_{args.algo}.acmi"


    # --- Minimal rollout and ACMI writer ---
    def extract_pos_vel(obs, info):
        # Try info first
        pos = info.get("drone_position")
        vel = info.get("drone_velocity")
        if pos is not None and vel is not None:
            return np.array(pos), np.array(vel)
        # Try obs dict
        if isinstance(obs, dict):
            pos = obs.get("position")
            vel = obs.get("velocity")
            if pos is not None and vel is not None:
                return np.array(pos), np.array(vel)
        # Try obs as flat array: assume [x, y, z, vx, vy, vz, ...]
        if isinstance(obs, (np.ndarray, list)) and len(obs) >= 6:
            arr = np.array(obs)
            return arr[:3], arr[3:6]
        # Fallback: zeros
        return np.zeros(3), np.zeros(3)

    def rollout_and_save_acmi(env, model, acmi_path, max_steps=1000):
        """Run a rollout and save trajectory as .acmi file. Returns (success, trajectory)."""
        obs, info = env.reset()
        trajectory = []
        t = 0.0
        dt = 0.05  # 20Hz default
        success = False
        for step in range(max_steps):
            if hasattr(model, 'predict'):
                action, _ = model.predict(obs, deterministic=True)
            else:
                # PID controller
                target_pos = info.get("target_position", np.zeros(3))
                current_pos, current_vel = extract_pos_vel(obs, info)
                current_time = t
                action = model(
                    target_pos, current_pos, current_vel, current_time,
                    target_yaw=info.get("target_yaw", 0.0),
                    current_yaw=info.get("drone_yaw", 0.0)
                )
            obs, reward, terminated, truncated, info = env.step(action)
            pos, vel = extract_pos_vel(obs, info)
            trajectory.append((t, pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]))
            t += dt
            if terminated or truncated:
                # Check for success flag in info, or define your own success condition
                success = info.get("is_success", False)
                break
        # Write ACMI file
        with open(acmi_path, "w") as f:
            f.write("FileType=text/acmi/tacview\n")
            f.write("FileVersion=2.1\n")
            f.write("0,ReferenceTime=2025-07-24T00:00:00Z\n")
            f.write("0,Title=FlyCraft RL Rollout\n")
            f.write("0,Author=DeepLearning612 RL Demo\n")
            f.write("0,DataSource=RL Simulation\n")
            f.write("0,Recorder=generate_acmi.py\n")
            f.write("0,\n")
            f.write("0,AddObject,Aircraft,1\n")
            f.write("0,1,Name=Drone\n")
            f.write("0,1,Type=UAV\n")
            f.write("0,1,Color=Blue\n")
            for t, x, y, z, vx, vy, vz in trajectory:
                f.write(f"{t:.2f},1,T={x:.2f}|{y:.2f}|{z:.2f}|{vx:.2f}|{vy:.2f}|{vz:.2f}\n")
            f.write("0,RemoveObject,1\n")
        print(f"Saved ACMI file to: {acmi_path}")
        return success

    # Minimal loop: keep running until success
    attempt = 0
    while True:
        attempt += 1
        acmi_path = save_dir / f"rollout_{args.algo}_attempt{attempt}.acmi"
        success = rollout_and_save_acmi(env, model, acmi_path, max_steps=config.get("max_steps", 1000))
        if success:
            print(f"Success on attempt {attempt}!")
            break
        else:
            print(f"Attempt {attempt} not successful, retrying...")

if __name__ == "__main__":
    main()
