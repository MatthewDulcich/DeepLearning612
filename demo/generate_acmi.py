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

# Try to import wrappers if available
try:
    from utils_my.sb3.my_wrappers import ScaledActionWrapper, ScaledObservationWrapper
except ImportError:
    ScaledActionWrapper = None
    ScaledObservationWrapper = None

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
    parser.add_argument("--origin-lon", type=float, default=0.0, help="Reference longitude in degrees for flat-world export (spherical anchor).")
    parser.add_argument("--origin-lat", type=float, default=0.0, help="Reference latitude in degrees for flat-world export (spherical anchor).")
    parser.add_argument("--alt-offset", type=float, default=0.0, help="Meters to add to altitude when exporting (useful if sim Z=AGL).")
    args = parser.parse_args()

    # Load config (YAML)
    import yaml
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Create environment with wrappers if available
    env = gym.make("FlyCraft", max_episode_steps=config.get("max_steps", 1000))
    if ScaledObservationWrapper is not None:
        env = ScaledObservationWrapper(env)
    if ScaledActionWrapper is not None:
        env = ScaledActionWrapper(env)

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
        # 1. Try info dict
        pos = info.get("drone_position")
        vel = info.get("drone_velocity")
        if pos is not None and vel is not None:
            print(f"[DEBUG] Extracted from info: pos={pos}, vel={vel}")
            return np.array(pos), np.array(vel)
        # 2. Try obs dict
        if isinstance(obs, dict):
            pos = obs.get("position")
            vel = obs.get("velocity")
            if pos is not None and vel is not None:
                print(f"[DEBUG] Extracted from obs dict: pos={pos}, vel={vel}")
                return np.array(pos), np.array(vel)
            # Try achieved_goal/observation fallback
            pos = obs.get("achieved_goal")
            obs_arr = obs.get("observation")
            if pos is not None and obs_arr is not None and len(obs_arr) >= 6:
                vel = obs_arr[3:6]
                print(f"[DEBUG] Extracted from obs['achieved_goal'] and obs['observation']: pos={pos}, vel={vel}")
                return np.array(pos), np.array(vel)
        # 3. Try flat obs array
        if isinstance(obs, (np.ndarray, list)) and len(obs) >= 6:
            arr = np.array(obs)
            print(f"[DEBUG] Extracted from flat obs: pos={arr[:3]}, vel={arr[3:6]}")
            return arr[:3], arr[3:6]
        # 4. Fallback
        print("[DEBUG] Could not extract pos/vel, returning zeros.")
        return np.zeros(3), np.zeros(3)

    def rollout_and_save_acmi(env, model, max_steps=1000):
        """Run a rollout and save trajectory as .acmi file. Returns (success, trajectory)."""
        obs, info = env.reset()
        trajectory = []
        t = 0.0
        dt = 0.05  # 20Hz default
        success = False
        for step in range(max_steps):
            print(f"[DEBUG] Step {step} obs: {obs}")
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
            print(f"[DEBUG] Step {step} pos: {pos}, vel: {vel}")
            trajectory.append((t, pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]))
            t += dt
            if terminated or truncated:
                # Check for success flag in info, or define your own success condition
                success = info.get("is_success", False)
                break
        return success, trajectory

    # Minimal loop: keep running until success
    attempt = 0
    save_every = 100  # Set your interval here

    while True:
        attempt += 1
        acmi_path = save_dir / f"rollout_{args.algo}_attempt{attempt}.acmi"
        success, trajectory = rollout_and_save_acmi(env, model, max_steps=config.get("max_steps", 1000))

        # Only save every x attempts or if successful
        if (attempt % save_every == 0) or success:
            with open(acmi_path, "w", encoding="utf-8", newline="\n") as f:
                # --- ACMI 2.2 header ---
                f.write("FileType=text/acmi/tacview\n")
                f.write("FileVersion=2.2\n")
                f.write(f"0,ReferenceTime={config.get('reference_time', '2025-07-24T00:00:00Z')}\n")
                f.write("0,Title=FlyCraft RL Rollout\n")
                f.write("0,Author=DeepLearning612 RL Demo\n")
                f.write("0,DataSource=RL Simulation\n")
                f.write("0,DataRecorder=generate_acmi.py\n")
                # Optional numeric properties to anchor flat-world U/V
                f.write(f"0,ReferenceLongitude={args.origin_lon}\n")
                f.write(f"0,ReferenceLatitude={args.origin_lat}\n")

                # --- Start of telemetry ---
                # Use id '1' (hex) for our drone
                if trajectory:
                    first_t, x0, y0, z0, vx0, vy0, vz0 = trajectory[0]
                else:
                    first_t, x0, y0, z0 = 0.0, 0.0, 0.0, 0.0

                # Initial frame and object creation with properties
                f.write(f"#{first_t:.2f}\n")
                f.write("1,Name=Drone,Type=Air+UAV,Color=Blue\n")
                # Flat-world notation: T = Lon|Lat|Alt|U|V (meters for Alt/U/V)
                f.write(
                    f"1,T={args.origin_lon}|{args.origin_lat}|{z0 + args.alt_offset:.2f}|{x0:.2f}|{y0:.2f}\n"
                )

                # Subsequent frames
                for (t, x, y, z, vx, vy, vz) in trajectory[1:]:
                    f.write(f"#{t:.2f}\n")
                    # Keep the same notation every time; provide full T for clarity
                    f.write(
                        f"1,T={args.origin_lon}|{args.origin_lat}|{z + args.alt_offset:.2f}|{x:.2f}|{y:.2f}\n"
                    )

                # Clean object removal at the end (instead of old RemoveObject command)
                if trajectory:
                    f.write(f"#{trajectory[-1][0]:.2f}\n")
                f.write("-1\n")
            print(f"Saved ACMI file to: {acmi_path}")

        if success:
            print(f"Success on attempt {attempt}!")
            break
        else:
            print(f"Attempt {attempt} not successful, retrying...")

if __name__ == "__main__":
    main()
