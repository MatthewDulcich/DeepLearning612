import numpy as np
import torch
from stable_baselines3 import PPO  # Change to your RL algorithm if needed

from src.drone_rl.utils import metrics

from drone_rl.train import make_env

def load_reference_trajectory():
    # TODO: Replace with your actual reference trajectory loading
    return np.load("reference_trajectory.npy")

def load_obstacles():
    # TODO: Replace with your actual obstacle loading
    # Example: [(np.array([x, y, z]), radius), ...]
    return []

def load_goal():
    # TODO: Replace with your actual goal position
    return np.array([0.0, 0.0, 0.0])

def evaluate_model(model_path, n_episodes=20, render=False):
    model = PPO.load(model_path)
    env = make_env()
    reference_trajectory = load_reference_trajectory()
    obstacles = load_obstacles()
    goal_pos = load_goal()

    all_metrics = []
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        episode_data = {
            "positions": [],
            "velocities": [],
            "timestamps": [],
        }
        t0 = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, done, info = env.step(action)
            # You may need to adjust these keys based on your env's info dict
            episode_data["positions"].append(info.get("position", np.zeros(3)))
            episode_data["velocities"].append(info.get("velocity", np.zeros(3)))
            episode_data["timestamps"].append(info.get("timestamp", 0.0))
            obs = next_obs
            if render:
                env.render()
        # Convert lists to arrays
        for k in episode_data:
            episode_data[k] = np.array(episode_data[k])
        metrics_result = metrics.compute_all_metrics(
            episode_data,
            reference_trajectory,
            obstacles,
            goal_pos
        )
        all_metrics.append(metrics_result)
        print(f"Episode {ep+1}: {metrics_result}")

    # Aggregate and print summary
    print("\n=== Final Evaluation Summary ===")
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics if m[key] is not None and isinstance(m[key], (float, int, np.floating, np.integer))]
        if vals:
            print(f"{key}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    evaluate_model(args.model, n_episodes=args.episodes, render=args.render)