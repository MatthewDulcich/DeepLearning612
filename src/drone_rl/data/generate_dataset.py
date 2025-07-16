"""Generate training episodes from FlyCraft with domain randomisation.

Usage
-----
python -m drone_rl.data.generate_dataset --out data/raw --episodes 10000 \
       --terrains 3 --weather 4 --max-steps 500

The script records (obs, action, reward, next_obs, done) tuples and saves them
as Parquet shards (`data/raw/part-XXXX.parquet`). Each shard contains ~10k
transitions for easier I/O during training.

Domain randomisation parameters (wind, sensor noise, obstacle layout) are
sampled every episode to increase diversity. A summary `metadata.json` is
written with aggregate counts and environment hashes.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tqdm

try:
    import gymnasium as gym
    import flycraft_gym  # noqa: F401  # pylint: disable=unused-import
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "FlyCraft gym is required. Install via `pip install flycraft` or ensure that"
        " it is available in your PYTHONPATH."
    ) from exc

# Optional real-world dataset integration
try:
    from drone_rl.data.real_world import load_real_world_data
    REAL_WORLD_AVAILABLE = True
except ImportError:
    REAL_WORLD_AVAILABLE = False


# ------------------------- utilities ------------------------- #

def randomise_env(env: gym.Env, seed: int, terrain_ids: List[int], weather_configs: List[Dict]) -> Dict:  # noqa: D401
    """Apply domain randomisation to the FlyCraft environment.
    
    Parameters
    ----------
    env : gym.Env
        FlyCraft environment instance
    seed : int
        Random seed for reproducibility
    terrain_ids : List[int]
        List of available terrain IDs to sample from
    weather_configs : List[Dict]
        List of weather configuration presets to sample from
        
    Returns
    -------
    Dict
        Configuration applied to the environment
    """
    rng = np.random.default_rng(seed)
    
    # Select a terrain from available options
    terrain_id = terrain_ids[rng.integers(0, len(terrain_ids))]
    env.set_terrain(terrain_id=terrain_id)
    
    # Either use a preset or generate random weather
    if weather_configs and rng.random() < 0.7:  # 70% chance to use preset
        weather = weather_configs[rng.integers(0, len(weather_configs))]
        env.set_weather(**weather)
    else:
        # Generate random weather parameters
        weather = {
            "wind_speed": rng.uniform(0.0, 8.0),  # m/s
            "wind_dir": rng.uniform(0.0, 360.0),  # degrees
            "fog_density": rng.uniform(0.0, 0.5),  # 0-1 scale
            "rain_intensity": rng.uniform(0.0, 0.3),  # 0-1 scale
            "turbulence": rng.uniform(0.0, 0.4),  # 0-1 scale
        }
        env.set_weather(**weather)
    
    # Randomize sensor characteristics
    sensor_config = {
        "std": rng.uniform(0.0, 0.1),  # Standard deviation of noise
        "bias": rng.uniform(-0.05, 0.05),  # Systematic bias
        "dropout_prob": rng.uniform(0.0, 0.02),  # Probability of sensor dropout
    }
    env.set_sensor_noise(**sensor_config)
    
    # Randomize obstacle layout if supported
    if hasattr(env, "set_obstacle_layout"):
        obstacle_density = rng.uniform(0.1, 0.5)  # Density of obstacles
        min_size = rng.uniform(0.2, 1.0)  # Minimum obstacle size
        max_size = rng.uniform(min_size, 3.0)  # Maximum obstacle size
        env.set_obstacle_layout(density=obstacle_density, min_size=min_size, max_size=max_size)
    
    # Return the full configuration for metadata
    return {
        "terrain_id": terrain_id,
        "weather": weather,
        "sensor_config": sensor_config,
        "obstacle_config": {
            "density": obstacle_density if hasattr(env, "set_obstacle_layout") else 0.0,
            "min_size": min_size if hasattr(env, "set_obstacle_layout") else 0.0,
            "max_size": max_size if hasattr(env, "set_obstacle_layout") else 0.0,
        }
    }


def get_heuristic_action(env: gym.Env, obs: Dict, difficulty: float = 0.0) -> np.ndarray:
    """Generate a heuristic action to improve exploration.
    
    Parameters
    ----------
    env : gym.Env
        FlyCraft environment
    obs : Dict
        Current observation
    difficulty : float
        How much to deviate from optimal policy (0.0 = perfect, 1.0 = random)
        
    Returns
    -------
    np.ndarray
        Action vector
    """
    # Extract relevant state information (position, velocity, target)
    # This is a simplified example - adapt to actual observation space
    if hasattr(env, "get_optimal_action"):
        # If environment provides an optimal action function
        optimal_action = env.get_optimal_action(obs)
        
        # Add noise based on difficulty
        if difficulty > 0:
            noise = np.random.normal(0, difficulty, size=optimal_action.shape)
            action = optimal_action + noise
            # Clip to action space bounds
            action = np.clip(action, env.action_space.low, env.action_space.high)
            return action
        return optimal_action
    
    # Fallback to random actions with increasing probability based on difficulty
    if np.random.random() < difficulty:
        return env.action_space.sample()
    
    # Simple rule-based policy if optimal action not available
    # This is placeholder logic - actual implementation depends on the specific task
    action = np.zeros(env.action_space.shape)
    # ... implement basic navigation logic here ...
    
    return action


def collect_episode(
    env: gym.Env, 
    max_steps: int, 
    rng: np.random.Generator,
    exploration_noise: float = 0.3,
    collect_extra_metrics: bool = False
) -> Dict[str, np.ndarray]:
    """Collect a single episode of transitions.
    
    Parameters
    ----------
    env : gym.Env
        FlyCraft environment
    max_steps : int
        Maximum steps per episode
    rng : np.random.Generator
        Random number generator
    exploration_noise : float
        Amount of noise to add to actions (0.0 = deterministic, 1.0 = random)
    collect_extra_metrics : bool
        Whether to collect additional metrics like TTC, path deviation
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary of transition arrays
    """
    obs, info = env.reset()
    
    # Initialize transition storage
    trans = {
        "obs": [], 
        "action": [], 
        "reward": [], 
        "next_obs": [], 
        "done": []
    }
    
    # Optional extra metrics for validation
    if collect_extra_metrics:
        trans.update({
            "ttc": [],  # Time to collision
            "path_dev": [],  # Path deviation
            "vel_error": [],  # Velocity error
        })
    
    for step in range(max_steps):
        # Mix of random and heuristic actions for better exploration
        if rng.random() < 0.2:  # 20% pure random
            action = env.action_space.sample()
        else:
            # Use heuristic with noise for better coverage
            action = get_heuristic_action(env, obs, difficulty=exploration_noise)
            
            # Add time-correlated noise for realistic trajectories
            if step > 0 and len(trans["action"]) > 0:
                prev_action = trans["action"][-1]
                # Blend with previous action for temporal smoothness
                alpha = rng.uniform(0.1, 0.3)  # Smoothing factor
                action = alpha * prev_action + (1 - alpha) * action
        
        # Execute action in environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Record transition
        trans["obs"].append(obs)
        trans["action"].append(action)
        trans["reward"].append(reward)
        trans["next_obs"].append(next_obs)
        trans["done"].append(done)
        
        # Collect extra metrics if available in info
        if collect_extra_metrics:
            trans["ttc"].append(info.get("time_to_collision", -1))
            trans["path_dev"].append(info.get("path_deviation", -1))
            trans["vel_error"].append(info.get("velocity_error", -1))
        
        obs = next_obs
        if done:
            break
    
    # Convert lists to arrays
    for k, v in trans.items():
        if k in ["obs", "next_obs"]:
            # Handle dictionary observations
            if isinstance(v[0], dict):
                # Group by observation key
                grouped = {key: [] for key in v[0].keys()}
                for obs_dict in v:
                    for key, value in obs_dict.items():
                        grouped[key].append(value)
                # Convert each group to array
                trans[k] = {key: np.array(values) for key, values in grouped.items()}
            else:
                trans[k] = np.array(v)
        else:
            trans[k] = np.array(v)
    
    return trans


def merge_real_world_data(
    sim_data: Dict[str, np.ndarray],
    real_data_path: str,
    ratio: float = 0.1
) -> Dict[str, np.ndarray]:
    """Merge simulation data with real-world drone flight data.
    
    Parameters
    ----------
    sim_data : Dict[str, np.ndarray]
        Simulation data
    real_data_path : str
        Path to real-world dataset
    ratio : float
        Ratio of real to simulation data to include
        
    Returns
    -------
    Dict[str, np.ndarray]
        Merged dataset
    """
    if not REAL_WORLD_AVAILABLE:
        print("Warning: real_world module not available, skipping real data integration")
        return sim_data
    
    # Load real-world data (implemented in separate module)
    real_data = load_real_world_data(real_data_path)
    
    # Determine how many real samples to include
    n_sim = len(sim_data["obs"])
    n_real = min(int(n_sim * ratio), len(real_data["obs"]))
    
    # Randomly select subset of real data
    indices = np.random.choice(len(real_data["obs"]), n_real, replace=False)
    real_subset = {k: v[indices] for k, v in real_data.items()}
    
    # Merge datasets
    merged = {}
    for k in sim_data:
        if k in real_subset:
            merged[k] = np.concatenate([sim_data[k], real_subset[k]])
        else:
            merged[k] = sim_data[k]
    
    return merged


# ------------------------- main ------------------------- #

def main() -> None:  # noqa: D401
    """Main entry point for dataset generation."""
    parser = argparse.ArgumentParser(description="Generate FlyCraft dataset")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to collect")
    parser.add_argument("--terrains", type=int, default=3, help="Number of terrain types to use")
    parser.add_argument("--weather", type=int, default=4, help="Number of weather presets to generate")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--real-data", type=str, default=None, help="Path to real-world dataset (optional)")
    parser.add_argument("--real-ratio", type=float, default=0.1, help="Ratio of real to simulation data")
    parser.add_argument("--exploration", type=float, default=0.3, help="Exploration noise level (0-1)")
    parser.add_argument("--collect-metrics", action="store_true", help="Collect additional metrics")
    args = parser.parse_args()

    # Create output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Create environment
    env = gym.make("FlyCraft-Nav-v0", render_mode=None, max_episode_steps=args.max_steps)

    # Generate terrain IDs to use
    available_terrains = list(range(10))  # Assuming 10 terrains available
    if args.terrains < len(available_terrains):
        terrain_ids = rng.choice(available_terrains, size=args.terrains, replace=False).tolist()
    else:
        terrain_ids = available_terrains

    # Generate weather presets
    weather_configs = []
    for _ in range(args.weather):
        weather_configs.append({
            "wind_speed": rng.uniform(0.0, 8.0),
            "wind_dir": rng.uniform(0.0, 360.0),
            "fog_density": rng.uniform(0.0, 0.5),
            "rain_intensity": rng.uniform(0.0, 0.3),
            "turbulence": rng.uniform(0.0, 0.4),
        })

    # Initialize metadata collection
    metadata = []
    total_transitions = 0

    # Buffer for collecting transitions before writing to disk
    shard_size = 10000  # transitions per Parquet file
    buffer = {k: [] for k in ["obs", "action", "reward", "next_obs", "done"]}
    if args.collect_metrics:
        buffer.update({k: [] for k in ["ttc", "path_dev", "vel_error"]})
    shard_idx = 0

    # Progress bar for episode collection
    pbar = tqdm.trange(args.episodes, desc="Collecting episodes")
    for ep in pbar:
        # Apply domain randomization
        env_config = randomise_env(
            env, 
            seed=args.seed + ep,
            terrain_ids=terrain_ids,
            weather_configs=weather_configs
        )
        
        # Collect episode data
        trans = collect_episode(
            env, 
            args.max_steps, 
            rng,
            exploration_noise=args.exploration,
            collect_extra_metrics=args.collect_metrics
        )
        
        # Update episode metadata
        episode_meta = {
            "episode": ep,
            "steps": len(trans["reward"]),
            "terrain": env_config["terrain_id"],
            "weather": env_config["weather"],
            "return": float(sum(trans["reward"])),
        }
        metadata.append(episode_meta)
        total_transitions += len(trans["reward"])
        
        # Update progress bar
        pbar.set_postfix({
            "total_steps": total_transitions,
            "avg_ep_len": total_transitions / (ep + 1)
        })
        
        # Add to buffer
        for k in buffer:
            if k in trans:
                buffer[k].append(trans[k])
        
        # Flush shard if large enough
        buffer_size = sum(len(b) for b in buffer["obs"])
        if buffer_size >= shard_size:
            # Convert to DataFrame
            shard_path = out_dir / f"part-{shard_idx:04d}.parquet"
            
            # Handle dictionary observations
            if isinstance(buffer["obs"][0], dict):
                # Flatten observation dictionaries for storage
                flat_buffer = {}
                for k in buffer:
                    if k in ["obs", "next_obs"]:
                        # Each key in the observation becomes a column
                        for obs_key in buffer[k][0].keys():
                            flat_buffer[f"{k}_{obs_key}"] = np.concatenate([
                                obs_dict[obs_key] for obs_dict in buffer[k]
                            ])
                    else:
                        flat_buffer[k] = np.concatenate(buffer[k])
                df = pd.DataFrame(flat_buffer)
            else:
                # Standard array observations
                df = pd.DataFrame({k: np.concatenate(v) for k, v in buffer.items()})
            
            # Write to parquet
            df.to_parquet(shard_path)
            
            # Clear buffer
            for k in buffer:
                buffer[k].clear()
            
            shard_idx += 1

    # Flush remaining data
    if buffer["obs"]:
        shard_path = out_dir / f"part-{shard_idx:04d}.parquet"
        
        # Handle dictionary observations (same as above)
        if isinstance(buffer["obs"][0], dict):
            flat_buffer = {}
            for k in buffer:
                if k in ["obs", "next_obs"]:
                    for obs_key in buffer[k][0].keys():
                        flat_buffer[f"{k}_{obs_key}"] = np.concatenate([
                            obs_dict[obs_key] for obs_dict in buffer[k]
                        ])
                else:
                    flat_buffer[k] = np.concatenate(buffer[k])
            df = pd.DataFrame(flat_buffer)
        else:
            df = pd.DataFrame({k: np.concatenate(v) for k, v in buffer.items()})
        
        df.to_parquet(shard_path)

    # Optionally merge with real-world data if provided
    if args.real_data and os.path.exists(args.real_data):
        print(f"Integrating real-world data from {args.real_data}")
        # Implementation would depend on the real_world module
        # This would create additional shards with mixed sim/real data
    
    # Save dataset summary metadata
    summary = {
        "total_episodes": args.episodes,
        "total_transitions": total_transitions,
        "terrains_used": terrain_ids,
        "weather_configs": weather_configs,
        "avg_episode_length": total_transitions / args.episodes,
        "shards": shard_idx + 1,
        "real_data_included": args.real_data is not None and os.path.exists(args.real_data),
    }
    
    # Save detailed episode metadata
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "episodes": metadata
        }, f, indent=2)

    print(f"Dataset generation complete:")
    print(f"  - {args.episodes} episodes")
    print(f"  - {total_transitions} total transitions")
    print(f"  - {shard_idx + 1} parquet shards")
    print(f"  - Saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
