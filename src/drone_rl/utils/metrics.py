"""Metric computation utilities for drone navigation evaluation.

This module provides functions to compute:
1. Safety metrics: Time-to-Collision (TTC)
2. Navigation metrics: Path deviation, velocity error
3. Performance metrics: Course completion time, success rate
4. System metrics: Inference latency, model complexity (FLOPs, params)
5. Distillation metrics: Knowledge retention

These metrics are used to evaluate drone performance against quantitative targets:
- TTC > 3s (safety buffer)
- Path deviation < 0.5m (route fidelity)
- Inference latency < 10ms on CPU (real-time capability)
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import torch
from torch import nn


def time_to_collision(
    position: np.ndarray, 
    velocity: np.ndarray, 
    obstacle_pos: np.ndarray, 
    obstacle_radius: float,
    safety_threshold: float = 3.0
) -> Tuple[float, bool]:
    """Compute predicted time to collision assuming constant velocity.

    Parameters
    ----------
    position : np.ndarray
        (3,) current drone position
    velocity : np.ndarray
        (3,) current velocity vector
    obstacle_pos : np.ndarray
        (3,) obstacle center position
    obstacle_radius : float
        obstacle radius
    safety_threshold : float
        Minimum safe TTC in seconds (default: 3.0s)

    Returns
    -------
    Tuple[float, bool]
        TTC in seconds (np.inf if no predicted collision) and
        safety flag (True if TTC > safety_threshold)
    """
    # Vector from drone to obstacle
    rel_pos = obstacle_pos - position
    
    # Normalize velocity
    vel_norm = np.linalg.norm(velocity)
    if vel_norm < 1e-6:  # Almost stationary
        return math.inf, True
    
    vel_unit = velocity / vel_norm
    
    # Project relative position onto velocity direction
    proj_len = np.dot(rel_pos, vel_unit)
    
    # If negative, moving away from obstacle
    if proj_len <= 0:
        return math.inf, True
    
    # Find closest approach point
    closest_point = position + proj_len * vel_unit
    
    # Distance at closest approach
    closest_dist = np.linalg.norm(closest_point - obstacle_pos)
    
    # If closest approach is outside obstacle, no collision
    if closest_dist > obstacle_radius:
        return math.inf, True
    
    # Find distance to collision point using Pythagorean theorem
    # Distance from current position to collision point
    dist_to_collision = proj_len - math.sqrt(obstacle_radius**2 - closest_dist**2)
    
    # Time to collision
    ttc = dist_to_collision / vel_norm
    
    # Return TTC and safety flag
    return max(ttc, 0.0), ttc > safety_threshold


def min_ttc_multiple_obstacles(
    position: np.ndarray, 
    velocity: np.ndarray, 
    obstacles: List[Tuple[np.ndarray, float]],
    safety_threshold: float = 3.0
) -> Tuple[float, bool, Optional[int]]:
    """Compute minimum TTC across multiple obstacles.

    Parameters
    ----------
    position : np.ndarray
        (3,) current drone position
    velocity : np.ndarray
        (3,) current velocity vector
    obstacles : List[Tuple[np.ndarray, float]]
        List of (obstacle_pos, obstacle_radius) tuples
    safety_threshold : float
        Minimum safe TTC in seconds (default: 3.0s)

    Returns
    -------
    Tuple[float, bool, Optional[int]]
        Minimum TTC, safety flag, and index of closest obstacle (None if no collision)
    """
    min_ttc = math.inf
    min_idx = None
    
    for i, (obs_pos, obs_radius) in enumerate(obstacles):
        ttc, _ = time_to_collision(position, velocity, obs_pos, obs_radius)
        if ttc < min_ttc:
            min_ttc = ttc
            min_idx = i
    
    return min_ttc, min_ttc > safety_threshold, min_idx


def path_deviation(
    traj: np.ndarray, 
    ref_traj: np.ndarray,
    target_threshold: float = 0.5
) -> Tuple[float, bool]:
    """Dynamic Time Warping distance between two 3D trajectories.
    
    Parameters
    ----------
    traj : np.ndarray
        (N, 3) trajectory to evaluate
    ref_traj : np.ndarray
        (M, 3) reference trajectory
    target_threshold : float
        Target maximum deviation in meters (default: 0.5m)
    
    Returns
    -------
    Tuple[float, bool]
        DTW distance and success flag (True if deviation < target_threshold)
    """
    try:
        from dtw import accelerated_dtw  # lazy import
    except ImportError:
        raise ImportError(
            "DTW package is required for path_deviation. "
            "Install with: pip install dtw-python"
        )
    
    # Compute DTW distance
    d, _, _, _ = accelerated_dtw(traj, ref_traj, dist="euclidean")
    
    # Normalize by path length for fair comparison across trajectories
    norm_factor = max(len(traj), len(ref_traj))
    normalized_d = d / norm_factor if norm_factor > 0 else d
    
    return float(normalized_d), normalized_d < target_threshold


def velocity_error(
    vel: np.ndarray, 
    target_vel: np.ndarray,
    relative: bool = True
) -> float:
    """Compute velocity error (absolute or relative).
    
    Parameters
    ----------
    vel : np.ndarray
        (3,) current velocity
    target_vel : np.ndarray
        (3,) target velocity
    relative : bool
        If True, return relative error (percentage)
    
    Returns
    -------
    float
        Velocity error (absolute in m/s or relative in percentage)
    """
    abs_error = np.linalg.norm(vel - target_vel)
    
    if not relative:
        return float(abs_error)
    
    target_norm = np.linalg.norm(target_vel)
    if target_norm < 1e-6:  # Avoid division by zero
        return 0.0 if abs_error < 1e-6 else 1.0
    
    return float(abs_error / target_norm)


def course_completion_time(
    timestamps: np.ndarray,
    positions: np.ndarray,
    goal_pos: np.ndarray,
    goal_radius: float = 0.5,
    timeout: Optional[float] = None
) -> Tuple[Optional[float], bool]:
    """Compute time to reach goal position.
    
    Parameters
    ----------
    timestamps : np.ndarray
        (N,) array of timestamps
    positions : np.ndarray
        (N, 3) array of positions
    goal_pos : np.ndarray
        (3,) goal position
    goal_radius : float
        Radius around goal to consider as reached
    timeout : Optional[float]
        Maximum allowed time (None for no limit)
    
    Returns
    -------
    Tuple[Optional[float], bool]
        Completion time (None if goal not reached) and success flag
    """
    for i, pos in enumerate(positions):
        dist = np.linalg.norm(pos - goal_pos)
        if dist <= goal_radius:
            completion_time = timestamps[i] - timestamps[0]
            if timeout is None or completion_time <= timeout:
                return float(completion_time), True
    
    return None, False


def success_rate_unseen(
    results: List[Dict[str, Any]],
    terrain_key: str = "terrain_id",
    success_key: str = "success"
) -> Dict[str, float]:
    """Compute success rate on known vs unseen terrains.
    
    Parameters
    ----------
    results : List[Dict[str, Any]]
        List of episode results with terrain_id and success flag
    terrain_key : str
        Key for terrain identifier
    success_key : str
        Key for success flag
    
    Returns
    -------
    Dict[str, float]
        Dictionary with overall, known, and unseen success rates
    """
    # Count unique terrains
    terrains = set(r[terrain_key] for r in results if terrain_key in r)
    
    # Split results by terrain
    terrain_results = {t: [] for t in terrains}
    for r in results:
        if terrain_key in r and r[terrain_key] in terrains:
            terrain_results[r[terrain_key]].append(r.get(success_key, False))
    
    # Compute success rates per terrain
    terrain_rates = {
        t: sum(results) / len(results) if results else 0.0
        for t, results in terrain_results.items()
    }
    
    # Compute overall success rate
    all_successes = [r.get(success_key, False) for r in results]
    overall_rate = sum(all_successes) / len(all_successes) if all_successes else 0.0
    
    # Sort terrains by success rate to identify "easy" vs "hard" terrains
    sorted_terrains = sorted(terrain_rates.items(), key=lambda x: x[1], reverse=True)
    
    # Consider top 50% as "known" and bottom 50% as "unseen"
    mid = len(sorted_terrains) // 2
    known_terrains = [t for t, _ in sorted_terrains[:mid]]
    unseen_terrains = [t for t, _ in sorted_terrains[mid:]]
    
    # Compute success rates for known vs unseen
    known_results = [
        r.get(success_key, False) for r in results 
        if terrain_key in r and r[terrain_key] in known_terrains
    ]
    unseen_results = [
        r.get(success_key, False) for r in results 
        if terrain_key in r and r[terrain_key] in unseen_terrains
    ]
    
    known_rate = sum(known_results) / len(known_results) if known_results else 0.0
    unseen_rate = sum(unseen_results) / len(unseen_results) if unseen_results else 0.0
    
    return {
        "overall": overall_rate,
        "known": known_rate,
        "unseen": unseen_rate,
        "generalization_gap": known_rate - unseen_rate
    }


def latency_ms(
    start_time: float, 
    end_time: float,
    target_ms: float = 10.0
) -> Tuple[float, bool]:
    """Return latency in milliseconds and real-time capability flag.
    
    Parameters
    ----------
    start_time : float
        Start timestamp (seconds)
    end_time : float
        End timestamp (seconds)
    target_ms : float
        Target maximum latency in milliseconds (default: 10ms)
    
    Returns
    -------
    Tuple[float, bool]
        Latency in milliseconds and real-time flag (True if latency < target_ms)
    """
    latency = (end_time - start_time) * 1e3
    return latency, latency < target_ms


def measure_inference_latency(
    model: nn.Module,
    sample_input: Union[torch.Tensor, Dict[str, torch.Tensor]],
    n_warmup: int = 10,
    n_runs: int = 100,
    device: str = "cpu"
) -> Dict[str, float]:
    """Measure inference latency of a model.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model to evaluate
    sample_input : Union[torch.Tensor, Dict[str, torch.Tensor]]
        Sample input for the model
    n_warmup : int
        Number of warmup runs
    n_runs : int
        Number of measured runs
    device : str
        Device to run inference on ("cpu" or "cuda")
    
    Returns
    -------
    Dict[str, float]
        Dictionary with mean, median, p95, and p99 latency in milliseconds
    """
    model.to(device)
    model.eval()
    
    # Move input to device
    if isinstance(sample_input, dict):
        inputs = {k: v.to(device) for k, v in sample_input.items()}
    else:
        inputs = sample_input.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(inputs)
    
    # Measure latency
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(inputs)
            # Synchronize CUDA if needed
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms
    
    return {
        "mean": np.mean(latencies),
        "median": np.median(latencies),
        "p95": np.percentile(latencies, 95),
        "p99": np.percentile(latencies, 99),
        "min": np.min(latencies),
        "max": np.max(latencies),
        "real_time_capable": np.percentile(latencies, 95) < 10.0  # p95 < 10ms
    }


def count_parameters(model: nn.Module) -> int:
    """Count number of trainable parameters in a model.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model
    
    Returns
    -------
    int
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(
    model: nn.Module,
    sample_input: Union[torch.Tensor, Dict[str, torch.Tensor]]
) -> int:
    """Estimate FLOPs for a PyTorch model.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model
    sample_input : Union[torch.Tensor, Dict[str, torch.Tensor]]
        Sample input for the model
    
    Returns
    -------
    int
        Estimated FLOPs per forward pass
    """
    try:
        from fvcore.nn import FlopCountAnalysis
        
        if isinstance(sample_input, dict):
            # For dictionary inputs (common in SB3), create a dummy forward
            # that unpacks the dict for FlopCountAnalysis
            original_forward = model.forward
            
            def new_forward(*args, **kwargs):
                if len(args) == 1 and isinstance(args[0], dict):
                    return original_forward(args[0])
                return original_forward(*args, **kwargs)
            
            model.forward = new_forward
        
        flops = FlopCountAnalysis(model, sample_input)
        return flops.total()
    except ImportError:
        print("Warning: fvcore not installed, FLOPs estimation not available")
        print("Install with: pip install fvcore")
        return -1
    finally:
        # Restore original forward if modified
        if isinstance(sample_input, dict) and 'original_forward' in locals():
            model.forward = original_forward


def distillation_accuracy_retention(
    teacher_metrics: Dict[str, float],
    student_metrics: Dict[str, float],
    key_metrics: List[str] = ["success_rate", "path_deviation", "ttc"]
) -> Dict[str, float]:
    """Calculate knowledge retention from teacher to student model.
    
    Parameters
    ----------
    teacher_metrics : Dict[str, float]
        Metrics from teacher model
    student_metrics : Dict[str, float]
        Metrics from student model
    key_metrics : List[str]
        List of metric keys to compare
    
    Returns
    -------
    Dict[str, float]
        Dictionary with retention percentages for each metric
    """
    retention = {}
    
    for key in key_metrics:
        if key not in teacher_metrics or key not in student_metrics:
            continue
            
        t_val = teacher_metrics[key]
        s_val = student_metrics[key]
        
        # Handle different metric types (higher is better vs lower is better)
        if key in ["success_rate", "ttc"]:  # Higher is better
            if t_val <= 0:  # Avoid division by zero
                retention[key] = 1.0 if s_val >= t_val else 0.0
            else:
                retention[key] = min(s_val / t_val, 1.0)  # Cap at 100%
        else:  # Lower is better (path_deviation, latency)
            if t_val <= 0:  # Avoid division by zero
                retention[key] = 1.0 if s_val <= t_val else 0.0
            else:
                # For lower-is-better metrics, retention is inverse ratio
                # (capped at 100% even if student outperforms teacher)
                retention[key] = min(t_val / s_val, 1.0) if s_val > 0 else 1.0
    
    # Overall retention (average across metrics)
    if retention:
        retention["overall"] = sum(retention.values()) / len(retention)
    
    return retention


def compute_all_metrics(
    episode_data: Dict[str, np.ndarray],
    reference_trajectory: np.ndarray,
    obstacles: List[Tuple[np.ndarray, float]],
    goal_pos: np.ndarray,
    target_thresholds: Dict[str, float] = {
        "ttc": 3.0,           # seconds
        "path_dev": 0.5,      # meters
        "completion": 60.0,   # seconds
        "latency": 10.0       # milliseconds
    }
) -> Dict[str, Any]:
    """Compute all metrics for an episode.
    
    Parameters
    ----------
    episode_data : Dict[str, np.ndarray]
        Dictionary with episode data (positions, velocities, timestamps, etc.)
    reference_trajectory : np.ndarray
        Reference trajectory for path deviation
    obstacles : List[Tuple[np.ndarray, float]]
        List of (obstacle_pos, obstacle_radius) tuples
    goal_pos : np.ndarray
        Goal position
    target_thresholds : Dict[str, float]
        Dictionary with target thresholds for each metric
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with all computed metrics
    """
    positions = episode_data["positions"]
    velocities = episode_data["velocities"]
    timestamps = episode_data["timestamps"]
    
    # Compute minimum TTC across all timesteps and obstacles
    min_ttc_value = float('inf')
    for i, (pos, vel) in enumerate(zip(positions, velocities)):
        ttc, _, _ = min_ttc_multiple_obstacles(pos, vel, obstacles)
        min_ttc_value = min(min_ttc_value, ttc)
    
    # Compute path deviation
    path_dev, path_success = path_deviation(
        positions, reference_trajectory, 
        target_threshold=target_thresholds["path_dev"]
    )
    
    # Compute course completion
    completion_time, reached_goal = course_completion_time(
        timestamps, positions, goal_pos,
        timeout=target_thresholds["completion"]
    )
    
    # Compute average velocity error if target velocities available
    vel_error = None
    if "target_velocities" in episode_data:
        vel_errors = [
            velocity_error(vel, target_vel) 
            for vel, target_vel in zip(velocities, episode_data["target_velocities"])
        ]
        vel_error = np.mean(vel_errors)
    
    # Compute latency if available
    latency = None
    latency_success = None
    if "inference_times" in episode_data:
        latencies = [
            latency_ms(start, end, target_thresholds["latency"])[0]
            for start, end in zip(
                episode_data["inference_start_times"],
                episode_data["inference_end_times"]
            )
        ]
        latency = np.mean(latencies)
        latency_success = latency < target_thresholds["latency"]
    
    # Overall success
    success = (
        min_ttc_value >= target_thresholds["ttc"] and
        path_success and
        reached_goal and
        (latency_success if latency_success is not None else True)
    )
    
    return {
        "ttc": min_ttc_value,
        "ttc_success": min_ttc_value >= target_thresholds["ttc"],
        "path_deviation": path_dev,
        "path_success": path_success,
        "completion_time": completion_time,
        "reached_goal": reached_goal,
        "velocity_error": vel_error,
        "latency": latency,
        "latency_success": latency_success,
        "success": success
    }
