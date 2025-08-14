"""Streamlit demo: fly drone with distilled student policy.

This interactive demo allows users to:
1. Load and visualize a trained drone navigation policy
2. Configure simulation parameters (obstacles, weather)
3. View real-time flight metrics (TTC, path deviation)
4. Compare different models (transformer, LSTM, PID)
5. Visualize attention patterns and state predictions
"""
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import streamlit as st
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import imageio
import io
import os
import json
import uuid
from dataclasses import dataclass, asdict

# Import local modules
try:
    import flycraft  # noqa: F401
except ImportError:
    st.error("FlyCraft gym not installed. Install with: pip install flycraft")
    st.stop()

from src.drone_rl.models.transformer_policy import TransformerActorCritic
from src.drone_rl.models.baselines import SimpleLSTMPolicy, DronePositionController
from src.drone_rl.utils.metrics import time_to_collision
from src.drone_rl.utils.model_compatibility import load_model_with_compatibility
from stable_baselines3 import PPO

# Try to import RecurrentPPO for compatibility with new LSTM models
try:
    from sb3_contrib import RecurrentPPO
    RECURRENT_PPO_AVAILABLE = True
except ImportError:
    RECURRENT_PPO_AVAILABLE = False
    RecurrentPPO = None
    # Only show warning if this is actually being run as a Streamlit app
    import sys
    if 'streamlit' in sys.modules:
        st.warning("RecurrentPPO not available. Install with: pip install sb3-contrib")

# Page configuration
st.set_page_config(
    page_title="Drone Transformer RL Demo",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling

st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
    }
    .metric-label {
        font-size: 14px;
        color: #555;
    }
    .success {
        color: #28a745;
    }
    .warning {
        color: #ffc107;
    }
    .danger {
        color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# --- Saved runs utilities ---
SAVE_DIR = Path("demo/saved_runs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class RunRecord:
    id: str
    timestamp: float
    model_type: str
    model_path: str
    config: dict
    metrics: dict
    positions: list
    velocities: list
    rewards: list
    ttc: list
    reference_trajectory: list | None
    video_path: str | None

def save_run(record: RunRecord) -> str:
    run_dir = SAVE_DIR / record.id
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / "run.json"
    with open(json_path, "w") as f:
        json.dump(asdict(record), f)
    return str(json_path)

def load_run(json_path: str) -> RunRecord:
    with open(json_path) as f:
        data = json.load(f)
    return RunRecord(**data)

# Cache model loading to avoid reloading on every interaction
@st.cache_resource
def load_model(model_path: str, model_type: str = "transformer"):
    """Load a trained model from checkpoint.
    
    Parameters
    ----------
    model_path : str
        Path to model checkpoint
    model_type : str
        Type of model (transformer, lstm, pid)
        
    Returns
    -------
    model
        Loaded model
    env
        Environment instance
    """
    try:
        # Create environment
        env = gym.make("FlyCraft", max_episode_steps=1000)
        
        # For PID controller, return a custom controller
        if model_type == "pid":
            controller = DronePositionController()
            return controller, env
        
        # Try to detect if this is a RecurrentPPO model by checking the path/filename
        is_recurrent = "recurrent" in model_path.lower() or "lstm_recurrent" in model_path.lower()
        
        # For transformer or LSTM, load from checkpoint
        if model_type == "transformer":
            model = load_model_with_compatibility(model_path, env, TransformerActorCritic, device="cpu")
        elif model_type == "lstm":
            if is_recurrent and RECURRENT_PPO_AVAILABLE:
                # Try to load as RecurrentPPO first
                try:
                    model = RecurrentPPO.load(model_path, env=env, device="cpu")
                    # Store that this is a recurrent model for later use
                    model._is_recurrent = True
                    return model, env
                except Exception as e:
                    st.warning(f"Failed to load as RecurrentPPO: {e}. Trying regular PPO...")
            
            # Fallback to regular PPO with SimpleLSTMPolicy
            model = load_model_with_compatibility(model_path, env, SimpleLSTMPolicy, device="cpu")
            model._is_recurrent = False
        else:
            custom_objects = {}
            model = PPO.load(model_path, env=env, device="cpu", custom_objects=custom_objects)
            model._is_recurrent = False
            
        return model, env
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def create_trajectory_plot(positions: np.ndarray, reference: Optional[np.ndarray] = None) -> Figure:
    """Create 3D trajectory visualization.
    
    Parameters
    ----------
    positions : np.ndarray
        Array of drone positions [N, 3]
    reference : Optional[np.ndarray]
        Optional reference trajectory [M, 3]
        
    Returns
    -------
    Figure
        Plotly figure with trajectory
    """
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scatter3d"}]])
    
    # Add drone trajectory (lines only, no markers)
    fig.add_trace(
        go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode="lines",
            name="Drone Path",
            line=dict(color="blue", width=4),
        )
    )
    
    # Add reference trajectory if provided
    if reference is not None:
        fig.add_trace(
            go.Scatter3d(
                x=reference[:, 0],
                y=reference[:, 1],
                z=reference[:, 2],
                mode="lines",
                name="Reference Path",
                line=dict(color="green", width=2, dash="dash"),
            )
        )
    
    # Add start and end points
    fig.add_trace(
        go.Scatter3d(
            x=[positions[0, 0]],
            y=[positions[0, 1]],
            z=[positions[0, 2]],
            mode="markers",
            name="Start",
            marker=dict(size=8, color="green"),
        )
    )
    
    fig.add_trace(
        go.Scatter3d(
            x=[positions[-1, 0]],
            y=[positions[-1, 1]],
            z=[positions[-1, 2]],
            mode="markers",
            name="End",
            marker=dict(size=8, color="red"),
        )
    )
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            zaxis_title="Z Position (m)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(x=0, y=1),
        height=500,
    )
    
    return fig

def visualize_attention(attention_weights: np.ndarray, step: int) -> Figure:
    """Visualize attention weights.
    
    Parameters
    ----------
    attention_weights : np.ndarray
        Attention weights [heads, seq_len, seq_len]
    step : int
        Current step to highlight
        
    Returns
    -------
    Figure
        Matplotlib figure with attention visualization
    """
    n_heads = attention_weights.shape[0]
    fig, axes = plt.subplots(1, n_heads, figsize=(n_heads * 3, 3))
    
    if n_heads == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        im = ax.imshow(attention_weights[i], cmap="viridis")
        ax.set_title(f"Head {i+1}")
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Highlight current timestep
        if step < attention_weights.shape[1]:
            ax.axvline(x=step, color="red", linestyle="--", alpha=0.7)
            ax.axhline(y=step, color="red", linestyle="--", alpha=0.7)
    
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.tight_layout()
    
    return fig

def display_metrics(metrics: Dict[str, float]) -> None:
    """Display metrics in styled cards.
    
    Parameters
    ----------
    metrics : Dict[str, float]
        Dictionary of metric names and values
    """
    cols = st.columns(len(metrics))
    
    for i, (name, value) in enumerate(metrics.items()):
        with cols[i]:
            # Determine status color based on metric thresholds
            status = "success"
            if name == "TTC (s)" and value < 3.0:
                status = "danger"
            elif name == "TTC (s)" and value < 10.0:
                status = "warning"

            val_str = "∞" if (isinstance(value, float) and not np.isfinite(value)) else f"{value:.2f}"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value {status}">{val_str}</div>
                <div class="metric-label">{name}</div>
            </div>
            """, unsafe_allow_html=True)

def display_metrics_explanation():
    """Display detailed explanations of flight metrics."""
    st.subheader("📊 Understanding Flight Metrics")
    
    with st.expander("Metric Explanations", expanded=False):
        st.markdown("""
        ### Time-to-Collision (TTC)
        **What it measures:** The time in seconds until the drone would collide with the nearest obstacle if it continues on its current trajectory.
        
        - **Good values:** > 3 seconds (safe flight)
        - **Warning values:** 1-3 seconds (caution needed)
        - **Dangerous values:** < 1 second (immediate collision risk)
        - **∞ (Infinity):** No collision detected on current path
        
        *This is a critical safety metric - the drone must maintain sufficient TTC to react to obstacles.*
        
        ### Average Reward
        **What it measures:** The mean reward signal from the reinforcement learning environment, indicating overall mission performance.
        
        - **High positive:** Mission objectives being met
        - **Near zero:** Mediocre performance
        - **Negative:** Poor performance, missing objectives
        
        *Higher rewards indicate better overall flight performance and objective completion.*
        
        ### Success Rate
        **What it measures:** Whether the drone successfully completed its navigation mission without crashing or failing.
        
        - ✅ **Success:** Mission completed safely
        - ❌ **Failure:** Crashed, exceeded time limit, or failed to reach target
        
        *This is the ultimate measure of mission effectiveness.*
        """)
        
        st.markdown("""
        ---
        ### Flight Patterns Analysis
        
        The **Action Analysis** section shows:
        - **Actions Over Time:** How the drone's control inputs (thrust, pitch, yaw) change during flight
        - **Action Statistics:** Which movement axes the drone uses most/least
        - **Movement Efficiency:** How direct the drone's path is compared to optimal
        
        The **3D Trajectory** visualization shows:
        - 🔵 **Blue line:** Actual drone flight path
        - 🟢 **Green dashed line:** Optimal reference trajectory (if available)
        - 🟢 **Green dot:** Mission start point
        - 🔴 **Red dot:** Mission end point
        """)

def run_simulation(model, env, config: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
    """Run simulation with given model and configuration.
    
    Parameters
    ----------
    model : Any
        Model or controller
    env : gym.Env
        Environment instance
    config : Dict[str, Any]
        Simulation configuration
        
    Returns
    -------
    Dict[str, Any]
        Simulation results
    """
    # Reset environment (optional seeding)
    try:
        obs, info = env.reset(seed=seed)
    except TypeError:
        # Older gym versions may not support seed kwarg here
        if seed is not None and hasattr(env, "seed"):
            try:
                env.seed(seed)
            except Exception:
                pass
        obs, info = env.reset()

    # Initialize hidden states for RecurrentPPO models
    if hasattr(model, "_is_recurrent") and model._is_recurrent and RECURRENT_PPO_AVAILABLE:
        # For RecurrentPPO, we need to track hidden states
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
    else:
        lstm_states = None
        episode_starts = None

    # Seed logs with initial state (so Start and Path render correctly)
    init_pos = info.get("drone_position", np.zeros(3))
    init_vel = info.get("drone_velocity", np.zeros(3))
    
    # Debug: Check what's in the initial info and obs
    print(f"DEBUG: Initial info keys: {list(info.keys())}")
    print(f"DEBUG: Initial obs type: {type(obs)}")
    if isinstance(obs, dict):
        print(f"DEBUG: Initial obs keys: {list(obs.keys())}")
    print(f"DEBUG: Initial position from info: {init_pos}")
    
    # Debug action space bounds
    print(f"DEBUG: Action space: {env.action_space}")
    if hasattr(env.action_space, 'low') and hasattr(env.action_space, 'high'):
        print(f"DEBUG: Action space low: {env.action_space.low}")
        print(f"DEBUG: Action space high: {env.action_space.high}")
        print(f"DEBUG: Action space shape: {env.action_space.shape}")
    
    # Try to extract position from plane_state in info
    if np.allclose(init_pos, 0) and "plane_state" in info:
        plane_state = info["plane_state"]
        print(f"DEBUG: plane_state type: {type(plane_state)}")
        if hasattr(plane_state, 'shape'):
            print(f"DEBUG: plane_state shape: {plane_state.shape}")
        elif hasattr(plane_state, '__len__'):
            print(f"DEBUG: plane_state length: {len(plane_state)}")
        
        # Handle dictionary plane_state
        if isinstance(plane_state, dict):
            print(f"DEBUG: plane_state keys: {list(plane_state.keys())}")
            # Print first few key-value pairs to understand the structure
            for i, (key, value) in enumerate(plane_state.items()):
                if i < 10:  # Show first 10 key-value pairs
                    print(f"DEBUG: {key}: {value} (type: {type(value).__name__})")
            
            # Look for common position keys in FlyCraft
            position_keys = ['x', 'y', 'z', 'pos_x', 'pos_y', 'pos_z', 'position', 'lat', 'lon', 'alt', 'north', 'east', 'down']
            pos_values = []
            for key in position_keys:
                if key in plane_state:
                    pos_values.append(plane_state[key])
                    print(f"DEBUG: Found {key}: {plane_state[key]}")
            
            if len(pos_values) >= 3:
                init_pos = np.array(pos_values[:3])
                print(f"DEBUG: Using position from plane_state dict: {init_pos}")
            else:
                # Try to extract from first 3 numeric values if they exist
                values = list(plane_state.values())
                numeric_values = []
                for val in values:
                    try:
                        numeric_values.append(float(val))
                    except (ValueError, TypeError):
                        pass
                
                if len(numeric_values) >= 3:
                    init_pos = np.array(numeric_values[:3])
                    print(f"DEBUG: Using first 3 numeric values from plane_state: {init_pos}")
                else:
                    print(f"DEBUG: Could not extract position from plane_state dict")
        
        elif hasattr(plane_state, 'shape') and len(plane_state) >= 3:
            # Position is typically the first 3 elements in plane state
            init_pos = plane_state[:3]
            print(f"DEBUG: Using position from plane_state: {init_pos}")
        elif hasattr(plane_state, '__len__') and len(plane_state) >= 3:
            # Handle case where plane_state is a list/tuple
            init_pos = np.array(plane_state[:3])
            print(f"DEBUG: Using position from plane_state (list): {init_pos}")
    
    # Try to extract from observation if available
    elif np.allclose(init_pos, 0) and isinstance(obs, dict) and "plane_state" in obs:
        plane_state = obs["plane_state"]
        if hasattr(plane_state, 'shape') and len(plane_state) >= 3:
            init_pos = plane_state[:3]
            print(f"DEBUG: Using position from obs plane_state: {init_pos}")
        elif hasattr(plane_state, '__len__') and len(plane_state) >= 3:
            init_pos = np.array(plane_state[:3])
            print(f"DEBUG: Using position from obs plane_state (list): {init_pos}")

    # Prepare result storage
    results = {
        "frames": [],
        "positions": [],
        "velocities": [],
        "actions": [],
        "rewards": [],
        "ttc": [],
        "attention": [] if hasattr(model, "policy") and hasattr(model.policy, "get_attention_weights") else None,
    }

    # Log initial state as step 0
    results["positions"].append(init_pos)
    results["velocities"].append(init_vel)
    # Placeholder action and reward at t=0
    try:
        zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
    except Exception:
        zero_action = 0.0
    results["actions"].append(zero_action)
    results["rewards"].append(0.0)

    # Initial metrics
    if "obstacles" in info and len(info["obstacles"]) > 0:
        ttc0, _ = time_to_collision(
            init_pos,
            init_vel,
            info["obstacles"][0][0],
            info["obstacles"][0][1],
        )
        results["ttc"].append(ttc0)
    else:
        results["ttc"].append(float("inf"))
    
    # Run simulation
    done = False
    total_reward = 0.0
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    max_steps = config.get("max_steps", 500)
    for step in range(max_steps):
        # Get action from model
        if isinstance(model, DronePositionController):
            # For PID controller
            target_pos = info.get("target_position", np.zeros(3))
            current_pos = info.get("drone_position", np.zeros(3))
            current_vel = info.get("drone_velocity", np.zeros(3))
            current_time = step * 0.05  # Assuming 20Hz simulation
            
            action = model(
                target_pos, current_pos, current_vel, current_time,
                target_yaw=info.get("target_yaw", 0.0),
                current_yaw=info.get("drone_yaw", 0.0)
            )
        else:
            # For RL policies
            if hasattr(model, "_is_recurrent") and model._is_recurrent and RECURRENT_PPO_AVAILABLE:
                # For RecurrentPPO, maintain hidden states
                action, lstm_states = model.predict(
                    obs, state=lstm_states, episode_start=episode_starts, deterministic=True
                )
                episode_starts = np.array([False])  # Only first step is episode start
            else:
                # For regular PPO
                action, _ = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Update results
        if "rgb_array" in info:
            results["frames"].append(info["rgb_array"])
        
        # Debug: Print info keys to see what's available
        if step < 3:  # Only print for first few steps to avoid spam
            print(f"DEBUG Step {step}: info keys: {list(info.keys())}")
            if "drone_position" in info:
                print(f"DEBUG Step {step}: drone_position: {info['drone_position']}")
            else:
                print(f"DEBUG Step {step}: drone_position not found in info")
                if isinstance(obs, dict):
                    print(f"DEBUG Step {step}: obs keys: {list(obs.keys())}")
        
        drone_pos = info.get("drone_position", None)
        
        # If drone_position not in info, try to extract from plane_state
        if drone_pos is None:
            # Check info first
            if "plane_state" in info:
                plane_state = info["plane_state"]
                
                # Handle dictionary plane_state
                if isinstance(plane_state, dict):
                    # Look for common position keys in FlyCraft
                    position_keys = ['x', 'y', 'z', 'pos_x', 'pos_y', 'pos_z', 'position']
                    pos_values = []
                    for key in position_keys:
                        if key in plane_state:
                            pos_values.append(plane_state[key])
                    
                    if len(pos_values) >= 3:
                        drone_pos = np.array(pos_values[:3])
                        if step < 3:
                            print(f"DEBUG Step {step}: Extracted position from info plane_state dict: {drone_pos}")
                    else:
                        # Try to extract from first 3 values if they exist
                        values = list(plane_state.values())
                        if len(values) >= 3:
                            try:
                                drone_pos = np.array([float(v) for v in values[:3]])
                                if step < 3:
                                    print(f"DEBUG Step {step}: Using first 3 values from info plane_state: {drone_pos}")
                            except (ValueError, TypeError):
                                if step < 3:
                                    print(f"DEBUG Step {step}: Could not convert first 3 values to float")
                
                elif hasattr(plane_state, 'shape') and len(plane_state) >= 3:
                    drone_pos = plane_state[:3]
                    if step < 3:
                        print(f"DEBUG Step {step}: Extracted position from info plane_state: {drone_pos}")
                elif hasattr(plane_state, '__len__') and len(plane_state) >= 3:
                    # Handle case where plane_state is a list/tuple
                    drone_pos = np.array(plane_state[:3])
                    if step < 3:
                        print(f"DEBUG Step {step}: Extracted position from info plane_state (list): {drone_pos}")
            
            # Check obs if still None
            elif isinstance(obs, dict) and "plane_state" in obs:
                plane_state = obs["plane_state"]
                
                # Handle dictionary plane_state
                if isinstance(plane_state, dict):
                    position_keys = ['x', 'y', 'z', 'pos_x', 'pos_y', 'pos_z', 'position']
                    pos_values = []
                    for key in position_keys:
                        if key in plane_state:
                            pos_values.append(plane_state[key])
                    
                    if len(pos_values) >= 3:
                        drone_pos = np.array(pos_values[:3])
                        if step < 3:
                            print(f"DEBUG Step {step}: Extracted position from obs plane_state dict: {drone_pos}")
                
                elif hasattr(plane_state, 'shape') and len(plane_state) >= 3:
                    drone_pos = plane_state[:3]
                    if step < 3:
                        print(f"DEBUG Step {step}: Extracted position from obs plane_state: {drone_pos}")
                elif hasattr(plane_state, '__len__') and len(plane_state) >= 3:
                    drone_pos = np.array(plane_state[:3])
                    if step < 3:
                        print(f"DEBUG Step {step}: Extracted position from obs plane_state (list): {drone_pos}")
            
            # Fallback
            if drone_pos is None:
                drone_pos = np.zeros(3)
                if step < 3:
                    print(f"DEBUG Step {step}: Using fallback zeros for position")
        
        results["positions"].append(drone_pos)
        results["velocities"].append(info.get("drone_velocity", np.zeros(3)))
        results["actions"].append(action)
        results["rewards"].append(reward)
        
        # Debug action patterns (track movement tendencies)
        if step < 10 or step % 50 == 0:  # Log periodically
            if hasattr(action, 'shape') and len(action) >= 3:
                print(f"DEBUG Step {step}: Action: {action[:3]} (type: {type(action).__name__})")
            else:
                print(f"DEBUG Step {step}: Action: {action} (type: {type(action).__name__})")
        
        # Track target information for analysis
        if "desired_goal" in info:
            desired_goal = info["desired_goal"]
            if step == 0:
                print(f"DEBUG: Desired goal: {desired_goal}")
                if hasattr(desired_goal, '__len__') and len(desired_goal) >= 3:
                    goal_diff = np.array(desired_goal[:3]) if init_pos is not None else None
                    if goal_diff is not None and not np.allclose(init_pos, 0):
                        movement_required = goal_diff - init_pos
                        print(f"DEBUG: Required movement [x,y,z]: {movement_required}")
                        print(f"DEBUG: Largest movement dimension: {np.argmax(np.abs(movement_required))} ({'x' if np.argmax(np.abs(movement_required))==0 else 'y' if np.argmax(np.abs(movement_required))==1 else 'z'})")
        
        # Compute TTC metric
        if "obstacles" in info and len(info["obstacles"]) > 0:
            ttc_val, _ = time_to_collision(
                info["drone_position"],
                info["drone_velocity"],
                info["obstacles"][0][0],  # First obstacle position
                info["obstacles"][0][1],  # First obstacle radius
            )
            results["ttc"].append(ttc_val)
        else:
            results["ttc"].append(float("inf"))
        
        # Capture attention weights if available
        if results["attention"] is not None and hasattr(model.policy, "get_attention_weights"):
            try:
                attn = model.policy.get_attention_weights(obs)
                results["attention"].append(attn)
            except:
                # If attention extraction fails, disable it
                results["attention"] = None
        
        # Update progress
        progress = (step + 1) / max_steps
        progress_bar.progress(progress)
        status_text.text(f"Step {step+1}/{max_steps} | Reward: {reward:.2f} | Total: {total_reward:.2f}")
        
        total_reward += reward
        done = terminated or truncated
        if done:
            break
    
    # Convert lists to arrays
    results["positions"] = np.array(results["positions"])
    results["velocities"] = np.array(results["velocities"])
    results["actions"] = np.array(results["actions"])
    results["rewards"] = np.array(results["rewards"])
    results["ttc"] = np.array(results["ttc"])
    
    # Debug position data
    print(f"DEBUG: Final positions shape: {results['positions'].shape}")
    print(f"DEBUG: Position range X: [{np.min(results['positions'][:, 0]):.3f}, {np.max(results['positions'][:, 0]):.3f}]")
    print(f"DEBUG: Position range Y: [{np.min(results['positions'][:, 1]):.3f}, {np.max(results['positions'][:, 1]):.3f}]")  
    print(f"DEBUG: Position range Z: [{np.min(results['positions'][:, 2]):.3f}, {np.max(results['positions'][:, 2]):.3f}]")
    print(f"DEBUG: Total movement: {np.linalg.norm(results['positions'][-1] - results['positions'][0]):.3f}")
    
    # Add summary metrics
    results["total_reward"] = total_reward
    # Try both possible success keys from environment
    results["success"] = info.get("success", info.get("is_success", False))
    results["steps"] = step + 1
    results["reference_trajectory"] = None  # Not available in FlyCraft
    
    # Analyze action patterns to understand model behavior
    actions_array = np.array(results["actions"])
    if len(actions_array.shape) > 1 and actions_array.shape[1] >= 3:
        action_means = np.mean(actions_array, axis=0)
        action_stds = np.std(actions_array, axis=0)
        print(f"DEBUG: Action means: {action_means[:3]} (x, y, z movement tendencies)")
        print(f"DEBUG: Action stds: {action_stds[:3]} (movement variability)")
        
        # Check if model is avoiding certain directions
        if len(action_means) >= 3:
            dominant_axis = np.argmax(np.abs(action_means[:3]))
            weakest_axis = np.argmin(np.abs(action_means[:3]))
            print(f"DEBUG: Dominant movement axis: {dominant_axis} ({'x' if dominant_axis==0 else 'y' if dominant_axis==1 else 'z'})")
            print(f"DEBUG: Weakest movement axis: {weakest_axis} ({'x' if weakest_axis==0 else 'y' if weakest_axis==1 else 'z'})")
    
    # Debug success detection
    print(f"DEBUG: Final info keys: {list(info.keys())}")
    print(f"DEBUG: Success status: {results['success']}")
    if "is_success" in info:
        print(f"DEBUG: is_success value: {info['is_success']}")
    
    # Clear progress bar
    progress_bar.empty()
    status_text.empty()
    
    return results

# Main application
def main():
    """Main Streamlit application."""
    rec = None  # Ensure rec is always defined

    st.title("🚁 Drone Transformer RL – Live Demo")
    st.markdown("""
    This demo showcases our transformer-based reinforcement learning approach for autonomous drone navigation.
    The model learns to navigate through complex environments, avoiding obstacles while following a reference path.
    """)
    
    # Sidebar configuration
    st.sidebar.title("Configuration")

    # Model selection
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["transformer", "lstm", "pid"],
        index=0,
        help="Select model architecture"
    )

    # Run / Replay mode and auto-retry settings
    mode = st.sidebar.radio("Mode", ["Live", "Replay"], index=0)
    auto_retry = st.sidebar.checkbox("Auto-retry until success", value=True)
    max_attempts = st.sidebar.number_input("Max attempts", min_value=1, max_value=1000, value=20, step=1)
    seed_base = st.sidebar.number_input("Seed base", min_value=0, value=123, step=1)
    vary_seed = st.sidebar.checkbox("Vary seed per attempt", value=True)

    # Saved runs
    saved_jsons = sorted(SAVE_DIR.glob("*/run.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    saved_choices = [str(p) for p in saved_jsons]
    selected_saved = st.sidebar.selectbox("Saved runs", saved_choices) if saved_choices else None
    
    # Show replay buttons regardless of mode for easier access
    replay_btn = st.sidebar.button("Replay selected run")
    replay_last_btn = st.sidebar.button("Replay last success")
    
    # Show status of last success
    if st.session_state.get("last_success_path"):
        last_success_name = Path(st.session_state.last_success_path).parent.name
        st.sidebar.success(f"Last success: {last_success_name}")
    else:
        st.sidebar.info("No successful runs yet")
    
    # Debug info (can be hidden in production)
    with st.sidebar.expander("Debug Info"):
        st.write(f"Saved runs found: {len(saved_choices)}")
        st.write(f"Last success path: {st.session_state.get('last_success_path', 'None')}")
        st.write(f"Save directory: {SAVE_DIR}")
        if st.button("Clear last success"):
            if "last_success_path" in st.session_state:
                del st.session_state.last_success_path
            st.rerun()

    # Model checkpoint
    if model_type != "pid":
        default_ckpt = (
            "runs/student_distilled/final_model.zip" if model_type == "transformer"
            else ("runs/baseline_lstm_quick/final_model.zip" if model_type == "lstm"
                  else "runs/baseline_lstm/final_model.zip")
        )
        checkpoint = st.sidebar.text_input(
            "Model Checkpoint",
            default_ckpt,
            help="Path to model checkpoint"
        )
    else:
        checkpoint = None

    max_steps = st.sidebar.slider(
        "Max Steps",
        100, 1000, 500,
        help="Maximum simulation steps"
    )

    # Create configuration dictionary
    config = {
        "max_steps": max_steps,
    }

    # Load model button
    load_btn = st.sidebar.button("Load Model")

    if load_btn:
        # Normalize extension for non-PID models
        ckpt_path = checkpoint
        if model_type != "pid" and ckpt_path is not None:
            low = ckpt_path.lower()
            if low.endswith(".zip.zip"):
                ckpt_path = ckpt_path[:-4]
            elif not low.endswith(".zip"):
                ckpt_path = ckpt_path + ".zip"
        with st.spinner(f"Loading {model_type.upper()} model..."):
            model, env = load_model(ckpt_path, model_type)

        if model is not None and env is not None:
            st.session_state.model = model
            st.session_state.env = env
            st.success(f"{model_type.capitalize()} model loaded successfully!")

            # Display model info
            if model_type != "pid":
                st.subheader("Model Information")

                # Get parameter count
                if hasattr(model, "policy"):
                    param_count = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
                    st.info(f"Trainable parameters: {param_count:,}")

                    # Display architecture details
                    if model_type == "transformer":
                        if hasattr(model.policy, "features_extractor"):
                            extractor = model.policy.features_extractor
                            if hasattr(extractor, "transformer"):
                                transformer = extractor.transformer
                                st.info(f"""
                                Transformer architecture:
                                - Embedding dimension: {extractor.embed_dim}
                                - Attention heads: {transformer.global_layers[0].num_heads if hasattr(transformer, 'global_layers') else 'N/A'}
                                - Layers: {len(transformer.global_layers) + len(transformer.local_layers) if hasattr(transformer, 'global_layers') else 'N/A'}
                                - Memory: {'Yes' if extractor.use_memory else 'No'}
                                """)

    # Run simulation button
    run_btn = st.sidebar.button("Run Simulation")

    # Main content
    if "model" in st.session_state and "env" in st.session_state:
        model = st.session_state.model
        env = st.session_state.env

        if mode == "Live" and run_btn:
            attempts = 0
            last_results = None
            while True:
                seed = (seed_base + attempts) if vary_seed else seed_base
                with st.spinner(f"Running simulation (attempt {attempts+1})..."):
                    results = run_simulation(model, env, config, seed=seed)
                last_results = results

                # Show quick summary per attempt
                st.write(f"Attempt {attempts+1}: reward {results['total_reward']:.2f}, success={results['success']}")

                if results["success"]:
                    # Save successful run
                    rec = RunRecord(
                        id=str(uuid.uuid4()),
                        timestamp=time.time(),
                        model_type=model_type,
                        model_path=checkpoint or "",
                        config=config,
                        metrics={
                            "total_reward": float(results["total_reward"]),
                            "steps": int(results["steps"]),
                            "success": bool(results["success"]),
                            "mean_ttc": float(np.nanmean(results["ttc"])) if np.isfinite(results["ttc"]).any() else float("inf"),
                        },
                        positions=results["positions"].tolist(),
                        velocities=results["velocities"].tolist(),
                        rewards=results["rewards"].tolist(),
                        ttc=results["ttc"].tolist(),
                        reference_trajectory=(results["reference_trajectory"].tolist() if results["reference_trajectory"] is not None else None),
                        video_path=None,
                    )
                    json_path = save_run(rec)
                    
                    try:
                        # Save MP4 if frames are available (optional)
                        if results.get("frames") and len(results["frames"]) > 0:
                            run_dir = Path(json_path).parent
                            mp4_path = run_dir / "video.mp4"
                            imageio.mimsave(str(mp4_path), results["frames"], format="mp4", fps=20)
                            rec.video_path = str(mp4_path)
                            # Update the saved record with video path
                            with open(json_path, "w") as f:
                                json.dump(asdict(rec), f)
                            save_message = f"✅ Saved successful run with video → {Path(json_path).parent.name}"
                        else:
                            save_message = f"✅ Saved successful run (trajectory data) → {Path(json_path).parent.name}"
                        
                        st.session_state.last_success_path = json_path
                        st.success(save_message)
                        
                        # Trigger rerun to update sidebar status
                        st.rerun()
                        
                    except Exception as save_error:
                        st.error(f"Failed to save run details: {save_error}")
                    
                    break

                attempts += 1
                if not auto_retry or attempts >= int(max_attempts):
                    break

            # Display the last attempt (success or not)
            if last_results is not None:
                st.subheader("Simulation Results")
                if last_results["success"]:
                    st.success(f"Mission successful! Completed in {last_results['steps']} steps with reward {last_results['total_reward']:.2f}")
                else:
                    st.error(f"Mission failed. Completed {last_results['steps']} steps with reward {last_results['total_reward']:.2f}")

                st.subheader("Flight Metrics")
                metrics = {
                    "TTC (s)": np.mean(last_results["ttc"]) if np.isfinite(last_results["ttc"]).any() else float("inf"),
                    "Avg Reward": np.mean(last_results["rewards"]),
                }
                display_metrics(metrics)
                
                # Add metrics explanation
                display_metrics_explanation()

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("3D Trajectory")
                    traj_fig = create_trajectory_plot(last_results["positions"], last_results["reference_trajectory"])
                    st.plotly_chart(traj_fig, use_container_width=True, key="training_trajectory")
                with col2:
                    st.subheader("Metrics Over Time")
                    metrics_df = pd.DataFrame({
                        "Step": np.arange(len(last_results["ttc"])),
                        "TTC (s)": np.clip(last_results["ttc"], 0, 10),
                        "Reward": last_results["rewards"],
                    })
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=metrics_df["Step"], y=metrics_df["TTC (s)"], mode="lines", name="TTC (s)"))
                    fig.add_trace(go.Scatter(x=metrics_df["Step"], y=metrics_df["Reward"], mode="lines", name="Reward"))
                    fig.update_layout(xaxis_title="Simulation Step", yaxis_title="Value", legend=dict(x=0, y=1, orientation="h"), margin=dict(l=0, r=0, b=0, t=30), height=400)
                    st.plotly_chart(fig, use_container_width=True, key="training_metrics")

                # Add action analysis plots
                st.subheader("Action Analysis")
                if len(last_results["actions"]) > 0:
                    actions_array = np.array(last_results["actions"])
                    
                    if len(actions_array.shape) > 1 and actions_array.shape[1] >= 3:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Action patterns over time
                            st.write("**Actions Over Time**")
                            action_fig = go.Figure()
                            steps = np.arange(len(actions_array))
                            
                            action_labels = ['X-axis', 'Y-axis', 'Z-axis']
                            colors = ['red', 'green', 'blue']
                            
                            for i in range(min(3, actions_array.shape[1])):
                                action_fig.add_trace(go.Scatter(
                                    x=steps, 
                                    y=actions_array[:, i],
                                    mode="lines", 
                                    name=f"Action {action_labels[i]}",
                                    line=dict(color=colors[i])
                                ))
                            
                            action_fig.update_layout(
                                xaxis_title="Simulation Step",
                                yaxis_title="Action Value",
                                legend=dict(x=0, y=1, orientation="h"),
                                margin=dict(l=0, r=0, b=0, t=30),
                                height=350
                            )
                            st.plotly_chart(action_fig, use_container_width=True, key="action_patterns")
                        
                        with col2:
                            # Action statistics
                            st.write("**Action Statistics**")
                            action_means = np.mean(actions_array[:, :3], axis=0)
                            action_stds = np.std(actions_array[:, :3], axis=0)
                            action_ranges = np.max(actions_array[:, :3], axis=0) - np.min(actions_array[:, :3], axis=0)
                            
                            stats_fig = go.Figure()
                            
                            x_labels = ['X-axis', 'Y-axis', 'Z-axis']
                            
                            stats_fig.add_trace(go.Bar(
                                x=x_labels,
                                y=np.abs(action_means),
                                name='Mean (abs)',
                                marker_color='lightblue'
                            ))
                            
                            stats_fig.add_trace(go.Bar(
                                x=x_labels,
                                y=action_stds,
                                name='Std Dev',
                                marker_color='orange'
                            ))
                            
                            stats_fig.add_trace(go.Bar(
                                x=x_labels,
                                y=action_ranges,
                                name='Range',
                                marker_color='lightgreen'
                            ))
                            
                            stats_fig.update_layout(
                                xaxis_title="Movement Axis",
                                yaxis_title="Action Magnitude",
                                barmode='group',
                                legend=dict(x=0, y=1, orientation="h"),
                                margin=dict(l=0, r=0, b=0, t=30),
                                height=350
                            )
                            st.plotly_chart(stats_fig, use_container_width=True, key="action_stats")
                        
                        # Movement analysis
                        if len(last_results["positions"]) > 1:
                            positions_array = np.array(last_results["positions"])
                            if positions_array.shape[1] >= 3:
                                st.subheader("Movement Analysis")
                                
                                # Calculate actual movement vs required movement
                                start_pos = positions_array[0]
                                end_pos = positions_array[-1]
                                actual_movement = end_pos - start_pos
                                
                                # Debug movement data
                                print(f"DEBUG: Start position: {start_pos}")
                                print(f"DEBUG: End position: {end_pos}")
                                print(f"DEBUG: Actual movement: {actual_movement}")
                                
                                # Get target if available
                                target_movement = None
                                if last_results.get("reference_trajectory") is not None:
                                    ref_traj = last_results["reference_trajectory"]
                                    if len(ref_traj) > 0:
                                        target_pos = ref_traj[-1]
                                        target_movement = target_pos - start_pos
                                        print(f"DEBUG: Target position: {target_pos}")
                                        print(f"DEBUG: Required movement: {target_movement}")
                                else:
                                    print(f"DEBUG: No reference trajectory available")
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Movement comparison
                                    movement_fig = go.Figure()
                                    
                                    if target_movement is not None:
                                        movement_fig.add_trace(go.Bar(
                                            x=['X', 'Y', 'Z'],
                                            y=target_movement,
                                            name='Required Movement',
                                            marker_color='red',
                                            opacity=0.7
                                        ))
                                    
                                    movement_fig.add_trace(go.Bar(
                                        x=['X', 'Y', 'Z'],
                                        y=actual_movement,
                                        name='Actual Movement',
                                        marker_color='blue',
                                        opacity=0.7
                                    ))
                                    
                                    movement_fig.update_layout(
                                        title="Required vs Actual Movement",
                                        xaxis_title="Axis",
                                        yaxis_title="Distance",
                                        barmode='group',
                                        height=300
                                    )
                                    st.plotly_chart(movement_fig, use_container_width=True, key="movement_comparison")
                                
                                with col2:
                                    # Movement efficiency metrics
                                    total_distance = np.sum(np.linalg.norm(np.diff(positions_array, axis=0), axis=1))
                                    direct_distance = np.linalg.norm(actual_movement)
                                    efficiency = direct_distance / total_distance if total_distance > 0 else 0
                                    
                                    st.metric("Movement Efficiency", f"{efficiency:.3f}")
                                    st.metric("Total Distance", f"{total_distance:.2f}")
                                    st.metric("Direct Distance", f"{direct_distance:.2f}")
                                    
                                    # Show which axis had largest movement requirement vs achievement
                                    if target_movement is not None:
                                        required_magnitudes = np.abs(target_movement)
                                        actual_magnitudes = np.abs(actual_movement)
                                        
                                        axis_names = ['X', 'Y', 'Z']
                                        largest_required = np.argmax(required_magnitudes)
                                        largest_actual = np.argmax(actual_magnitudes)
                                        
                                        st.write("**Movement Analysis:**")
                                        st.write(f"• Largest required: {axis_names[largest_required]}-axis ({required_magnitudes[largest_required]:.2f})")
                                        st.write(f"• Largest achieved: {axis_names[largest_actual]}-axis ({actual_magnitudes[largest_actual]:.2f})")
                                        
                                        # Check if the model struggled with a particular axis
                                        achievement_ratios = actual_magnitudes / (required_magnitudes + 1e-6)
                                        worst_axis = np.argmin(achievement_ratios)
                                        st.write(f"• Weakest performance: {axis_names[worst_axis]}-axis ({achievement_ratios[worst_axis]:.2f} ratio)")

                if last_results.get("frames"):
                    st.subheader("Flight Video")
                    video_bytes = io.BytesIO()
                    imageio.mimsave(video_bytes, last_results["frames"], format="mp4", fps=20)
                    video_bytes.seek(0)
                    st.video(video_bytes, format="video/mp4", caption="Flight Animation (MP4)")

        # Check for replay button clicks (works in any mode)
        replay_target = None
        if replay_last_btn:
            if st.session_state.get("last_success_path"):
                replay_target = st.session_state.last_success_path
                st.info(f"Replaying last successful run...")
            else:
                st.warning("No successful run available to replay. Run a simulation first!")
        elif replay_btn:
            if selected_saved:
                replay_target = selected_saved
                st.info(f"Replaying selected run: {Path(selected_saved).parent.name}")
            else:
                st.warning("Please select a saved run to replay!")
        
        # Handle replay
        if replay_target:
            try:
                rec = load_run(replay_target)
                # Adapt record to results-like dict
                results = {
                    "positions": np.array(rec.positions),
                    "reference_trajectory": (np.array(rec.reference_trajectory) if rec.reference_trajectory is not None else None),
                    "ttc": np.array(rec.ttc),
                    "rewards": np.array(rec.rewards),
                    "success": bool(rec.metrics.get("success", False)),
                    "steps": int(rec.metrics.get("steps", len(rec.rewards))),
                    "total_reward": float(rec.metrics.get("total_reward", np.sum(rec.rewards))),
                    "frames": [],
                }
                
                st.subheader("Replayed Results")
                if results["success"]:
                    st.success(f"Mission successful! Completed in {results['steps']} steps with reward {results['total_reward']:.2f}")
                else:
                    st.error(f"Mission failed. Completed {results['steps']} steps with reward {results['total_reward']:.2f}")

                st.subheader("Flight Metrics")
                metrics = {
                    "TTC (s)": np.mean(results["ttc"]) if np.isfinite(results["ttc"]).any() else float("inf"),
                    "Avg Reward": np.mean(results["rewards"]),
                }
                display_metrics(metrics)
                
                # Add metrics explanation
                display_metrics_explanation()
                
                # Create visualization columns
                col1, col2 = st.columns(2)
                
                with col1:
                    # Display trajectory plot
                    st.subheader("3D Trajectory")
                    trajectory_fig = create_trajectory_plot(
                        results["positions"],
                        results["reference_trajectory"]
                    )
                    st.plotly_chart(trajectory_fig, use_container_width=True, key="replay_trajectory")
                
                with col2:
                    # Display metrics over time
                    st.subheader("Metrics Over Time")
                    metrics_df = pd.DataFrame({
                        "Step": np.arange(len(results["ttc"])),
                        "TTC (s)": np.clip(results["ttc"], 0, 10),
                        "Reward": results["rewards"],
                    })
                    
                    # Create multi-line chart
                    metrics_fig = go.Figure()
                    
                    metrics_fig.add_trace(go.Scatter(
                        x=metrics_df["Step"], y=metrics_df["TTC (s)"],
                        mode="lines", name="TTC (s)", line=dict(color="green")
                    ))
                    
                    metrics_fig.add_trace(go.Scatter(
                        x=metrics_df["Step"], y=metrics_df["Reward"],
                        mode="lines", name="Reward", line=dict(color="purple")
                    ))
                    
                    metrics_fig.update_layout(
                        xaxis_title="Simulation Step",
                        yaxis_title="Value",
                        legend=dict(x=0, y=1, orientation="h"),
                        margin=dict(l=0, r=0, b=0, t=30),
                        height=400,
                    )
                    
                    st.plotly_chart(metrics_fig, use_container_width=True, key="replay_metrics")
                
                # Load and display video if available (optional)
                if hasattr(rec, 'video_path') and rec.video_path and Path(rec.video_path).exists():
                    st.subheader("Flight Video")
                    with open(rec.video_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                        st.video(video_bytes, format="video/mp4", caption="Replayed Flight Animation")
                
                # Always show replay complete message
                st.success("🎬 Replay complete! The trajectory and metrics above show the flight path.")
                    
            except Exception as e:
                st.error(f"Failed to load replay: {e}")

    # Display instructions if no model loaded
    if "model" not in st.session_state:
        st.info("👈 Configure and load a model using the sidebar to begin.")

        # Display project information
        st.subheader("About the Project")
        st.markdown("""
        ### Drone Transformer RL

        This project implements a transformer-based reinforcement learning approach for autonomous drone navigation.
        The system combines:

        1. **Hierarchical Transformer Policy** - multi-scale attention with relative position encodings and memory
        2. **Reinforcement Learning** - PPO algorithm for policy optimization
        3. **Knowledge Distillation** - large teacher → lightweight student for real-time inference
        4. **Curriculum Learning** - progressive difficulty for robust performance

        The model learns to navigate through complex environments, avoiding obstacles while maintaining desired
        trajectories and velocities.

        **Key metrics:**
        - Time-to-Collision (TTC) > 3s (safety)
        - Average Reward (performance indicator)
        - Success Rate (mission completion)
        - Inference Latency < 10ms (real-time capability)
        """)

if __name__ == "__main__":
    main()

