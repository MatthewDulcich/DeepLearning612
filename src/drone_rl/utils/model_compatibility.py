"""Compatibility utilities for loading models with different architectures."""

import torch
import warnings
from typing import Dict, Any
from stable_baselines3 import PPO


def load_model_with_compatibility(model_path: str, env, policy_class, device: str = "cpu") -> PPO:
    """Load a model with compatibility handling for architecture changes.
    
    This function handles cases where saved models have components that are not
    present in the current model definition (e.g., deprecated state_predictor).
    
    Parameters
    ----------
    model_path : str
        Path to the saved model
    env : gym.Env
        Environment instance
    policy_class : type
        Policy class to use
    device : str
        Device to load model on
        
    Returns
    -------
    PPO
        Loaded model with compatible state dict
    """
    try:
        # First try normal loading
        custom_objects = {"policy_class": policy_class}
        return PPO.load(model_path, env=env, device=device, custom_objects=custom_objects)
    
    except Exception as e:
        error_str = str(e)
        print(f"Normal loading failed: {e}")
        
        # Handle various compatibility issues
        if any(keyword in error_str for keyword in [
            "Unexpected key(s) in state_dict", 
            "state_predictor", 
            "unexpected keyword argument 'n_envs'",
            "LSTMFeatureExtractor",
            "__init__() got an unexpected keyword argument"
        ]):
            print(f"Model contains incompatible components. Loading with compatibility mode...")
            
            # Create a patched policy class that handles old parameter signatures
            patched_policy_class = create_compatible_policy_class(policy_class)
            
            try:
                # Try with patched policy class
                custom_objects = {"policy_class": patched_policy_class}
                return PPO.load(model_path, env=env, device=device, custom_objects=custom_objects)
                
            except Exception as inner_e:
                print(f"Patched policy loading failed: {inner_e}")
                # Use manual state dict loading approach
                return load_with_manual_state_dict(model_path, env, policy_class, device)
        else:
            # Re-raise if it's a different error
            raise e


def create_compatible_policy_class(base_policy_class):
    """Create a policy class that's compatible with old model signatures."""
    
    class CompatiblePolicy(base_policy_class):
        def __init__(self, *args, **kwargs):
            # Filter out any problematic kwargs that might come from old models
            filtered_kwargs = {}
            for key, value in kwargs.items():
                # Skip problematic feature extractor kwargs
                if key == "features_extractor_kwargs" and isinstance(value, dict):
                    # Filter n_envs from feature extractor kwargs
                    filtered_value = {k: v for k, v in value.items() if k != "n_envs"}
                    filtered_kwargs[key] = filtered_value
                else:
                    filtered_kwargs[key] = value
            
            super().__init__(*args, **filtered_kwargs)
    
    return CompatiblePolicy


def load_with_manual_state_dict(model_path: str, env, policy_class, device: str = "cpu") -> PPO:
    """Load model by manually handling state dict loading."""
    import zipfile
    import io
    
    print("Attempting manual state dict loading...")
    
    try:
        # Create a fresh model instance with the current architecture
        model = PPO(policy_class, env, device=device)
        
        # Load the saved model's state dict using torch directly
        with zipfile.ZipFile(model_path, 'r') as zip_file:
            # Load policy state dict from the pytorch_variables.pth file
            try:
                with zip_file.open('policy.pth') as f:
                    policy_state_dict = torch.load(io.BytesIO(f.read()), map_location=device)
            except KeyError:
                # Fallback: try pytorch_variables.pth (different SB3 versions)
                with zip_file.open('pytorch_variables.pth') as f:
                    checkpoint = torch.load(io.BytesIO(f.read()), map_location=device)
                    policy_state_dict = checkpoint
        
        # Filter out deprecated keys
        filtered_state_dict = filter_state_dict_for_compatibility(policy_state_dict)
        
        # Load the filtered state dict with strict=False
        missing_keys, unexpected_keys = model.policy.load_state_dict(filtered_state_dict, strict=False)
        
        if unexpected_keys:
            print(f"Ignored unexpected keys: {unexpected_keys}")
        if missing_keys:
            print(f"Missing keys (will use random initialization): {missing_keys}")
        
        print("Model loaded successfully with manual state dict loading")
        return model
        
    except Exception as e:
        print(f"Manual state dict loading failed: {e}")
        # Try the strict=False fallback
        return load_with_strict_false(model_path, env, policy_class, device)


def load_with_strict_false(model_path: str, env, policy_class, device: str = "cpu") -> PPO:
    """Fallback loading method that uses strict=False during state dict loading."""
    import zipfile
    import pickle
    import tempfile
    import os
    
    print("Attempting fallback loading with strict=False...")
    
    # This is a more direct approach that modifies SB3's loading process
    # by monkey-patching the load_state_dict method temporarily
    original_load_state_dict = torch.nn.Module.load_state_dict
    
    def patched_load_state_dict(self, state_dict, strict=True):
        # Force strict=False and filter deprecated keys
        filtered_dict = filter_state_dict_for_compatibility(state_dict)
        return original_load_state_dict(self, filtered_dict, strict=False)
    
    try:
        # Temporarily patch the method
        torch.nn.Module.load_state_dict = patched_load_state_dict
        
        # Now try loading normally
        custom_objects = {"policy_class": policy_class}
        model = PPO.load(model_path, env=env, device=device, custom_objects=custom_objects)
        
        print("Model loaded with fallback method")
        return model
        
    finally:
        # Always restore the original method
        torch.nn.Module.load_state_dict = original_load_state_dict


def filter_state_dict_for_compatibility(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Filter state dict to remove deprecated components.
    
    Parameters
    ----------
    state_dict : Dict[str, torch.Tensor]
        Original state dictionary
        
    Returns
    -------
    Dict[str, torch.Tensor]
        Filtered state dictionary
    """
    deprecated_prefixes = [
        'state_predictor',
        'policy.state_predictor'
    ]
    
    filtered_dict = {}
    removed_keys = []
    
    for key, value in state_dict.items():
        if any(key.startswith(prefix) for prefix in deprecated_prefixes):
            removed_keys.append(key)
            continue
        filtered_dict[key] = value
    
    if removed_keys:
        warnings.warn(f"Removed deprecated keys from state dict: {removed_keys}")
    
    return filtered_dict
