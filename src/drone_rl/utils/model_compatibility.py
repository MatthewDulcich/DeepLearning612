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
        if "Unexpected key(s) in state_dict" in error_str or "state_predictor" in error_str:
            print(f"Model contains deprecated components. Loading with compatibility mode...")
            print(f"Original error: {e}")
            
            # Use a simpler approach: create model and manually load filtered state dict
            try:
                # Create a fresh model instance with the current architecture
                model = PPO(policy_class, env, device=device)
                
                # Load the saved model's state dict using torch directly
                import zipfile
                import io
                
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
                
                print("Model loaded successfully in compatibility mode")
                return model
                
            except Exception as inner_e:
                print(f"Compatibility mode also failed: {inner_e}")
                # Try even simpler approach - just load with strict=False
                return load_with_strict_false(model_path, env, policy_class, device)
        else:
            # Re-raise if it's a different error
            raise e


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
