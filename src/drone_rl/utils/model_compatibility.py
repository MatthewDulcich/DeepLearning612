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
        if "Unexpected key(s) in state_dict" in str(e) or "state_predictor" in str(e):
            warnings.warn(
                f"Model contains deprecated components. Loading with compatibility mode.\n"
                f"Original error: {e}"
            )
            
            # Load manually with filtering
            import zipfile
            import pickle
            import tempfile
            import os
            
            # Extract and modify the model data
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract the zip file
                with zipfile.ZipFile(model_path, 'r') as zip_file:
                    zip_file.extractall(temp_dir)
                
                # Load the data pickle
                data_path = os.path.join(temp_dir, 'data')
                with open(data_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Filter out deprecated state_dict keys
                if 'state_dict' in data:
                    original_state_dict = data['state_dict']
                    filtered_state_dict = {}
                    
                    deprecated_prefixes = [
                        'policy.state_predictor',
                        'state_predictor'
                    ]
                    
                    for key, value in original_state_dict.items():
                        # Skip deprecated keys
                        if any(key.startswith(prefix) for prefix in deprecated_prefixes):
                            continue
                        filtered_state_dict[key] = value
                    
                    data['state_dict'] = filtered_state_dict
                
                # Save the modified data
                with open(data_path, 'wb') as f:
                    pickle.dump(data, f)
                
                # Create a new zip file
                temp_model_path = os.path.join(temp_dir, 'model_compatible.zip')
                with zipfile.ZipFile(temp_model_path, 'w') as zip_file:
                    for file_name in os.listdir(temp_dir):
                        if file_name != 'model_compatible.zip':
                            zip_file.write(
                                os.path.join(temp_dir, file_name),
                                file_name
                            )
                
                # Load the compatible model
                custom_objects = {"policy_class": policy_class}
                model = PPO.load(temp_model_path, env=env, device=device, custom_objects=custom_objects)
                
                print("Model loaded successfully in compatibility mode")
                return model
        else:
            # Re-raise if it's a different error
            raise e


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
