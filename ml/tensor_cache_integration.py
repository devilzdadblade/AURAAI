"""
DeviceAwareTensorCache integration for RL model.

This module provides functions to integrate DeviceAwareTensorCache into the RL model
for efficient tensor operations and device transfers.
"""

import torch
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

from src.ml.memory_management import DeviceAwareTensorCache

# Create seeded random generator for reproducibility
rng = np.random.default_rng(42)


def batch_to_tensor(tensor_cache: DeviceAwareTensorCache, 
                   batch_data: List[Tuple], 
                   indices: Optional[List[int]] = None, 
                   weights: Optional[torch.Tensor] = None,
                   device: torch.device = None) -> Tuple[torch.Tensor, ...]:
    """
    Convert batch data to tensors efficiently using DeviceAwareTensorCache.
    This method batches tensor operations to minimize individual device transfers.
    
    Args:
        tensor_cache: DeviceAwareTensorCache instance
        batch_data: List of experience tuples
        indices: Optional indices for prioritized replay
        weights: Optional importance sampling weights
        device: Target device for tensors
        
    Returns:
        Tuple of tensors (states, actions, rewards, next_states, dones, weights)
    """
    # Unpack batch data
    states, actions, rewards, next_states, dones = zip(*batch_data)
    
    # Convert to numpy arrays first for efficient batching
    states_np = np.array(states)
    actions_np = np.array(actions)
    rewards_np = np.array(rewards)
    next_states_np = np.array(next_states)
    dones_np = np.array(dones)
    
    # Use DeviceAwareTensorCache for efficient tensor conversion and caching
    states_tensor = tensor_cache.to_tensor(states_np, key="batch_states", dtype=torch.float32)
    actions_tensor = tensor_cache.to_tensor(actions_np, key="batch_actions", dtype=torch.long)
    rewards_tensor = tensor_cache.to_tensor(rewards_np, key="batch_rewards", dtype=torch.float32)
    next_states_tensor = tensor_cache.to_tensor(next_states_np, key="batch_next_states", dtype=torch.float32)
    dones_tensor = tensor_cache.to_tensor(dones_np, key="batch_dones", dtype=torch.bool)
    
    # Handle weights for prioritized replay
    if weights is not None:
        weights_tensor = weights.to(device) if device else weights
    else:
        weights_tensor = torch.ones(len(batch_data), device=device)
        
    return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor, weights_tensor


def warm_tensor_cache(tensor_cache: DeviceAwareTensorCache, 
                     batch_size: int, 
                     input_size: int, 
                     action_size: int) -> None:
    """
    Warm the tensor cache with frequently accessed tensors to improve performance.
    This reduces cache misses during training and inference.
    
    Args:
        tensor_cache: DeviceAwareTensorCache instance
        batch_size: Batch size for training
        input_size: Input state dimension
        action_size: Number of possible actions
    """
    # Common tensor patterns for warming
    common_tensors = []
    
    # Zero tensors for various operations
    common_tensors.append(("zero_scalar", np.array(0.0)))
    common_tensors.append(("zero_vector", np.zeros(action_size)))
    common_tensors.append(("zero_batch", np.zeros((batch_size, input_size))))
    
    # Identity and unit tensors
    common_tensors.append(("ones_batch", np.ones(batch_size)))
    common_tensors.append(("unit_vector", np.ones(action_size)))
    
    # Common batch size patterns
    common_tensors.append(("batch_indices", np.arange(batch_size)))
    common_tensors.append(("action_indices", np.arange(action_size)))
    
    # Small random tensors for initialization
    common_tensors.append(("small_random", rng.standard_normal((batch_size, input_size)) * 0.01))
    
    # Common masks for operations
    common_tensors.append(("done_mask_false", np.zeros(batch_size, dtype=bool)))
    common_tensors.append(("done_mask_true", np.ones(batch_size, dtype=bool)))
    
    # Common weights for prioritized replay
    common_tensors.append(("uniform_weights", np.ones(batch_size)))
    
    # Common action masks
    for action in range(min(5, action_size)):  # Cache first few actions for quick access
        action_mask = np.zeros(action_size)
        action_mask[action] = 1.0
        common_tensors.append((f"action_mask_{action}", action_mask))
    
    # Prefill cache with common patterns
    tensor_cache.prefill_cache(common_tensors, dtype=torch.float32)


def optimize_state_tensor(tensor_cache: DeviceAwareTensorCache, 
                         state: np.ndarray, 
                         dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Optimize state tensor conversion using DeviceAwareTensorCache.
    
    Args:
        tensor_cache: DeviceAwareTensorCache instance
        state: State as numpy array
        dtype: Target tensor dtype
        
    Returns:
        State tensor on the target device
    """
    # Use a hash of the state bytes as a cache key
    cache_key = f"state_{hash(state.tobytes())}"
    return tensor_cache.to_tensor(state, key=cache_key, dtype=dtype).unsqueeze(0)