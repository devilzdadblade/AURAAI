"""
Inference optimization for reinforcement learning models.

This module provides tools for optimizing the inference pipeline to meet real-time
constraints. It implements torch.no_grad() context for all inference operations,
DataLoader with pin_memory=True for efficient data transfer, and batched inference
for multiple simultaneous requests.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Create a random number generator for modern numpy random usage
rng = np.random.default_rng(seed=42)
import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
from collections import deque

from src.ml.memory_management import DeviceAwareTensorCache

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Represents a single inference request."""
    state: np.ndarray
    request_id: str
    timestamp: float
    priority: float = 1.0


@dataclass
class InferenceResult:
    """Represents the result of an inference request."""
    request_id: str
    action: int
    config_params: np.ndarray
    info: Dict[str, Any]
    latency_ms: float


class OptimizedInferencePipeline:
    """
    Optimizes the inference pipeline for real-time constraints.
    
    This class provides methods for:
    - Using torch.no_grad() context for all inference operations
    - Implementing DataLoader with pin_memory=True for efficient data transfer
    - Optimizing action selection pipeline to meet 10ms latency requirement
    - Implementing inference batching for multiple simultaneous requests
    """
    
    def __init__(self, model: nn.Module, 
                 device: torch.device,
                 tensor_cache: DeviceAwareTensorCache,
                 batch_size: int = 8,
                 max_queue_size: int = 32,
                 latency_target_ms: float = 10.0):
        """
        Initialize the OptimizedInferencePipeline.
        
        Args:
            model: PyTorch model for inference
            device: Device the model is running on
            tensor_cache: DeviceAwareTensorCache for efficient tensor operations
            batch_size: Maximum batch size for batched inference
            max_queue_size: Maximum size of the inference request queue
            latency_target_ms: Target latency for inference in milliseconds
        """
        self.model = model
        self.device = device
        self.tensor_cache = tensor_cache
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.latency_target_ms = latency_target_ms
        
        # Request queue for batched inference
        self.request_queue = deque(maxlen=max_queue_size)
        
        # Pre-allocate tensors for batched inference
        self._preallocate_inference_tensors()
        
        # Set model to evaluation mode
        self.model.eval()
        
        logger.info(f"OptimizedInferencePipeline initialized with batch_size={batch_size}, "
                   f"latency_target={latency_target_ms}ms")
    
    def _preallocate_inference_tensors(self):
        """Pre-allocate tensors for batched inference to minimize memory allocations."""
        # Get input size from model
        input_size = getattr(self.model, 'input_size', None)
        if input_size is None:
            logger.warning("Could not determine input_size from model, skipping tensor pre-allocation")
            return
        
        # Pre-allocate batch tensor
        self.preallocated_batch = torch.zeros((self.batch_size, input_size), 
                                             device=self.device)
    
    def optimize_single_inference(self, state: np.ndarray) -> Tuple[int, np.ndarray, Dict[str, Any]]:
        """
        Optimize inference for a single state input.
        
        Args:
            state: Input state as numpy array
            
        Returns:
            Tuple of (action, config_params, info_dict)
        """
        # Use DeviceAwareTensorCache for efficient tensor conversion
        cache_key = f"state_{hash(state.tobytes())}"
        state_tensor = self.tensor_cache.to_tensor(state, key=cache_key).unsqueeze(0)
        
        # Perform inference with torch.no_grad()
        with torch.no_grad():
            # Forward pass
            q_values, config_params = self.model.forward(state_tensor)
            
            # Get action probabilities with numerical stability
            action_probs = torch.softmax(q_values, dim=1)
            
            # Select action (greedy)
            action = torch.argmax(q_values, dim=1).item()
            
            # Calculate policy entropy for monitoring
            log_probs = torch.log_softmax(q_values, dim=1)
            policy_entropy = -torch.sum(action_probs * log_probs, dim=1)
        
        # Convert tensors to numpy
        config_params_np = config_params.cpu().numpy()
        q_values_np = q_values.cpu().numpy()
        action_probs_np = action_probs.cpu().numpy()
        policy_entropy_np = policy_entropy.cpu().numpy()
        
        # Create info dict
        info_dict = {
            'uncertainty': 0.0,  # Default value if uncertainty estimator not used
            'q_values': q_values_np.squeeze(),
            'action_probs': action_probs_np.squeeze(),
            'policy_entropy': policy_entropy_np.item()
        }
        
        return action, config_params_np, info_dict
    
    def add_inference_request(self, request: InferenceRequest) -> bool:
        """
        Add an inference request to the queue for batched processing.
        
        Args:
            request: InferenceRequest object
            
        Returns:
            True if request was added, False if queue is full
        """
        if len(self.request_queue) >= self.max_queue_size:
            logger.warning(f"Inference request queue full ({self.max_queue_size} requests)")
            return False
        
        self.request_queue.append(request)
        return True
    
    def process_inference_batch(self) -> List[InferenceResult]:
        """
        Process a batch of inference requests.
        
        Returns:
            List of InferenceResult objects
        """
        if not self.request_queue:
            return []
        
        # Determine batch size (up to self.batch_size)
        batch_size = min(len(self.request_queue), self.batch_size)
        
        # Extract requests from queue
        requests = [self.request_queue.popleft() for _ in range(batch_size)]
        
        # Extract states and create batch
        states = np.array([req.state for req in requests])
        
        # Use DataLoader with pin_memory for efficient transfer
        dataset = TensorDataset(torch.from_numpy(states).float())
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            pin_memory=True,
            num_workers=0  # No additional workers for small batches
        )
        
        # Get batch tensor
        for batch in dataloader:
            # Move to device efficiently
            states_tensor = batch[0].to(self.device, non_blocking=True)
            break
        
        # Perform inference with torch.no_grad()
        with torch.no_grad():
            # Forward pass
            q_values, config_params = self.model.forward(states_tensor)
            
            # Get action probabilities
            action_probs = torch.softmax(q_values, dim=1)
            
            # Select actions (greedy)
            actions = torch.argmax(q_values, dim=1)
            
            # Calculate policy entropy for monitoring
            log_probs = torch.log_softmax(q_values, dim=1)
            policy_entropy = -torch.sum(action_probs * log_probs, dim=1)
        
        # Convert tensors to numpy
        actions_np = actions.cpu().numpy()
        config_params_np = config_params.cpu().numpy()
        q_values_np = q_values.cpu().numpy()
        action_probs_np = action_probs.cpu().numpy()
        policy_entropy_np = policy_entropy.cpu().numpy()
        
        # Create results
        results = []
        for i, request in enumerate(requests):
            # Create info dict
            info_dict = {
                'uncertainty': 0.0,  # Default value if uncertainty estimator not used
                'q_values': q_values_np[i],
                'action_probs': action_probs_np[i],
                'policy_entropy': policy_entropy_np[i].item()
            }
            
            # Calculate latency (simplified for now)
            latency_ms = 0.0  # Will be set by caller
            
            # Create result
            result = InferenceResult(
                request_id=request.request_id,
                action=actions_np[i],
                config_params=config_params_np[i],
                info=info_dict,
                latency_ms=latency_ms
            )
            
            results.append(result)
        
        return results


class InferenceOptimizer:
    """
    Optimizes inference operations for the RL model.
    
    This class provides methods for:
    - Wrapping model methods with torch.no_grad() for inference
    - Implementing efficient tensor operations
    - Optimizing the action selection pipeline
    """
    
    @staticmethod
    def optimize_act_method(model, state: np.ndarray, epsilon: float = 0.0,
                          use_uncertainty: bool = True) -> Tuple[int, np.ndarray, Dict[str, Any]]:
        """
        Optimized version of the act method using torch.no_grad() and efficient tensor operations.
        
        Args:
            model: SelfImprovingRLModel instance
            state: Input state as numpy array
            epsilon: Exploration rate
            use_uncertainty: Whether to use uncertainty estimation
            
        Returns:
            Tuple of (action, config_params, info_dict)
        """
        # Use DeviceAwareTensorCache for efficient tensor conversion
        cache_key = f"state_{hash(state.tobytes())}"
        state_tensor = model.tensor_cache.to_tensor(state, key=cache_key)
        
        with torch.no_grad():
            # Cache features for this state to avoid recomputation
            features_key = f"features_{cache_key}"
            if features_key in model.tensor_cache.cache:
                features = model.tensor_cache.cache[features_key]
            else:
                features = model.feature_net(state_tensor)
                # Cache the features for future use
                model.tensor_cache.cache[features_key] = features
            
            uncertainty_bonus = 0.0
            # If uncertainty estimation is enabled AND requested for this action
            if model.use_uncertainty and model.uncertainty_estimator and use_uncertainty:
                # Get mean Q-values and their uncertainty (standard deviation) from ensemble
                q_values_for_action_selection, q_uncertainty_std = model.uncertainty_estimator(features)
                uncertainty_bonus = q_uncertainty_std.mean().item()  # Average uncertainty across all actions
            else:
                # Otherwise, use the standard Q-network's predictions
                q_values_for_action_selection, config_params = model.forward(state_tensor)
            
            # Add numerical stability checks for Q-values
            q_values_for_action_selection = model.stability_manager.handle_numerical_instability(
                q_values_for_action_selection, replacement_value=0.0, name="act_q_values"
            )
            
            # Epsilon-greedy exploration
            if rng.random() < epsilon:
                action = rng.integers(0, model.action_size)
            else:
                # Add uncertainty bonus to encourage exploration of uncertain actions
                exploration_q_values = q_values_for_action_selection + 0.1 * uncertainty_bonus
                
                # Add numerical stability check for exploration Q-values
                exploration_q_values = model.stability_manager.handle_numerical_instability(
                    exploration_q_values, replacement_value=0.0, name="exploration_q_values"
                )
                
                # Select action based on Q-values (argmax for greedy selection)
                action = torch.argmax(exploration_q_values, dim=1).item()
            
            # Get config parameters
            if not model.use_uncertainty or not use_uncertainty:
                _, config_params = model.forward(state_tensor)
            
            # Convert to numpy
            config_params_np = config_params.cpu().numpy()
            
            # Calculate policy entropy for diagnostic monitoring
            probs = torch.softmax(q_values_for_action_selection, dim=1)
            probs = model.stability_manager.clamp_probabilities(probs)  # Clamp and renormalize probabilities
            
            # Calculate entropy using log_softmax for numerical stability
            log_probs = torch.log_softmax(q_values_for_action_selection, dim=1)
            policy_entropy = -torch.sum(probs * log_probs, dim=1)
            
            # Validate entropy computation results
            if not torch.isfinite(policy_entropy).all():
                # Handle numerical instability with fallback value
                logger.warning("Numerical instability detected in entropy calculation")
                policy_entropy = torch.tensor(0.5, device=policy_entropy.device)  # Fallback to moderate entropy value
            
            policy_entropy_value = policy_entropy.item()
            q_values_np = q_values_for_action_selection.cpu().numpy()
        
        return action, config_params_np, {
            'uncertainty': uncertainty_bonus,
            'q_values': q_values_np.squeeze(),
            'policy_entropy': policy_entropy_value
        }


def create_inference_data_loader(states: np.ndarray, batch_size: int = 32) -> DataLoader:
    """
    Create a DataLoader for efficient inference data transfer.
    
    Args:
        states: Numpy array of states
        batch_size: Batch size for the DataLoader
        
    Returns:
        DataLoader configured for efficient inference
    """
    # Convert to tensor dataset
    dataset = TensorDataset(torch.from_numpy(states).float())
    
    # Create DataLoader with pin_memory for efficient transfer
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0  # Adjust based on system capabilities
    )


def optimize_batch_inference(model: nn.Module, 
                           states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimize batch inference using torch.no_grad() and efficient tensor operations.
    
    Args:
        model: PyTorch model
        states: Batch of states as tensor
        
    Returns:
        Tuple of (q_values, config_params)
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Perform inference with torch.no_grad()
    with torch.no_grad():
        # Forward pass
        q_values, config_params = model(states)
    
    return q_values, config_params