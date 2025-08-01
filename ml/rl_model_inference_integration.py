"""
Integration of inference optimization into the RL model.

This module provides functions to integrate the inference optimization
components into the RL model for real-time trading requirements.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import time
from typing import List, Optional, Tuple, Any, Union, Callable

from src.ml.inference_optimizer import (
    OptimizedInferencePipeline,
    InferenceOptimizer,
    InferenceRequest,
    create_inference_data_loader,
    optimize_batch_inference
)

logger = logging.getLogger(__name__)


def integrate_inference_optimizer(model, latency_target_ms=10.0):
    """
    Integrate inference optimization with the RL model.
    
    Args:
        model: SelfImprovingRLModel instance
        latency_target_ms: Target latency for inference in milliseconds
        
    Returns:
        OptimizedInferencePipeline instance
    """
    # Create inference pipeline
    pipeline = OptimizedInferencePipeline(
        model=model,
        device=model.device,
        tensor_cache=model.tensor_cache,
        batch_size=8,
        max_queue_size=32,
        latency_target_ms=latency_target_ms
    )
    
    # Store pipeline in model
    model.inference_pipeline = pipeline
    
    # Monkey patch the act method for optimization
    original_act = model.act
    
    def optimized_act(state, epsilon=0.0, use_uncertainty=True):
        """Optimized version of the act method."""
        start_time = time.time()
        
        # Use the optimized act method
        action, config_params, info = InferenceOptimizer.optimize_act_method(
            model, state, epsilon, use_uncertainty
        )
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Log latency if it exceeds target
        if latency_ms > latency_target_ms:
            logger.warning(f"Inference latency ({latency_ms:.2f}ms) exceeds target ({latency_target_ms}ms)")
        
        # Add latency to info
        info['latency_ms'] = latency_ms
        
        return action, config_params, info
    
    # Replace the act method with the optimized version
    model.original_act = original_act
    model.act = optimized_act
    
    logger.info(f"Inference optimizer integrated with RL model, latency target: {latency_target_ms}ms")
    return pipeline


def batch_inference(model, states):
    """
    Perform batch inference with optimized performance.
    
    Args:
        model: SelfImprovingRLModel instance
        states: Batch of states as numpy array
        
    Returns:
        Tuple of (actions, config_params, info_dict)
    """
    # Convert states to tensor
    if isinstance(states, np.ndarray):
        # Create DataLoader with pin_memory for efficient transfer
        dataloader = create_inference_data_loader(states, batch_size=len(states))
        
        # Get batch tensor
        for batch in dataloader:
            # Move to device efficiently
            states_tensor = batch[0].to(model.device, non_blocking=True)
            break
    else:
        # Assume states is already a tensor
        states_tensor = states
    
    # Perform inference with torch.no_grad()
    with torch.no_grad():
        # Forward pass
        q_values, config_params = model.forward(states_tensor)
        
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
    
    # Create info dict
    info_dict = {
        'q_values': q_values_np,
        'action_probs': action_probs_np,
        'policy_entropy': policy_entropy_np
    }
    
    return actions_np, config_params_np, info_dict


def create_batched_inference_method(model):
    """
    Create a batched inference method for the model.
    
    Args:
        model: SelfImprovingRLModel instance
        
    Returns:
        Batched inference method
    """
    def act_batch(states, epsilon=0.0, use_uncertainty=True):
        """
        Perform batched inference for multiple states.
        
        Args:
            states: Batch of states as numpy array
            epsilon: Exploration rate
            use_uncertainty: Whether to use uncertainty estimation
            
        Returns:
            Tuple of (actions, config_params, info_dict)
        """
        return batch_inference(model, states)
    
    # Add the method to the model
    model.act_batch = act_batch
    
    return act_batch


def optimize_inference_for_real_time(model):
    """
    Apply all inference optimizations for real-time trading requirements.
    
    Args:
        model: SelfImprovingRLModel instance
        
    Returns:
        Optimized model
    """
    # Integrate inference optimizer
    integrate_inference_optimizer(model)
    
    # Create batched inference method
    create_batched_inference_method(model)
    
    # Set model to evaluation mode for inference
    model.eval()
    
    # Pre-compile critical paths with torch.jit.script if possible
    try:
        # Try to script the forward method for faster execution
        # This may fail if the model uses dynamic control flow
        scripted_forward = torch.jit.script(model.forward)
        
        # Replace the forward method with the scripted version
        model.scripted_forward = scripted_forward
        
        # Monkey patch the forward method to use the scripted version
        original_forward = model.forward
        
        def optimized_forward(x):
            """Use the scripted forward method for inference."""
            if model.training:
                # Use original forward in training mode
                return original_forward(x)
            else:
                # Use scripted forward in evaluation mode
                return scripted_forward(x)
        
        # Replace the forward method with the optimized version
        model.original_forward = original_forward
        model.forward = optimized_forward
        
        logger.info("Successfully compiled model forward pass with torch.jit.script")
    except Exception as e:
        logger.warning(f"Failed to compile model with torch.jit.script: {str(e)}")
        logger.warning("Falling back to standard inference")
    
    logger.info("Applied all inference optimizations for real-time trading requirements")
    
    return model