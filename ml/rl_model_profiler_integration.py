"""
Integration of PerformanceProfiler into the RL model.

This module provides functions to integrate the PerformanceProfiler
into the RL model for performance monitoring and optimization.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, Tuple

from src.ml.performance_profiler import PerformanceProfiler, PerformanceMetrics, InferenceMetrics

logger = logging.getLogger(__name__)


def integrate_profiler_with_rl_model(model, tensorboard_writer=None, latency_target_ms=10.0):
    """
    Integrate PerformanceProfiler with the RL model.
    
    Args:
        model: SelfImprovingRLModel instance
        tensorboard_writer: Optional TensorBoard SummaryWriter
        latency_target_ms: Target latency for inference in milliseconds
        
    Returns:
        PerformanceProfiler instance
    """
    # Create profiler
    profiler = PerformanceProfiler(
        model=model,
        device=model.device,
        tensorboard_writer=tensorboard_writer,
        latency_target_ms=latency_target_ms
    )
    
    # Store profiler in model
    model.performance_profiler = profiler
    
    logger.info(f"PerformanceProfiler integrated with RL model, latency target: {latency_target_ms}ms")
    return profiler


def profile_training_step(model, batch_data, step_in_episode_count=0):
    """
    Profile a training step using the integrated profiler.
    
    Args:
        model: SelfImprovingRLModel instance
        batch_data: Batch data for training
        step_in_episode_count: Current step in episode
        
    Returns:
        TrainingMetrics and PerformanceMetrics
    """
    if not hasattr(model, 'performance_profiler'):
        # If profiler not integrated, just run the update
        return model.update(step_in_episode_count)
    
    # Unpack batch data
    states, actions, rewards, next_states, dones, weights = batch_data
    batch_size = states.shape[0]
    
    # Define forward function
    def forward_fn():
        # Get current Q-values
        current_q_values, _ = model.forward(states)
        state_action_values = current_q_values.gather(1, actions.unsqueeze(1))
        
        # Get target Q-values (Double DQN)
        next_q_values_online, _ = model.forward(next_states)
        next_actions = next_q_values_online.argmax(dim=1, keepdim=True)
        next_q_values_target, _ = model.target_net.forward(next_states)
        next_state_values = next_q_values_target.gather(1, next_actions).squeeze(1)
        
        # Zero out value for terminal states
        next_state_values[dones] = 0.0
        
        # Get current hyperparameters
        hyperparams = model._get_current_hyperparams()
        gamma = hyperparams['gamma']
        
        # Compute target Q-values
        target_q_values = rewards + (gamma * next_state_values)
        
        return current_q_values, state_action_values, target_q_values
    
    # Define backward function
    def backward_fn():
        # Compute TD error
        td_errors = torch.abs(target_q_values - state_action_values)
        
        # Compute weighted loss
        losses = model.q_criterion(state_action_values, target_q_values.unsqueeze(1))
        weighted_loss = (losses * weights.unsqueeze(1)).mean()
        
        # Backward pass
        weighted_loss.backward()
        return weighted_loss
    
    # Define optimizer step function
    def optimizer_step_fn():
        # Update all components in parallel
        loss_fns = {
            "main": lambda: weighted_loss
        }
        
        # Add auxiliary task loss if enabled
        if model.use_auxiliary and model.auxiliary_heads:
            loss_fns["auxiliary"] = lambda: model.auxiliary_heads.compute_loss(
                model.feature_net(states), next_states
            )
        
        # Add curiosity loss if enabled
        if model.use_curiosity and model.curiosity_module:
            loss_fns["curiosity"] = lambda: model.curiosity_module.compute_loss(
                states, actions, next_states
            )
        
        # Add uncertainty loss if enabled
        if model.use_uncertainty and model.uncertainty_estimator:
            loss_fns["uncertainty"] = lambda: model.uncertainty_estimator.compute_loss(
                model.feature_net(states), target_q_values
            )
        
        # Update all components in parallel
        model.component_updater.update_all(loss_fns)
    
    # Profile training step
    current_q_values, state_action_values, target_q_values, weighted_loss = None, None, None, None
    performance_metrics = model.performance_profiler.profile_training_step(
        batch_size=batch_size,
        forward_fn=lambda: globals().update(
            (current_q_values, state_action_values, target_q_values) = forward_fn()
        ) or (current_q_values, state_action_values, target_q_values),
        backward_fn=lambda: globals().update(weighted_loss=backward_fn()) or weighted_loss,
        optimizer_step_fn=optimizer_step_fn
    )
    
    # Update target network if needed
    hyperparams = model._get_current_hyperparams()
    tau = hyperparams['tau']
    model._soft_update_target_network(tau)
    
    # Create training metrics
    training_metrics = model._create_training_metrics(
        td_errors=torch.abs(target_q_values - state_action_values.squeeze(1)),
        loss=weighted_loss.item(),
        q_values=current_q_values,
        step_in_episode=step_in_episode_count
    )
    
    # Add metrics to storage
    model.training_metrics.add_metric(training_metrics)
    
    # Check for performance regression
    if len(model.performance_profiler.training_history) >= 20:
        regression_result = model.performance_profiler.detect_performance_regression()
        if regression_result["has_regression"]:
            logger.warning("Performance regression detected during training")
            for reg in regression_result["regressions"]:
                logger.warning(f"  {reg['metric']}: {reg['baseline']:.2f} -> {reg['recent']:.2f} "
                             f"({reg['percent_change']:.1f}% change)")
    
    return training_metrics, performance_metrics


def profile_inference(model, state):
    """
    Profile model inference using the integrated profiler.
    
    Args:
        model: SelfImprovingRLModel instance
        state: Input state
        
    Returns:
        Action, config_params, info_dict, and InferenceMetrics
    """
    if not hasattr(model, 'performance_profiler'):
        # If profiler not integrated, just run the act method
        return model.act(state)
    
    # Define preprocess function
    def preprocess_fn():
        # Use DeviceAwareTensorCache for efficient tensor conversion
        cache_key = f"state_{hash(state.tobytes())}"
        return model.tensor_cache.to_tensor(state, key=cache_key)
    
    # Define inference function
    def inference_fn(state_tensor):
        with torch.no_grad():
            # Forward pass
            q_values, config_params = model.forward(state_tensor)
            
            # Get action probabilities
            action_probs = torch.softmax(q_values, dim=1)
            
            # Select action (greedy)
            action = torch.argmax(q_values, dim=1).item()
            
            # Calculate policy entropy for monitoring
            log_probs = torch.log_softmax(q_values, dim=1)
            policy_entropy = -torch.sum(action_probs * log_probs, dim=1)
            
            return action, config_params, q_values, action_probs, policy_entropy
    
    # Define postprocess function
    def postprocess_fn(results):
        action, config_params, q_values, action_probs, policy_entropy = results
        
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
    
    # Profile inference
    inference_metrics = model.performance_profiler.profile_inference(
        preprocess_fn=preprocess_fn,
        inference_fn=inference_fn,
        postprocess_fn=postprocess_fn
    )
    
    # Check if inference meets latency target
    if not inference_metrics.meets_latency_target:
        logger.warning(f"Inference latency ({inference_metrics.total_time_ms:.2f}ms) "
                     f"exceeds target ({model.performance_profiler.latency_target_ms}ms)")
    
    # Return the results from postprocess_fn
    return *postprocess_fn.results, inference_metrics