"""
Integration of error handling components into the SelfImprovingRLModel.

This module provides functions to integrate input validation, error handling,
and fallback mechanisms into the SelfImprovingRLModel class.
"""

import logging
import traceback
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

# Create a random number generator for modern numpy random usage with fixed seed for reproducibility
rng = np.random.default_rng(42)  # Fixed seed 42 for reproducible results
import torch

from src.ml.input_validator import InputValidator, ValidationResult
from src.ml.error_handler import ErrorHandler, ErrorSeverity, ErrorContext
from src.ml.fallback_manager import FallbackManager, FailureContext, FallbackLevel
from src.ml.model_health_monitor import ModelHealthMonitor

logger = logging.getLogger(__name__)


def integrate_error_handling(model):
    """
    Integrate error handling components into a SelfImprovingRLModel instance.
    
    This function adds input validation, error handling, and fallback mechanisms
    to an existing SelfImprovingRLModel instance by monkey patching its methods.
    
    Args:
        model: SelfImprovingRLModel instance to enhance
    """
    # Create components if they don't exist
    if not hasattr(model, 'input_validator'):
        model.input_validator = InputValidator(
            expected_shape=(model.input_size,),
            value_range=(-10.0, 10.0),  # Adjust based on expected input range
            allow_sanitization=True
        )
    
    if not hasattr(model, 'error_handler'):
        model.error_handler = ErrorHandler(
            model_health_monitor=getattr(model, 'model_health_monitor', None),
            fallback_manager=getattr(model, 'fallback_manager', None),
            max_consecutive_errors=5
        )
    
    if not hasattr(model, 'fallback_manager') and hasattr(model, 'model_health_monitor'):
        # Create a simplified model for fallback
        try:
            simplified_model = create_simplified_model(model)
            model.fallback_manager = FallbackManager(
                simplified_model=simplified_model,
                liquidate_on_halt=True
            )
            
            # Update error handler with fallback manager
            model.error_handler.fallback_manager = model.fallback_manager
        except Exception as e:
            logger.warning(f"Failed to create fallback manager: {e}")
    
    # Monkey patch methods with error handling
    _patch_act_method(model)
    _patch_update_method(model)
    _patch_add_experience_method(model)
    _patch_forward_method(model)


def create_simplified_model(model):
    """
    Create a simplified version of the model for fallback.
    
    Args:
        model: Original SelfImprovingRLModel instance
        
    Returns:
        Simplified model for fallback
    """
    import torch.nn as nn
    
    # Create a simple feed-forward network with the same input/output dimensions
    simplified_model = nn.Sequential(
        nn.Linear(model.input_size, model.hidden_size // 2),
        nn.ReLU(),
        nn.Linear(model.hidden_size // 2, model.action_size)
    ).to(model.device)
    
    return simplified_model


def _patch_act_method(model):
    """
    Patch the act method with error handling.
    
    Args:
        model: SelfImprovingRLModel instance to patch
    """
    original_act = model.act
    
    def act_with_error_handling(state, epsilon=0.0, use_uncertainty=True):
        # Validate input
        validation_result = model.input_validator.validate_state(state)
        if not validation_result.is_valid:
            logger.warning(
                f"Input validation failed in act(): {validation_result.error_message}. "
                f"Corrective action: {validation_result.corrective_action}"
            )
            
            # Try to sanitize input if possible
            if validation_result.sanitized_input is not None:
                state = validation_result.sanitized_input
            elif model.fallback_manager is not None:
                # Use fallback if sanitization not possible
                failure_context = FailureContext(
                    error_type="input_validation",
                    error_message=validation_result.error_message,
                    component="act",
                    input_state=state
                )
                
                fallback_action = model.fallback_manager.get_fallback_action(state, failure_context)
                return fallback_action.selected_action, np.zeros(model.config_size), {
                    'uncertainty': 1.0,  # High uncertainty for fallback
                    'q_values': fallback_action.action_values,
                    'policy_entropy': 0.0,
                    'fallback_used': True,
                    'fallback_level': fallback_action.strategy_level.name
                }
            else:
                # No fallback available, raise error
                raise ValueError(f"Invalid input state: {validation_result.error_message}")
        
        # Try to execute the original method with error handling
        result, error_context = model.error_handler.try_execute(
            func=original_act,
            component="act",
            args=(state, epsilon, use_uncertainty),
            fallback_func=lambda s, e, u: _fallback_act(model, s, e, u)
        )
        
        if error_context is not None and model.fallback_manager is not None:
            # Use fallback if original method failed
            failure_context = FailureContext(
                error_type=error_context.error_type,
                error_message=error_context.error_message,
                component=error_context.component,
                stack_trace=error_context.stack_trace,
                input_state=state
            )
            
            fallback_action = model.fallback_manager.get_fallback_action(state, failure_context)
            return fallback_action.selected_action, np.zeros(model.config_size), {
                'uncertainty': 1.0,  # High uncertainty for fallback
                'q_values': fallback_action.action_values,
                'policy_entropy': 0.0,
                'fallback_used': True,
                'fallback_level': fallback_action.strategy_level.name
            }
        
        return result
    
    # Replace the original method
    model.act = act_with_error_handling


def _fallback_act(model, state, epsilon=0.0, _use_uncertainty=True):
    """
    Fallback implementation of act method.
    
    Args:
        model: SelfImprovingRLModel instance
        state: Input state
        epsilon: Exploration parameter
        _use_uncertainty: Whether to use uncertainty estimation (unused in fallback)
        
    Returns:
        Action, config parameters, and metadata
    """
    # Simple epsilon-greedy implementation
    if rng.random() < epsilon:
        action = rng.integers(0, model.action_size)
    else:
        # Use the feature network and advantage head only
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().to(model.device)
            features = model.feature_net(state_tensor)
            advantage = model.advantage_head(features)
            action = torch.argmax(advantage).item()
    
    # Return action with zero config params and minimal metadata
    return action, np.zeros(model.config_size), {
        'uncertainty': 0.5,
        'q_values': np.zeros(model.action_size),
        'policy_entropy': 0.0,
        'fallback_used': True
    }


def _patch_update_method(model):
    """
    Patch the update method with error handling.
    
    Args:
        model: SelfImprovingRLModel instance to patch
    """
    original_update = model.update
    
    def update_with_error_handling(step_in_episode_count=0):
        # Try to execute the original method with error handling
        result, error_context = model.error_handler.try_execute(
            func=original_update,
            component="update",
            args=(step_in_episode_count,),
            fallback_func=lambda s: _fallback_update(model, s),
            severity=ErrorSeverity.ERROR
        )
        
        if error_context is not None:
            # Log the error
            logger.error(f"Error in update(): {error_context.error_message}")
            
            # Check model health
            if hasattr(model, 'model_health_monitor'):
                health_metrics = model.model_health_monitor.assess_health(model)
                logger.info(
                    f"Model health after error: {health_metrics.status.name} "
                    f"(score: {health_metrics.overall_health:.2f})"
                )
        
        return result
    
    # Replace the original method
    model.update = update_with_error_handling


def _fallback_update(model, _step_in_episode_count=0):
    """
    Fallback implementation of update method.
    
    Args:
        model: SelfImprovingRLModel instance
        _step_in_episode_count: Current step in episode (unused in fallback)
        
    Returns:
        Empty training metrics
    """
    # Return empty metrics
    from src.ml.metrics import TrainingMetrics
    
    return TrainingMetrics(
        loss=0.0,
        q_value_mean=0.0,
        q_value_std=0.0,
        td_error_mean=0.0,
        reward_mean=0.0,
        step=model.total_steps
    )


def _patch_add_experience_method(model):
    """
    Patch the add_experience method with error handling.
    
    Args:
        model: SelfImprovingRLModel instance to patch
    """
    original_add_experience = model.add_experience
    
    def add_experience_with_error_handling(state, action, reward, next_state, done):
        # Validate inputs
        state_validation = model.input_validator.validate_state(state)
        next_state_validation = model.input_validator.validate_state(next_state)
        
        # Handle validation failures
        if not state_validation.is_valid:
            logger.warning(
                f"State validation failed in add_experience(): {state_validation.error_message}. "
                f"Corrective action: {state_validation.corrective_action}"
            )
            if state_validation.sanitized_input is not None:
                state = state_validation.sanitized_input
        
        if not next_state_validation.is_valid:
            logger.warning(
                f"Next state validation failed in add_experience(): {next_state_validation.error_message}. "
                f"Corrective action: {next_state_validation.corrective_action}"
            )
            if next_state_validation.sanitized_input is not None:
                next_state = next_state_validation.sanitized_input
        
        # Try to execute the original method with error handling
        result, error_context = model.error_handler.try_execute(
            func=original_add_experience,
            component="add_experience",
            args=(state, action, reward, next_state, done),
            severity=ErrorSeverity.WARNING
        )
        
        if error_context is not None:
            logger.warning(f"Error in add_experience(): {error_context.error_message}")
        
        return result
    
    # Replace the original method
    model.add_experience = add_experience_with_error_handling


def _patch_forward_method(model):
    """
    Patch the forward method with error handling.
    
    Args:
        model: SelfImprovingRLModel instance to patch
    """
    original_forward = model.forward

    def _validate_and_sanitize_numpy(x, model):
        validation_result = model.input_validator.validate_state(x)
        if not validation_result.is_valid:
            logger.warning(
                f"Input validation failed in forward(): {validation_result.error_message}. "
                f"Corrective action: {validation_result.corrective_action}"
            )
            if validation_result.sanitized_input is not None:
                x = validation_result.sanitized_input
                x = torch.from_numpy(x).float().to(model.device)
        return x

    def _validate_and_sanitize_tensor(x, model):
        validation_result = model.input_validator.validate_tensor(x)
        if not validation_result.is_valid:
            logger.warning(
                f"Input validation failed in forward(): {validation_result.error_message}. "
                f"Corrective action: {validation_result.corrective_action}"
            )
            if validation_result.sanitized_input is not None:
                x = validation_result.sanitized_input
        return x

    def forward_with_error_handling(x):
        # Validate input tensor
        if isinstance(x, np.ndarray):
            x = _validate_and_sanitize_numpy(x, model)
        elif isinstance(x, torch.Tensor):
            x = _validate_and_sanitize_tensor(x, model)

        # Try to execute the original method with error handling
        result, error_context = model.error_handler.try_execute(
            func=original_forward,
            component="forward",
            args=(x,),
            fallback_func=lambda x: _fallback_forward(model, x),
            severity=ErrorSeverity.ERROR
        )

        if error_context is not None:
            logger.error(f"Error in forward(): {error_context.error_message}")

        return result

    # Replace the original method
    model.forward = forward_with_error_handling


def _fallback_forward(model, x):
    """
    Fallback implementation of forward method.
    
    Args:
        model: SelfImprovingRLModel instance
        x: Input tensor
        
    Returns:
        Q-values and config parameters
    """
    # Simple forward pass using only essential components
    with torch.no_grad():
        try:
            # Try to use feature network
            features = model.feature_net(x)
            
            # Compute value and advantage
            value = model.value_head(features)
            advantage = model.advantage_head(features)
            
            # Combine for Q-values
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
            
            # Simple config params
            config_params = torch.zeros((x.shape[0], model.config_size), device=x.device)
            
            return q_values, config_params
        except Exception:
            # Even more simplified fallback
            batch_size = x.shape[0] if len(x.shape) > 1 else 1
            q_values = torch.zeros((batch_size, model.action_size), device=x.device)
            config_params = torch.zeros((batch_size, model.config_size), device=x.device)
            
            return q_values, config_params