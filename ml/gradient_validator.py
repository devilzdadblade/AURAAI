"""
Gradient validation utilities for reinforcement learning models.

This module provides tools to ensure gradient stability in RL computations,
preventing issues like exploding gradients, NaN/Inf values, and other numerical instabilities.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class GradientStats:
    """Statistics about gradients in a model."""
    mean_grad_norm: float
    max_grad_norm: float
    min_grad_norm: float
    has_nan: bool
    has_inf: bool
    num_params_with_nan: int
    num_params_with_inf: int
    num_params_total: int
    layer_norms: Dict[str, float]


@dataclass
class ValidationResult:
    """Result of gradient validation."""
    is_valid: bool
    error_message: Optional[str] = None
    affected_params: Optional[List[str]] = None


class GradientValidator:
    """
    Validates gradients before optimizer steps to prevent numerical instability.
    
    This class provides methods for:
    - Validating gradients for NaN/Inf values
    - Clipping gradients to prevent exploding gradients
    - Logging gradient statistics for monitoring
    """
    
    def __init__(self, max_grad_norm: float = 1.0):
        """
        Initialize the GradientValidator.
        
        Args:
            max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
        """
        self.max_grad_norm = max_grad_norm
        logger.info(f"GradientValidator initialized with max_grad_norm={max_grad_norm}")
    
    def validate_gradients(self, model: nn.Module) -> ValidationResult:
        """
        Validate gradients of a model for NaN/Inf values.
        
        Args:
            model: PyTorch model to validate gradients for
            
        Returns:
            ValidationResult with validation status and error message if invalid
        """
        affected_params = []
        has_nan = False
        has_inf = False
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Check for NaN values
                if torch.isnan(param.grad).any():
                    has_nan = True
                    affected_params.append(f"{name} (NaN)")
                    logger.warning(f"NaN gradient detected in parameter: {name}")
                
                # Check for Inf values
                if torch.isinf(param.grad).any():
                    has_inf = True
                    affected_params.append(f"{name} (Inf)")
                    logger.warning(f"Inf gradient detected in parameter: {name}")
        
        is_valid = not (has_nan or has_inf)
        error_message = None
        
        if not is_valid:
            error_message = f"Invalid gradients detected: {len(affected_params)} parameters affected"
            logger.error(error_message)
            
            # Log gradient statistics for debugging
            stats = self.log_gradient_stats(model)
            logger.debug(f"Gradient statistics: mean={stats.mean_grad_norm:.6f}, "
                        f"max={stats.max_grad_norm:.6f}, min={stats.min_grad_norm:.6f}")
        
        return ValidationResult(
            is_valid=is_valid,
            error_message=error_message,
            affected_params=affected_params
        )
    
    def clip_gradients(self, model: nn.Module, max_norm: Optional[float] = None) -> float:
        """
        Clip gradients to prevent exploding gradients.
        
        Args:
            model: PyTorch model to clip gradients for
            max_norm: Maximum gradient norm (defaults to self.max_grad_norm if None)
            
        Returns:
            Total norm of the gradients after clipping
        """
        if max_norm is None:
            max_norm = self.max_grad_norm
            
        # Use PyTorch's built-in gradient clipping utility
        total_norm = torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(),
            max_norm=max_norm
        )
        
        if total_norm > max_norm:
            logger.info(f"Gradients clipped from {total_norm:.4f} to {max_norm:.4f}")
            
        return total_norm
    
    def log_gradient_stats(self, model: nn.Module) -> GradientStats:
        """
        Calculate and log gradient statistics for monitoring.
        
        Args:
            model: PyTorch model to calculate gradient statistics for
            
        Returns:
            GradientStats object with gradient statistics
        """
        grad_norms = []
        layer_norms = {}
        has_nan = False
        has_inf = False
        num_params_with_nan = 0
        num_params_with_inf = 0
        num_params_total = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                num_params_total += 1
                
                # Check for NaN/Inf values
                if torch.isnan(param.grad).any():
                    has_nan = True
                    num_params_with_nan += 1
                
                if torch.isinf(param.grad).any():
                    has_inf = True
                    num_params_with_inf += 1
                
                # Calculate gradient norm for this parameter
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                layer_norms[name] = grad_norm
        
        # Calculate statistics
        mean_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
        max_grad_norm = max(grad_norms) if grad_norms else 0.0
        min_grad_norm = min(grad_norms) if grad_norms else 0.0
        
        # Create and return GradientStats object
        stats = GradientStats(
            mean_grad_norm=mean_grad_norm,
            max_grad_norm=max_grad_norm,
            min_grad_norm=min_grad_norm,
            has_nan=has_nan,
            has_inf=has_inf,
            num_params_with_nan=num_params_with_nan,
            num_params_with_inf=num_params_with_inf,
            num_params_total=num_params_total,
            layer_norms=layer_norms
        )
        
        # Log summary statistics
        logger.debug(f"Gradient stats - Mean: {mean_grad_norm:.6f}, Max: {max_grad_norm:.6f}, "
                    f"Min: {min_grad_norm:.6f}, NaN: {num_params_with_nan}, Inf: {num_params_with_inf}")
        
        return stats