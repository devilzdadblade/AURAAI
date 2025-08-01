"""
Numerical stability utilities for reinforcement learning models.

This module provides tools to ensure numerical stability in RL computations,
preventing issues like NaN/Inf values, log(0) errors, and other numerical instabilities.
"""

import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class NumericalStabilityManager:
    """
    Ensures numerical stability across all computations in RL models.
    
    This class provides methods for stable computations, including:
    - Safe logarithm operations
    - Probability clamping and renormalization
    - Handling of NaN/Inf values
    - Gradient validation and clipping
    """
    
    def __init__(self, epsilon: float = 1e-10, max_grad_norm: float = 1.0):
        """
        Initialize the NumericalStabilityManager.
        
        Args:
            epsilon: Small constant to prevent numerical instability (default: 1e-10)
            max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
        """
        self.epsilon = epsilon
        self.max_grad_norm = max_grad_norm
        logger.info(f"NumericalStabilityManager initialized with epsilon={epsilon}, "
                   f"max_grad_norm={max_grad_norm}")
    
    def stable_log_softmax(self, logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Compute log softmax in a numerically stable way.
        
        Args:
            logits: Input tensor
            dim: Dimension along which to compute log softmax (default: -1)
            
        Returns:
            Tensor with log softmax values
        """
        return F.log_softmax(logits, dim=dim)
    
    def stable_log(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute logarithm in a numerically stable way by adding epsilon.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with safe logarithm values
        """
        return torch.log(torch.clamp(x, min=self.epsilon))
    
    def clamp_probabilities(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Clamp probability values to prevent numerical instability and renormalize.
        
        Args:
            probs: Probability tensor
            
        Returns:
            Clamped and renormalized probability tensor
        """
        # Clamp values to [epsilon, 1.0]
        clamped_probs = torch.clamp(probs, min=self.epsilon, max=1.0)
        
        # Renormalize to ensure sum is 1.0 along the last dimension
        sum_probs = clamped_probs.sum(dim=-1, keepdim=True)
        normalized_probs = clamped_probs / sum_probs
        
        return normalized_probs
    
    def handle_numerical_instability(self, 
                                    tensor: torch.Tensor, 
                                    replacement_value: Optional[float] = 0.0,
                                    name: str = "tensor") -> torch.Tensor:
        """
        Handle NaN/Inf values in tensors by replacing them with a specified value.
        
        Args:
            tensor: Input tensor to check
            replacement_value: Value to use for replacing NaN/Inf (default: 0.0)
            name: Name of the tensor for logging purposes
            
        Returns:
            Tensor with NaN/Inf values replaced
        """
        if torch.isfinite(tensor).all():
            return tensor
        
        # Log and analyze the numerical instability
        self._log_numerical_instability(tensor, name)
        self._log_tensor_statistics(tensor, name)
        self._log_value_distribution(tensor, name)
        self._log_debug_info(name)
        
        # Replace NaN/Inf values and return result
        return self._replace_invalid_values(tensor, replacement_value)

    def _log_numerical_instability(self, tensor: torch.Tensor, name: str) -> None:
        """Log basic information about numerical instability."""
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        total_elements = tensor.numel()
        
        nan_percent = (nan_count / total_elements) * 100 if total_elements > 0 else 0
        inf_percent = (inf_count / total_elements) * 100 if total_elements > 0 else 0
        
        logger.warning(
            f"Numerical instability detected in {name}: "
            f"{nan_count} NaN values ({nan_percent:.2f}%), "
            f"{inf_count} Inf values ({inf_percent:.2f}%) "
            f"out of {total_elements} elements"
        )

    def _log_tensor_statistics(self, tensor: torch.Tensor, name: str) -> None:
        """Log statistical information about valid tensor values."""
        valid_mask = torch.isfinite(tensor)
        if not valid_mask.any():
            logger.error(f"No valid values in {name} tensor!")
            return
        
        valid_tensor = tensor[valid_mask]
        stats = self._calculate_tensor_stats(valid_tensor)
        logger.info(f"Valid {name} statistics: {stats}")

    def _calculate_tensor_stats(self, valid_tensor: torch.Tensor) -> dict:
        """Calculate comprehensive statistics for valid tensor values."""
        return {
            "min": valid_tensor.min().item(),
            "max": valid_tensor.max().item(),
            "mean": valid_tensor.mean().item(),
            "std": valid_tensor.std().item() if valid_tensor.numel() > 1 else 0.0,
            "median": valid_tensor.median().item() if valid_tensor.numel() > 0 else 0.0,
            "25th_percentile": torch.quantile(valid_tensor, 0.25, dim=None).item() if valid_tensor.numel() > 0 else 0.0,
            "75th_percentile": torch.quantile(valid_tensor, 0.75, dim=None).item() if valid_tensor.numel() > 0 else 0.0
        }

    def _log_value_distribution(self, tensor: torch.Tensor, name: str) -> None:
        """Log histogram-like distribution for debugging."""
        valid_mask = torch.isfinite(tensor)
        if not valid_mask.any() or valid_mask.sum() <= 10:
            return
        
        valid_tensor = tensor[valid_mask]
        try:
            hist_values, hist_edges = torch.histogram(
                valid_tensor.flatten(), 
                bins=10, 
                range=(valid_tensor.min().item(), valid_tensor.max().item())
            )
            hist_info = self._format_histogram_info(hist_values, hist_edges)
            logger.info(hist_info)
        except Exception as e:
            logger.debug(f"Could not generate histogram for {name}: {e}")

    def _format_histogram_info(self, hist_values: torch.Tensor, 
                             hist_edges: torch.Tensor) -> str:
        """Format histogram information for logging."""
        hist_info = "\nValue distribution histogram:"
        for i in range(len(hist_values)):
            bin_start = hist_edges[i].item()
            bin_end = hist_edges[i+1].item()
            count = hist_values[i].item()
            hist_info += f"\n  [{bin_start:.4f}, {bin_end:.4f}): {count}"
        return hist_info

    def _log_debug_info(self, name: str) -> None:
        """Log debug information including stack trace."""
        import traceback
        logger.debug(
            f"Stack trace for numerical instability in {name}:\n"
            f"{''.join(traceback.format_stack()[-5:-1])}"
        )

    def _replace_invalid_values(self, tensor: torch.Tensor, 
                              replacement_value: float) -> torch.Tensor:
        """Replace NaN/Inf values with the specified replacement value."""
        replacement = torch.ones_like(tensor) * replacement_value
        return torch.where(torch.isfinite(tensor), tensor, replacement)