"""
Parallel component updater for reinforcement learning models.

This module provides tools for concurrent optimization of multiple model components,
improving training performance by parallelizing optimizer updates.
"""

import logging
import math
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple, Callable, Any, List

import torch
import torch.nn as nn
import torch.optim as optim

from src.ml.gradient_validator import GradientValidator

logger = logging.getLogger(__name__)


class ParallelComponentUpdater:
    """
    Enables parallel updates for multiple model components using ThreadPoolExecutor.
    
    This class provides methods for:
    - Concurrent optimization of multiple model components
    - Proper error handling and result collection from parallel updates
    - Efficient resource utilization during training
    """
    
    def __init__(self, components: Dict[str, Tuple[nn.Module, optim.Optimizer]], max_workers: int = None):
        """
        Initialize the ParallelComponentUpdater.
        
        Args:
            components: Dictionary mapping component names to (model, optimizer) tuples
            max_workers: Maximum number of worker threads (default: number of components)
        """
        self.components = components
        self.max_workers = max_workers or len(components)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.gradient_validator = GradientValidator(max_grad_norm=1.0)
        
        logger.info(f"ParallelComponentUpdater initialized with {len(components)} components "
                   f"and {self.max_workers} workers")
        
    def update_component(self, 
                        name: str, 
                        model: nn.Module, 
                        optimizer: optim.Optimizer, 
                        loss_fn: Callable, 
                        *args, **kwargs) -> Tuple[str, float, Any]:
        """
        Update a single component with error handling.
        
        Args:
            name: Component name for logging
            model: PyTorch model to update
            optimizer: PyTorch optimizer for the model
            loss_fn: Loss function that computes the component's loss
            *args, **kwargs: Arguments to pass to the loss function
            
        Returns:
            Tuple of (component_name, loss_value, additional_info)
        """
        try:
            # Clear gradients
            optimizer.zero_grad()
            
            # Compute loss
            loss = loss_fn(*args, **kwargs)
            
            # Skip update if loss is not a tensor or is zero
            if not isinstance(loss, torch.Tensor) or math.isclose(loss.item(), 0.0, abs_tol=1e-9):
                return name, 0.0, {"status": "skipped", "reason": "zero_loss"}
            
            # Backpropagate
            loss.backward()
            
            # Validate gradients
            validation_result = self.gradient_validator.validate_gradients(model)
            
            if validation_result.is_valid:
                # Clip gradients
                grad_norm = self.gradient_validator.clip_gradients(model, max_norm=1.0)
                
                # Update parameters
                optimizer.step()
                
                return name, loss.item(), {
                    "status": "success", 
                    "grad_norm": grad_norm,
                    "validation": "passed"
                }
            else:
                # Log error and skip update
                logger.error(f"Skipping {name} optimizer step due to invalid gradients: "
                           f"{validation_result.error_message}")
                
                # Log detailed gradient statistics for debugging
                grad_stats = self.gradient_validator.log_gradient_stats(model)
                logger.error(f"{name} gradient stats: mean={grad_stats.mean_grad_norm:.6f}, "
                           f"max={grad_stats.max_grad_norm:.6f}, "
                           f"NaN={grad_stats.num_params_with_nan}, "
                           f"Inf={grad_stats.num_params_with_inf}")
                
                return name, loss.item(), {
                    "status": "error", 
                    "reason": "invalid_gradients",
                    "validation": "failed",
                    "error_message": validation_result.error_message
                }
                
        except Exception as e:
            # Log exception
            logger.error(f"Error updating {name}: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            
            return name, 0.0, {
                "status": "error", 
                "reason": "exception", 
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def update_all(self, loss_fns: Dict[str, Callable]) -> Dict[str, Any]:
        """
        Update all components in parallel using ThreadPoolExecutor.
        
        Args:
            loss_fns: Dictionary mapping component names to loss functions
            
        Returns:
            Dictionary with update results for each component
        """
        # Validate that all components have corresponding loss functions
        missing_components = set(self.components.keys()) - set(loss_fns.keys())
        if missing_components:
            logger.warning(f"Missing loss functions for components: {missing_components}")
        
        # Submit tasks to the executor
        futures = {}
        for name, (model, optimizer) in self.components.items():
            if name in loss_fns:
                future = self.executor.submit(
                    self.update_component,
                    name, model, optimizer, loss_fns[name]
                )
                futures[future] = name
        
        # Collect results
        results = {}
        for future in as_completed(futures):
            name = futures[future]
            try:
                component_name, loss_value, info = future.result()
                results[component_name] = {
                    "loss": loss_value,
                    "info": info
                }
            except Exception as e:
                logger.error(f"Exception in component {name}: {str(e)}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                results[name] = {
                    "loss": 0.0,
                    "info": {
                        "status": "error",
                        "reason": "future_exception",
                        "error": str(e)
                    }
                }
        
        return results
    
    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)
        logger.info("ParallelComponentUpdater shutdown complete")