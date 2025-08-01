"""
Error handling system for reinforcement learning models.

This module provides comprehensive error handling and recovery mechanisms
for RL models, including contextual error recovery, detailed logging,
and error escalation paths.
"""

import logging
import traceback
import time
import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import sys
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Constants for history management
MAX_ERROR_HISTORY_SIZE = 1000


class ErrorSeverity(Enum):
    """Enumeration of error severity levels."""
    INFO = 0
    WARNING = 1
    ERROR = 2
    CRITICAL = 3


@dataclass
class ErrorContext:
    """Context information about an error."""
    
    error_type: str  # Type of error (e.g., 'numerical', 'input_validation')
    error_message: str  # Error message
    component: str  # Component where the error occurred
    severity: ErrorSeverity  # Severity level
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    stack_trace: Optional[str] = None  # Stack trace if available
    input_state: Optional[Any] = None  # Input state that caused the error
    additional_info: Dict[str, Any] = field(default_factory=dict)  # Additional context


class ErrorHandler:
    """
    Handles errors in RL models with contextual recovery.
    
    This class provides comprehensive error handling, including
    detailed logging, recovery mechanisms, and error escalation.
    """
    
    def __init__(self, 
                 model_health_monitor=None,
                 fallback_manager=None,
                 max_consecutive_errors: int = 5,
                 error_cooldown_sec: float = 60.0):
        """
        Initialize the error handler.
        
        Args:
            model_health_monitor: Optional ModelHealthMonitor instance
            fallback_manager: Optional FallbackManager instance
            max_consecutive_errors: Maximum number of consecutive errors before escalation
            error_cooldown_sec: Cooldown period between error count resets
        """
        self.model_health_monitor = model_health_monitor
        self.fallback_manager = fallback_manager
        self.max_consecutive_errors = max_consecutive_errors
        self.error_cooldown_sec = error_cooldown_sec
        
        # Error tracking
        self.error_history = []
        self.consecutive_errors = 0
        self.last_error_time = None
        self.error_counts = {
            ErrorSeverity.INFO: 0,
            ErrorSeverity.WARNING: 0,
            ErrorSeverity.ERROR: 0,
            ErrorSeverity.CRITICAL: 0
        }
        
        # Recovery tracking
        self.recovery_attempts = []
        
        logger.info(
            "ErrorHandler initialized with max_consecutive_errors=%d, error_cooldown_sec=%.1f",
            max_consecutive_errors, error_cooldown_sec
        )
    
    def handle_error(self, 
                    error: Exception, 
                    component: str,
                    severity: ErrorSeverity = ErrorSeverity.ERROR,
                    input_state: Optional[Any] = None,
                    additional_info: Optional[Dict[str, Any]] = None) -> ErrorContext:
        """
        Handle an error with detailed logging and tracking.
        
        Args:
            error: Exception that occurred
            component: Component where the error occurred
            severity: Severity level of the error
            input_state: Optional input state that caused the error
            additional_info: Optional additional context information
            
        Returns:
            ErrorContext with error details
        """
        # Create error context
        error_context = ErrorContext(
            error_type=error.__class__.__name__,
            error_message=str(error),
            component=component,
            severity=severity,
            stack_trace=traceback.format_exc(),
            input_state=input_state,
            additional_info=additional_info or {}
        )
        
        # Update error tracking
        self._update_error_tracking(error_context)
        
        # Log the error with appropriate severity
        self._log_error(error_context)
        
        # Record error in model health monitor if available
        if self.model_health_monitor is not None:
            self.model_health_monitor.record_error(
                error_context.error_type,
                {
                    "message": error_context.error_message,
                    "component": error_context.component,
                    "severity": error_context.severity.name,
                    "timestamp": error_context.timestamp.isoformat()
                }
            )
        
        # Check for escalation
        self._check_for_escalation(error_context)
        
        return error_context
    
    def try_execute(self, 
                   func: Callable, 
                   component: str,
                   args: Tuple = (),
                   kwargs: Dict[str, Any] = None,
                   fallback_func: Optional[Callable] = None,
                   severity: ErrorSeverity = ErrorSeverity.ERROR) -> Tuple[Any, Optional[ErrorContext]]:
        """
        Execute a function with error handling.
        
        Args:
            func: Function to execute
            component: Component name for error tracking
            args: Arguments to pass to the function
            kwargs: Keyword arguments to pass to the function
            fallback_func: Optional fallback function to call if the main function fails
            severity: Severity level if an error occurs
            
        Returns:
            Tuple of (result, error_context)
            If successful, error_context will be None
            If an error occurs and fallback_func is provided, result will be from fallback_func
            If an error occurs and no fallback_func is provided, result will be None
        """
        kwargs = kwargs or {}
        
        try:
            result = func(*args, **kwargs)
            return result, None
        except Exception as e:
            # Handle the error
            error_context = self.handle_error(
                error=e,
                component=component,
                severity=severity,
                input_state=args[0] if args else None,
                additional_info={"function": func.__name__}
            )
            
            # Try fallback if provided
            if fallback_func is not None:
                try:
                    fallback_result = fallback_func(*args, **kwargs)
                    
                    # Record successful recovery
                    self._record_recovery(
                        strategy=f"fallback_to_{fallback_func.__name__}",
                        success=True,
                        error_context=error_context
                    )
                    
                    return fallback_result, error_context
                except Exception as fallback_error:
                    # Fallback also failed
                    fallback_error_context = self.handle_error(
                        error=fallback_error,
                        component=f"{component}_fallback",
                        severity=ErrorSeverity.CRITICAL,
                        input_state=args[0] if args else None,
                        additional_info={"function": fallback_func.__name__}
                    )
                    
                    # Record failed recovery
                    self._record_recovery(
                        strategy=f"fallback_to_{fallback_func.__name__}",
                        success=False,
                        error_context=error_context
                    )
                    
                    return None, fallback_error_context
            
            return None, error_context
    
    def safe_tensor_operation(self,
                             operation: Callable,
                             tensors: List[torch.Tensor],
                             component: str,
                             default_value: Optional[Union[float, torch.Tensor]] = None) -> torch.Tensor:
        """
        Safely perform a tensor operation with error handling.
        
        Args:
            operation: Function that performs the tensor operation
            tensors: List of input tensors
            component: Component name for error tracking
            default_value: Default value to return if operation fails
            
        Returns:
            Result of the operation, or default_value if the operation fails
        """
        try:
            result = operation(*tensors)
            
            # Check for NaN/Inf values
            if torch.isnan(result).any() or torch.isinf(result).any():
                raise ValueError("Operation produced NaN/Inf values")
            
            return result
        except Exception as e:
            # Handle the error
            self.handle_error(
                error=e,
                component=component,
                severity=ErrorSeverity.WARNING,
                additional_info={"operation": operation.__name__}
            )
            
            # Return default value if provided
            if default_value is not None:
                if isinstance(default_value, float):
                    return torch.full_like(tensors[0], default_value)
                else:
                    return default_value
            
            # Otherwise, return zeros
            return torch.zeros_like(tensors[0])
    
    def _update_error_tracking(self, error_context: ErrorContext) -> None:
        """
        Update error tracking metrics.
        
        Args:
            error_context: Error context to track
        """
        # Add to error history
        self.error_history.append(error_context)
        
        # Keep history bounded
        if len(self.error_history) > MAX_ERROR_HISTORY_SIZE:
            self.error_history = self.error_history[-MAX_ERROR_HISTORY_SIZE:]
        
        # Update error counts
        self.error_counts[error_context.severity] += 1
        
        # Update consecutive error tracking
        current_time = time.time()
        
        if self.last_error_time is None or (current_time - self.last_error_time) > self.error_cooldown_sec:
            # Reset consecutive errors if cooldown period has passed
            self.consecutive_errors = 1
        else:
            self.consecutive_errors += 1
        
        self.last_error_time = current_time
    
    def _log_error(self, error_context: ErrorContext) -> None:
        """
        Log an error with appropriate severity.
        
        Args:
            error_context: Error context to log
        """
        log_message = (
            f"[{error_context.component}] {error_context.error_type}: {error_context.error_message} "
            f"(consecutive={self.consecutive_errors})"
        )
        
        # Log with appropriate severity
        if error_context.severity == ErrorSeverity.INFO:
            logger.info(log_message)
        elif error_context.severity == ErrorSeverity.WARNING:
            logger.warning(log_message)
        elif error_context.severity == ErrorSeverity.ERROR:
            logger.error(log_message)
            logger.error(f"Stack trace: {error_context.stack_trace}")
        elif error_context.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
            logger.critical(f"Stack trace: {error_context.stack_trace}")
            
            # Log additional details for critical errors
            if error_context.input_state is not None:
                try:
                    if isinstance(error_context.input_state, np.ndarray):
                        logger.critical(
                            f"Input state: shape={error_context.input_state.shape}, "
                            f"min={np.min(error_context.input_state)}, "
                            f"max={np.max(error_context.input_state)}, "
                            f"has_nan={np.isnan(error_context.input_state).any()}, "
                            f"has_inf={np.isinf(error_context.input_state).any()}"
                        )
                    elif isinstance(error_context.input_state, torch.Tensor):
                        logger.critical(
                            f"Input tensor: shape={error_context.input_state.shape}, "
                            f"min={torch.min(error_context.input_state).item()}, "
                            f"max={torch.max(error_context.input_state).item()}, "
                            f"has_nan={torch.isnan(error_context.input_state).any().item()}, "
                            f"has_inf={torch.isinf(error_context.input_state).any().item()}, "
                            f"device={error_context.input_state.device}"
                        )
                except Exception as e:
                    logger.critical(f"Error logging input state: {e}")
    
    def _check_for_escalation(self, error_context: ErrorContext) -> None:
        """
        Check if error should trigger escalation.
        
        Args:
            error_context: Error context to check
        """
        # Automatic escalation for critical errors
        if error_context.severity == ErrorSeverity.CRITICAL and self.fallback_manager is not None:
            logger.critical(
                "Critical error triggered automatic fallback escalation: %s",
                error_context.error_message
            )
            self.fallback_manager.escalate_fallback(f"Critical error: {error_context.error_type}")
            return
        
        # Escalation based on consecutive errors
        if self.consecutive_errors >= self.max_consecutive_errors and self.fallback_manager is not None:
            logger.error(
                "Maximum consecutive errors (%d) reached, escalating fallback",
                self.max_consecutive_errors
            )
            self.fallback_manager.escalate_fallback(f"Max consecutive errors: {self.consecutive_errors}")
    
    def _record_recovery(self, strategy: str, success: bool, error_context: ErrorContext) -> None:
        """
        Record a recovery attempt.
        
        Args:
            strategy: Recovery strategy used
            success: Whether the recovery was successful
            error_context: Error context that triggered the recovery
        """
        recovery_record = {
            "timestamp": datetime.datetime.now(),
            "strategy": strategy,
            "success": success,
            "error_type": error_context.error_type,
            "error_message": error_context.error_message,
            "component": error_context.component
        }
        
        self.recovery_attempts.append(recovery_record)
        
        # Keep history bounded
        if len(self.recovery_attempts) > MAX_ERROR_HISTORY_SIZE:
            self.recovery_attempts = self.recovery_attempts[-MAX_ERROR_HISTORY_SIZE:]
        
        # Record in model health monitor if available
        if self.model_health_monitor is not None:
            self.model_health_monitor.record_recovery_attempt(
                strategy=strategy,
                success=success,
                details={
                    "error_type": error_context.error_type,
                    "component": error_context.component
                }
            )
    
    def get_error_stats(self) -> Dict[str, Any]:
        """
        Get error statistics.
        
        Returns:
            Dictionary with error statistics
        """
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_counts": {k.name: v for k, v in self.error_counts.items()},
            "consecutive_errors": self.consecutive_errors,
            "recovery_attempts": len(self.recovery_attempts),
            "successful_recoveries": sum(1 for r in self.recovery_attempts if r["success"]),
            "failed_recoveries": sum(1 for r in self.recovery_attempts if not r["success"])
        }
    
    def reset_consecutive_errors(self) -> None:
        """Reset consecutive error counter."""
        self.consecutive_errors = 0
        self.last_error_time = None