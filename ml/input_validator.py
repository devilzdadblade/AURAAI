"""
Input validation system for reinforcement learning models.

This module provides robust input validation and sanitization for RL models,
ensuring that inputs meet expected shapes, types, and value ranges.
"""

import logging
import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of input validation."""
    
    is_valid: bool
    error_message: Optional[str] = None
    corrective_action: Optional[str] = None
    sanitized_input: Optional[Any] = None


class InputValidator:
    """
    Validates and sanitizes inputs to RL models.
    
    This class provides comprehensive validation for state inputs,
    ensuring they meet expected shapes, types, and value ranges.
    It can also attempt to sanitize inputs with minor issues.
    """

    ERROR_MSG_NAN_INF = "Input contains NaN/Inf values"
    
    def __init__(self, 
                 expected_shape: Optional[Tuple[int, ...]] = None,
                 value_range: Optional[Tuple[float, float]] = None,
                 allow_sanitization: bool = True,
                 strict_mode: bool = False):
        """
        Initialize the input validator.
        
        Args:
            expected_shape: Expected shape of input arrays/tensors
            value_range: Expected range of values (min, max)
            allow_sanitization: Whether to attempt sanitization of inputs
            strict_mode: If True, reject any input that doesn't exactly match expectations
        """
        self.expected_shape = expected_shape
        self.value_range = value_range
        self.allow_sanitization = allow_sanitization
        self.strict_mode = strict_mode
        
        # Track validation statistics
        self.validation_count = 0
        self.error_count = 0
        self.sanitization_count = 0
        
        logger.info(
            "InputValidator initialized with shape=%s, range=%s, sanitization=%s, strict=%s",
            expected_shape, value_range, allow_sanitization, strict_mode
        )
    
    def validate_state(self, state: np.ndarray) -> ValidationResult:
        """
        Validate a state array (refactored for lower complexity).
        
        Args:
            state: NumPy array representing the state
            
        Returns:
            ValidationResult with validation status and details
        """
        from .validation_strategies import (
            BasicPropertiesValidator, ShapeValidator, 
            ValueValidator, RangeValidator
        )
        
        self.validation_count += 1
        
        # Create validation pipeline
        validators = [
            BasicPropertiesValidator(self.allow_sanitization),
        ]
        
        if self.expected_shape is not None:
            validators.append(ShapeValidator(self.expected_shape, self.allow_sanitization))
        
        validators.append(ValueValidator(self.allow_sanitization))
        
        if self.value_range is not None:
            validators.append(RangeValidator(self.value_range, self.allow_sanitization))
        
        # Execute validation pipeline
        current_state = state
        for validator in validators:
            result = validator.validate(current_state)
            
            if not result.is_valid:
                self.error_count += 1
                return result
            
            if result.sanitized_input is not None:
                self.sanitization_count += 1
                return result
        
        # All checks passed
        return ValidationResult(is_valid=True)
        
    def _validate_basic_state_properties(self, state: np.ndarray) -> ValidationResult:
        """Validate the basic properties of the state array."""
        # Check if input is None
        if state is None:
            self.error_count += 1
            return ValidationResult(
                is_valid=False,
                error_message="Input state is None",
                corrective_action="Provide a valid state array"
            )
        
        # Check type
        if not isinstance(state, np.ndarray):
            self.error_count += 1
            return ValidationResult(
                is_valid=False,
                error_message=f"Input state must be a numpy array, got {type(state)}",
                corrective_action="Convert input to numpy array"
            )
            
        return ValidationResult(is_valid=True)
        
    def _validate_state_shape(self, state: np.ndarray) -> ValidationResult:
        """Validate the shape of the state array."""
        if state.shape != self.expected_shape:
            # Check if shape can be sanitized
            if self.allow_sanitization and len(state.shape) == len(self.expected_shape):
                try:
                    # Try to reshape or pad/truncate
                    sanitized_state = self._sanitize_shape(state)
                    self.sanitization_count += 1
                    return ValidationResult(
                        is_valid=True,
                        error_message=f"Input shape {state.shape} doesn't match expected {self.expected_shape}",
                        corrective_action="Sanitized to match expected shape",
                        sanitized_input=sanitized_state
                    )
                except (ValueError, IndexError, MemoryError):
                    # Shape sanitization failed - continue to error handling
                    pass
            
            self.error_count += 1
            return ValidationResult(
                is_valid=False,
                error_message=f"Input shape {state.shape} doesn't match expected {self.expected_shape}",
                corrective_action=f"Reshape input to {self.expected_shape}"
            )
            
        return ValidationResult(is_valid=True)
        
        if np.isnan(state).any() or np.isinf(state).any():
            if self.allow_sanitization:
                # Replace NaN/Inf with zeros
                sanitized_state = state.copy()
                sanitized_state[np.isnan(sanitized_state)] = 0.0
                sanitized_state[np.isinf(sanitized_state)] = 0.0
                self.sanitization_count += 1
                return ValidationResult(
                    is_valid=True,
                    error_message=self.ERROR_MSG_NAN_INF,
                    corrective_action="Replaced NaN/Inf values with zeros",
                    sanitized_input=sanitized_state
                )
            else:
                self.error_count += 1
                return ValidationResult(
                    is_valid=False,
                    error_message=self.ERROR_MSG_NAN_INF,
                    corrective_action="Remove NaN/Inf values from input"
                )
                
        return ValidationResult(is_valid=True)
        
    def _validate_value_range(self, state: np.ndarray) -> ValidationResult:
        """Validate that values in the state array are within the expected range."""
        min_val, max_val = self.value_range
        if np.min(state) < min_val or np.max(state) > max_val:
            if self.allow_sanitization:
                # Clip values to range
                sanitized_state = np.clip(state, min_val, max_val)
                self.sanitization_count += 1
                return ValidationResult(
                    is_valid=True,
                    error_message=f"Input values outside range [{min_val}, {max_val}]",
                    corrective_action="Clipped values to range",
                    sanitized_input=sanitized_state
                )
            else:
                self.error_count += 1
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Input values outside range [{min_val}, {max_val}]",
                    corrective_action="Ensure values are within range"
                )
                
        return ValidationResult(is_valid=True)
    
    def validate_tensor(self, tensor: torch.Tensor) -> ValidationResult:
        """
        Validate a PyTorch tensor.
        
        Args:
            tensor: PyTorch tensor to validate
            
        Returns:
            ValidationResult with validation status and details
        """
        self.validation_count += 1
        
        # Basic validation checks
        basic_result = self._validate_basic_tensor_properties(tensor)
        if not basic_result.is_valid:
            return basic_result
            
        # Shape validation
        if self.expected_shape is not None:
            shape_result = self._validate_tensor_shape(tensor)
            if not shape_result.is_valid or shape_result.sanitized_input is not None:
                return shape_result
                
        # Value validation (NaN/Inf)
        nan_inf_result = self._validate_tensor_values(tensor)
        if not nan_inf_result.is_valid or nan_inf_result.sanitized_input is not None:
            return nan_inf_result
            
        # Range validation
        if self.value_range is not None:
            range_result = self._validate_tensor_range(tensor)
            if not range_result.is_valid or range_result.sanitized_input is not None:
                return range_result
        
        # All checks passed
        return ValidationResult(is_valid=True)
        
    def _validate_basic_tensor_properties(self, tensor: torch.Tensor) -> ValidationResult:
        """Validate the basic properties of the tensor."""
        # Check if input is None
        if tensor is None:
            self.error_count += 1
            return ValidationResult(
                is_valid=False,
                error_message="Input tensor is None",
                corrective_action="Provide a valid tensor"
            )
        
        # Check type
        if not isinstance(tensor, torch.Tensor):
            self.error_count += 1
            return ValidationResult(
                is_valid=False,
                error_message=f"Input must be a PyTorch tensor, got {type(tensor)}",
                corrective_action="Convert input to PyTorch tensor"
            )
            
        return ValidationResult(is_valid=True)
        
    def _validate_tensor_shape(self, tensor: torch.Tensor) -> ValidationResult:
        """Validate the shape of the tensor."""
        if tensor.shape != self.expected_shape:
            # Check if shape can be sanitized
            if self.allow_sanitization and len(tensor.shape) == len(self.expected_shape):
                try:
                    # Try to reshape or pad/truncate
                    sanitized_tensor = self._sanitize_tensor_shape(tensor)
                    self.sanitization_count += 1
                    return ValidationResult(
                        is_valid=True,
                        error_message=f"Input shape {tensor.shape} doesn't match expected {self.expected_shape}",
                        corrective_action="Sanitized to match expected shape",
                        sanitized_input=sanitized_tensor
                    )
                except (ValueError, IndexError, RuntimeError):
                    # Tensor shape sanitization failed - continue to error handling
                    pass
            
            self.error_count += 1
            return ValidationResult(
                is_valid=False,
                error_message=f"Input shape {tensor.shape} doesn't match expected {self.expected_shape}",
                corrective_action=f"Reshape input to {self.expected_shape}"
            )
            
        return ValidationResult(is_valid=True)
        
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            if self.allow_sanitization:
                # Replace NaN/Inf with zeros
                sanitized_tensor = tensor.clone()
                sanitized_tensor[torch.isnan(sanitized_tensor)] = 0.0
                sanitized_tensor[torch.isinf(sanitized_tensor)] = 0.0
                self.sanitization_count += 1
                return ValidationResult(
                    is_valid=True,
                    error_message=self.ERROR_MSG_NAN_INF,
                    corrective_action="Replaced NaN/Inf values with zeros",
                    sanitized_input=sanitized_tensor
                )
            else:
                self.error_count += 1
                return ValidationResult(
                    is_valid=False,
                    error_message=self.ERROR_MSG_NAN_INF,
                    corrective_action="Remove NaN/Inf values from input"
                )
                
        return ValidationResult(is_valid=True)
        
    def _validate_tensor_range(self, tensor: torch.Tensor) -> ValidationResult:
        """Validate that values in the tensor are within the expected range."""
        min_val, max_val = self.value_range
        if torch.min(tensor).item() < min_val or torch.max(tensor).item() > max_val:
            if self.allow_sanitization:
                # Clip values to range
                sanitized_tensor = torch.clamp(tensor, min_val, max_val)
                self.sanitization_count += 1
                return ValidationResult(
                    is_valid=True,
                    error_message=f"Input values outside range [{min_val}, {max_val}]",
                    corrective_action=f"Clipped values to range [{min_val}, {max_val}]",
                    sanitized_input=sanitized_tensor
                )
            else:
                self.error_count += 1
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Input values outside range [{min_val}, {max_val}]",
                    corrective_action="Ensure values are within range"
                )
                
        return ValidationResult(is_valid=True)
    
    def sanitize_input(self, state: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Attempt to sanitize input to meet requirements.
        
        Args:
            state: Input state array or tensor
            
        Returns:
            Sanitized state array or tensor
            
        Raises:
            ValueError: If sanitization is not possible
        """
        if not self.allow_sanitization:
            raise ValueError("Sanitization is disabled")
        
        if isinstance(state, np.ndarray):
            result = self.validate_state(state)
            if result.is_valid and result.sanitized_input is not None:
                return result.sanitized_input
            elif result.is_valid:
                return state
        elif isinstance(state, torch.Tensor):
            result = self.validate_tensor(state)
            if result.is_valid and result.sanitized_input is not None:
                return result.sanitized_input
            elif result.is_valid:
                return state
        
        # If we get here, sanitization failed
        raise ValueError(f"Cannot sanitize input: {result.error_message}")
    
    def _sanitize_shape(self, state: np.ndarray) -> np.ndarray:
        """
        Sanitize array shape to match expected shape.
        
        Args:
            state: Input state array
            
        Returns:
            Sanitized state array
            
        Raises:
            ValueError: If sanitization is not possible
        """
        if self.expected_shape is None:
            return state
        
        # Check if dimensions match
        if len(state.shape) != len(self.expected_shape):
            raise ValueError(f"Cannot sanitize shape {state.shape} to {self.expected_shape}")
        
        # Create sanitized array
        sanitized = np.zeros(self.expected_shape, dtype=state.dtype)
        
        # Copy data with shape constraints
        for idx in range(len(self.expected_shape)):
            copy_size = min(state.shape[idx], self.expected_shape[idx])
            
            # Create slices for each dimension
            src_slices = tuple(slice(0, state.shape[i]) if i != idx else slice(0, copy_size) 
                              for i in range(len(state.shape)))
            dst_slices = tuple(slice(0, self.expected_shape[i]) if i != idx else slice(0, copy_size) 
                              for i in range(len(self.expected_shape)))
            
            # Copy partial data
            if idx == 0:
                sanitized[dst_slices] = state[src_slices]
        
        return sanitized
    
    def _sanitize_tensor_shape(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Sanitize tensor shape to match expected shape.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Sanitized tensor
            
        Raises:
            ValueError: If sanitization is not possible
        """
        if self.expected_shape is None:
            return tensor
        
        # Check if dimensions match
        if len(tensor.shape) != len(self.expected_shape):
            raise ValueError(f"Cannot sanitize shape {tensor.shape} to {self.expected_shape}")
        
        # Create sanitized tensor
        sanitized = torch.zeros(self.expected_shape, dtype=tensor.dtype, device=tensor.device)
        
        # Copy data with shape constraints
        for idx in range(len(self.expected_shape)):
            copy_size = min(tensor.shape[idx], self.expected_shape[idx])
            
            # Create slices for each dimension
            src_slices = tuple(slice(0, tensor.shape[i]) if i != idx else slice(0, copy_size) 
                              for i in range(len(tensor.shape)))
            dst_slices = tuple(slice(0, self.expected_shape[i]) if i != idx else slice(0, copy_size) 
                              for i in range(len(self.expected_shape)))
            
            # Copy partial data
            if idx == 0:
                sanitized[dst_slices] = tensor[src_slices]
        
        return sanitized
    
    def get_validation_stats(self) -> Dict[str, int]:
        """
        Get validation statistics.
        
        Returns:
            Dictionary with validation statistics
        """
        return {
            "validation_count": self.validation_count,
            "error_count": self.error_count,
            "sanitization_count": self.sanitization_count,
            "error_rate": self.error_count / max(1, self.validation_count),
            "sanitization_rate": self.sanitization_count / max(1, self.validation_count)
        }
    
    def reset_stats(self) -> None:
        """Reset validation statistics."""
        self.validation_count = 0
        self.error_count = 0
        self.sanitization_count = 0