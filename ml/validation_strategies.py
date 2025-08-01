"""Validation strategies for ML input validation."""
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any
import numpy as np
import torch
from dataclasses import dataclass

@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    error_message: Optional[str] = None
    corrective_action: Optional[str] = None
    sanitized_input: Optional[Any] = None


class ValidationStrategy(ABC):
    """Base class for validation strategies."""
    
    def __init__(self, allow_sanitization: bool = False):
        self.allow_sanitization = allow_sanitization
    
    @abstractmethod
    def validate(self, data: Any) -> ValidationResult:
        """Validate the input data."""
        pass


class BasicPropertiesValidator(ValidationStrategy):
    """Validates basic properties of input data."""
    
    def validate(self, state: np.ndarray) -> ValidationResult:
        # Check if input is None
        if state is None:
            return ValidationResult(
                is_valid=False,
                error_message="Input state is None",
                corrective_action="Provide a valid state array"
            )
        
        # Check type
        if not isinstance(state, np.ndarray):
            return ValidationResult(
                is_valid=False,
                error_message=f"Input state must be a numpy array, got {type(state)}",
                corrective_action="Convert input to numpy array"
            )
            
        return ValidationResult(is_valid=True)


class ShapeValidator(ValidationStrategy):
    """Validates array shape."""
    
    def __init__(self, expected_shape: Tuple, allow_sanitization: bool = False):
        super().__init__(allow_sanitization)
        self.expected_shape = expected_shape
    
    def validate(self, state: np.ndarray) -> ValidationResult:
        if state.shape != self.expected_shape:
            if self.allow_sanitization and len(state.shape) == len(self.expected_shape):
                try:
                    sanitized_state = self._sanitize_shape(state)
                    return ValidationResult(
                        is_valid=True,
                        error_message=f"Input shape {state.shape} doesn't match expected {self.expected_shape}",
                        corrective_action="Sanitized to match expected shape",
                        sanitized_input=sanitized_state
                    )
                except Exception:
                    pass
            
            return ValidationResult(
                is_valid=False,
                error_message=f"Input shape {state.shape} doesn't match expected {self.expected_shape}",
                corrective_action=f"Reshape input to {self.expected_shape}"
            )
        
        return ValidationResult(is_valid=True)
    
    def _sanitize_shape(self, state: np.ndarray) -> np.ndarray:
        """Attempt to sanitize the shape by reshaping or padding/truncating."""
        # Simple reshape attempt
        return state.reshape(self.expected_shape)


class ValueValidator(ValidationStrategy):
    """Validates array values for NaN/Inf."""
    
    def validate(self, state: np.ndarray) -> ValidationResult:
        if np.isnan(state).any() or np.isinf(state).any():
            if self.allow_sanitization:
                sanitized_state = state.copy()
                sanitized_state[np.isnan(sanitized_state)] = 0.0
                sanitized_state[np.isinf(sanitized_state)] = 0.0
                return ValidationResult(
                    is_valid=True,
                    error_message="Input contains NaN/Inf values",
                    corrective_action="Replaced NaN/Inf values with zeros",
                    sanitized_input=sanitized_state
                )
            else:
                return ValidationResult(
                    is_valid=False,
                    error_message="Input contains NaN/Inf values",
                    corrective_action="Remove NaN/Inf values from input"
                )
        
        return ValidationResult(is_valid=True)


class RangeValidator(ValidationStrategy):
    """Validates value ranges."""
    
    def __init__(self, value_range: Tuple[float, float], allow_sanitization: bool = False):
        super().__init__(allow_sanitization)
        self.value_range = value_range
    
    def validate(self, state: np.ndarray) -> ValidationResult:
        min_val, max_val = self.value_range
        if np.min(state) < min_val or np.max(state) > max_val:
            if self.allow_sanitization:
                sanitized_state = np.clip(state, min_val, max_val)
                return ValidationResult(
                    is_valid=True,
                    error_message=f"Input values outside range [{min_val}, {max_val}]",
                    corrective_action="Clipped values to range",
                    sanitized_input=sanitized_state
                )
            else:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Input values outside range [{min_val}, {max_val}]",
                    corrective_action="Ensure values are within range"
                )
        
        return ValidationResult(is_valid=True)
