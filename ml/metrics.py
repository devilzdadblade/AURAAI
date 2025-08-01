"""
Metrics tracking for reinforcement learning models.

This module provides classes for tracking and storing training metrics.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    
    loss: float
    q_value_mean: float
    q_value_std: float
    td_error_mean: float
    reward_mean: float
    step: int
    
    # Optional metrics
    policy_loss: Optional[float] = None
    value_loss: Optional[float] = None
    entropy: Optional[float] = None
    exploration_rate: Optional[float] = None
    gradient_norm: Optional[float] = None
    learning_rate: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "loss": self.loss,
            "q_value_mean": self.q_value_mean,
            "q_value_std": self.q_value_std,
            "td_error_mean": self.td_error_mean,
            "reward_mean": self.reward_mean,
            "step": self.step,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "entropy": self.entropy,
            "exploration_rate": self.exploration_rate,
            "gradient_norm": self.gradient_norm,
            "learning_rate": self.learning_rate
        }