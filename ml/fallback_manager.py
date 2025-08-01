"""
Hierarchical fallback system for reinforcement learning models.

This module provides a fallback management system that can progressively
escalate through different fallback strategies when model issues are detected.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable
import datetime
import numpy as np

# Create a random number generator for modern numpy random usage with fixed seed for reproducibility
rng = np.random.default_rng(42)  # Fixed seed 42 for reproducible results
import torch

logger = logging.getLogger(__name__)


class FallbackLevel(Enum):
    """Enumeration of fallback strategy levels."""
    NORMAL = 0  # Normal operation, no fallback
    SIMPLIFIED_RL = 1  # Use simplified RL model
    RULE_BASED = 2  # Use rule-based strategy
    CONSERVATIVE = 3  # Use conservative strategy (minimal trading)
    HALT = 4  # Halt all trading


@dataclass
class FailureContext:
    """Context information about the failure that triggered fallback."""
    
    error_type: str  # Type of error (e.g., 'numerical', 'input_validation')
    error_message: str  # Error message
    component: str  # Component where the error occurred
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    stack_trace: Optional[str] = None  # Stack trace if available
    input_state: Optional[Any] = None  # Input state that caused the error
    additional_info: Dict[str, Any] = field(default_factory=dict)  # Additional context


@dataclass
class FallbackAction:
    """Action recommended by a fallback strategy."""
    
    action_values: np.ndarray  # Action values (e.g., Q-values)
    selected_action: int  # Selected action index
    confidence: float  # Confidence in the action (0.0 to 1.0)
    strategy_level: FallbackLevel  # Strategy that produced this action
    is_safe: bool = True  # Whether the action is considered safe
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


@dataclass
class FallbackEffectiveness:
    """Tracking information about fallback strategy effectiveness."""
    
    strategy_level: FallbackLevel
    activation_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_recovery_time_sec: float = 0.0
    last_activation: Optional[datetime.datetime] = None
    cumulative_reward: float = 0.0


class FallbackStrategy(ABC):
    """Abstract base class for fallback strategies."""
    
    def __init__(self, level: FallbackLevel):
        self.level = level
        self.effectiveness = FallbackEffectiveness(strategy_level=level)
    
    @abstractmethod
    def get_action(self, state: np.ndarray, failure_context: FailureContext) -> FallbackAction:
        """
        Get a fallback action for the given state and failure context.
        
        Args:
            state: Current environment state
            failure_context: Context about the failure
            
        Returns:
            FallbackAction with recommended action
        """
        pass
    
    def record_outcome(self, success: bool, reward: float = 0.0) -> None:
        """
        Record the outcome of using this fallback strategy.
        
        Args:
            success: Whether the fallback action was successful
            reward: Reward received from the action
        """
        self.effectiveness.activation_count += 1
        if success:
            self.effectiveness.success_count += 1
        else:
            self.effectiveness.failure_count += 1
        
        self.effectiveness.cumulative_reward += reward
        self.effectiveness.last_activation = datetime.datetime.now()


class SimplifiedRLStrategy(FallbackStrategy):
    """
    Fallback to a simplified RL model with reduced complexity.
    
    This strategy uses a smaller, more stable version of the RL model
    that sacrifices some performance for reliability.
    """
    
    def __init__(self, simplified_model=None):
        super().__init__(FallbackLevel.SIMPLIFIED_RL)
        self.simplified_model = simplified_model
    
    def get_action(self, state: np.ndarray, failure_context: FailureContext) -> FallbackAction:
        """Get action from simplified RL model."""
        try:
            if self.simplified_model is None:
                # If no simplified model is provided, return a random action
                # with low confidence
                action_values = rng.normal(0, 0.1, (len(state), ))
                selected_action = np.argmax(action_values)
                return FallbackAction(
                    action_values=action_values,
                    selected_action=selected_action,
                    confidence=0.3,
                    strategy_level=self.level,
                    metadata={"method": "random_fallback"}
                )
            
            # Use the simplified model to get an action
            with torch.no_grad():
                if isinstance(state, np.ndarray):
                    state_tensor = torch.from_numpy(state).float()
                else:
                    state_tensor = state
                
                action_values = self.simplified_model(state_tensor).cpu().numpy()
                selected_action = np.argmax(action_values)
                
                return FallbackAction(
                    action_values=action_values,
                    selected_action=selected_action,
                    confidence=0.7,
                    strategy_level=self.level,
                    metadata={"method": "simplified_model"}
                )
        except Exception as e:
            logger.error(f"SimplifiedRLStrategy failed: {str(e)}")
            # If the simplified model fails, escalate to the next level
            # by returning a very low confidence action
            action_values = np.zeros((len(state), ))
            return FallbackAction(
                action_values=action_values,
                selected_action=0,  # Default to first action
                confidence=0.0,  # Zero confidence will trigger escalation
                strategy_level=self.level,
                is_safe=False,
                metadata={"error": str(e), "method": "simplified_model_failed"}
            )


class RuleBasedStrategy(FallbackStrategy):
    """
    Rule-based fallback strategy using predefined heuristics.
    
    This strategy uses simple, robust rules to make decisions when
    the RL models are not functioning properly.
    """
    
    def __init__(self, rules: Optional[Dict[str, Callable]] = None):
        super().__init__(FallbackLevel.RULE_BASED)
        self.rules = rules or {}
    
    def get_action(self, state: np.ndarray, failure_context: FailureContext) -> FallbackAction:
        """Get action based on predefined rules."""
        try:
            # Default to a neutral action if no rules match
            action_values = np.zeros((len(state), ))
            selected_action = 0  # Default to first action (often "hold")
            confidence = 0.5
            
            # Check if we have a rule for this specific error type
            error_type = failure_context.error_type
            if error_type in self.rules:
                rule_result = self.rules[error_type](state, failure_context)
                if rule_result is not None:
                    action_values, selected_action, confidence = rule_result
            
            return FallbackAction(
                action_values=action_values,
                selected_action=selected_action,
                confidence=confidence,
                strategy_level=self.level,
                metadata={"method": f"rule_{error_type}"}
            )
        except Exception as e:
            logger.error(f"RuleBasedStrategy failed: {str(e)}")
            # If rule-based strategy fails, return a conservative action
            # with low confidence
            action_values = np.zeros((len(state), ))
            return FallbackAction(
                action_values=action_values,
                selected_action=0,  # Default to first action
                confidence=0.1,  # Very low confidence will likely trigger escalation
                strategy_level=self.level,
                is_safe=False,
                metadata={"error": str(e), "method": "rule_based_failed"}
            )


class ConservativeStrategy(FallbackStrategy):
    """
    Conservative fallback strategy that minimizes risk.
    
    This strategy takes minimal-risk actions, such as reducing position sizes
    or holding current positions without taking new ones.
    """
    
    def __init__(self):
        super().__init__(FallbackLevel.CONSERVATIVE)
    
    def get_action(self, state: np.ndarray, failure_context: FailureContext) -> FallbackAction:
        """Get conservative action (usually hold or minimal position)."""
        try:
            # In a conservative strategy, we typically default to "hold"
            # which is often represented by zeros or a specific action index
            action_values = np.zeros((len(state), ))
            
            # Assuming action 0 is the most conservative (e.g., "hold")
            # This would need to be adjusted based on the actual action space
            selected_action = 0
            
            return FallbackAction(
                action_values=action_values,
                selected_action=selected_action,
                confidence=0.9,  # High confidence in this conservative action
                strategy_level=self.level,
                metadata={"method": "conservative_hold"}
            )
        except Exception as e:
            logger.error(f"ConservativeStrategy failed: {str(e)}")
            # If even the conservative strategy fails, we need to halt
            action_values = np.zeros((len(state), ))
            return FallbackAction(
                action_values=action_values,
                selected_action=0,
                confidence=0.0,  # Zero confidence will trigger escalation to HALT
                strategy_level=self.level,
                is_safe=False,
                metadata={"error": str(e), "method": "conservative_failed"}
            )


class TradingHaltStrategy(FallbackStrategy):
    """
    Emergency fallback that halts all trading activity.
    
    This is the final fallback level when all other strategies have failed.
    It stops all trading and may initiate position liquidation if configured.
    """
    
    def __init__(self, liquidate_positions: bool = False):
        super().__init__(FallbackLevel.HALT)
        self.liquidate_positions = liquidate_positions
    
    def get_action(self, state: np.ndarray, failure_context: FailureContext) -> FallbackAction:
        """Get halt action (no trading or liquidate)."""
        # For a halt strategy, we always return the same action
        # regardless of the state
        action_values = np.zeros((len(state), ))
        
        # The selected action depends on whether we want to liquidate positions
        selected_action = 1 if self.liquidate_positions else 0  # Assuming 1 might be "liquidate"
        
        logger.critical(
            "TRADING HALT ACTIVATED: %s - %s",
            failure_context.error_type,
            failure_context.error_message
        )
        
        return FallbackAction(
            action_values=action_values,
            selected_action=selected_action,
            confidence=1.0,  # Maximum confidence as this is our last resort
            strategy_level=self.level,
            metadata={
                "method": "trading_halt",
                "liquidate_positions": self.liquidate_positions
            }
        )


class FallbackManager:
    """
    Manages hierarchical fallback strategies for RL models.
    
    This class coordinates multiple fallback strategies at different levels,
    escalating through them as needed when issues are detected.
    """
    
    def __init__(self, 
                 simplified_model=None,
                 rule_set: Optional[Dict[str, Callable]] = None,
                 liquidate_on_halt: bool = True,
                 confidence_threshold: float = 0.2):
        """
        Initialize the fallback manager.
        
        Args:
            simplified_model: Optional simplified RL model for first-level fallback
            rule_set: Optional dictionary of rule functions for rule-based fallback
            liquidate_on_halt: Whether to liquidate positions when halting
            confidence_threshold: Threshold below which to escalate fallback level
        """
        # Initialize fallback strategies
        self.fallback_strategies = [
            SimplifiedRLStrategy(simplified_model),
            RuleBasedStrategy(rule_set),
            ConservativeStrategy(),
            TradingHaltStrategy(liquidate_on_halt)
        ]
        
        self.confidence_threshold = confidence_threshold
        self.current_level = FallbackLevel.NORMAL
        self.activation_history = []
        self.recovery_attempts = 0
        self.last_escalation_time = None
        self.backoff_factor = 1.0  # For exponential backoff
        
        logger.info("FallbackManager initialized with %d strategies", len(self.fallback_strategies))
    
    def get_fallback_action(self, 
                           state: np.ndarray, 
                           failure_context: FailureContext) -> FallbackAction:
        """
        Get a fallback action for the given state and failure context.
        
        This method will try strategies starting from the current fallback level,
        escalating if necessary until a suitable action is found.
        
        Args:
            state: Current environment state
            failure_context: Context about the failure
            
        Returns:
            FallbackAction with recommended action
        """
        # Record this activation
        self._record_activation(failure_context)
        
        # Start with the current fallback level
        current_index = self.current_level.value
        if current_index == 0:  # If we're in NORMAL mode, start with first fallback
            current_index = 1
        
        # Try each fallback strategy in order until we get a confident action
        while current_index <= FallbackLevel.HALT.value:
            strategy = self.fallback_strategies[current_index - 1]  # Adjust for 0-indexing
            
            logger.info(
                "Trying fallback strategy: %s (level %d)",
                strategy.__class__.__name__,
                current_index
            )
            
            # Get action from this strategy
            action = strategy.get_action(state, failure_context)
            
            # Record the outcome (we'll assume success for now)
            strategy.record_outcome(success=True)
            
            # If the action has sufficient confidence, return it
            if action.confidence >= self.confidence_threshold:
                # Update current level if it changed
                if current_index != self.current_level.value:
                    self.current_level = FallbackLevel(current_index)
                    logger.warning(
                        "Fallback level set to %s",
                        self.current_level.name
                    )
                
                return action
            
            # Otherwise, escalate to the next level
            current_index += 1
        
        # If we've exhausted all strategies, use the HALT strategy
        halt_strategy = self.fallback_strategies[-1]
        logger.critical(
            "All fallback strategies failed, using HALT strategy as last resort"
        )
        self.current_level = FallbackLevel.HALT
        return halt_strategy.get_action(state, failure_context)
    
    def escalate_fallback(self, reason: str = "manual_escalation") -> FallbackLevel:
        """
        Manually escalate to the next fallback level.
        
        Args:
            reason: Reason for the escalation
            
        Returns:
            New fallback level
        """
        current_value = self.current_level.value
        
        # Don't escalate beyond HALT
        if current_value >= FallbackLevel.HALT.value:
            logger.warning("Already at maximum fallback level (HALT), cannot escalate further")
            return self.current_level
        
        # Escalate to next level
        new_level = FallbackLevel(current_value + 1)
        self.current_level = new_level
        self.last_escalation_time = datetime.datetime.now()
        
        logger.warning(
            "Fallback escalated to %s: %s",
            new_level.name,
            reason
        )
        
        # Record this escalation
        self._record_activation(FailureContext(
            error_type="manual_escalation",
            error_message=reason,
            component="FallbackManager"
        ))
        
        return new_level
    
    def attempt_recovery(self) -> bool:
        """
        Attempt to recover to a lower fallback level.
        
        Uses exponential backoff to avoid rapid oscillation between levels.
        
        Returns:
            True if recovery was successful, False otherwise
        """
        # Don't attempt recovery if we're already at NORMAL
        if self.current_level == FallbackLevel.NORMAL:
            return True
        
        # Check if enough time has passed since last escalation
        if self.last_escalation_time is not None:
            elapsed_seconds = (datetime.datetime.now() - self.last_escalation_time).total_seconds()
            required_wait = 60 * self.backoff_factor  # Base wait is 1 minute
            
            if elapsed_seconds < required_wait:
                logger.info(
                    "Too soon to attempt recovery, waiting %d more seconds",
                    required_wait - elapsed_seconds
                )
                return False
        
        # Attempt to recover to one level lower
        current_value = self.current_level.value
        if current_value > 1:  # Don't go below SIMPLIFIED_RL
            new_level = FallbackLevel(current_value - 1)
            self.current_level = new_level
            self.recovery_attempts += 1
            self.backoff_factor *= 2  # Exponential backoff
            
            logger.info(
                "Recovered to fallback level %s (attempt %d)",
                new_level.name,
                self.recovery_attempts
            )
            return True
        
        return False
    
    def reset(self) -> None:
        """
        Reset to normal operation mode.
        
        This should only be called when the system is confirmed to be stable.
        """
        self.current_level = FallbackLevel.NORMAL
        self.backoff_factor = 1.0
        self.recovery_attempts = 0
        self.last_escalation_time = None
        
        logger.info("FallbackManager reset to NORMAL operation")
    
    def get_effectiveness_report(self) -> Dict[FallbackLevel, FallbackEffectiveness]:
        """
        Get effectiveness report for all fallback strategies.
        
        Returns:
            Dictionary mapping fallback levels to their effectiveness metrics
        """
        return {
            strategy.level: strategy.effectiveness
            for strategy in self.fallback_strategies
        }
    
    def _record_activation(self, failure_context: FailureContext) -> None:
        """
        Record a fallback activation for analysis.
        
        Args:
            failure_context: Context about the failure
        """
        activation_record = {
            "timestamp": datetime.datetime.now(),
            "level": self.current_level,
            "error_type": failure_context.error_type,
            "error_message": failure_context.error_message,
            "component": failure_context.component
        }
        
        self.activation_history.append(activation_record)
        
        # Keep history bounded
        if len(self.activation_history) > 1000:
            self.activation_history = self.activation_history[-1000:]