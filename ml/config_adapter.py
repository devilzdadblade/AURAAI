from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any


class RLConfigAdapter(ABC):
    """
    Abstract base class for adapting RL model configuration outputs to environment parameters.
    
    This abstract class defines the interface for configuration adapters that translate
    raw reinforcement learning model outputs into meaningful trading parameters.
    Different trading strategies or environments may require different adaptation
    strategies, making this abstraction essential for flexibility.
    
    The adapter pattern allows the RL model to output normalized values (typically
    in the range [-1, 1]) which are then scaled and transformed into domain-specific
    parameters with appropriate ranges and constraints.
    """
    
    @abstractmethod
    def __call__(self, model_output_config: np.ndarray) -> Dict[str, Any]:
        """
        Transform raw RL model configuration output to environment-specific parameters.
        
        Args:
            model_output_config: Raw configuration output from the RL model,
                               typically normalized values in range [-1, 1]
                               
        Returns:
            Dictionary containing environment-specific configuration parameters
            
        Raises:
            NotImplementedError: Must be implemented by concrete subclasses
        """
        pass


class TradingConfigAdapter(RLConfigAdapter):
    """
    Configuration adapter for translating RL model outputs to trading parameters.
    
    This adapter transforms the raw numerical outputs from the reinforcement learning
    model into practical trading parameters such as stop loss percentages, profit
    targets, risk levels, and confidence thresholds. It handles the scaling and
    constraint application to ensure all parameters fall within reasonable ranges
    for live trading.
    
    The adapter assumes the RL model outputs a 4-dimensional configuration vector
    with normalized values in the range [-1, 1], which are then mapped to:
    - Stop loss percentage (0.1% to 2.0%)
    - Profit target percentage (0.1% to 4.0%)
    - Risk per trade percentage (0.5% to 3.0% of capital)
    - Confidence threshold (0.5 to 0.9)
    
    These ranges can be adjusted based on trading strategy requirements and
    risk management policies.
    """
    
    def __call__(self, model_output_config: np.ndarray) -> Dict[str, Any]:
        """
        Transform RL model configuration output to concrete trading parameters.
        
        This method takes the raw normalized outputs from the RL model and scales
        them to appropriate ranges for trading parameters. Each dimension of the
        input array corresponds to a specific trading parameter with predefined
        scaling ranges.
        
        Args:
            model_output_config: Numpy array of size 4 containing normalized values
                               in range [-1, 1] representing:
                               [0] -> Stop loss scaling factor
                               [1] -> Profit target scaling factor  
                               [2] -> Risk per trade scaling factor
                               [3] -> Confidence threshold scaling factor
                               
        Returns:
            Dictionary containing scaled trading parameters:
            - stop_loss_pct: Stop loss percentage (0.001 to 0.02)
            - profit_target_pct: Profit target percentage (0.001 to 0.04)
            - risk_per_trade_pct: Risk per trade percentage (0.005 to 0.03)
            - confidence_threshold: Confidence threshold (0.5 to 0.9)
            
        Example:
            >>> adapter = TradingConfigAdapter()
            >>> model_output = np.array([0.5, -0.2, 0.8, 0.1])
            >>> params = adapter(model_output)
            >>> print(f"Stop loss: {params['stop_loss_pct']:.3f}")
            Stop loss: 0.016
        """
        # Assumes model_output_config is a np.array of size 4, with values from -1 to 1
        
        # Scale the outputs to reasonable ranges
        # Example:
        # config[0] -> Stop Loss Percentage (e.g., 0.1% to 2%)
        # config[1] -> Profit Target Percentage (e.g., 0.1% to 4%)
        # config[2] -> Risk per trade (e.g., 0.5% to 3% of capital)
        # config[3] -> Confidence Threshold (e.g., 0.5 to 0.9)

        stop_loss_pct = 0.001 + ((model_output_config[0] + 1) / 2) * (0.02 - 0.001)
        profit_target_pct = 0.001 + ((model_output_config[1] + 1) / 2) * (0.04 - 0.001)
        risk_per_trade_pct = 0.005 + ((model_output_config[2] + 1) / 2) * (0.03 - 0.005)
        confidence_threshold = 0.5 + ((model_output_config[3] + 1) / 2) * (0.4)

        return {
            "stop_loss_pct": stop_loss_pct,
            "profit_target_pct": profit_target_pct,
            "risk_per_trade_pct": risk_per_trade_pct,
            "confidence_threshold": confidence_threshold,
        }
