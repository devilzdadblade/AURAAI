"""
ImprovedDuelingNetwork implementation with modern architecture features.

This module provides an enhanced version of the DuelingNetwork architecture
with residual connections, attention mechanisms, and advanced normalization
techniques for improved performance and training stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, List, Optional

from src.ml.residual_block import ResidualBlock
from src.ml.attention_mechanism import AttentionMechanism
from src.ml.noisy_linear import NoisyLinear
from src.ml.normalization_monitor import NormalizationConfig, NormalizationMonitor


class ImprovedDuelingNetwork(nn.Module):
    """
    Enhanced dueling network architecture with modern improvements.
    
    This implementation includes:
    - ResidualBlock components for improved gradient flow
    - Multi-Head Attention for dynamic feature importance weighting
    - Layer normalization and spectral normalization for stability
    - Optional noisy layers for parameter space exploration
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        action_size: int,
        num_residual_blocks: int = 3,
        attention_heads: int = 8,
        dropout_rate: float = 0.1,
        use_spectral_norm: bool = True,
        use_noisy_layers: bool = False,
        normalization_type: str = 'layer',  # 'layer', 'batch', 'spectral', 'none'
        enable_norm_monitoring: bool = True
    ):
        """
        Initialize the improved dueling network.
        
        Args:
            input_size: Dimension of input features
            hidden_size: Dimension of hidden layers
            action_size: Number of possible actions
            num_residual_blocks: Number of residual blocks to use
            attention_heads: Number of attention heads
            dropout_rate: Dropout probability
            use_spectral_norm: Whether to use spectral normalization
            use_noisy_layers: Whether to use noisy linear layers for exploration
            normalization_type: Type of normalization to use ('layer', 'batch', 'spectral', 'none')
            enable_norm_monitoring: Whether to enable normalization parameter monitoring
        """
        super().__init__()
        
        # Store configuration
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.use_noisy_layers = use_noisy_layers
        self.normalization_type = normalization_type
        
        # Create normalization configuration
        self.norm_config = NormalizationConfig(
            normalization_type=normalization_type,
            use_spectral_norm=use_spectral_norm,
            use_layer_norm=(normalization_type == 'layer'),
            use_batch_norm=(normalization_type == 'batch')
        )
        
        # Input projection layer with normalization
        input_projection = nn.Linear(input_size, hidden_size)
        self.input_projection = self.norm_config.apply_normalization(
            input_projection, hidden_size, normalization_type
        )
        
        # Feature extraction with residual blocks
        self.feature_blocks = nn.ModuleList([
            ResidualBlock(
                hidden_size, 
                hidden_size, 
                dropout_rate=dropout_rate,
                use_spectral_norm=use_spectral_norm,
                use_layer_norm=(normalization_type == 'layer')
            ) for _ in range(num_residual_blocks)
        ])
        
        # Multi-head attention for feature importance
        self.attention = AttentionMechanism(
            embed_dim=hidden_size,
            num_heads=attention_heads,
            dropout=dropout_rate,
            use_residual=True,
            use_layer_norm=(normalization_type == 'layer')
        )
        
        # Value stream with normalization
        self.value_stream = self._build_stream(
            hidden_size, 
            1, 
            dropout_rate, 
            use_spectral_norm, 
            use_noisy=False
        )
        
        # Advantage stream with optional noisy layers
        self.advantage_stream = self._build_stream(
            hidden_size, 
            action_size, 
            dropout_rate, 
            use_spectral_norm, 
            use_noisy=use_noisy_layers
        )
        
        # Initialize normalization monitor if enabled
        self.norm_monitor = None
        if enable_norm_monitoring:
            self.norm_monitor = NormalizationMonitor(self)
    
    def _build_stream(
        self, 
        input_size: int, 
        output_size: int, 
        dropout_rate: float, 
        use_spectral_norm: bool, 
        use_noisy: bool
    ) -> nn.Sequential:
        """
        Build a value or advantage stream with the specified configuration.
        
        Args:
            input_size: Input dimension
            output_size: Output dimension
            dropout_rate: Dropout probability
            use_spectral_norm: Whether to use spectral normalization
            use_noisy: Whether to use noisy linear layers
            
        Returns:
            Sequential module representing the stream
        """
        layers = []
        
        # First layer with appropriate normalization
        if use_noisy:
            # NoisyLinear doesn't support spectral norm directly
            first_layer = NoisyLinear(input_size, input_size // 2)
            layers.append(first_layer)
            
            # Add normalization after the noisy layer
            norm_layer = self.norm_config.get_normalization_layer(input_size // 2)
            if norm_layer:
                layers.append(norm_layer)
        else:
            # Regular linear layer with configurable normalization
            linear1 = nn.Linear(input_size, input_size // 2)
            normalized_layer = self.norm_config.apply_normalization(
                linear1, input_size // 2, self.normalization_type
            )
            layers.append(normalized_layer)
        
        # Activation and dropout
        layers.append(nn.ReLU())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer with appropriate normalization
        if use_noisy:
            # NoisyLinear for the output layer
            output_layer = NoisyLinear(input_size // 2, output_size)
            layers.append(output_layer)
        else:
            # Regular linear layer with configurable normalization
            linear2 = nn.Linear(input_size // 2, output_size)
            if use_spectral_norm:
                linear2 = nn.utils.spectral_norm(linear2)
            layers.append(linear2)
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the improved dueling network.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple containing:
            - Q-values tensor
            - Dictionary with intermediate outputs and attention weights
        """
        # Store intermediate outputs for diagnostics
        intermediates = {}
        
        # Input projection
        x = self.input_projection(x)
        intermediates['post_projection'] = x
        
        # Feature extraction with residual blocks
        for i, block in enumerate(self.feature_blocks):
            x = block(x)
            intermediates[f'residual_block_{i}'] = x
        
        # Apply attention mechanism
        # For attention, we need to reshape to (batch_size, 1, hidden_size)
        # to treat the entire feature vector as a single token
        batch_size = x.shape[0]
        x_reshaped = x.view(batch_size, 1, self.hidden_size)
        
        # Apply attention and get attention weights
        x_attended, attention_info = self.attention(x_reshaped)
        
        # Reshape back to (batch_size, hidden_size)
        x = x_attended.view(batch_size, self.hidden_size)
        
        # Store attention information
        intermediates['attention_weights'] = attention_info['attention_weights']
        intermediates['feature_importance'] = attention_info['feature_importance']
        
        # Value and advantage streams
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Store value and advantage
        intermediates['value'] = value
        intermediates['advantage'] = advantage
        
        # Combine value and advantage to get Q-values
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_values = value + (advantage - advantage_mean)
        
        # Reset noise if using noisy layers and in training mode
        if self.training and self.use_noisy_layers:
            self.reset_noise()
        
        # Update normalization monitor if enabled
        if self.training and self.norm_monitor is not None:
            self.norm_monitor.step()
        
        return q_values, intermediates
    
    def reset_noise(self):
        """Reset noise for all noisy layers if they exist."""
        if not self.use_noisy_layers:
            return
            
        # Reset noise in advantage stream
        for module in self.advantage_stream:
            if isinstance(module, NoisyLinear):
                module.reset_noise()
                
        # Reset noise in value stream (if any)
        for module in self.value_stream:
            if isinstance(module, NoisyLinear):
                module.reset_noise()
    
    def get_normalization_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get normalization statistics from the monitor.
        
        Returns:
            Dictionary containing normalization statistics or empty dict if monitoring is disabled
        """
        if self.norm_monitor is not None:
            return self.norm_monitor.get_statistics()
        return {}
    
    def detect_normalization_anomalies(self, threshold: float = 3.0) -> List[Dict[str, Any]]:
        """
        Detect anomalies in normalization parameters.
        
        Args:
            threshold: Z-score threshold for anomaly detection
            
        Returns:
            List of detected anomalies or empty list if monitoring is disabled
        """
        if self.norm_monitor is not None:
            return self.norm_monitor.detect_anomalies(threshold)
        return []
    
    def adjust_normalization_parameters(self):
        """
        Automatically adjust normalization parameters based on detected anomalies.
        """
        if self.norm_monitor is not None:
            anomalies = self.norm_monitor.detect_anomalies()
            if anomalies:
                self.norm_monitor.adjust_parameters(anomalies)