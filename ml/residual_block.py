"""
ResidualBlock implementation for improved gradient flow in neural networks.

This module provides a ResidualBlock class that implements skip connections
to improve gradient flow during backpropagation, allowing for training of
deeper neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Union


class ResidualBlock(nn.Module):
    """
    Residual block with skip connections for improved gradient flow.
    
    This implementation includes:
    - Skip connections (x + F(x))
    - Optional spectral normalization for weight matrices
    - Configurable activation functions
    - Optional dropout for regularization
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout_rate: float = 0.1,
        use_spectral_norm: bool = True,
        activation: Union[str, Callable] = "relu",
        use_layer_norm: bool = True
    ):
        """
        Initialize a residual block.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            dropout_rate: Dropout probability (0.0 to 1.0)
            use_spectral_norm: Whether to apply spectral normalization
            activation: Activation function to use ("relu", "gelu", "elu", or callable)
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.dropout_rate = dropout_rate
        self.use_spectral_norm = use_spectral_norm
        self.use_layer_norm = use_layer_norm

        self.main_branch = self._build_main_branch(
            in_features, out_features, dropout_rate, use_spectral_norm, activation, use_layer_norm
        )
        self.skip_connection = self._build_skip_connection(
            in_features, out_features, use_spectral_norm
        )
        self.final_activation = self._get_activation(activation)

    def _get_activation(self, activation: Union[str, Callable]) -> nn.Module:
        if isinstance(activation, str):
            act = activation.lower()
            if act == "relu":
                return nn.ReLU()
            elif act == "gelu":
                return nn.GELU()
            elif act == "elu":
                return nn.ELU()
            else:
                raise ValueError(f"Unsupported activation: {activation}")
        elif callable(activation):
            return activation()
        else:
            raise ValueError(f"Unsupported activation type: {type(activation)}")

    def _build_main_branch(
        self,
        in_features: int,
        out_features: int,
        dropout_rate: float,
        use_spectral_norm: bool,
        activation: Union[str, Callable],
        use_layer_norm: bool
    ) -> nn.Sequential:
        layers = []
        linear1 = nn.Linear(in_features, out_features)
        if use_spectral_norm:
            linear1 = nn.utils.spectral_norm(linear1)
        layers.append(linear1)
        if use_layer_norm:
            layers.append(nn.LayerNorm(out_features))
        layers.append(self._get_activation(activation))
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        linear2 = nn.Linear(out_features, out_features)
        if use_spectral_norm:
            linear2 = nn.utils.spectral_norm(linear2)
        layers.append(linear2)
        if use_layer_norm:
            layers.append(nn.LayerNorm(out_features))
        return nn.Sequential(*layers)

    def _build_skip_connection(
        self,
        in_features: int,
        out_features: int,
        use_spectral_norm: bool
    ) -> Optional[nn.Module]:
        if in_features != out_features:
            skip_connection = nn.Linear(in_features, out_features)
            if use_spectral_norm:
                skip_connection = nn.utils.spectral_norm(skip_connection)
            return skip_connection
        return None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after residual computation (F(x) + x)
        """
        # Main branch computation
        main_output = self.main_branch(x)
        
        # Skip connection
        if self.skip_connection is not None:
            skip_output = self.skip_connection(x)
        else:
            skip_output = x
        
        # Residual connection: F(x) + x
        output = main_output + skip_output
        
        # Final activation
        output = self.final_activation(output)
        
        return output