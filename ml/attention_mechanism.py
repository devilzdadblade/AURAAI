"""
Multi-Head Attention mechanism for dynamic feature importance weighting.

This module provides an AttentionMechanism class that implements multi-head
attention for capturing complex relationships between features and improving
model performance through dynamic feature importance weighting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class AttentionMechanism(nn.Module):
    """
    Multi-head attention mechanism for dynamic feature importance weighting.
    
    This implementation includes:
    - Multi-head attention using PyTorch's nn.MultiheadAttention
    - Attention weight extraction for interpretability
    - Residual connections for stable training
    - Layer normalization for improved convergence
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_residual: bool = True,
        use_layer_norm: bool = True
    ):
        """
        Initialize the attention mechanism.
        
        Args:
            embed_dim: Dimension of the input features
            num_heads: Number of attention heads
            dropout: Dropout probability
            use_residual: Whether to use residual connections
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        # Ensure embed_dim is divisible by num_heads
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        # Store configuration
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        
        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Use batch_first=True for more intuitive tensor shapes
        )
        
        # Layer normalization (if enabled)
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Linear projection for feature importance weighting
        self.feature_importance = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the attention mechanism.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
               For feature attention, seq_len represents the number of features
               
        Returns:
            Tuple containing:
            - Output tensor after attention
            - Dictionary with attention weights and other diagnostic information
        """
        # Ensure input has 3 dimensions (batch_size, seq_len, embed_dim)
        if x.dim() == 2:
            # If input is (batch_size, embed_dim), add a sequence dimension
            x = x.unsqueeze(1)
        
        _, _, _ = x.shape  # batch_size, seq_len, embed_dim unused in computation
        
        # Apply layer normalization if enabled
        if self.use_layer_norm:
            x_norm = self.layer_norm(x)
        else:
            x_norm = x
        
        # Self-attention: use the same tensor for query, key, and value
        # Output shape: (batch_size, seq_len, embed_dim)
        attn_output, attn_weights = self.attention(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            need_weights=True  # Return attention weights for interpretability
        )
        
        # Apply residual connection if enabled
        if self.use_residual:
            output = x + attn_output
        else:
            output = attn_output
        
        # Apply feature importance weighting
        feature_weights = torch.sigmoid(self.feature_importance(output))
        weighted_output = output * feature_weights
        
        # Prepare diagnostic information
        diagnostics = {
            'attention_weights': attn_weights,  # Shape: (batch_size, seq_len, seq_len)
            'feature_importance': feature_weights,  # Shape: (batch_size, seq_len, embed_dim)
            'pre_residual': attn_output,  # Output before residual connection
        }
        
        return weighted_output, diagnostics
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract attention weights for a given input.
        
        Args:
            x: Input tensor
            
        Returns:
            Attention weights tensor
        """
        # Ensure input has 3 dimensions
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Apply layer normalization if enabled
        if self.use_layer_norm:
            x_norm = self.layer_norm(x)
        else:
            x_norm = x
        
        # Get attention weights only
        with torch.no_grad():
            _, attn_weights = self.attention(
                query=x_norm,
                key=x_norm,
                value=x_norm,
                need_weights=True
            )
        
        return attn_weights