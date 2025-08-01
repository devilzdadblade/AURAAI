import torch
import torch.nn as nn

from typing import Dict

class AuxiliaryTaskHeads(nn.Module):
    """
    Auxiliary prediction heads for better representation learning.
    This version primarily focuses on 'next_state_prediction' due to clear target availability.
    """

    def __init__(self, hidden_size: int, state_size: int):
        super().__init__()

        # Next state predictor: predicts the *next observation* from the current state features.
        self.next_state_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_size)  # Output matches original observation size
        )

    def forward(self, features: torch.Tensor, next_features: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Compute auxiliary predictions."""
        predictions = {}
        predictions['next_state_prediction'] = self.next_state_predictor(features)
        return predictions