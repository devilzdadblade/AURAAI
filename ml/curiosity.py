import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ImprovedCuriosityModule(nn.Module):
    """Enhanced Intrinsic Curiosity Module (ICM) with normalization and robust dynamics models."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Feature network with LayerNorm for stable embeddings
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )

        # Inverse dynamics model: Predicts action from (state_feature, next_state_feature)
        self.inverse_model = nn.Sequential(
            nn.Linear(hidden_dim // 2 + hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )

        # Forward dynamics model: Predicts next_state_feature from (state_feature, action)
        self.forward_model = nn.Sequential(
            nn.Linear(hidden_dim // 2 + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        # Running statistics for normalizing intrinsic rewards
        self.register_buffer('reward_mean', torch.zeros(1, dtype=torch.float32))
        self.register_buffer('reward_std', torch.ones(1, dtype=torch.float32))
        self.register_buffer('update_count', torch.zeros(1, dtype=torch.float32))

    def forward(self, state: torch.Tensor, next_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_feat = self.feature_net(state)
        next_state_feat = self.feature_net(next_state)

        # Inverse dynamics loss
        pred_action = self.inverse_model(torch.cat([state_feat, next_state_feat], dim=1))
        inverse_loss = F.cross_entropy(pred_action, action.long())

        # Forward dynamics loss
        action_onehot = F.one_hot(action.long(), self.action_dim).float()
        pred_next_state_feat = self.forward_model(torch.cat([state_feat, action_onehot], dim=1))
        forward_loss = F.mse_loss(pred_next_state_feat, next_state_feat.detach())

        # Intrinsic reward
        intrinsic_reward = forward_loss.detach()
        with torch.no_grad():
            self.update_count += 1.0
            delta = intrinsic_reward.mean() - self.reward_mean
            self.reward_mean += delta / self.update_count
            M2 = self.reward_std ** 2 * (self.update_count - 1)
            m2_new = M2 + delta * (intrinsic_reward.mean() - self.reward_mean)
            self.reward_std = torch.sqrt(m2_new / (self.update_count + 1e-8) + 1e-8)

        normalized_reward = (intrinsic_reward - self.reward_mean) / (self.reward_std + 1e-8)
        return normalized_reward, inverse_loss, forward_loss