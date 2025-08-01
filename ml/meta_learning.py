import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from collections import deque

class MetaLearningHyperparams(nn.Module):
    """Learnable hyperparameters using meta-gradients for increased stability."""

    def __init__(self, initial_lr=0.001, initial_gamma=0.99, initial_tau=0.005,
                 initial_epsilon=0.1, meta_lr=0.0001, adaptation_steps=10):
        super().__init__()

        self.log_lr = nn.Parameter(torch.log(torch.tensor(initial_lr, dtype=torch.float32)))
        self.logit_gamma = nn.Parameter(torch.logit(torch.tensor(initial_gamma, dtype=torch.float32)))
        self.log_tau = nn.Parameter(torch.log(torch.tensor(initial_tau, dtype=torch.float32)))
        self.logit_epsilon = nn.Parameter(torch.logit(torch.tensor(initial_epsilon, dtype=torch.float32)))

        self.meta_optimizer = optim.Adam(self.parameters(), lr=meta_lr, weight_decay=1e-4)
        self.adaptation_steps = adaptation_steps
        self.meta_update_counter = 0
        self.last_meta_loss = 0.0

    def get_hyperparams(self):
        with torch.no_grad():
            return {
                'lr': torch.exp(self.log_lr).item(),
                'gamma': torch.sigmoid(self.logit_gamma).item(),
                'tau': torch.exp(self.log_tau).item(),
                'epsilon': torch.sigmoid(self.logit_epsilon).item()
            }

    def update_meta_params(self, episode_rewards_history: deque):
        required_history_len = self.adaptation_steps + 20
        if len(episode_rewards_history) < required_history_len:
            self.last_meta_loss = 0.0
            return 0.0

        self.meta_update_counter += 1
        if self.meta_update_counter < self.adaptation_steps:
            self.last_meta_loss = 0.0
            return 0.0

        self.meta_update_counter = 0

        recent_window_size = 10
        older_window_offset = 10
        older_window_size = 10

        if len(episode_rewards_history) < (recent_window_size + older_window_offset + older_window_size):
            self.last_meta_loss = 0.0
            return 0.0

        recent_rewards = list(episode_rewards_history)[-recent_window_size:]
        older_rewards = list(episode_rewards_history)[
            -(recent_window_size + older_window_offset + older_window_size):
            -(recent_window_size + older_window_offset)
        ]

        recent_avg_reward = np.mean(recent_rewards)
        older_avg_reward = np.mean(older_rewards)
        performance_delta = recent_avg_reward - older_avg_reward

        meta_loss = -torch.tensor(performance_delta, requires_grad=True, dtype=torch.float32)

        import logging
        from src.ml.gradient_validator import GradientValidator
        from src.ml.numerical_stability import NumericalStabilityManager
        
        logger = logging.getLogger(__name__)
        gradient_validator = GradientValidator(max_grad_norm=0.1)  # Using smaller norm for meta-learning
        stability_manager = NumericalStabilityManager(epsilon=1e-10)
        
        # Add numerical stability check for meta loss
        meta_loss = stability_manager.handle_numerical_instability(
            meta_loss, replacement_value=0.0, name="meta_learning_loss"
        )
        
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        
        # Add gradient validation before optimizer step
        validation_result = gradient_validator.validate_gradients(self)
        
        if validation_result.is_valid:
            # Use the existing clip_grad_norm_ call
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.1)
            self.meta_optimizer.step()
        else:
            logger.error(f"Skipping meta-learning optimizer step due to invalid gradients: {validation_result.error_message}")
            # Log detailed gradient statistics for debugging
            grad_stats = gradient_validator.log_gradient_stats(self)
            logger.error(f"Meta-learning gradient stats: mean={grad_stats.mean_grad_norm:.6f}, "
                       f"max={grad_stats.max_grad_norm:.6f}, NaN={grad_stats.num_params_with_nan}, "
                       f"Inf={grad_stats.num_params_with_inf}")

        with torch.no_grad():
            self.log_lr.clamp_(math.log(1e-5), math.log(5e-2))
            self.logit_gamma.clamp_(
                torch.logit(torch.tensor(0.9, dtype=torch.float32)),
                torch.logit(torch.tensor(0.999, dtype=torch.float32))
            )
            self.log_tau.clamp_(math.log(1e-4), math.log(0.1))
            self.logit_epsilon.clamp_(
                torch.logit(torch.tensor(1e-3, dtype=torch.float32)),
                torch.logit(torch.tensor(0.5, dtype=torch.float32))
            )

        self.last_meta_loss = meta_loss.item()
        return self.last_meta_loss
