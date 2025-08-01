import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple

class ImprovedUncertaintyEstimator(nn.Module):
    """Uncertainty estimation using an ensemble of networks."""

    def __init__(self, network_factory, num_networks=5, learning_rate: float = 0.001):
        super().__init__()
        self.networks = nn.ModuleList([network_factory() for _ in range(num_networks)])
        self.num_networks = num_networks
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-4)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = []
        for network in self.networks:
            network.eval()
            with torch.no_grad():
                output = network(x)
                outputs.append(output)
        outputs_stacked = torch.stack(outputs)
        mean_prediction = outputs_stacked.mean(dim=0)
        uncertainty_std = outputs_stacked.std(dim=0)
        return mean_prediction, uncertainty_std

    def update_networks(self, features: torch.Tensor, target_values: torch.Tensor):
        import logging
        from src.ml.gradient_validator import GradientValidator
        from src.ml.numerical_stability import NumericalStabilityManager
        
        logger = logging.getLogger(__name__)
        gradient_validator = GradientValidator(max_grad_norm=1.0)
        stability_manager = NumericalStabilityManager(epsilon=1e-10)
        
        self.optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=features.device)
        
        for network in self.networks:
            network.train()
            pred = network(features.detach())
            
            # Add numerical stability check for predictions
            pred = stability_manager.handle_numerical_instability(
                pred, replacement_value=0.0, name="uncertainty_network_pred"
            )
            
            loss = nn.functional.mse_loss(pred, target_values.detach())
            
            # Add numerical stability check for loss
            loss = stability_manager.handle_numerical_instability(
                loss, replacement_value=0.1, name="uncertainty_network_loss"
            )
            
            total_loss += loss
            
        # Add numerical stability check for total loss
        total_loss = stability_manager.handle_numerical_instability(
            total_loss, replacement_value=0.1, name="uncertainty_total_loss"
        )
        
        total_loss.backward()
        
        # Add gradient validation before optimizer step
        validation_result = gradient_validator.validate_gradients(self)
        
        if validation_result.is_valid:
            # Clip gradients to prevent exploding gradients
            gradient_validator.clip_gradients(self, max_norm=1.0)
            self.optimizer.step()
        else:
            logger.error(f"Skipping uncertainty optimizer step due to invalid gradients: {validation_result.error_message}")
            # Log detailed gradient statistics for debugging
            grad_stats = gradient_validator.log_gradient_stats(self)
            logger.error(f"Uncertainty gradient stats: mean={grad_stats.mean_grad_norm:.6f}, "
                       f"max={grad_stats.max_grad_norm:.6f}, NaN={grad_stats.num_params_with_nan}, "
                       f"Inf={grad_stats.num_params_with_inf}")
            
        self.optimizer.zero_grad()