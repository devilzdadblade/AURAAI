"""
Normalization parameter monitoring and adjustment capabilities.

This module provides tools for monitoring and adjusting normalization parameters
in neural networks, including spectral normalization, layer normalization, and
batch normalization.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import numpy as np


class NormalizationMonitor:
    """
    Monitor and adjust normalization parameters in neural networks.
    
    This class provides tools for:
    - Tracking normalization parameter statistics
    - Detecting anomalies in normalization parameters
    - Adjusting normalization parameters dynamically
    - Logging normalization parameter information
    """
    
    def __init__(self, model: nn.Module, log_frequency: int = 100):
        """
        Initialize the normalization monitor.
        
        Args:
            model: The neural network model to monitor
            log_frequency: How often to log normalization statistics (in steps)
        """
        self.model = model
        self.log_frequency = log_frequency
        self.step_counter = 0
        self.logger = logging.getLogger(__name__)
        
        # Statistics tracking
        self.spectral_norm_stats = {}
        self.layer_norm_stats = {}
        self.batch_norm_stats = {}
        
        # Register hooks for parameter tracking
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to track normalization parameters."""
        for name, module in self.model.named_modules():
            # Track spectral normalization
            if hasattr(module, 'weight_orig'):
                module.register_forward_hook(
                    lambda mod, inp, out, name=name: self._track_spectral_norm(mod, name)
                )
            
            # Track layer normalization
            if isinstance(module, nn.LayerNorm):
                module.register_forward_hook(
                    lambda mod, inp, out, name=name: self._track_layer_norm(mod, out, name)
                )
            
            # Track batch normalization
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.register_forward_hook(
                    lambda mod, inp, out, name=name: self._track_batch_norm(mod, name)
                )
    
    def _track_spectral_norm(self, module: nn.Module, name: str):
        """Track spectral normalization parameters."""
        if hasattr(module, 'weight_orig') and hasattr(module, 'weight'):
            weight_orig = module.weight_orig
            weight = module.weight
            
            # Calculate spectral norm (sigma)
            with torch.no_grad():
                sigma = torch.norm(weight_orig, dim=None) / torch.norm(weight, dim=None)
            
            # Store statistics
            if name not in self.spectral_norm_stats:
                self.spectral_norm_stats[name] = {
                    'sigma_history': [],
                    'min_sigma': float('inf'),
                    'max_sigma': float('-inf'),
                    'mean_sigma': 0.0,
                }
            
            stats = self.spectral_norm_stats[name]
            sigma_val = sigma.item()
            stats['sigma_history'].append(sigma_val)
            stats['min_sigma'] = min(stats['min_sigma'], sigma_val)
            stats['max_sigma'] = max(stats['max_sigma'], sigma_val)
            stats['mean_sigma'] = np.mean(stats['sigma_history'][-100:])  # Moving average
    
    def _track_layer_norm(self, module: nn.LayerNorm, output: torch.Tensor, name: str):
        """Track layer normalization statistics."""
        with torch.no_grad():
            # Calculate statistics on the normalized output
            mean = output.mean().item()
            std = output.std().item()
            
            # Store statistics
            if name not in self.layer_norm_stats:
                self.layer_norm_stats[name] = {
                    'mean_history': [],
                    'std_history': [],
                    'weight_mean': 0.0,
                    'weight_std': 0.0,
                    'bias_mean': 0.0,
                    'bias_std': 0.0,
                }
            
            stats = self.layer_norm_stats[name]
            stats['mean_history'].append(mean)
            stats['std_history'].append(std)
            
            if module.weight is not None and module.bias is not None:
                stats['weight_mean'] = module.weight.mean().item()
                stats['weight_std'] = module.weight.std().item()
                stats['bias_mean'] = module.bias.mean().item()
                stats['bias_std'] = module.bias.std().item()
    
    def _track_batch_norm(self, module: nn.Module, name: str):
        """Track batch normalization statistics."""
        if hasattr(module, 'running_mean') and hasattr(module, 'running_var'):
            with torch.no_grad():
                running_mean = module.running_mean
                running_var = module.running_var
                
                # Store statistics
                if name not in self.batch_norm_stats:
                    self.batch_norm_stats[name] = {
                        'mean_history': [],
                        'var_history': [],
                        'weight_mean': 0.0,
                        'weight_std': 0.0,
                        'bias_mean': 0.0,
                        'bias_std': 0.0,
                    }
                
                stats = self.batch_norm_stats[name]
                stats['mean_history'].append(running_mean.mean().item())
                stats['var_history'].append(running_var.mean().item())
                
                if module.weight is not None and module.bias is not None:
                    stats['weight_mean'] = module.weight.mean().item()
                    stats['weight_std'] = module.weight.std().item()
                    stats['bias_mean'] = module.bias.mean().item()
                    stats['bias_std'] = module.bias.std().item()
    
    def step(self):
        """
        Update step counter and log statistics if needed.
        
        Call this method after each training step.
        """
        self.step_counter += 1
        
        # Log statistics periodically
        if self.step_counter % self.log_frequency == 0:
            self._log_statistics()
    
    def _log_statistics(self):
        """Log normalization statistics."""
        # Log spectral norm statistics
        if self.spectral_norm_stats:
            self.logger.info(f"Spectral Norm Statistics (Step {self.step_counter}):")
            for name, stats in self.spectral_norm_stats.items():
                self.logger.info(f"  {name}: sigma={stats['mean_sigma']:.4f} (min={stats['min_sigma']:.4f}, max={stats['max_sigma']:.4f})")
        
        # Log layer norm statistics
        if self.layer_norm_stats:
            self.logger.info(f"Layer Norm Statistics (Step {self.step_counter}):")
            for name, stats in self.layer_norm_stats.items():
                mean_mean = np.mean(stats['mean_history'][-10:])
                mean_std = np.mean(stats['std_history'][-10:])
                self.logger.info(f"  {name}: output_mean={mean_mean:.4f}, output_std={mean_std:.4f}, "
                               f"weight_mean={stats['weight_mean']:.4f}, bias_mean={stats['bias_mean']:.4f}")
        
        # Log batch norm statistics
        if self.batch_norm_stats:
            self.logger.info(f"Batch Norm Statistics (Step {self.step_counter}):")
            for name, stats in self.batch_norm_stats.items():
                mean_mean = np.mean(stats['mean_history'][-10:])
                mean_var = np.mean(stats['var_history'][-10:])
                self.logger.info(f"  {name}: running_mean={mean_mean:.4f}, running_var={mean_var:.4f}, "
                               f"weight_mean={stats['weight_mean']:.4f}, bias_mean={stats['bias_mean']:.4f}")
    
    def get_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all normalization statistics.
        
        Returns:
            Dictionary containing all normalization statistics
        """
        return {
            'spectral_norm': self.spectral_norm_stats,
            'layer_norm': self.layer_norm_stats,
            'batch_norm': self.batch_norm_stats,
        }
    
    def detect_anomalies(self, threshold: float = 3.0) -> List[Dict[str, Any]]:
        """
        Detect anomalies in normalization parameters.
        
        Args:
            threshold: Z-score threshold for anomaly detection
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Check different types of anomalies
        anomalies.extend(self._detect_spectral_norm_anomalies(threshold))
        anomalies.extend(self._detect_layer_norm_anomalies(threshold))
        anomalies.extend(self._detect_batch_norm_anomalies(threshold))
        
        return anomalies

    def _detect_spectral_norm_anomalies(self, threshold: float) -> List[Dict[str, Any]]:
        """Detect anomalies in spectral normalization parameters."""
        anomalies = []
        
        for name, stats in self.spectral_norm_stats.items():
            if len(stats['sigma_history']) <= 10:
                continue
                
            sigma_history = stats['sigma_history'][-100:]
            mean = np.mean(sigma_history)
            std = np.std(sigma_history)
            
            if std > 0:
                current = sigma_history[-1]
                z_score = abs(current - mean) / std
                
                if z_score > threshold:
                    anomalies.append({
                        'type': 'spectral_norm',
                        'name': name,
                        'value': current,
                        'mean': mean,
                        'std': std,
                        'z_score': z_score,
                    })
        
        return anomalies

    def _detect_layer_norm_anomalies(self, threshold: float) -> List[Dict[str, Any]]:
        """Detect anomalies in layer normalization parameters."""
        anomalies = []
        
        for name, stats in self.layer_norm_stats.items():
            if len(stats['mean_history']) <= 10:
                continue
                
            anomalies.extend(self._check_layer_norm_mean_anomalies(
                name, stats, threshold
            ))
            anomalies.extend(self._check_layer_norm_std_anomalies(
                name, stats, threshold
            ))
        
        return anomalies

    def _check_layer_norm_mean_anomalies(self, name: str, stats: Dict[str, Any], 
                                       threshold: float) -> List[Dict[str, Any]]:
        """Check for anomalies in layer norm mean values."""
        anomalies = []
        mean_history = stats['mean_history'][-100:]
        
        mean_mean = np.mean(mean_history)
        mean_std = np.std(mean_history)
        
        if mean_std > 0:
            current_mean = mean_history[-1]
            z_score_mean = abs(current_mean - mean_mean) / mean_std
            
            if z_score_mean > threshold:
                anomalies.append({
                    'type': 'layer_norm_mean',
                    'name': name,
                    'value': current_mean,
                    'mean': mean_mean,
                    'std': mean_std,
                    'z_score': z_score_mean,
                })
        
        return anomalies

    def _check_layer_norm_std_anomalies(self, name: str, stats: Dict[str, Any], 
                                      threshold: float) -> List[Dict[str, Any]]:
        """Check for anomalies in layer norm standard deviation values."""
        anomalies = []
        std_history = stats['std_history'][-100:]
        
        std_mean = np.mean(std_history)
        std_std = np.std(std_history)
        
        if std_std > 0:
            current_std = std_history[-1]
            z_score_std = abs(current_std - std_mean) / std_std
            
            if z_score_std > threshold:
                anomalies.append({
                    'type': 'layer_norm_std',
                    'name': name,
                    'value': current_std,
                    'mean': std_mean,
                    'std': std_std,
                    'z_score': z_score_std,
                })
        
        return anomalies

    def _detect_batch_norm_anomalies(self, threshold: float) -> List[Dict[str, Any]]:
        """Detect anomalies in batch normalization parameters."""
        anomalies = []
        
        for name, stats in self.batch_norm_stats.items():
            if len(stats['mean_history']) <= 10:
                continue
                
            anomalies.extend(self._check_batch_norm_mean_anomalies(
                name, stats, threshold
            ))
            anomalies.extend(self._check_batch_norm_var_anomalies(
                name, stats, threshold
            ))
        
        return anomalies

    def _check_batch_norm_mean_anomalies(self, name: str, stats: Dict[str, Any], 
                                       threshold: float) -> List[Dict[str, Any]]:
        """Check for anomalies in batch norm mean values."""
        anomalies = []
        mean_history = stats['mean_history'][-100:]
        
        mean_mean = np.mean(mean_history)
        mean_std = np.std(mean_history)
        
        if mean_std > 0:
            current_mean = mean_history[-1]
            z_score_mean = abs(current_mean - mean_mean) / mean_std
            
            if z_score_mean > threshold:
                anomalies.append({
                    'type': 'batch_norm_mean',
                    'name': name,
                    'value': current_mean,
                    'mean': mean_mean,
                    'std': mean_std,
                    'z_score': z_score_mean,
                })
        
        return anomalies

    def _check_batch_norm_var_anomalies(self, name: str, stats: Dict[str, Any], 
                                      threshold: float) -> List[Dict[str, Any]]:
        """Check for anomalies in batch norm variance values."""
        anomalies = []
        var_history = stats['var_history'][-100:]
        
        var_mean = np.mean(var_history)
        var_std = np.std(var_history)
        
        if var_std > 0:
            current_var = var_history[-1]
            z_score_var = abs(current_var - var_mean) / var_std
            
            if z_score_var > threshold:
                anomalies.append({
                    'type': 'batch_norm_var',
                    'name': name,
                    'value': current_var,
                    'mean': var_mean,
                    'std': var_std,
                    'z_score': z_score_var,
                })
        
        return anomalies
    
    def adjust_parameters(self, anomalies: List[Dict[str, Any]]):
        """
        Adjust normalization parameters based on detected anomalies.
        
        Args:
            anomalies: List of detected anomalies
        """
        for anomaly in anomalies:
            self._process_single_anomaly(anomaly)

    def _process_single_anomaly(self, anomaly: Dict[str, Any]) -> None:
        """Process and adjust parameters for a single anomaly."""
        name = anomaly['name']
        anomaly_type = anomaly['type']
        
        # Log the anomaly
        self.logger.warning(
            f"Adjusting parameters for anomaly: {anomaly_type} in {name}, "
            f"value={anomaly['value']:.4f}, z_score={anomaly['z_score']:.4f}"
        )
        
        # Find the module
        module = self._find_module_by_name(name)
        if module is None:
            return
        
        # Adjust parameters based on anomaly type
        self._adjust_parameters_by_type(module, anomaly_type)

    def _find_module_by_name(self, name: str) -> Optional[nn.Module]:
        """Find a module by its name in the model."""
        for module_name, module in self.model.named_modules():
            if module_name == name:
                return module
        return None

    def _adjust_parameters_by_type(self, module: nn.Module, anomaly_type: str) -> None:
        """Adjust module parameters based on the anomaly type."""
        if anomaly_type.startswith('spectral_norm'):
            self._adjust_spectral_norm_parameters(module)
        elif anomaly_type.startswith('layer_norm'):
            self._adjust_layer_norm_parameters(module)
        elif anomaly_type.startswith('batch_norm'):
            self._adjust_batch_norm_parameters(module)

    def _adjust_spectral_norm_parameters(self, module: nn.Module) -> None:
        """Adjust spectral normalization parameters."""
        # For spectral norm, we can't directly adjust the parameters
        # Just log the anomaly for now
        pass

    def _adjust_layer_norm_parameters(self, module: nn.Module) -> None:
        """Adjust layer normalization parameters."""
        if not (hasattr(module, 'weight') and hasattr(module, 'bias')):
            return
            
        with torch.no_grad():
            # Reduce the variance of the weight
            if module.weight.std() > 1.0:
                module.weight.data = module.weight.data * (1.0 / module.weight.std())
            
            # Center the bias
            if abs(module.bias.mean()) > 0.1:
                module.bias.data = module.bias.data - module.bias.mean()

    def _adjust_batch_norm_parameters(self, module: nn.Module) -> None:
        """Adjust batch normalization parameters."""
        if not (hasattr(module, 'weight') and hasattr(module, 'bias')):
            return
            
        with torch.no_grad():
            # Reduce the variance of the weight
            if module.weight.std() > 1.0:
                module.weight.data = module.weight.data * (1.0 / module.weight.std())
            
            # Center the bias
            if abs(module.bias.mean()) > 0.1:
                module.bias.data = module.bias.data - module.bias.mean()


class NormalizationConfig:
    """
    Configuration for normalization layers in neural networks.
    
    This class provides a centralized way to configure normalization options
    for different parts of a neural network.
    """
    
    def __init__(
        self,
        normalization_type: str = 'layer',  # 'layer', 'batch', 'spectral', 'none'
        use_spectral_norm: bool = True,
        use_layer_norm: bool = True,
        use_batch_norm: bool = False,
        spectral_norm_n_power_iterations: int = 1,
        layer_norm_eps: float = 1e-5,
        batch_norm_eps: float = 1e-5,
        batch_norm_momentum: float = 0.1,
    ):
        """
        Initialize normalization configuration.
        
        Args:
            normalization_type: Primary normalization type to use
            use_spectral_norm: Whether to use spectral normalization
            use_layer_norm: Whether to use layer normalization
            use_batch_norm: Whether to use batch normalization
            spectral_norm_n_power_iterations: Number of power iterations for spectral norm
            layer_norm_eps: Epsilon value for layer normalization
            batch_norm_eps: Epsilon value for batch normalization
            batch_norm_momentum: Momentum value for batch normalization
        """
        self.normalization_type = normalization_type
        self.use_spectral_norm = use_spectral_norm
        self.use_layer_norm = use_layer_norm
        self.use_batch_norm = use_batch_norm
        self.spectral_norm_n_power_iterations = spectral_norm_n_power_iterations
        self.layer_norm_eps = layer_norm_eps
        self.batch_norm_eps = batch_norm_eps
        self.batch_norm_momentum = batch_norm_momentum
    
    def apply_normalization(
        self,
        module: nn.Module,
        features: int,
        normalization_type: Optional[str] = None
    ) -> nn.Module:
        """
        Apply normalization to a module based on configuration.
        
        Args:
            module: Module to apply normalization to
            features: Number of features for normalization layers
            normalization_type: Override the default normalization type
            
        Returns:
            Module with normalization applied
        """
        norm_type = normalization_type or self.normalization_type
        
        # Apply spectral normalization if enabled
        if self.use_spectral_norm and hasattr(module, 'weight'):
            module = nn.utils.spectral_norm(
                module,
                n_power_iterations=self.spectral_norm_n_power_iterations
            )
        
        # Create a sequential module with the original module and normalization
        layers = [module]
        
        # Add appropriate normalization layer based on type
        norm_layer = self._create_normalization_layer(features, norm_type)
        if norm_layer is not None:
            layers.append(norm_layer)
        
        # If only one layer, return it directly
        if len(layers) == 1:
            return layers[0]
        
        # Otherwise, return a sequential module
        return nn.Sequential(*layers)
    
    def get_normalization_layer(self, features: int, normalization_type: Optional[str] = None) -> Optional[nn.Module]:
        """
        Get a standalone normalization layer based on configuration.
        
        Args:
            features: Number of features for the normalization layer
            normalization_type: Override the default normalization type
            
        Returns:
            Normalization layer or None if normalization is disabled
        """
        norm_type = normalization_type or self.normalization_type
        return self._create_normalization_layer(features, norm_type) 
   
    def _create_normalization_layer(self, features: int, norm_type: str) -> Optional[nn.Module]:
        """Create a normalization layer based on type and configuration."""
        if norm_type == 'layer' and self.use_layer_norm:
            return nn.LayerNorm(features, eps=self.layer_norm_eps)
        elif norm_type == 'batch' and self.use_batch_norm:
            return nn.BatchNorm1d(features, eps=self.batch_norm_eps, momentum=self.batch_norm_momentum)
        else:
            return None