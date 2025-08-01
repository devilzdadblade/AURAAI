"""
Comprehensive training and runtime monitoring for RL models.

This module provides tools for monitoring training metrics, detecting convergence,
and visualizing model performance during training and inference.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import logging
import os
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import deque
import json
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    step: int
    timestamp: float
    policy_loss: float
    value_loss: float
    total_loss: float
    episode_reward: Optional[float] = None
    episode_length: Optional[int] = None
    exploration_rate: Optional[float] = None
    learning_rate: Optional[float] = None
    gradient_norm: Optional[float] = None
    entropy: Optional[float] = None
    q_values_mean: Optional[float] = None
    q_values_std: Optional[float] = None
    q_values_min: Optional[float] = None
    q_values_max: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ConvergenceMetrics:
    """Metrics for convergence detection."""
    is_converged: bool
    confidence: float
    stable_episodes: int
    reward_mean: float
    reward_std: float
    value_loss_trend: float  # Negative means decreasing (good)
    policy_loss_trend: float  # Negative means decreasing (good)


class TrainingMonitor:
    """
    Comprehensive training monitoring for RL models.
    
    This class provides tools for:
    - Tracking training metrics (loss, rewards, etc.)
    - Detecting convergence
    - Visualizing training progress
    - TensorBoard integration
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        model_name: str = "rl_model",
        metrics_buffer_size: int = 10000,
        convergence_window: int = 100,
        convergence_threshold: float = 0.05,
        use_tensorboard: bool = True,
        log_frequency: int = 10
    ):
        """
        Initialize the training monitor.
        
        Args:
            log_dir: Directory for logs and visualizations
            model_name: Name of the model being monitored
            metrics_buffer_size: Maximum number of metrics to store in memory
            convergence_window: Window size for convergence detection
            convergence_threshold: Threshold for convergence detection
            use_tensorboard: Whether to use TensorBoard for visualization
            log_frequency: How often to log metrics (in steps)
        """
        self.log_dir = log_dir
        self.model_name = model_name
        self.metrics_buffer_size = metrics_buffer_size
        self.convergence_window = convergence_window
        self.convergence_threshold = convergence_threshold
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.log_frequency = log_frequency
        
        # Create log directory
        self.run_dir = os.path.join(
            log_dir, 
            f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics = deque(maxlen=metrics_buffer_size)
        self.episode_rewards = deque(maxlen=metrics_buffer_size)
        self.episode_lengths = deque(maxlen=metrics_buffer_size)
        self.value_losses = deque(maxlen=metrics_buffer_size)
        self.policy_losses = deque(maxlen=metrics_buffer_size)
        self.q_value_stats = deque(maxlen=metrics_buffer_size)
        self.gradient_norms = deque(maxlen=metrics_buffer_size)
        self.entropies = deque(maxlen=metrics_buffer_size)
        
        # Initialize TensorBoard writer
        self.writer = None
        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir=self.run_dir)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize step counter
        self.step_counter = 0
        
        # Initialize convergence detection
        self.is_converged = False
        self.convergence_confidence = 0.0
        self.stable_episodes = 0
        
        self.logger.info(f"Training monitor initialized. Logs will be saved to {self.run_dir}")
    
    def log_metrics(
        self,
        policy_loss: float,
        value_loss: float,
        total_loss: float,
        q_values: Optional[torch.Tensor] = None,
        episode_reward: Optional[float] = None,
        episode_length: Optional[int] = None,
        exploration_rate: Optional[float] = None,
        learning_rate: Optional[float] = None,
        model: Optional[nn.Module] = None,
        entropy: Optional[float] = None
    ) -> TrainingMetrics:
        """
        Log training metrics.
        
        Args:
            policy_loss: Policy loss value
            value_loss: Value loss value
            total_loss: Total loss value
            q_values: Q-values tensor (for statistics)
            episode_reward: Episode reward (if available)
            episode_length: Episode length (if available)
            exploration_rate: Exploration rate (if available)
            learning_rate: Learning rate (if available)
            model: Model (for gradient statistics)
            entropy: Action entropy (if available)
            
        Returns:
            TrainingMetrics object with logged metrics
        """
        self.step_counter += 1
        
        # Calculate gradient norm if model is provided
        gradient_norm = None
        if model is not None:
            gradient_norm = self._calculate_gradient_norm(model)
        
        # Calculate Q-value statistics if provided
        q_values_mean = None
        q_values_std = None
        q_values_min = None
        q_values_max = None
        
        if q_values is not None:
            with torch.no_grad():
                q_values_mean = q_values.mean().item()
                q_values_std = q_values.std().item()
                q_values_min = q_values.min().item()
                q_values_max = q_values.max().item()
        
        # Create metrics object
        metrics = TrainingMetrics(
            step=self.step_counter,
            timestamp=time.time(),
            policy_loss=policy_loss,
            value_loss=value_loss,
            total_loss=total_loss,
            episode_reward=episode_reward,
            episode_length=episode_length,
            exploration_rate=exploration_rate,
            learning_rate=learning_rate,
            gradient_norm=gradient_norm,
            entropy=entropy,
            q_values_mean=q_values_mean,
            q_values_std=q_values_std,
            q_values_min=q_values_min,
            q_values_max=q_values_max
        )
        
        # Store metrics
        self.metrics.append(metrics)
        
        # Store specific metrics in their respective deques
        if episode_reward is not None:
            self.episode_rewards.append(episode_reward)
        
        if episode_length is not None:
            self.episode_lengths.append(episode_length)
        
        self.value_losses.append(value_loss)
        self.policy_losses.append(policy_loss)
        
        if gradient_norm is not None:
            self.gradient_norms.append(gradient_norm)
        
        if entropy is not None:
            self.entropies.append(entropy)
        
        if q_values_mean is not None:
            self.q_value_stats.append({
                "mean": q_values_mean,
                "std": q_values_std,
                "min": q_values_min,
                "max": q_values_max
            })
        
        # Log metrics periodically
        if self.step_counter % self.log_frequency == 0:
            self._log_periodic_metrics(metrics)
        
        # Check for convergence if we have enough data
        if len(self.episode_rewards) >= self.convergence_window:
            self._check_convergence()
        
        return metrics
    
    def _calculate_gradient_norm(self, model: nn.Module) -> float:
        """
        Calculate gradient norm for a model.
        
        Args:
            model: PyTorch model
            
        Returns:
            Gradient norm
        """
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        
        total_norm = total_norm ** 0.5
        return total_norm
    
    def _log_periodic_metrics(self, metrics: TrainingMetrics):
        """
        Log metrics periodically.
        
        Args:
            metrics: Current metrics
        """
        self._log_to_console(metrics)
        self._log_to_tensorboard(metrics)

    def _log_to_console(self, metrics: TrainingMetrics):
        """Helper to log metrics to console."""
        log_str = f"Step {metrics.step}: Loss={metrics.total_loss:.4f} "
        if metrics.episode_reward is not None:
            log_str += f"Reward={metrics.episode_reward:.2f} "
        if metrics.gradient_norm is not None:
            log_str += f"GradNorm={metrics.gradient_norm:.4f} "
        if metrics.entropy is not None:
            log_str += f"Entropy={metrics.entropy:.4f} "
        if metrics.q_values_mean is not None:
            log_str += f"Q-mean={metrics.q_values_mean:.4f} "
        self.logger.info(log_str)

    def _log_to_tensorboard(self, metrics: TrainingMetrics):
        """Helper to log metrics to TensorBoard."""
        if self.writer is None:
            return
        self.writer.add_scalar("Loss/policy", metrics.policy_loss, metrics.step)
        self.writer.add_scalar("Loss/value", metrics.value_loss, metrics.step)
        self.writer.add_scalar("Loss/total", metrics.total_loss, metrics.step)
        if metrics.episode_reward is not None:
            self.writer.add_scalar("Reward/episode", metrics.episode_reward, metrics.step)
        if metrics.episode_length is not None:
            self.writer.add_scalar("Episode/length", metrics.episode_length, metrics.step)
        if metrics.exploration_rate is not None:
            self.writer.add_scalar("Training/exploration_rate", metrics.exploration_rate, metrics.step)
        if metrics.learning_rate is not None:
            self.writer.add_scalar("Training/learning_rate", metrics.learning_rate, metrics.step)
        if metrics.gradient_norm is not None:
            self.writer.add_scalar("Gradients/norm", metrics.gradient_norm, metrics.step)
        if metrics.entropy is not None:
            self.writer.add_scalar("Policy/entropy", metrics.entropy, metrics.step)
        if metrics.q_values_mean is not None:
            self.writer.add_scalar("Q-values/mean", metrics.q_values_mean, metrics.step)
            self.writer.add_scalar("Q-values/std", metrics.q_values_std, metrics.step)
            self.writer.add_scalar("Q-values/min", metrics.q_values_min, metrics.step)
            self.writer.add_scalar("Q-values/max", metrics.q_values_max, metrics.step)
            self.writer.add_scalar("Q-values/range", metrics.q_values_max - metrics.q_values_min, metrics.step)
    
    def _check_convergence(self) -> ConvergenceMetrics:
        """
        Check for convergence based on recent metrics.
        
        Returns:
            ConvergenceMetrics object with convergence information
        """
        # Get recent rewards
        recent_rewards = list(self.episode_rewards)[-self.convergence_window:]
        
        # Calculate reward statistics
        reward_mean = np.mean(recent_rewards)
        reward_std = np.std(recent_rewards)
        reward_cv = reward_std / (abs(reward_mean) + 1e-8)  # Coefficient of variation
        
        # Calculate loss trends
        recent_value_losses = list(self.value_losses)[-self.convergence_window:]
        recent_policy_losses = list(self.policy_losses)[-self.convergence_window:]
        
        # Use linear regression to estimate trend
        x = np.arange(len(recent_value_losses))
        value_loss_trend = np.polyfit(x, recent_value_losses, 1)[0] if len(recent_value_losses) > 1 else 0
        policy_loss_trend = np.polyfit(x, recent_policy_losses, 1)[0] if len(recent_policy_losses) > 1 else 0
        
        # Check for convergence
        is_stable = reward_cv < self.convergence_threshold
        losses_decreasing = value_loss_trend < 0 and policy_loss_trend < 0
        
        if is_stable:
            self.stable_episodes += 1
        else:
            self.stable_episodes = 0
        
        # Calculate convergence confidence
        confidence = min(1.0, self.stable_episodes / (self.convergence_window / 2))
        
        # Update convergence status
        self.is_converged = confidence > 0.8 and losses_decreasing
        self.convergence_confidence = confidence
        
        # Create convergence metrics
        convergence_metrics = ConvergenceMetrics(
            is_converged=self.is_converged,
            confidence=confidence,
            stable_episodes=self.stable_episodes,
            reward_mean=reward_mean,
            reward_std=reward_std,
            value_loss_trend=value_loss_trend,
            policy_loss_trend=policy_loss_trend
        )
        
        # Log convergence status if changed
        if self.is_converged and confidence > 0.9:
            self.logger.info(
                f"Convergence detected at step {self.step_counter} "
                f"with confidence {confidence:.2f}. "
                f"Reward mean: {reward_mean:.2f}, std: {reward_std:.2f}"
            )
            
            # Log to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar("Convergence/confidence", confidence, self.step_counter)
                self.writer.add_scalar("Convergence/reward_mean", reward_mean, self.step_counter)
                self.writer.add_scalar("Convergence/reward_std", reward_std, self.step_counter)
        
        return convergence_metrics
    
    def get_convergence_status(self) -> ConvergenceMetrics:
        """
        Get current convergence status.
        
        Returns:
            ConvergenceMetrics object with convergence information
        """
        if len(self.episode_rewards) >= self.convergence_window:
            return self._check_convergence()
        else:
            # Not enough data for convergence detection
            return ConvergenceMetrics(
                is_converged=False,
                confidence=0.0,
                stable_episodes=0,
                reward_mean=0.0,
                reward_std=0.0,
                value_loss_trend=0.0,
                policy_loss_trend=0.0
            )
    
    def get_recent_metrics(self, n: int = 100) -> List[TrainingMetrics]:
        """
        Get recent metrics.
        
        Args:
            n: Number of recent metrics to return
            
        Returns:
            List of recent TrainingMetrics objects
        """
        return list(self.metrics)[-n:]
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics of training.
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            "steps": self.step_counter,
            "convergence": {
                "is_converged": self.is_converged,
                "confidence": self.convergence_confidence,
                "stable_episodes": self.stable_episodes
            }
        }
        
        # Add reward statistics if available
        if self.episode_rewards:
            summary["rewards"] = {
                "mean": np.mean(self.episode_rewards),
                "std": np.std(self.episode_rewards),
                "min": np.min(self.episode_rewards),
                "max": np.max(self.episode_rewards),
                "last": self.episode_rewards[-1]
            }
        
        # Add loss statistics
        if self.value_losses:
            summary["value_loss"] = {
                "mean": np.mean(self.value_losses),
                "std": np.std(self.value_losses),
                "min": np.min(self.value_losses),
                "max": np.max(self.value_losses),
                "last": self.value_losses[-1]
            }
        
        if self.policy_losses:
            summary["policy_loss"] = {
                "mean": np.mean(self.policy_losses),
                "std": np.std(self.policy_losses),
                "min": np.min(self.policy_losses),
                "max": np.max(self.policy_losses),
                "last": self.policy_losses[-1]
            }
        
        # Add gradient statistics if available
        if self.gradient_norms:
            summary["gradient_norm"] = {
                "mean": np.mean(self.gradient_norms),
                "std": np.std(self.gradient_norms),
                "min": np.min(self.gradient_norms),
                "max": np.max(self.gradient_norms),
                "last": self.gradient_norms[-1]
            }
        
        # Add entropy statistics if available
        if self.entropies:
            summary["entropy"] = {
                "mean": np.mean(self.entropies),
                "std": np.std(self.entropies),
                "min": np.min(self.entropies),
                "max": np.max(self.entropies),
                "last": self.entropies[-1]
            }
        
        # Add Q-value statistics if available
        if self.q_value_stats:
            means = [stats["mean"] for stats in self.q_value_stats]
            stds = [stats["std"] for stats in self.q_value_stats]
            mins = [stats["min"] for stats in self.q_value_stats]
            maxs = [stats["max"] for stats in self.q_value_stats]
            
            summary["q_values"] = {
                "mean": {
                    "mean": np.mean(means),
                    "std": np.std(means),
                    "min": np.min(means),
                    "max": np.max(means),
                    "last": means[-1]
                },
                "std": {
                    "mean": np.mean(stds),
                    "std": np.std(stds),
                    "min": np.min(stds),
                    "max": np.max(stds),
                    "last": stds[-1]
                },
                "min": {
                    "mean": np.mean(mins),
                    "std": np.std(mins),
                    "min": np.min(mins),
                    "max": np.max(mins),
                    "last": mins[-1]
                },
                "max": {
                    "mean": np.mean(maxs),
                    "std": np.std(maxs),
                    "min": np.min(maxs),
                    "max": np.max(maxs),
                    "last": maxs[-1]
                }
            }
        
        return summary
    
    def save_metrics(self, filename: str = "training_metrics.json"):
        """
        Save metrics to a file.
        
        Args:
            filename: Name of the file to save metrics to
        """
        filepath = os.path.join(self.run_dir, filename)
        
        # Convert metrics to list of dictionaries
        metrics_list = [m.to_dict() for m in self.metrics]
        
        # Save to file
        with open(filepath, "w") as f:
            json.dump(metrics_list, f, indent=2)
        
        self.logger.info(f"Metrics saved to {filepath}")
    
    def plot_metrics(self, save_dir: Optional[str] = None):
        """
        Plot training metrics.
        
        Args:
            save_dir: Directory to save plots to (defaults to run_dir)
        """
        if not self.metrics:
            self.logger.warning("No metrics to plot")
            return
        
        save_dir = save_dir or self.run_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Extract data for plotting
        steps = [m.step for m in self.metrics]
        policy_losses = [m.policy_loss for m in self.metrics]
        value_losses = [m.value_loss for m in self.metrics]
        total_losses = [m.total_loss for m in self.metrics]
        
        # Plot losses
        plt.figure(figsize=(10, 6))
        plt.plot(steps, policy_losses, label="Policy Loss")
        plt.plot(steps, value_losses, label="Value Loss")
        plt.plot(steps, total_losses, label="Total Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training Losses")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "losses.png"))
        plt.close()
        
        # Plot rewards if available
        episode_rewards = [m.episode_reward for m in self.metrics if m.episode_reward is not None]
        if episode_rewards:
            reward_steps = [m.step for m in self.metrics if m.episode_reward is not None]
            
            plt.figure(figsize=(10, 6))
            plt.plot(reward_steps, episode_rewards)
            plt.xlabel("Steps")
            plt.ylabel("Reward")
            plt.title("Episode Rewards")
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, "rewards.png"))
            plt.close()
        
        # Plot gradient norms if available
        gradient_norms = [m.gradient_norm for m in self.metrics if m.gradient_norm is not None]
        if gradient_norms:
            norm_steps = [m.step for m in self.metrics if m.gradient_norm is not None]
            
            plt.figure(figsize=(10, 6))
            plt.plot(norm_steps, gradient_norms)
            plt.xlabel("Steps")
            plt.ylabel("Gradient Norm")
            plt.title("Gradient Norms")
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, "gradient_norms.png"))
            plt.close()
        
        # Plot Q-value statistics if available
        q_means = [m.q_values_mean for m in self.metrics if m.q_values_mean is not None]
        q_stds = [m.q_values_std for m in self.metrics if m.q_values_std is not None]
        q_mins = [m.q_values_min for m in self.metrics if m.q_values_min is not None]
        q_maxs = [m.q_values_max for m in self.metrics if m.q_values_max is not None]
        
        if q_means:
            q_steps = [m.step for m in self.metrics if m.q_values_mean is not None]
            
            plt.figure(figsize=(10, 6))
            plt.plot(q_steps, q_means, label="Mean")
            plt.fill_between(q_steps, 
                            [m - s for m, s in zip(q_means, q_stds)],
                            [m + s for m, s in zip(q_means, q_stds)],
                            alpha=0.3)
            plt.plot(q_steps, q_mins, label="Min", linestyle="--")
            plt.plot(q_steps, q_maxs, label="Max", linestyle="--")
            plt.xlabel("Steps")
            plt.ylabel("Q-Value")
            plt.title("Q-Value Statistics")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, "q_values.png"))
            plt.close()
        
        # Plot entropy if available
        entropies = [m.entropy for m in self.metrics if m.entropy is not None]
        if entropies:
            entropy_steps = [m.step for m in self.metrics if m.entropy is not None]
            
            plt.figure(figsize=(10, 6))
            plt.plot(entropy_steps, entropies)
            plt.xlabel("Steps")
            plt.ylabel("Entropy")
            plt.title("Action Entropy")
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, "entropy.png"))
            plt.close()
        
        self.logger.info(f"Plots saved to {save_dir}")
    
    def close(self):
        """Close the monitor and clean up resources."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        
        # Save metrics before closing
        self.save_metrics()
        
        # Plot metrics before closing
        self.plot_metrics()
        
        self.logger.info("Training monitor closed")


class RuntimeMonitor:
    """
    Runtime monitoring for RL models during inference.
    
    This class provides tools for:
    - Tracking inference latency
    - Monitoring action distributions
    - Detecting anomalies in model behavior
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        model_name: str = "rl_model",
        metrics_buffer_size: int = 1000,
        log_frequency: int = 100,
        latency_threshold_ms: float = 10.0
    ):
        """
        Initialize the runtime monitor.
        
        Args:
            log_dir: Directory for logs and visualizations
            model_name: Name of the model being monitored
            metrics_buffer_size: Maximum number of metrics to store in memory
            log_frequency: How often to log metrics (in steps)
            latency_threshold_ms: Threshold for latency alerts (in milliseconds)
        """
        self.log_dir = log_dir
        self.model_name = model_name
        self.metrics_buffer_size = metrics_buffer_size
        self.log_frequency = log_frequency
        self.latency_threshold_ms = latency_threshold_ms
        
        # Create log directory
        self.run_dir = os.path.join(
            log_dir, 
            f"{model_name}_runtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.inference_times = deque(maxlen=metrics_buffer_size)
        self.action_distributions = deque(maxlen=metrics_buffer_size)
        self.q_value_distributions = deque(maxlen=metrics_buffer_size)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize step counter
        self.step_counter = 0
        
        # Initialize TensorBoard writer
        self.writer = None
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=self.run_dir)
        
        self.logger.info(f"Runtime monitor initialized. Logs will be saved to {self.run_dir}")
    
    def log_inference(
        self,
        inference_time_ms: float,
        action: int,
        q_values: Optional[torch.Tensor] = None,
        _state: Optional[torch.Tensor] = None
    ):
        """
        Log inference metrics.
        
        Args:
            inference_time_ms: Inference time in milliseconds
            action: Selected action
            q_values: Q-values tensor (if available)
            _state: State tensor (if available) - currently unused
        """
        self.step_counter += 1
        
        # Store inference time
        self.inference_times.append(inference_time_ms)
        
        # Store action
        self.action_distributions.append(action)
        
        # Store Q-value distribution if available
        if q_values is not None:
            with torch.no_grad():
                self.q_value_distributions.append(q_values.cpu().numpy())
        
        # Check for latency issues
        if inference_time_ms > self.latency_threshold_ms:
            self.logger.warning(
                f"Inference latency ({inference_time_ms:.2f} ms) exceeded threshold "
                f"({self.latency_threshold_ms:.2f} ms) at step {self.step_counter}"
            )
        
        # Log metrics periodically
        if self.step_counter % self.log_frequency == 0:
            self._log_periodic_metrics()
    
    def _log_periodic_metrics(self):
        """Log metrics periodically."""
        # Calculate statistics
        if self.inference_times:
            mean_time = np.mean(self.inference_times)
            max_time = np.max(self.inference_times)
            p95_time = np.percentile(self.inference_times, 95)
            p99_time = np.percentile(self.inference_times, 99)
            
            # Log to console
            self.logger.info(
                f"Inference stats (step {self.step_counter}): "
                f"Mean={mean_time:.2f} ms, "
                f"Max={max_time:.2f} ms, "
                f"P95={p95_time:.2f} ms, "
                f"P99={p99_time:.2f} ms"
            )
            
            # Log to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar("Inference/mean_time_ms", mean_time, self.step_counter)
                self.writer.add_scalar("Inference/max_time_ms", max_time, self.step_counter)
                self.writer.add_scalar("Inference/p95_time_ms", p95_time, self.step_counter)
                self.writer.add_scalar("Inference/p99_time_ms", p99_time, self.step_counter)
        
        # Log action distribution
        if self.action_distributions:
            action_counts = {}
            for action in self.action_distributions:
                if action not in action_counts:
                    action_counts[action] = 0
                action_counts[action] += 1
            
            # Log to TensorBoard
            if self.writer is not None:
                for action, count in action_counts.items():
                    self.writer.add_scalar(
                        f"Actions/frequency_{action}", 
                        count / len(self.action_distributions), 
                        self.step_counter
                    )
    
    def get_latency_stats(self) -> Dict[str, float]:
        """
        Get latency statistics.
        
        Returns:
            Dictionary with latency statistics
        """
        if not self.inference_times:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0
            }
        
        return {
            "mean": np.mean(self.inference_times),
            "std": np.std(self.inference_times),
            "min": np.min(self.inference_times),
            "max": np.max(self.inference_times),
            "p50": np.percentile(self.inference_times, 50),
            "p95": np.percentile(self.inference_times, 95),
            "p99": np.percentile(self.inference_times, 99)
        }
    
    def get_action_distribution(self) -> Dict[int, float]:
        """
        Get action distribution.
        
        Returns:
            Dictionary mapping actions to frequencies
        """
        if not self.action_distributions:
            return {}
        
        action_counts = {}
        for action in self.action_distributions:
            if action not in action_counts:
                action_counts[action] = 0
            action_counts[action] += 1
        
        # Convert counts to frequencies
        total = len(self.action_distributions)
        return {action: count / total for action, count in action_counts.items()}
    
    def detect_anomalies(self, threshold: float = 3.0) -> List[Dict[str, Any]]:
        """
        Detect anomalies in runtime behavior.

        Args:
            threshold: Z-score threshold for anomaly detection

        Returns:
            List of detected anomalies
        """
        anomalies = []
        anomalies.extend(self._detect_latency_anomalies(threshold))
        anomalies.extend(self._detect_action_distribution_anomalies())
        return anomalies

    def _detect_latency_anomalies(self, threshold: float) -> List[Dict[str, Any]]:
        """Detect latency anomalies based on z-score."""
        anomalies = []
        if len(self.inference_times) > 10:
            mean = np.mean(self.inference_times)
            std = np.std(self.inference_times)
            if std > 0:
                for i, time in enumerate(self.inference_times):
                    z_score = abs(time - mean) / std
                    if z_score > threshold:
                        anomalies.append({
                            "type": "latency",
                            "value": time,
                            "mean": mean,
                            "std": std,
                            "z_score": z_score,
                            "step": self.step_counter - len(self.inference_times) + i
                        })
        return anomalies

    def _detect_action_distribution_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies in action distribution."""
        anomalies = []
        if len(self.action_distributions) > 10:
            action_counts = {}
            for action in self.action_distributions:
                action_counts[action] = action_counts.get(action, 0) + 1
            total = len(self.action_distributions)
            if len(action_counts) > 0:
                expected_freq = 1.0 / len(action_counts)
                for action, count in action_counts.items():
                    freq = count / total
                    if freq > expected_freq * 3:
                        anomalies.append({
                            "type": "action_distribution",
                            "action": action,
                            "frequency": freq,
                            "expected_frequency": expected_freq,
                            "ratio": freq / expected_freq
                        })
        return anomalies
    
    def save_metrics(self, filename: str = "runtime_metrics.json"):
        """
        Save metrics to a file.
        
        Args:
            filename: Name of the file to save metrics to
        """
        filepath = os.path.join(self.run_dir, filename)
        
        # Create metrics dictionary
        metrics = {
            "latency_stats": self.get_latency_stats(),
            "action_distribution": self.get_action_distribution(),
            "steps": self.step_counter
        }
        
        # Save to file
        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Metrics saved to {filepath}")
    
    def plot_metrics(self, save_dir: Optional[str] = None):
        """
        Plot runtime metrics.
        
        Args:
            save_dir: Directory to save plots to (defaults to run_dir)
        """
        save_dir = save_dir or self.run_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Plot inference times
        if self.inference_times:
            plt.figure(figsize=(10, 6))
            plt.plot(list(self.inference_times))
            plt.axhline(y=self.latency_threshold_ms, color='r', linestyle='--', label=f"Threshold ({self.latency_threshold_ms} ms)")
            plt.xlabel("Inference Step")
            plt.ylabel("Time (ms)")
            plt.title("Inference Latency")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, "inference_latency.png"))
            plt.close()
        
        # Plot action distribution
        if self.action_distributions:
            action_counts = {}
            for action in self.action_distributions:
                if action not in action_counts:
                    action_counts[action] = 0
                action_counts[action] += 1
            
            plt.figure(figsize=(10, 6))
            plt.bar(action_counts.keys(), action_counts.values())
            plt.xlabel("Action")
            plt.ylabel("Count")
            plt.title("Action Distribution")
            plt.grid(True, axis='y')
            plt.savefig(os.path.join(save_dir, "action_distribution.png"))
            plt.close()
        
        self.logger.info(f"Plots saved to {save_dir}")
    
    def close(self):
        """Close the monitor and clean up resources."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        
        # Save metrics before closing
        self.save_metrics()
        
        # Plot metrics before closing
        self.plot_metrics()
        
        self.logger.info("Runtime monitor closed")