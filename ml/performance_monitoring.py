"""
Performance monitoring for reinforcement learning models.

This module provides tools for monitoring performance metrics during training
and detecting performance regressions.
"""

import torch
import numpy as np
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class PerformanceAlert:
    """Performance alert for training loop."""
    metric_name: str
    current_value: float
    threshold_value: float
    severity: str  # 'warning', 'critical'
    message: str
    timestamp: float


class PerformanceMonitor:
    """
    Monitors performance metrics during training and detects regressions.
    
    This class provides methods for:
    - Collecting real-time performance metrics during training
    - Detecting performance regressions compared to baselines
    - Generating performance alerts when thresholds are exceeded
    - Suggesting performance optimizations based on monitoring results
    """
    
    def __init__(self, 
                 latency_threshold_ms: float = 10.0,
                 history_size: int = 1000,
                 alert_callback: Optional[Callable[[PerformanceAlert], None]] = None):
        """
        Initialize the PerformanceMonitor.
        
        Args:
            latency_threshold_ms: Threshold for inference latency in milliseconds
            history_size: Size of the metrics history
            alert_callback: Optional callback function for alerts
        """
        self.latency_threshold_ms = latency_threshold_ms
        self.history_size = history_size
        self.alert_callback = alert_callback
        
        # Performance metrics history
        self.training_step_times = deque(maxlen=history_size)
        self.forward_times = deque(maxlen=history_size)
        self.backward_times = deque(maxlen=history_size)
        self.optimizer_times = deque(maxlen=history_size)
        self.inference_times = deque(maxlen=history_size)
        self.batch_sizes = deque(maxlen=history_size)
        self.samples_per_second = deque(maxlen=history_size)
        
        # Baseline metrics for regression detection
        self.baseline_metrics = {}
        
        # Alert history
        self.alerts = deque(maxlen=history_size)
        
        logger.info(f"PerformanceMonitor initialized with latency threshold: {latency_threshold_ms}ms")
    
    def record_training_step(self, 
                            batch_size: int,
                            forward_time_ms: float,
                            backward_time_ms: float,
                            optimizer_time_ms: float,
                            total_time_ms: float) -> Dict[str, float]:
        """
        Record performance metrics for a training step.
        
        Args:
            batch_size: Size of the batch
            forward_time_ms: Time for forward pass in milliseconds
            backward_time_ms: Time for backward pass in milliseconds
            optimizer_time_ms: Time for optimizer step in milliseconds
            total_time_ms: Total time for the training step in milliseconds
            
        Returns:
            Dictionary with performance metrics
        """
        # Calculate samples per second
        samples_per_second = (batch_size * 1000) / total_time_ms
        
        # Record metrics
        self.training_step_times.append(total_time_ms)
        self.forward_times.append(forward_time_ms)
        self.backward_times.append(backward_time_ms)
        self.optimizer_times.append(optimizer_time_ms)
        self.batch_sizes.append(batch_size)
        self.samples_per_second.append(samples_per_second)
        
        # Create metrics dictionary
        metrics = {
            'batch_size': batch_size,
            'forward_time_ms': forward_time_ms,
            'backward_time_ms': backward_time_ms,
            'optimizer_time_ms': optimizer_time_ms,
            'total_time_ms': total_time_ms,
            'samples_per_second': samples_per_second
        }
        
        # Check for performance regressions
        if len(self.training_step_times) >= 10:
            self._check_for_regressions(metrics)
        
        return metrics
    
    def record_inference(self, latency_ms: float) -> Dict[str, float]:
        """
        Record inference latency.
        
        Args:
            latency_ms: Inference latency in milliseconds
            
        Returns:
            Dictionary with inference metrics
        """
        # Record latency
        self.inference_times.append(latency_ms)
        
        # Create metrics dictionary
        metrics = {
            'inference_latency_ms': latency_ms,
            'meets_latency_target': latency_ms <= self.latency_threshold_ms
        }
        
        # Check if latency exceeds threshold
        if latency_ms > self.latency_threshold_ms:
            self._create_alert(
                metric_name='inference_latency_ms',
                current_value=latency_ms,
                threshold_value=self.latency_threshold_ms,
                severity='warning',
                message=f"Inference latency ({latency_ms:.2f}ms) exceeds threshold ({self.latency_threshold_ms}ms)"
            )
        
        return metrics
    
    def update_baseline(self, window_size: int = 100) -> Dict[str, float]:
        """
        Update baseline metrics for regression detection.
        
        Args:
            window_size: Number of recent metrics to use for baseline
            
        Returns:
            Dictionary with baseline metrics
        """
        if len(self.training_step_times) < window_size:
            logger.warning(f"Not enough metrics to update baseline (need {window_size}, have {len(self.training_step_times)})")
            return self.baseline_metrics
        
        # Calculate baseline metrics from recent history
        self.baseline_metrics = {
            'mean_training_step_time': np.mean(list(self.training_step_times)[-window_size:]),
            'mean_forward_time': np.mean(list(self.forward_times)[-window_size:]),
            'mean_backward_time': np.mean(list(self.backward_times)[-window_size:]),
            'mean_optimizer_time': np.mean(list(self.optimizer_times)[-window_size:]),
            'mean_samples_per_second': np.mean(list(self.samples_per_second)[-window_size:]),
            'mean_inference_latency': np.mean(list(self.inference_times)[-window_size:]) if self.inference_times else 0.0
        }
        
        logger.info(f"Updated performance baseline: {self.baseline_metrics['mean_training_step_time']:.2f}ms step time, "
                   f"{self.baseline_metrics['mean_samples_per_second']:.2f} samples/sec")
        
        return self.baseline_metrics
    
    def detect_regression(self, 
                         window_size: int = 10,
                         threshold_percent: float = 20.0) -> Dict[str, Any]:
        """
        Detect performance regressions by comparing recent metrics to baseline.
        
        Args:
            window_size: Number of recent metrics to compare
            threshold_percent: Percentage change to consider a regression
            
        Returns:
            Dictionary with regression analysis results
        """
        if not self.baseline_metrics:
            logger.warning("No baseline metrics available for regression detection")
            return {"has_regression": False, "message": "No baseline metrics available"}
        
        if len(self.training_step_times) < window_size:
            return {"has_regression": False, "message": "Not enough recent metrics for analysis"}
        
        recent_metrics = self._get_recent_metrics(window_size)
        regressions = []
        regressions.extend(self._check_higher_is_worse_metrics(recent_metrics, threshold_percent))
        regressions.extend(self._check_lower_is_worse_metrics(recent_metrics, threshold_percent))
        
        has_regression = bool(regressions)
        result = {
            'has_regression': has_regression,
            'regressions': regressions,
            'baseline_metrics': self.baseline_metrics,
            'recent_metrics': recent_metrics
        }
        
        if has_regression:
            self._log_and_alert_regressions(regressions, threshold_percent)
        
        return result

    def _get_recent_metrics(self, window_size: int) -> Dict[str, float]:
        """Helper to calculate recent metrics."""
        return {
            'mean_training_step_time': np.mean(list(self.training_step_times)[-window_size:]),
            'mean_forward_time': np.mean(list(self.forward_times)[-window_size:]),
            'mean_backward_time': np.mean(list(self.backward_times)[-window_size:]),
            'mean_optimizer_time': np.mean(list(self.optimizer_times)[-window_size:]),
            'mean_samples_per_second': np.mean(list(self.samples_per_second)[-window_size:]),
            'mean_inference_latency': np.mean(list(self.inference_times)[-window_size:]) if self.inference_times else 0.0
        }

    def _check_higher_is_worse_metrics(self, recent_metrics: Dict[str, float], threshold_percent: float) -> List[Dict[str, Any]]:
        """Helper to check regressions for metrics where higher is worse."""
        regressions = []
        metrics = ['mean_training_step_time', 'mean_forward_time', 'mean_backward_time', 
                   'mean_optimizer_time', 'mean_inference_latency']
        for metric in metrics:
            baseline = self.baseline_metrics.get(metric, 0)
            recent = recent_metrics.get(metric, 0)
            if baseline == 0:
                continue
            percent_change = ((recent - baseline) / baseline) * 100
            if percent_change > threshold_percent:
                regressions.append({
                    'metric': metric,
                    'baseline': baseline,
                    'recent': recent,
                    'percent_change': percent_change
                })
        return regressions

    def _check_lower_is_worse_metrics(self, recent_metrics: Dict[str, float], threshold_percent: float) -> List[Dict[str, Any]]:
        """Helper to check regressions for metrics where lower is worse."""
        regressions = []
        metrics = ['mean_samples_per_second']
        for metric in metrics:
            baseline = self.baseline_metrics.get(metric, 0)
            recent = recent_metrics.get(metric, 0)
            if baseline == 0:
                continue
            percent_change = ((baseline - recent) / baseline) * 100
            if percent_change > threshold_percent:
                regressions.append({
                    'metric': metric,
                    'baseline': baseline,
                    'recent': recent,
                    'percent_change': percent_change
                })
        return regressions

    def _log_and_alert_regressions(self, regressions: List[Dict[str, Any]], threshold_percent: float) -> None:
        """Helper to log and alert for regressions."""
        logger.warning(f"Performance regression detected: {len(regressions)} metrics degraded")
        for reg in regressions:
            logger.warning("  %s: %.2f -> %.2f (%.1f%% change), threshold=%.1f", reg['metric'], reg['baseline'], reg['recent'], reg['percent_change'], threshold_percent)
            severity = 'critical' if reg['percent_change'] > threshold_percent * 2 else 'warning'
            self._create_alert(
                metric_name=reg['metric'],
                current_value=reg['recent'],
                threshold_value=reg['baseline'],
                severity=severity,
                message=f"Performance regression in {reg['metric']}: {reg['baseline']:.2f} -> {reg['recent']:.2f} "
                        f"({reg['percent_change']:.1f}% change)"
            )
    
    def generate_optimization_suggestions(self) -> List[str]:
        """
        Generate performance optimization suggestions based on monitoring data.
        
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        # Check if we have enough data
        if len(self.training_step_times) < 10:
            return ["Collect more performance data to generate suggestions."]
        
        # Calculate average metrics
        avg_forward_time = np.mean(list(self.forward_times))
        avg_backward_time = np.mean(list(self.backward_times))
        avg_optimizer_time = np.mean(list(self.optimizer_times))
        avg_total_time = np.mean(list(self.training_step_times))
        avg_inference_time = np.mean(list(self.inference_times)) if self.inference_times else 0.0
        
        # Check forward/backward ratio
        if avg_forward_time > avg_backward_time * 2:
            suggestions.append("Forward pass is taking significantly longer than backward pass. "
                             "Consider optimizing model architecture or using torch.jit.script.")
        
        # Check optimizer time
        if avg_optimizer_time > avg_total_time * 0.5:
            suggestions.append("Optimizer step is taking a large portion of training time. "
                             "Consider using a more efficient optimizer or reducing update frequency.")
        
        # Check inference latency
        if self.inference_times and avg_inference_time > self.latency_threshold_ms:
            suggestions.append(f"Inference latency ({avg_inference_time:.2f}ms) exceeds target "
                             f"({self.latency_threshold_ms}ms). Consider model quantization, "
                             f"pruning, or using torch.no_grad() for inference.")
        
        # Add general optimization tips if few specific suggestions
        if len(suggestions) < 2:
            suggestions.append("Use DataLoader with pin_memory=True for faster data transfer to GPU.")
            suggestions.append("Consider using mixed precision training with torch.cuda.amp.")
            suggestions.append("Batch tensor operations to minimize individual device transfers.")
            suggestions.append("Use torch.no_grad() for all inference operations.")
        
        return suggestions
    
    def _check_for_regressions(self, metrics: Dict[str, float]) -> None:
        """
        Check for performance regressions and create alerts if needed.
        
        Args:
            metrics: Current performance metrics
        """
        # Update baseline if not set
        if not self.baseline_metrics and len(self.training_step_times) >= 100:
            self.update_baseline()
        
        # Detect regressions every 100 steps
        if len(self.training_step_times) % 100 == 0:
            self.detect_regression()
    
    def _create_alert(self, 
                     metric_name: str,
                     current_value: float,
                     threshold_value: float,
                     severity: str,
                     message: str) -> None:
        """
        Create a performance alert.
        
        Args:
            metric_name: Name of the metric
            current_value: Current value of the metric
            threshold_value: Threshold value for the alert
            severity: Alert severity ('warning', 'critical')
            message: Alert message
        """
        # Create alert
        alert = PerformanceAlert(
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            severity=severity,
            message=message,
            timestamp=time.time()
        )
        
        # Add to alert history
        self.alerts.append(alert)
        
        # Log alert
        if severity == 'critical':
            logger.error(f"CRITICAL PERFORMANCE ALERT: {message}")
        else:
            logger.warning(f"Performance alert: {message}")
        
        # Call alert callback if provided
        if self.alert_callback:
            self.alert_callback(alert)