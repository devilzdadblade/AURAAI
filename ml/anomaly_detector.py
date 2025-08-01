"""
AnomalyDetector for behavioral monitoring of RL models.

This module provides tools for detecting anomalies in model behavior using
statistical process control methods, including monitoring of action distributions,
Q-value magnitudes, and trading patterns.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os
import json
import logging
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from datetime import datetime
from enum import Enum


class AnomalySeverity(Enum):
    """Severity levels for anomalies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Anomaly:
    """Container for anomaly information."""
    metric_name: str
    current_value: float
    expected_range: Tuple[float, float]
    z_score: float
    timestamp: float
    severity: AnomalySeverity
    context: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["severity"] = self.severity.value
        return result


@dataclass
class BaselineStats:
    """Container for baseline statistics."""
    mean: float
    std: float
    min: float
    max: float
    p25: float
    p50: float
    p75: float
    sample_count: int
    last_updated: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class AnomalyDetector:
    """
    Detector for anomalies in model behavior.
    
    This class provides tools for:
    - Establishing baseline statistics for model behavior
    - Detecting anomalies using statistical process control methods
    - Monitoring action distributions, Q-value magnitudes, and trading patterns
    - Alerting operators about detected anomalies
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        model_name: str = "rl_model",
        window_size: int = 1000,
        z_score_threshold: float = 3.0,
        alert_callback: Optional[Callable[[Anomaly], None]] = None,
        metrics_buffer_size: int = 10000
    ):
        """
        Initialize the anomaly detector.
        
        Args:
            log_dir: Directory for logs and visualizations
            model_name: Name of the model being monitored
            window_size: Window size for baseline statistics
            z_score_threshold: Z-score threshold for anomaly detection
            alert_callback: Callback function for anomaly alerts
            metrics_buffer_size: Maximum number of metrics to store in memory
        """
        self.log_dir = log_dir
        self.model_name = model_name
        self.window_size = window_size
        self.z_score_threshold = z_score_threshold
        self.alert_callback = alert_callback
        self.metrics_buffer_size = metrics_buffer_size
        
        # Create log directory
        self.run_dir = os.path.join(
            log_dir, 
            f"{model_name}_anomaly_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics_history = defaultdict(lambda: deque(maxlen=metrics_buffer_size))
        self.anomalies = deque(maxlen=metrics_buffer_size)
        self.baseline_stats = {}
        self.action_history = deque(maxlen=metrics_buffer_size)
        self.q_value_history = deque(maxlen=metrics_buffer_size)
        self.trading_pattern_history = deque(maxlen=metrics_buffer_size)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize anomaly counts
        self.anomaly_counts = defaultdict(int)
        
        self.logger.info(f"Anomaly detector initialized. Logs will be saved to {self.run_dir}")
    
    def update_metric(self, metric_name: str, value: float, context: Optional[Dict[str, Any]] = None) -> Optional[Anomaly]:
        """
        Update a metric and check for anomalies.
        
        Args:
            metric_name: Name of the metric
            value: Current value of the metric
            context: Additional context for the metric
            
        Returns:
            Anomaly object if an anomaly is detected, None otherwise
        """
        # Store metric value
        self.metrics_history[metric_name].append((time.time(), value, context))
        
        # Check for anomaly
        anomaly = self._check_anomaly(metric_name, value, context)
        
        # Update baseline statistics if we have enough data
        if len(self.metrics_history[metric_name]) >= self.window_size:
            self._update_baseline_stats(metric_name)
        
        return anomaly
    
    def update_action(self, action: int, q_values: Optional[torch.Tensor] = None, context: Optional[Dict[str, Any]] = None) -> List[Anomaly]:
        """
        Update action history and check for anomalies.
        
        Args:
            action: Selected action
            q_values: Q-values tensor (if available)
            context: Additional context for the action
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        timestamp = time.time()
        
        # Store action
        self.action_history.append((timestamp, action, context))
        
        # Check for action distribution anomalies
        if len(self.action_history) >= self.window_size:
            anomaly = self._check_action_distribution_anomaly(action)
            if anomaly:
                anomalies.append(anomaly)
        
        # Store and check Q-values if available
        if q_values is not None:
            with torch.no_grad():
                q_mean = q_values.mean().item()
                q_std = q_values.std().item()
                q_min = q_values.min().item()
                q_max = q_values.max().item()
                q_range = q_max - q_min
                
                # Store Q-value statistics
                self.q_value_history.append((timestamp, {
                    "mean": q_mean,
                    "std": q_std,
                    "min": q_min,
                    "max": q_max,
                    "range": q_range,
                    "action": action,
                    "context": context
                }))
                
                # Check for Q-value anomalies
                q_anomalies = self._check_q_value_anomalies(q_mean, q_std, q_min, q_max, q_range)
                anomalies.extend(q_anomalies)
        
        return anomalies
    
    def update_trading_pattern(
        self,
        pattern_type: str,
        value: float,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Anomaly]:
        """
        Update trading pattern history and check for anomalies.
        
        Args:
            pattern_type: Type of trading pattern (e.g., 'trade_frequency', 'position_size')
            value: Current value of the pattern
            context: Additional context for the pattern
            
        Returns:
            Anomaly object if an anomaly is detected, None otherwise
        """
        # Store trading pattern
        self.trading_pattern_history.append((time.time(), pattern_type, value, context))
        
        # Check for trading pattern anomalies
        anomaly = self._check_trading_pattern_anomaly(pattern_type, value, context)
        
        return anomaly
    
    def _check_anomaly(self, metric_name: str, value: float, context: Optional[Dict[str, Any]] = None) -> Optional[Anomaly]:
        """
        Check if a metric value is anomalous (refactored for lower complexity).
        
        Args:
            metric_name: Name of the metric
            value: Current value of the metric
            context: Additional context for the metric
            
        Returns:
            Anomaly object if an anomaly is detected, None otherwise
        """
        # Skip if we don't have baseline statistics
        if not self._has_baseline_stats(metric_name):
            return None
        
        # Calculate anomaly score
        z_score = self._calculate_z_score(metric_name, value)
        
        # Check if anomalous
        if self._is_anomalous(z_score):
            anomaly = self._create_anomaly_object(metric_name, value, z_score, context)
            self._process_detected_anomaly(anomaly)
            return anomaly
        
        return None

    def _has_baseline_stats(self, metric_name: str) -> bool:
        """Check if baseline statistics exist for the metric."""
        return metric_name in self.baseline_stats

    def _calculate_z_score(self, metric_name: str, value: float) -> float:
        """Calculate z-score for the metric value."""
        stats = self.baseline_stats[metric_name]
        if stats.std > 0:
            return abs(value - stats.mean) / stats.std
        return 0

    def _is_anomalous(self, z_score: float) -> bool:
        """Check if z-score indicates an anomaly."""
        return z_score > self.z_score_threshold

    def _create_anomaly_object(self, metric_name: str, value: float, z_score: float, 
                              context: Optional[Dict[str, Any]]) -> Anomaly:
        """Create an anomaly object with appropriate severity."""
        stats = self.baseline_stats[metric_name]
        severity = self._determine_severity(z_score)
        expected_range = (stats.mean - stats.std * 2, stats.mean + stats.std * 2)
        
        return Anomaly(
            metric_name=metric_name,
            current_value=value,
            expected_range=expected_range,
            z_score=z_score,
            timestamp=time.time(),
            severity=severity,
            context=context
        )

    def _determine_severity(self, z_score: float) -> AnomalySeverity:
        """Determine anomaly severity based on z-score."""
        if z_score > 5.0:
            return AnomalySeverity.CRITICAL
        elif z_score > 4.0:
            return AnomalySeverity.HIGH
        elif z_score > 3.0:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW

    def _process_detected_anomaly(self, anomaly: Anomaly) -> None:
        """Process a detected anomaly by storing, logging, and alerting."""
        # Store anomaly
        self.anomalies.append(anomaly)
        
        # Update anomaly count
        self.anomaly_counts[anomaly.metric_name] += 1
        
        # Log anomaly
        self._log_anomaly(anomaly)
        
        # Call alert callback if provided
        if self.alert_callback:
            self.alert_callback(anomaly)

    def _log_anomaly(self, anomaly: Anomaly) -> None:
        """Log the detected anomaly."""
        self.logger.warning(
            f"Anomaly detected in {anomaly.metric_name}: "
            f"value={anomaly.current_value:.4f}, z-score={anomaly.z_score:.2f}, "
            f"expected range=[{anomaly.expected_range[0]:.4f}, {anomaly.expected_range[1]:.4f}], "
            f"severity={anomaly.severity.value}"
        )
    
    def _check_action_distribution_anomaly(self, action: int) -> Optional[Anomaly]:
        """
        Check if the action distribution is anomalous.
        
        Args:
            action: Current action
            
        Returns:
            Anomaly object if an anomaly is detected, None otherwise
        """
        # Count actions in history
        action_counts = defaultdict(int)
        for _, act, _ in self.action_history:
            action_counts[act] += 1
        
        # Calculate action frequencies
        total_actions = len(self.action_history)
        action_frequencies = {a: count / total_actions for a, count in action_counts.items()}
        
        # Calculate expected frequency (uniform distribution)
        num_actions = len(action_counts)
        if num_actions == 0:
            return None
        
        expected_frequency = 1.0 / num_actions
        
        # Check if current action frequency is anomalous
        current_frequency = action_counts[action] / total_actions
        
        # Calculate z-score based on binomial distribution
        # For a binomial distribution with n trials and probability p,
        # the standard deviation is sqrt(n*p*(1-p))
        std_dev = np.sqrt(total_actions * expected_frequency * (1 - expected_frequency))
        
        if std_dev > 0:
            z_score = abs(current_frequency - expected_frequency) * np.sqrt(total_actions) / std_dev
        else:
            z_score = 0
        
        # Check if frequency is anomalous
        if z_score > self.z_score_threshold:
            # Determine severity
            if z_score > 5.0:
                severity = AnomalySeverity.CRITICAL
            elif z_score > 4.0:
                severity = AnomalySeverity.HIGH
            elif z_score > 3.0:
                severity = AnomalySeverity.MEDIUM
            else:
                severity = AnomalySeverity.LOW
            
            # Create anomaly
            anomaly = Anomaly(
                metric_name=f"action_distribution_{action}",
                current_value=current_frequency,
                expected_range=(
                    max(0, expected_frequency - 2 * std_dev / np.sqrt(total_actions)),
                    min(1, expected_frequency + 2 * std_dev / np.sqrt(total_actions))
                ),
                z_score=z_score,
                timestamp=time.time(),
                severity=severity,
                context={
                    "action": action,
                    "action_counts": dict(action_counts),
                    "action_frequencies": action_frequencies,
                    "expected_frequency": expected_frequency
                }
            )
            
            # Store anomaly
            self.anomalies.append(anomaly)
            
            # Update anomaly count
            self.anomaly_counts[f"action_distribution_{action}"] += 1
            
            # Log anomaly
            self.logger.warning(
                f"Action distribution anomaly detected for action {action}: "
                f"frequency={current_frequency:.4f}, z-score={z_score:.2f}, "
                f"expected frequency={expected_frequency:.4f}, "
                f"severity={severity.value}"
            )
            
            # Call alert callback if provided
            if self.alert_callback:
                self.alert_callback(anomaly)
            
            return anomaly
        
        return None
    
    def _check_q_value_anomalies(
        self,
        q_mean: float,
        q_std: float,
        q_min: float,
        q_max: float,
        q_range: float
    ) -> List[Anomaly]:
        """
        Check if Q-value statistics are anomalous.
        
        Args:
            q_mean: Mean Q-value
            q_std: Standard deviation of Q-values
            q_min: Minimum Q-value
            q_max: Maximum Q-value
            q_range: Range of Q-values
            
        Returns:
            List of detected anomalies
        """
        # Skip if insufficient history
        if not self._has_sufficient_history():
            return []
            
        # Collect anomalies from all checks
        return self._collect_q_value_anomalies(q_mean, q_std, q_min, q_max, q_range)
    
    def _has_sufficient_history(self) -> bool:
        """Check if we have enough history data for anomaly detection."""
        return len(self.q_value_history) >= self.window_size
        
    def _collect_q_value_anomalies(
        self, 
        q_mean: float, 
        q_std: float, 
        q_min: float, 
        q_max: float, 
        q_range: float
    ) -> List[Anomaly]:
        """Collect anomalies from all Q-value checks."""
        anomalies = []
        
        # Check Q-value mean anomalies
        self._add_anomaly_if_exists(
            anomalies,
            self._check_q_value_mean_anomaly(q_mean, q_std, q_min, q_max, q_range)
        )
        
        # Check Q-value range anomalies
        self._add_anomaly_if_exists(
            anomalies,
            self._check_q_value_range_anomaly(q_range, q_mean, q_std, q_min, q_max)
        )
        
        return anomalies
        
    def _add_anomaly_if_exists(self, anomalies: List[Anomaly], anomaly: Optional[Anomaly]) -> None:
        """Add anomaly to list if it exists."""
        if anomaly:
            anomalies.append(anomaly)

    def _check_q_value_mean_anomaly(
        self, q_mean: float, q_std: float, q_min: float, q_max: float, q_range: float
    ) -> Optional[Anomaly]:
        """Check for Q-value mean anomalies."""
        q_means = [entry[1]["mean"] for entry in self.q_value_history]
        
        mean_mean = np.mean(q_means)
        mean_std = np.std(q_means)
        
        if mean_std <= 0:
            return None
        
        mean_z_score = abs(q_mean - mean_mean) / mean_std
        
        if mean_z_score <= self.z_score_threshold:
            return None
        
        severity = self._determine_anomaly_severity(mean_z_score)
        context = self._create_q_value_context(q_mean, q_std, q_min, q_max, q_range, mean_mean, mean_std)
        
        anomaly = Anomaly(
            metric_name="q_value_mean",
            current_value=q_mean,
            expected_range=(mean_mean - mean_std * 2, mean_mean + mean_std * 2),
            z_score=mean_z_score,
            timestamp=time.time(),
            severity=severity,
            context=context
        )
        
        self._record_and_log_anomaly(anomaly, "Q-value mean")
        return anomaly

    def _check_q_value_range_anomaly(
        self, q_range: float, q_mean: float, q_std: float, q_min: float, q_max: float
    ) -> Optional[Anomaly]:
        """Check for Q-value range anomalies."""
        q_ranges = [entry[1]["range"] for entry in self.q_value_history]
        
        range_mean = np.mean(q_ranges)
        range_std = np.std(q_ranges)
        
        if range_std <= 0:
            return None
        
        range_z_score = abs(q_range - range_mean) / range_std
        
        if range_z_score <= self.z_score_threshold:
            return None
        
        severity = self._determine_anomaly_severity(range_z_score)
        context = self._create_q_value_context(q_mean, q_std, q_min, q_max, q_range, range_mean, range_std)
        
        anomaly = Anomaly(
            metric_name="q_value_range",
            current_value=q_range,
            expected_range=(range_mean - range_std * 2, range_mean + range_std * 2),
            z_score=range_z_score,
            timestamp=time.time(),
            severity=severity,
            context=context
        )
        
        self._record_and_log_anomaly(anomaly, "Q-value range")
        return anomaly

    def _determine_anomaly_severity(self, z_score: float) -> AnomalySeverity:
        """Determine anomaly severity for Q-value anomalies using different thresholds."""
        if z_score > 6.0:
            return AnomalySeverity.CRITICAL
        elif z_score > 4.5:
            return AnomalySeverity.HIGH
        elif z_score > 2.5:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW

    def _create_q_value_context(
        self, q_mean: float, q_std: float, q_min: float, q_max: float, 
        q_range: float, baseline_mean: float, baseline_std: float
    ) -> Dict[str, Any]:
        """Create context dictionary for Q-value anomaly."""
        return {
            "q_mean": q_mean,
            "q_std": q_std,
            "q_min": q_min,
            "q_max": q_max,
            "q_range": q_range,
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std
        }

    def _record_and_log_anomaly(self, anomaly: Anomaly, anomaly_type: str):
        """Record anomaly and log the detection."""
        # Store anomaly
        self.anomalies.append(anomaly)
        
        # Update anomaly count
        self.anomaly_counts[anomaly.metric_name] += 1
        
        # Log anomaly
        self.logger.warning(
            f"{anomaly_type} anomaly detected: "
            f"value={anomaly.current_value:.4f}, z-score={anomaly.z_score:.2f}, "
            f"expected range=[{anomaly.expected_range[0]:.4f}, {anomaly.expected_range[1]:.4f}], "
            f"severity={anomaly.severity.value}"
        )
        
        # Call alert callback if provided
        if self.alert_callback:
            self.alert_callback(anomaly)
    
    def _check_trading_pattern_anomaly(
        self,
        pattern_type: str,
        value: float,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Anomaly]:
        """
        Check if a trading pattern is anomalous.
        
        Args:
            pattern_type: Type of trading pattern
            value: Current value of the pattern
            context: Additional context for the pattern
            
        Returns:
            Anomaly object if an anomaly is detected, None otherwise
        """
        # Skip if we don't have enough history
        pattern_history = [(t, v) for t, pt, v, _ in self.trading_pattern_history if pt == pattern_type]
        if len(pattern_history) < self.window_size:
            return None
        
        # Calculate baseline statistics
        values = [v for _, v in pattern_history]
        mean_value = np.mean(values)
        std_value = np.std(values)
        
        # Calculate z-score
        if std_value > 0:
            z_score = abs(value - mean_value) / std_value
        else:
            z_score = 0
        
        # Check if value is anomalous
        if z_score > self.z_score_threshold:
            # Determine severity
            if z_score > 5.0:
                severity = AnomalySeverity.CRITICAL
            elif z_score > 4.0:
                severity = AnomalySeverity.HIGH
            elif z_score > 3.0:
                severity = AnomalySeverity.MEDIUM
            else:
                severity = AnomalySeverity.LOW
            
            # Create anomaly
            anomaly = Anomaly(
                metric_name=f"trading_pattern_{pattern_type}",
                current_value=value,
                expected_range=(mean_value - std_value * 2, mean_value + std_value * 2),
                z_score=z_score,
                timestamp=time.time(),
                severity=severity,
                context=context
            )
            
            # Store anomaly
            self.anomalies.append(anomaly)
            
            # Update anomaly count
            self.anomaly_counts[f"trading_pattern_{pattern_type}"] += 1
            
            # Log anomaly
            self.logger.warning(
                f"Trading pattern anomaly detected for {pattern_type}: "
                f"value={value:.4f}, z-score={z_score:.2f}, "
                f"expected range=[{anomaly.expected_range[0]:.4f}, {anomaly.expected_range[1]:.4f}], "
                f"severity={severity.value}"
            )
            
            # Call alert callback if provided
            if self.alert_callback:
                self.alert_callback(anomaly)
            
            return anomaly
        
        return None
    
    def _update_baseline_stats(self, metric_name: str):
        """
        Update baseline statistics for a metric.
        
        Args:
            metric_name: Name of the metric
        """
        # Get metric values
        values = [value for _, value, _ in self.metrics_history[metric_name]]
        
        # Calculate statistics
        mean_value = np.mean(values)
        std_value = np.std(values)
        min_value = np.min(values)
        max_value = np.max(values)
        p25_value = np.percentile(values, 25)
        p50_value = np.percentile(values, 50)
        p75_value = np.percentile(values, 75)
        
        # Update baseline statistics
        self.baseline_stats[metric_name] = BaselineStats(
            mean=mean_value,
            std=std_value,
            min=min_value,
            max=max_value,
            p25=p25_value,
            p50=p50_value,
            p75=p75_value,
            sample_count=len(values),
            last_updated=time.time()
        )
    
    def get_recent_anomalies(self, n: int = 10) -> List[Anomaly]:
        """
        Get recent anomalies.
        
        Args:
            n: Number of recent anomalies to return
            
        Returns:
            List of recent anomalies
        """
        return list(self.anomalies)[-n:]
    
    def get_anomaly_counts(self) -> Dict[str, int]:
        """
        Get anomaly counts by metric.
        
        Returns:
            Dictionary mapping metric names to anomaly counts
        """
        return dict(self.anomaly_counts)
    
    def get_baseline_stats(self, metric_name: str) -> Optional[BaselineStats]:
        """
        Get baseline statistics for a metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            BaselineStats object for the metric, or None if not available
        """
        return self.baseline_stats.get(metric_name)
    
    def get_all_baseline_stats(self) -> Dict[str, BaselineStats]:
        """
        Get all baseline statistics.
        
        Returns:
            Dictionary mapping metric names to BaselineStats objects
        """
        return self.baseline_stats
    
    def reset_baseline_stats(self, metric_name: Optional[str] = None):
        """
        Reset baseline statistics.
        
        Args:
            metric_name: Name of the metric to reset, or None to reset all
        """
        if metric_name is None:
            self.baseline_stats = {}
        elif metric_name in self.baseline_stats:
            del self.baseline_stats[metric_name]
    
    def save_anomalies(self, filename: str = "anomalies.json"):
        """
        Save anomalies to a file.
        
        Args:
            filename: Name of the file to save anomalies to
        """
        filepath = os.path.join(self.run_dir, filename)
        
        # Convert anomalies to dictionaries
        anomalies_dict = [anomaly.to_dict() for anomaly in self.anomalies]
        
        # Save to file
        with open(filepath, "w") as f:
            json.dump(anomalies_dict, f, indent=2)
        
        self.logger.info(f"Anomalies saved to {filepath}")
    
    def save_baseline_stats(self, filename: str = "baseline_stats.json"):
        """
        Save baseline statistics to a file.
        
        Args:
            filename: Name of the file to save baseline statistics to
        """
        filepath = os.path.join(self.run_dir, filename)
        
        # Convert baseline statistics to dictionaries
        stats_dict = {name: stats.to_dict() for name, stats in self.baseline_stats.items()}
        
        # Save to file
        with open(filepath, "w") as f:
            json.dump(stats_dict, f, indent=2)
        
        self.logger.info(f"Baseline statistics saved to {filepath}")
    
    def plot_anomalies(self, save_dir: Optional[str] = None):
        """
        Plot anomalies.
        
        Args:
            save_dir: Directory to save plots to (defaults to run_dir)
        """
        save_dir = save_dir or self.run_dir
        os.makedirs(save_dir, exist_ok=True)
        
        if not self._validate_anomalies_for_plotting():
            return
        
        # Plot individual metric anomalies
        self._plot_metric_anomalies(save_dir)
        
        # Plot anomaly count summary
        self._plot_anomaly_counts(save_dir)
        
        self.logger.info(f"Anomaly plots saved to {save_dir}")

    def _validate_anomalies_for_plotting(self) -> bool:
        """Validate that there are anomalies to plot."""
        if not self.anomalies:
            self.logger.info("No anomalies to plot")
            return False
        return True

    def _plot_metric_anomalies(self, save_dir: str) -> None:
        """Plot anomalies for each metric type."""
        # Group anomalies by metric
        anomalies_by_metric = defaultdict(list)
        for anomaly in self.anomalies:
            anomalies_by_metric[anomaly.metric_name].append(anomaly)
        
        # Plot each metric
        for metric_name, anomalies in anomalies_by_metric.items():
            if anomalies:
                self._plot_single_metric_anomalies(metric_name, anomalies, save_dir)

    def _plot_single_metric_anomalies(self, metric_name: str, anomalies: List, save_dir: str) -> None:
        """Plot anomalies for a single metric."""
        # Get metric history
        history = self._get_metric_history(metric_name)
        if not history:
            return
        
        # Prepare data for plotting
        plot_data = self._prepare_plot_data(history, anomalies)
        
        # Create and save plot
        self._create_metric_plot(metric_name, plot_data, save_dir)

    def _get_metric_history(self, metric_name: str) -> List[Tuple[float, float]]:
        """Get history data for a specific metric."""
        if metric_name.startswith("action_distribution_"):
            return self._get_action_distribution_history(metric_name)
        elif metric_name.startswith("trading_pattern_"):
            return self._get_trading_pattern_history(metric_name)
        elif metric_name == "q_value_mean":
            return [(entry[0], entry[1]["mean"]) for entry in self.q_value_history]
        elif metric_name == "q_value_range":
            return [(entry[0], entry[1]["range"]) for entry in self.q_value_history]
        else:
            return [(t, v) for t, v, _ in self.metrics_history[metric_name]]

    def _get_action_distribution_history(self, metric_name: str) -> List[Tuple[float, float]]:
        """Get action distribution history for specific action."""
        action = int(metric_name.split("_")[-1])
        return [(t, 1 if a == action else 0) for t, a, _ in self.action_history]

    def _get_trading_pattern_history(self, metric_name: str) -> List[Tuple[float, float]]:
        """Get trading pattern history for specific pattern type."""
        pattern_type = metric_name[len("trading_pattern_"):]
        return [(t, v) for t, pt, v, _ in self.trading_pattern_history if pt == pattern_type]

    def _prepare_plot_data(self, history: List[Tuple[float, float]], anomalies: List) -> Dict:
        """Prepare data for plotting."""
        timestamps = [t for t, _ in history]
        values = [v for _, v in history]
        
        # Convert to relative time
        if timestamps:
            first_timestamp = timestamps[0]
            timestamps = [(t - first_timestamp) for t in timestamps]
            anomaly_timestamps = [(a.timestamp - first_timestamp) for a in anomalies]
        else:
            anomaly_timestamps = []
        
        return {
            'timestamps': timestamps,
            'values': values,
            'anomaly_timestamps': anomaly_timestamps,
            'anomaly_values': [a.current_value for a in anomalies],
            'first_timestamp': timestamps[0] if timestamps else 0
        }

    def _create_metric_plot(self, metric_name: str, plot_data: Dict, save_dir: str) -> None:
        """Create and save metric plot with anomalies."""
        plt.figure(figsize=(10, 6))
        
        # Plot metric history
        plt.plot(plot_data['timestamps'], plot_data['values'], label=metric_name)
        
        # Plot anomalies
        plt.scatter(plot_data['anomaly_timestamps'], plot_data['anomaly_values'], 
                   color='red', marker='x', s=100, label='Anomalies')
        
        # Add baseline statistics if available
        self._add_baseline_lines(metric_name)
        
        # Format plot
        plt.xlabel("Time (seconds)")
        plt.ylabel("Value")
        plt.title(f"Anomalies for {metric_name}")
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.savefig(os.path.join(save_dir, f"anomalies_{metric_name.replace('/', '_')}.png"))
        plt.close()

    def _add_baseline_lines(self, metric_name: str) -> None:
        """Add baseline statistical lines to the plot."""
        if metric_name in self.baseline_stats:
            stats = self.baseline_stats[metric_name]
            plt.axhline(y=stats.mean, color='green', linestyle='-', label='Mean')
            plt.axhline(y=stats.mean + stats.std * 2, color='orange', linestyle='--', label='2σ Upper Bound')
            plt.axhline(y=stats.mean - stats.std * 2, color='orange', linestyle='--', label='2σ Lower Bound')

    def _plot_anomaly_counts(self, save_dir: str) -> None:
        """Plot anomaly counts by metric."""
        if not self.anomaly_counts:
            return
        
        plt.figure(figsize=(12, 6))
        
        # Prepare data
        metrics, counts = self._prepare_count_data()
        
        # Create bar plot
        plt.bar(metrics, counts)
        plt.xlabel("Metric")
        plt.ylabel("Anomaly Count")
        plt.title("Anomaly Counts by Metric")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(save_dir, "anomaly_counts.png"))
        plt.close()

    def _prepare_count_data(self) -> Tuple[List[str], List[int]]:
        """Prepare and sort anomaly count data."""
        metrics = list(self.anomaly_counts.keys())
        counts = list(self.anomaly_counts.values())
        
        # Sort by count (descending)
        sorted_indices = np.argsort(counts)[::-1]
        sorted_metrics = [metrics[i] for i in sorted_indices]
        sorted_counts = [counts[i] for i in sorted_indices]
        
        return sorted_metrics, sorted_counts
    
    def close(self):
        """Close the detector and clean up resources."""
        # Save anomalies
        self.save_anomalies()
        
        # Save baseline statistics
        self.save_baseline_stats()
        
        # Plot anomalies
        self.plot_anomalies()
        
        self.logger.info("Anomaly detector closed")


