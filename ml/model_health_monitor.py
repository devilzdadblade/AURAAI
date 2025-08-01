"""
Model health monitoring system for reinforcement learning models.

This module provides tools to continuously monitor the health of RL models,
detecting issues with training stability, prediction quality, and overall performance.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any, Deque
import datetime

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Enumeration of model health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthThresholds:
    """Configurable thresholds for model health assessment."""
    
    # Error rate thresholds (percentage of errors in recent operations)
    error_rate_warning: float = 0.05  # 5% errors trigger warning
    error_rate_critical: float = 0.15  # 15% errors trigger critical status
    
    # Q-value distribution thresholds
    q_value_range_warning: float = 100.0  # Warning if Q-values exceed this range
    q_value_range_critical: float = 1000.0  # Critical if Q-values exceed this range
    q_value_std_warning: float = 50.0  # Warning if Q-value std dev exceeds this
    q_value_std_critical: float = 200.0  # Critical if Q-value std dev exceeds this
    
    # Gradient norm thresholds
    gradient_norm_warning: float = 10.0  # Warning if gradient norm exceeds this
    gradient_norm_critical: float = 100.0  # Critical if gradient norm exceeds this
    
    # Prediction stability thresholds (std dev of predictions for same input)
    prediction_stability_warning: float = 0.2  # Warning if prediction std dev exceeds 20%
    prediction_stability_critical: float = 0.5  # Critical if prediction std dev exceeds 50%
    
    # Loss divergence thresholds (rate of change in loss)
    loss_increase_warning: float = 1.5  # Warning if loss increases by 50%
    loss_increase_critical: float = 3.0  # Critical if loss increases by 200%


@dataclass
class HealthMetrics:
    """Container for model health metrics."""
    
    # Overall health score (0.0 to 1.0, higher is better)
    overall_health: float = 1.0
    
    # Component health scores
    error_rate: float = 0.0
    q_value_health: float = 1.0
    gradient_health: float = 1.0
    prediction_stability: float = 1.0
    loss_stability: float = 1.0
    
    # Raw metrics
    recent_error_count: int = 0
    total_operations: int = 0
    q_value_mean: float = 0.0
    q_value_std: float = 0.0
    q_value_min: float = 0.0
    q_value_max: float = 0.0
    gradient_norm_mean: float = 0.0
    gradient_norm_max: float = 0.0
    prediction_std: float = 0.0
    recent_loss_values: List[float] = field(default_factory=list)
    
    # Status and recommendations
    status: HealthStatus = HealthStatus.HEALTHY
    recommended_action: str = "No action needed"
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)


@dataclass
class HealthReport:
    """Comprehensive health report for model monitoring."""
    
    current_metrics: HealthMetrics
    historical_trend: List[Tuple[datetime.datetime, float]] = field(default_factory=list)
    error_history: List[Dict[str, Any]] = field(default_factory=list)
    recovery_attempts: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    critical_issues: List[str] = field(default_factory=list)


class ModelHealthMonitor:
    """
    Monitors the health of reinforcement learning models.
    
    This class continuously assesses model health by tracking error rates,
    Q-value distributions, gradient norms, and other metrics to detect
    potential issues before they cause model failure.
    """
    
    def __init__(self, 
                 health_thresholds: Optional[HealthThresholds] = None,
                 history_size: int = 1000):
        """
        Initialize the model health monitor.
        
        Args:
            health_thresholds: Configurable thresholds for health assessment
            history_size: Maximum number of health records to maintain
        """
        self.thresholds = health_thresholds or HealthThresholds()
        self.health_history: Deque[HealthMetrics] = deque(maxlen=history_size)
        self.error_history: Deque[Dict[str, Any]] = deque(maxlen=history_size)
        self.recovery_attempts: Deque[Dict[str, Any]] = deque(maxlen=history_size)
        
        # Tracking metrics
        self.recent_errors = 0
        self.recent_operations = 0
        self.q_values_history: Deque[np.ndarray] = deque(maxlen=1000)
        self.gradient_norms: Deque[float] = deque(maxlen=100)
        self.prediction_history: Dict[str, Deque[np.ndarray]] = {}
        self.loss_history: Deque[float] = deque(maxlen=100)
        
        # Last assessment time and result
        self.last_assessment_time = time.time()
        self.last_health_metrics = HealthMetrics()
        
        logger.info("ModelHealthMonitor initialized with history size %d", history_size)
    
    def record_error(self, error_type: str, details: Dict[str, Any]) -> None:
        """
        Record an error occurrence for health assessment.
        
        Args:
            error_type: Type of error (e.g., 'numerical', 'input_validation')
            details: Additional error details
        """
        self.recent_errors += 1
        self.recent_operations += 1
        
        error_record = {
            "timestamp": datetime.datetime.now(),
            "error_type": error_type,
            "details": details
        }
        self.error_history.append(error_record)
        
        if self.recent_errors > 5:
            # Trigger health assessment if multiple errors occur
            self.assess_health()
    
    def record_operation(self, _operation_type: str) -> None:
        """
        Record a successful operation for error rate calculation.
        
        Args:
            _operation_type: Type of operation (e.g., 'inference', 'training_step') - currently unused
        """
        self.recent_operations += 1
    
    def record_q_values(self, q_values: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Record Q-values for distribution analysis.
        
        Args:
            q_values: Q-values from model prediction
        """
        if isinstance(q_values, torch.Tensor):
            q_values = q_values.detach().cpu().numpy()
        
        self.q_values_history.append(q_values)
    
    def record_gradient_norm(self, model: nn.Module) -> None:
        """
        Record gradient norm for stability analysis.
        
        Args:
            model: PyTorch model with gradients
        """
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        self.gradient_norms.append(total_norm)
    
    def record_prediction(self, state_key: str, prediction: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Record prediction for stability analysis.
        
        Args:
            state_key: Unique identifier for the input state
            prediction: Model prediction for the state
        """
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach().cpu().numpy()
        
        if state_key not in self.prediction_history:
            self.prediction_history[state_key] = deque(maxlen=10)
        
        self.prediction_history[state_key].append(prediction)
    
    def record_loss(self, loss: float) -> None:
        """
        Record loss value for trend analysis.
        
        Args:
            loss: Training loss value
        """
        self.loss_history.append(loss)
    
    def assess_health(self, model: Optional[nn.Module] = None) -> HealthMetrics:
        """
        Perform comprehensive health assessment of the model.
        
        Args:
            model: Optional model to assess gradient health
            
        Returns:
            HealthMetrics containing assessment results
        """
        metrics = HealthMetrics(timestamp=datetime.datetime.now())
        
        # Assess individual components
        self._assess_error_rate(metrics)
        self._assess_q_value_health(metrics)
        self._assess_gradient_health(metrics, model)
        self._assess_prediction_stability(metrics)
        self._assess_loss_stability(metrics)
        
        # Calculate overall health and determine status
        self._calculate_overall_health(metrics)
        self._determine_health_status(metrics)
        
        # Finalize assessment
        self._finalize_assessment(metrics)
        
        return metrics

    def _assess_error_rate(self, metrics: HealthMetrics) -> None:
        """Assess error rate and update metrics."""
        if self.recent_operations > 0:
            metrics.error_rate = self.recent_errors / self.recent_operations
            metrics.recent_error_count = self.recent_errors
            metrics.total_operations = self.recent_operations

    def _assess_q_value_health(self, metrics: HealthMetrics) -> None:
        """Assess Q-value distribution health."""
        if not self.q_values_history:
            return
            
        q_values = np.concatenate([qv.flatten() for qv in self.q_values_history])
        metrics.q_value_mean = float(np.mean(q_values))
        metrics.q_value_std = float(np.std(q_values))
        metrics.q_value_min = float(np.min(q_values))
        metrics.q_value_max = float(np.max(q_values))
        
        q_value_range = metrics.q_value_max - metrics.q_value_min
        
        # Calculate Q-value health score (0.0 to 1.0)
        if (q_value_range > self.thresholds.q_value_range_critical or 
            metrics.q_value_std > self.thresholds.q_value_std_critical):
            metrics.q_value_health = 0.0
        elif (q_value_range > self.thresholds.q_value_range_warning or 
              metrics.q_value_std > self.thresholds.q_value_std_warning):
            metrics.q_value_health = 0.5
        else:
            metrics.q_value_health = 1.0

    def _assess_gradient_health(self, metrics: HealthMetrics, model: Optional[nn.Module]) -> None:
        """Assess gradient health if model and gradient history available."""
        if model is None or not self.gradient_norms:
            return
            
        metrics.gradient_norm_mean = float(np.mean(self.gradient_norms))
        metrics.gradient_norm_max = float(np.max(self.gradient_norms))
        
        # Calculate gradient health score (0.0 to 1.0)
        if metrics.gradient_norm_max > self.thresholds.gradient_norm_critical:
            metrics.gradient_health = 0.0
        elif metrics.gradient_norm_max > self.thresholds.gradient_norm_warning:
            metrics.gradient_health = 0.5
        else:
            metrics.gradient_health = 1.0

    def _assess_prediction_stability(self, metrics: HealthMetrics) -> None:
        """Assess prediction stability across repeated inputs."""
        prediction_stds = []
        
        for state_key, predictions in self.prediction_history.items():
            if len(predictions) > 1:
                # Calculate standard deviation of predictions for the same input
                pred_array = np.stack(predictions)
                std_dev = np.mean(np.std(pred_array, axis=0))
                prediction_stds.append(std_dev)
        
        if not prediction_stds:
            return
            
        metrics.prediction_std = float(np.mean(prediction_stds))
        
        # Calculate prediction stability score (0.0 to 1.0)
        if metrics.prediction_std > self.thresholds.prediction_stability_critical:
            metrics.prediction_stability = 0.0
        elif metrics.prediction_std > self.thresholds.prediction_stability_warning:
            metrics.prediction_stability = 0.5
        else:
            metrics.prediction_stability = 1.0

    def _assess_loss_stability(self, metrics: HealthMetrics) -> None:
        """Assess loss trend stability."""
        if len(self.loss_history) <= 5:
            return
            
        recent_losses = list(self.loss_history)[-5:]
        metrics.recent_loss_values = recent_losses
        
        # Check for increasing loss trend
        if recent_losses[-1] > recent_losses[0] * self.thresholds.loss_increase_critical:
            metrics.loss_stability = 0.0
        elif recent_losses[-1] > recent_losses[0] * self.thresholds.loss_increase_warning:
            metrics.loss_stability = 0.5
        else:
            metrics.loss_stability = 1.0

    def _calculate_overall_health(self, metrics: HealthMetrics) -> None:
        """Calculate overall health score as weighted average."""
        health_scores = [
            (metrics.q_value_health, 0.25),
            (metrics.gradient_health, 0.25),
            (metrics.prediction_stability, 0.25),
            (metrics.loss_stability, 0.25)
        ]
        
        # Filter out None values
        valid_scores = [(score, weight) for score, weight in health_scores if score is not None]
        if valid_scores:
            total_weight = sum(weight for _, weight in valid_scores)
            metrics.overall_health = sum(score * weight for score, weight in valid_scores) / total_weight

    def _determine_health_status(self, metrics: HealthMetrics) -> None:
        """Determine overall status and recommended action."""
        if (metrics.error_rate > self.thresholds.error_rate_critical or 
            metrics.overall_health < 0.3):
            metrics.status = HealthStatus.CRITICAL
            metrics.recommended_action = "Halt training and investigate issues"
        elif (metrics.error_rate > self.thresholds.error_rate_warning or 
              metrics.overall_health < 0.7):
            metrics.status = HealthStatus.WARNING
            metrics.recommended_action = "Monitor closely and consider adjusting hyperparameters"
        else:
            metrics.status = HealthStatus.HEALTHY
            metrics.recommended_action = "Continue normal operation"

    def _finalize_assessment(self, metrics: HealthMetrics) -> None:
        """Finalize assessment by storing metrics and resetting counters."""
        # Store metrics in history
        self.health_history.append(metrics)
        self.last_health_metrics = metrics
        self.last_assessment_time = time.time()
        
        # Reset counters after assessment
        self.recent_errors = 0
        self.recent_operations = 0
        
        # Log health status
        logger.info(
            "Model health assessment: %s (score: %.2f) - %s",
            metrics.status.value,
            metrics.overall_health,
            metrics.recommended_action
        )
    
    def is_healthy(self) -> bool:
        """
        Get binary health status of the model.
        
        Returns:
            True if model is healthy, False otherwise
        """
        # If assessment is stale (>10 minutes), perform a new assessment
        if time.time() - self.last_assessment_time > 600:
            self.assess_health()
        
        return self.last_health_metrics.status == HealthStatus.HEALTHY
    
    def get_health_report(self) -> HealthReport:
        """
        Generate comprehensive health report.
        
        Returns:
            HealthReport with detailed metrics and history
        """
        # Ensure we have current metrics
        if time.time() - self.last_assessment_time > 60:
            self.assess_health()
        
        # Extract historical trend (last 10 assessments)
        historical_trend = [
            (metrics.timestamp, metrics.overall_health)
            for metrics in list(self.health_history)[-10:]
        ]
        
        # Extract recent errors
        recent_errors = list(self.error_history)[-10:]
        
        # Extract recent recovery attempts
        recent_recoveries = list(self.recovery_attempts)[-5:]
        
        # Generate warnings and critical issues
        warnings = []
        critical_issues = []
        
        metrics = self.last_health_metrics
        
        if metrics.error_rate > self.thresholds.error_rate_warning:
            warnings.append(f"High error rate: {metrics.error_rate:.2%}")
        
        if metrics.error_rate > self.thresholds.error_rate_critical:
            critical_issues.append(f"Critical error rate: {metrics.error_rate:.2%}")
        
        if metrics.q_value_health < 0.7:
            warnings.append(f"Q-value instability detected: range={metrics.q_value_max-metrics.q_value_min:.2f}, std={metrics.q_value_std:.2f}")
        
        if metrics.q_value_health < 0.3:
            critical_issues.append(f"Critical Q-value instability: range={metrics.q_value_max-metrics.q_value_min:.2f}, std={metrics.q_value_std:.2f}")
        
        if metrics.gradient_health < 0.7:
            warnings.append(f"High gradient norms: mean={metrics.gradient_norm_mean:.2f}, max={metrics.gradient_norm_max:.2f}")
        
        if metrics.gradient_health < 0.3:
            critical_issues.append(f"Critical gradient instability: max={metrics.gradient_norm_max:.2f}")
        
        if metrics.prediction_stability < 0.7:
            warnings.append(f"Prediction instability: std={metrics.prediction_std:.2f}")
        
        if metrics.prediction_stability < 0.3:
            critical_issues.append(f"Critical prediction instability: std={metrics.prediction_std:.2f}")
        
        if metrics.loss_stability < 0.7 and metrics.recent_loss_values:
            warnings.append(f"Loss increasing trend: {metrics.recent_loss_values[0]:.2f} → {metrics.recent_loss_values[-1]:.2f}")
        
        if metrics.loss_stability < 0.3 and metrics.recent_loss_values:
            critical_issues.append(f"Critical loss divergence: {metrics.recent_loss_values[0]:.2f} → {metrics.recent_loss_values[-1]:.2f}")
        
        return HealthReport(
            current_metrics=metrics,
            historical_trend=historical_trend,
            error_history=recent_errors,
            recovery_attempts=recent_recoveries,
            warnings=warnings,
            critical_issues=critical_issues
        )
    
    def record_recovery_attempt(self, strategy: str, success: bool, details: Dict[str, Any]) -> None:
        """
        Record a recovery attempt for tracking.
        
        Args:
            strategy: Recovery strategy used
            success: Whether the recovery was successful
            details: Additional details about the recovery attempt
        """
        recovery_record = {
            "timestamp": datetime.datetime.now(),
            "strategy": strategy,
            "success": success,
            "details": details
        }
        self.recovery_attempts.append(recovery_record)
        
        logger.info(
            "Recovery attempt: %s - %s",
            "SUCCESS" if success else "FAILED",
            strategy
        )
    
    def reset_counters(self) -> None:
        """Reset error and operation counters."""
        self.recent_errors = 0
        self.recent_operations = 0