"""
Integration of performance monitoring into the training loop.

This module provides functions to integrate performance monitoring
into the RL model training loop.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Callable

from src.ml.performance_monitoring import PerformanceMonitor, PerformanceAlert
from src.ml.performance_profiler import PerformanceProfiler

logger = logging.getLogger(__name__)


def setup_monitor_and_profiler(model, latency_threshold_ms, tensorboard_writer, alert_callback):
    monitor = PerformanceMonitor(
        latency_threshold_ms=latency_threshold_ms,
        alert_callback=alert_callback
    )
    model.performance_monitor = monitor
    if not hasattr(model, 'performance_profiler'):
        from src.ml.rl_model_profiler_integration import integrate_profiler_with_rl_model
        profiler = integrate_profiler_with_rl_model(
            model=model,
            tensorboard_writer=tensorboard_writer,
            latency_target_ms=latency_threshold_ms
        )
    else:
        profiler = model.performance_profiler
    return monitor, profiler

def monitored_update(model, monitor, tensorboard_writer, original_update, step_in_episode_count=0):
    start_time = time.time()
    training_metrics = original_update(step_in_episode_count)
    total_time = (time.time() - start_time) * 1000  # ms
    forward_time = total_time * 0.4
    backward_time = total_time * 0.4
    optimizer_time = total_time * 0.2
    batch_size = model.batch_size
    performance_metrics = monitor.record_training_step(
        batch_size=batch_size,
        forward_time_ms=forward_time,
        backward_time_ms=backward_time,
        optimizer_time_ms=optimizer_time,
        total_time_ms=total_time
    )
    if tensorboard_writer is not None:
        for name, value in performance_metrics.items():
            tensorboard_writer.add_scalar(f"performance/training/{name}", value, model.total_steps)
    check_regression_and_baseline(model, monitor, tensorboard_writer)
    return training_metrics

def check_regression_and_baseline(model, monitor, tensorboard_writer):
    if model.total_steps % 100 == 0:
        regression_result = monitor.detect_regression()
        if regression_result.get("has_regression"):
            logger.warning("Performance regression detected during training")
            if tensorboard_writer is not None:
                for reg in regression_result.get("regressions", []):
                    tensorboard_writer.add_scalar(
                        f"performance/regression/{reg['metric']}",
                        reg['percent_change'],
                        model.total_steps
                    )
            suggestions = monitor.generate_optimization_suggestions()
            logger.info("Performance optimization suggestions:")
            for suggestion in suggestions:
                logger.info(f"  - {suggestion}")
    if model.total_steps % 1000 == 0:
        monitor.update_baseline()

def monitored_act(model, monitor, tensorboard_writer, original_act, state, epsilon=0.0, use_uncertainty=True):
    start_time = time.time()
    action, config_params, info = original_act(state, epsilon, use_uncertainty)
    latency_ms = (time.time() - start_time) * 1000
    inference_metrics = monitor.record_inference(latency_ms)
    info['latency_ms'] = latency_ms
    info['meets_latency_target'] = inference_metrics['meets_latency_target']
    if tensorboard_writer is not None and hasattr(model, 'total_steps'):
        tensorboard_writer.add_scalar(
            "performance/inference/latency_ms",
            latency_ms,
            model.total_steps
        )
    return action, config_params, info

def integrate_performance_monitoring(model, 
                                   latency_threshold_ms=10.0,
                                   tensorboard_writer=None,
                                   alert_callback=None):
    """
    Integrate performance monitoring into the RL model training loop.
    
    Args:
        model: SelfImprovingRLModel instance
        latency_threshold_ms: Target latency for inference in milliseconds
        tensorboard_writer: Optional TensorBoard SummaryWriter
        alert_callback: Optional callback function for alerts
        
    Returns:
        PerformanceMonitor instance
    """
    monitor, _ = setup_monitor_and_profiler(model, latency_threshold_ms, tensorboard_writer, alert_callback)

    # Patch update method
    original_update = model.update
    def _monitored_update(step_in_episode_count=0):
        return monitored_update(model, monitor, tensorboard_writer, original_update, step_in_episode_count)
    model.original_update = original_update
    model.update = _monitored_update

    # Patch act method
    original_act = model.act
    if not hasattr(model, 'original_act'):
        def _monitored_act(state, epsilon=0.0, use_uncertainty=True):
            return monitored_act(model, monitor, tensorboard_writer, original_act, state, epsilon, use_uncertainty)
        model.original_act = original_act
        model.act = _monitored_act

    logger.info(f"Performance monitoring integrated into training loop, latency threshold: {latency_threshold_ms}ms")
    return monitor


def create_alert_callback(model, tensorboard_writer=None):
    """
    Create a callback function for performance alerts.
    
    Args:
        model: SelfImprovingRLModel instance
        tensorboard_writer: Optional TensorBoard SummaryWriter
        
    Returns:
        Alert callback function
    """
    def alert_callback(alert: PerformanceAlert):
        """Handle performance alerts."""
        # Log alert
        logger.warning(f"Performance alert: {alert.message}")
        
        # Log to TensorBoard if available
        if tensorboard_writer is not None and hasattr(model, 'total_steps'):
            tensorboard_writer.add_scalar(
                f"performance/alerts/{alert.metric_name}",
                alert.current_value,
                model.total_steps
            )
            
            # Add text summary of alert
            tensorboard_writer.add_text(
                "performance/alert_messages",
                f"{alert.severity.upper()}: {alert.message}",
                model.total_steps
            )
        
        # Take action based on severity
        if alert.severity == 'critical':
            # For critical alerts, we might want to take more drastic action
            # such as reducing batch size, learning rate, etc.
            logger.error(f"Critical performance alert: {alert.message}")
            
            # Example: Reduce batch size if training step time is too high
            if alert.metric_name == 'mean_training_step_time' and hasattr(model, 'batch_size'):
                new_batch_size = max(1, model.batch_size // 2)
                logger.warning(f"Reducing batch size from {model.batch_size} to {new_batch_size}")
                model.batch_size = new_batch_size
    
    return alert_callback


def detect_performance_regression(model, window_size=10, threshold_percent=20.0):
    """
    Detect performance regression in the model.
    
    Args:
        model: SelfImprovingRLModel instance
        window_size: Number of recent metrics to compare
        threshold_percent: Percentage change to consider a regression
        
    Returns:
        Dictionary with regression analysis results
    """
    if not hasattr(model, 'performance_monitor'):
        logger.warning("Performance monitor not integrated with model")
        return {"has_regression": False, "message": "Performance monitor not integrated"}
    
    return model.performance_monitor.detect_regression(
        window_size=window_size,
        threshold_percent=threshold_percent
    )


def get_optimization_suggestions(model):
    """
    Get performance optimization suggestions for the model.
    
    Args:
        model: SelfImprovingRLModel instance
        
    Returns:
        List of optimization suggestions
    """
    if not hasattr(model, 'performance_monitor'):
        logger.warning("Performance monitor not integrated with model")
        return ["Integrate performance monitor to get optimization suggestions."]
    
    return model.performance_monitor.generate_optimization_suggestions()