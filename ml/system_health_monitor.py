"""
Comprehensive system health monitoring for RL models.

This module provides tools for monitoring system health metrics, including
CPU and GPU memory usage, model inference latency, error rates, and fallback
activation frequency.
"""

# Constants
import torch
import numpy as np
import time
import os
import json
import logging
import psutil
import threading
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from datetime import datetime
from enum import Enum
import gc

# Constants
TIME_SECONDS_LABEL = "Time (seconds)"
ERROR_RATE_LABEL = "Error Rate"


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


@dataclass
class MemoryMetrics:
    """Container for memory metrics."""
    timestamp: float
    cpu_total_gb: float
    cpu_available_gb: float
    cpu_used_gb: float
    cpu_percent: float
    process_memory_gb: float
    gpu_total_gb: Optional[float] = None
    gpu_used_gb: Optional[float] = None
    gpu_percent: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class LatencyMetrics:
    """Container for latency metrics."""
    timestamp: float
    inference_time_ms: float
    preprocessing_time_ms: float
    postprocessing_time_ms: float
    total_time_ms: float
    batch_size: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ErrorMetrics:
    """Container for error metrics."""
    timestamp: float
    error_count: int
    error_rate: float
    error_type: str
    error_message: str
    context: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class FallbackMetrics:
    """Container for fallback metrics."""
    timestamp: float
    fallback_level: int
    fallback_reason: str
    duration_ms: float
    success: bool
    context: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SystemHealthMetrics:
    """Container for system health metrics."""
    timestamp: float
    status: HealthStatus
    cpu_health: float  # 0.0 to 1.0
    memory_health: float  # 0.0 to 1.0
    gpu_health: Optional[float] = None  # 0.0 to 1.0
    latency_health: Optional[float] = None  # 0.0 to 1.0
    error_health: Optional[float] = None  # 0.0 to 1.0
    overall_health: float  # 0.0 to 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["status"] = self.status.value
        return result


@dataclass
class HealthMonitorConfig:
    """Configuration for SystemHealthMonitor."""
    log_dir: str = "logs"
    model_name: str = "rl_model"
    metrics_buffer_size: int = 1000
    memory_warning_threshold: float = 0.8  # 80% memory usage
    memory_critical_threshold: float = 0.9  # 90% memory usage
    latency_warning_threshold_ms: float = 8.0  # 8ms
    latency_critical_threshold_ms: float = 10.0  # 10ms
    error_rate_warning_threshold: float = 0.01  # 1% error rate
    error_rate_critical_threshold: float = 0.05  # 5% error rate
    fallback_warning_threshold: float = 0.1  # 10% fallback rate
    fallback_critical_threshold: float = 0.3  # 30% fallback rate
    monitoring_interval_sec: float = 1.0  # 1 second
    alert_callback: Optional[Callable[[str, HealthStatus, Dict[str, Any]], None]] = None
    use_gpu: bool = torch.cuda.is_available()


class SystemHealthMonitor:
    """
    Comprehensive system health monitoring for RL models.
    
    This class provides tools for:
    - Monitoring CPU and GPU memory usage
    - Tracking model inference latency
    - Monitoring error rates and fallback activation frequency
    - Visualizing system health metrics
    - Alerting on critical health issues
    """
    
    def __init__(
        self,
        config: Optional[HealthMonitorConfig] = None,
        **kwargs
    ):
        """
        Initialize the system health monitor.
        
        Args:
            config: Configuration object with monitoring settings
            **kwargs: Individual configuration parameters (will be merged with config)
        """
        # Use provided config or create default, then update with kwargs
        if config is None:
            config = HealthMonitorConfig(**kwargs)
        else:
            # Update config with any provided kwargs
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # Store configuration
        self.config = config
        
        # Assign configuration values to instance attributes for easier access
        self.log_dir = config.log_dir
        self.model_name = config.model_name
        self.metrics_buffer_size = config.metrics_buffer_size
        self.memory_warning_threshold = config.memory_warning_threshold
        self.memory_critical_threshold = config.memory_critical_threshold
        self.latency_warning_threshold_ms = config.latency_warning_threshold_ms
        self.latency_critical_threshold_ms = config.latency_critical_threshold_ms
        self.error_rate_warning_threshold = config.error_rate_warning_threshold
        self.error_rate_critical_threshold = config.error_rate_critical_threshold
        self.fallback_warning_threshold = config.fallback_warning_threshold
        self.fallback_critical_threshold = config.fallback_critical_threshold
        self.monitoring_interval_sec = config.monitoring_interval_sec
        self.alert_callback = config.alert_callback
        self.use_gpu = config.use_gpu
        
        # Create log directory
        self.run_dir = os.path.join(
            self.log_dir, 
            f"{self.model_name}_health_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.memory_metrics = deque(maxlen=self.metrics_buffer_size)
        self.latency_metrics = deque(maxlen=self.metrics_buffer_size)
        self.error_metrics = deque(maxlen=self.metrics_buffer_size)
        self.fallback_metrics = deque(maxlen=self.metrics_buffer_size)
        self.health_metrics = deque(maxlen=self.metrics_buffer_size)
        
        # Initialize counters
        self.total_inferences = 0
        self.total_errors = 0
        self.total_fallbacks = 0
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize monitoring thread
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        
        # Initialize current health status
        self.current_health_status = HealthStatus.HEALTHY
        
        self.logger.info(f"System health monitor initialized. Logs will be saved to {self.run_dir}")
    
    def start_monitoring(self):
        """Start periodic monitoring."""
        if self.monitoring_thread is not None and self.monitoring_thread.is_alive():
            self.logger.warning("Monitoring thread is already running")
            return
        
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.logger.info("System health monitoring started")
    
    def stop_monitoring(self):
        """Stop periodic monitoring."""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.logger.warning("Monitoring thread is not running")
            return
        
        self.stop_monitoring.set()
        self.monitoring_thread.join(timeout=5.0)
        
        if self.monitoring_thread.is_alive():
            self.logger.warning("Monitoring thread did not stop gracefully")
        else:
            self.logger.info("System health monitoring stopped")
    
    def _monitoring_loop(self):
        """Periodic monitoring loop."""
        while not self.stop_monitoring.is_set():
            try:
                # Monitor memory usage
                self.monitor_memory()
                
                # Assess overall health
                self.assess_health()
                
                # Sleep for monitoring interval
                time.sleep(self.monitoring_interval_sec)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def monitor_memory(self) -> MemoryMetrics:
        """
        Monitor memory usage.
        
        Returns:
            MemoryMetrics object with memory usage information
        """
        # Get CPU memory usage
        cpu_memory = psutil.virtual_memory()
        cpu_total_gb = cpu_memory.total / (1024 ** 3)
        cpu_available_gb = cpu_memory.available / (1024 ** 3)
        cpu_used_gb = cpu_total_gb - cpu_available_gb
        cpu_percent = cpu_memory.percent
        
        # Get process memory usage
        process = psutil.Process()
        process_memory_gb = process.memory_info().rss / (1024 ** 3)
        
        # Get GPU memory usage if available
        gpu_total_gb = None
        gpu_used_gb = None
        gpu_percent = None
        
        if self.use_gpu and torch.cuda.is_available():
            try:
                gpu_used_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                gpu_reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
                
                # Get total GPU memory
                if hasattr(torch.cuda, 'get_device_properties'):
                    gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                else:
                    # Fallback to reserved memory if total memory is not available
                    gpu_total_gb = gpu_reserved_gb
                
                # Calculate GPU memory percentage
                if gpu_total_gb > 0:
                    gpu_percent = (gpu_used_gb / gpu_total_gb) * 100
            except Exception as e:
                self.logger.warning(f"Error getting GPU memory usage: {e}")
        
        # Create memory metrics
        metrics = MemoryMetrics(
            timestamp=time.time(),
            cpu_total_gb=cpu_total_gb,
            cpu_available_gb=cpu_available_gb,
            cpu_used_gb=cpu_used_gb,
            cpu_percent=cpu_percent,
            process_memory_gb=process_memory_gb,
            gpu_total_gb=gpu_total_gb,
            gpu_used_gb=gpu_used_gb,
            gpu_percent=gpu_percent
        )
        
        # Store metrics
        self.memory_metrics.append(metrics)
        
        # Check for memory issues
        self._check_memory_health(metrics)
        
        return metrics
    
    def log_inference_latency(
        self,
        inference_time_ms: float,
        preprocessing_time_ms: float = 0.0,
        postprocessing_time_ms: float = 0.0,
        batch_size: int = 1
    ) -> LatencyMetrics:
        """
        Log inference latency.
        
        Args:
            inference_time_ms: Inference time in milliseconds
            preprocessing_time_ms: Preprocessing time in milliseconds
            postprocessing_time_ms: Postprocessing time in milliseconds
            batch_size: Batch size
            
        Returns:
            LatencyMetrics object with latency information
        """
        # Calculate total time
        total_time_ms = inference_time_ms + preprocessing_time_ms + postprocessing_time_ms
        
        # Create latency metrics
        metrics = LatencyMetrics(
            timestamp=time.time(),
            inference_time_ms=inference_time_ms,
            preprocessing_time_ms=preprocessing_time_ms,
            postprocessing_time_ms=postprocessing_time_ms,
            total_time_ms=total_time_ms,
            batch_size=batch_size
        )
        
        # Store metrics
        self.latency_metrics.append(metrics)
        
        # Update total inferences
        self.total_inferences += 1
        
        # Check for latency issues
        self._check_latency_health(metrics)
        
        return metrics
    
    def log_error(
        self,
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorMetrics:
        """
        Log an error.
        
        Args:
            error_type: Type of error
            error_message: Error message
            context: Additional context for the error
            
        Returns:
            ErrorMetrics object with error information
        """
        # Update total errors
        self.total_errors += 1
        
        # Calculate error rate
        error_rate = self.total_errors / max(1, self.total_inferences)
        
        # Create error metrics
        metrics = ErrorMetrics(
            timestamp=time.time(),
            error_count=self.total_errors,
            error_rate=error_rate,
            error_type=error_type,
            error_message=error_message,
            context=context
        )
        
        # Store metrics
        self.error_metrics.append(metrics)
        
        # Check for error rate issues
        self._check_error_health(metrics)
        
        return metrics
    
    def log_fallback(
        self,
        fallback_level: int,
        fallback_reason: str,
        duration_ms: float,
        success: bool,
        context: Optional[Dict[str, Any]] = None
    ) -> FallbackMetrics:
        """
        Log a fallback activation.
        
        Args:
            fallback_level: Fallback level (higher means more severe)
            fallback_reason: Reason for fallback
            duration_ms: Duration of fallback in milliseconds
            success: Whether the fallback was successful
            context: Additional context for the fallback
            
        Returns:
            FallbackMetrics object with fallback information
        """
        # Update total fallbacks
        self.total_fallbacks += 1
        
        # Create fallback metrics
        metrics = FallbackMetrics(
            timestamp=time.time(),
            fallback_level=fallback_level,
            fallback_reason=fallback_reason,
            duration_ms=duration_ms,
            success=success,
            context=context
        )
        
        # Store metrics
        self.fallback_metrics.append(metrics)
        
        # Check for fallback rate issues
        self._check_fallback_health(metrics)
        
        return metrics
    
    def assess_health(self) -> SystemHealthMetrics:
        """
        Assess overall system health.
        
        Returns:
            SystemHealthMetrics object with health information
        """
        # Calculate CPU health (0.0 to 1.0, higher is better)
        cpu_health = 1.0
        if self.memory_metrics:
            latest_memory = self.memory_metrics[-1]
            cpu_usage = latest_memory.cpu_percent / 100.0
            cpu_health = 1.0 - cpu_usage
        
        # Calculate memory health (0.0 to 1.0, higher is better)
        memory_health = 1.0
        if self.memory_metrics:
            latest_memory = self.memory_metrics[-1]
            process_memory_ratio = latest_memory.process_memory_gb / latest_memory.cpu_total_gb
            memory_health = 1.0 - process_memory_ratio
        
        # Calculate GPU health (0.0 to 1.0, higher is better)
        gpu_health = None
        if self.use_gpu and self.memory_metrics and self.memory_metrics[-1].gpu_percent is not None:
            latest_memory = self.memory_metrics[-1]
            gpu_usage = latest_memory.gpu_percent / 100.0
            gpu_health = 1.0 - gpu_usage
        
        # Calculate latency health (0.0 to 1.0, higher is better)
        latency_health = None
        if self.latency_metrics:
            latest_latency = self.latency_metrics[-1]
            latency_ratio = latest_latency.total_time_ms / self.latency_critical_threshold_ms
            latency_health = max(0.0, 1.0 - latency_ratio)
        
        # Calculate error health (0.0 to 1.0, higher is better)
        error_health = None
        if self.total_inferences > 0:
            error_rate = self.total_errors / self.total_inferences
            error_ratio = error_rate / self.error_rate_critical_threshold
            error_health = max(0.0, 1.0 - error_ratio)
        
        # Calculate overall health (0.0 to 1.0, higher is better)
        health_factors = [cpu_health, memory_health]
        if gpu_health is not None:
            health_factors.append(gpu_health)
        if latency_health is not None:
            health_factors.append(latency_health)
        if error_health is not None:
            health_factors.append(error_health)
        
        overall_health = sum(health_factors) / len(health_factors)
        
        # Determine health status
        if overall_health < 0.5:
            status = HealthStatus.CRITICAL
        elif overall_health < 0.7:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY
        
        # Create health metrics
        metrics = SystemHealthMetrics(
            timestamp=time.time(),
            status=status,
            cpu_health=cpu_health,
            memory_health=memory_health,
            gpu_health=gpu_health,
            latency_health=latency_health,
            error_health=error_health,
            overall_health=overall_health
        )
        
        # Store metrics
        self.health_metrics.append(metrics)
        
        # Check for status change
        if status != self.current_health_status:
            self._handle_health_status_change(self.current_health_status, status, metrics)
            self.current_health_status = status
        
        return metrics
    
    def _check_memory_health(self, metrics: MemoryMetrics):
        """
        Check memory health and trigger alerts if needed.
        
        Args:
            metrics: Memory metrics
        """
        # Check CPU memory usage
        cpu_usage = metrics.cpu_percent / 100.0
        if cpu_usage > self.memory_critical_threshold:
            self._trigger_alert(
                "CPU memory usage critical",
                HealthStatus.CRITICAL,
                {
                    "cpu_usage": cpu_usage,
                    "threshold": self.memory_critical_threshold,
                    "cpu_total_gb": metrics.cpu_total_gb,
                    "cpu_used_gb": metrics.cpu_used_gb
                }
            )
        elif cpu_usage > self.memory_warning_threshold:
            self._trigger_alert(
                "CPU memory usage warning",
                HealthStatus.WARNING,
                {
                    "cpu_usage": cpu_usage,
                    "threshold": self.memory_warning_threshold,
                    "cpu_total_gb": metrics.cpu_total_gb,
                    "cpu_used_gb": metrics.cpu_used_gb
                }
            )
        
        # Check process memory usage
        process_memory_ratio = metrics.process_memory_gb / metrics.cpu_total_gb
        if process_memory_ratio > self.memory_critical_threshold:
            self._trigger_alert(
                "Process memory usage critical",
                HealthStatus.CRITICAL,
                {
                    "process_memory_ratio": process_memory_ratio,
                    "threshold": self.memory_critical_threshold,
                    "process_memory_gb": metrics.process_memory_gb,
                    "cpu_total_gb": metrics.cpu_total_gb
                }
            )
        elif process_memory_ratio > self.memory_warning_threshold:
            self._trigger_alert(
                "Process memory usage warning",
                HealthStatus.WARNING,
                {
                    "process_memory_ratio": process_memory_ratio,
                    "threshold": self.memory_warning_threshold,
                    "process_memory_gb": metrics.process_memory_gb,
                    "cpu_total_gb": metrics.cpu_total_gb
                }
            )
        
        # Check GPU memory usage
        if self.use_gpu and metrics.gpu_percent is not None:
            gpu_usage = metrics.gpu_percent / 100.0
            if gpu_usage > self.memory_critical_threshold:
                self._trigger_alert(
                    "GPU memory usage critical",
                    HealthStatus.CRITICAL,
                    {
                        "gpu_usage": gpu_usage,
                        "threshold": self.memory_critical_threshold,
                        "gpu_total_gb": metrics.gpu_total_gb,
                        "gpu_used_gb": metrics.gpu_used_gb
                    }
                )
            elif gpu_usage > self.memory_warning_threshold:
                self._trigger_alert(
                    "GPU memory usage warning",
                    HealthStatus.WARNING,
                    {
                        "gpu_usage": gpu_usage,
                        "threshold": self.memory_warning_threshold,
                        "gpu_total_gb": metrics.gpu_total_gb,
                        "gpu_used_gb": metrics.gpu_used_gb
                    }
                )
    
    def _check_latency_health(self, metrics: LatencyMetrics):
        """
        Check latency health and trigger alerts if needed.
        
        Args:
            metrics: Latency metrics
        """
        if metrics.total_time_ms > self.latency_critical_threshold_ms:
            self._trigger_alert(
                "Inference latency critical",
                HealthStatus.CRITICAL,
                {
                    "latency_ms": metrics.total_time_ms,
                    "threshold_ms": self.latency_critical_threshold_ms,
                    "inference_time_ms": metrics.inference_time_ms,
                    "preprocessing_time_ms": metrics.preprocessing_time_ms,
                    "postprocessing_time_ms": metrics.postprocessing_time_ms,
                    "batch_size": metrics.batch_size
                }
            )
        elif metrics.total_time_ms > self.latency_warning_threshold_ms:
            self._trigger_alert(
                "Inference latency warning",
                HealthStatus.WARNING,
                {
                    "latency_ms": metrics.total_time_ms,
                    "threshold_ms": self.latency_warning_threshold_ms,
                    "inference_time_ms": metrics.inference_time_ms,
                    "preprocessing_time_ms": metrics.preprocessing_time_ms,
                    "postprocessing_time_ms": metrics.postprocessing_time_ms,
                    "batch_size": metrics.batch_size
                }
            )
    
    def _check_error_health(self, metrics: ErrorMetrics):
        """
        Check error health and trigger alerts if needed.
        
        Args:
            metrics: Error metrics
        """
        if metrics.error_rate > self.error_rate_critical_threshold:
            self._trigger_alert(
                "Error rate critical",
                HealthStatus.CRITICAL,
                {
                    "error_rate": metrics.error_rate,
                    "threshold": self.error_rate_critical_threshold,
                    "error_count": metrics.error_count,
                    "total_inferences": self.total_inferences,
                    "error_type": metrics.error_type,
                    "error_message": metrics.error_message
                }
            )
        elif metrics.error_rate > self.error_rate_warning_threshold:
            self._trigger_alert(
                "Error rate warning",
                HealthStatus.WARNING,
                {
                    "error_rate": metrics.error_rate,
                    "threshold": self.error_rate_warning_threshold,
                    "error_count": metrics.error_count,
                    "total_inferences": self.total_inferences,
                    "error_type": metrics.error_type,
                    "error_message": metrics.error_message
                }
            )
    
    def _check_fallback_health(self, metrics: FallbackMetrics):
        """
        Check fallback health and trigger alerts if needed.
        
        Args:
            metrics: Fallback metrics
        """
        fallback_rate = self.total_fallbacks / max(1, self.total_inferences)
        
        if fallback_rate > self.fallback_critical_threshold:
            self._trigger_alert(
                "Fallback rate critical",
                HealthStatus.CRITICAL,
                {
                    "fallback_rate": fallback_rate,
                    "threshold": self.fallback_critical_threshold,
                    "fallback_count": self.total_fallbacks,
                    "total_inferences": self.total_inferences,
                    "fallback_level": metrics.fallback_level,
                    "fallback_reason": metrics.fallback_reason
                }
            )
        elif fallback_rate > self.fallback_warning_threshold:
            self._trigger_alert(
                "Fallback rate warning",
                HealthStatus.WARNING,
                {
                    "fallback_rate": fallback_rate,
                    "threshold": self.fallback_warning_threshold,
                    "fallback_count": self.total_fallbacks,
                    "total_inferences": self.total_inferences,
                    "fallback_level": metrics.fallback_level,
                    "fallback_reason": metrics.fallback_reason
                }
            )
    
    def _handle_health_status_change(
        self,
        old_status: HealthStatus,
        new_status: HealthStatus,
        metrics: SystemHealthMetrics
    ):
        """
        Handle health status change.
        
        Args:
            old_status: Old health status
            new_status: New health status
            metrics: Health metrics
        """
        if new_status == HealthStatus.CRITICAL:
            self._trigger_alert(
                "System health critical",
                HealthStatus.CRITICAL,
                {
                    "old_status": old_status.value,
                    "new_status": new_status.value,
                    "overall_health": metrics.overall_health,
                    "cpu_health": metrics.cpu_health,
                    "memory_health": metrics.memory_health,
                    "gpu_health": metrics.gpu_health,
                    "latency_health": metrics.latency_health,
                    "error_health": metrics.error_health
                }
            )
        elif new_status == HealthStatus.WARNING:
            self._trigger_alert(
                "System health warning",
                HealthStatus.WARNING,
                {
                    "old_status": old_status.value,
                    "new_status": new_status.value,
                    "overall_health": metrics.overall_health,
                    "cpu_health": metrics.cpu_health,
                    "memory_health": metrics.memory_health,
                    "gpu_health": metrics.gpu_health,
                    "latency_health": metrics.latency_health,
                    "error_health": metrics.error_health
                }
            )
        elif new_status == HealthStatus.HEALTHY and old_status != HealthStatus.HEALTHY:
            self._trigger_alert(
                "System health recovered",
                HealthStatus.HEALTHY,
                {
                    "old_status": old_status.value,
                    "new_status": new_status.value,
                    "overall_health": metrics.overall_health,
                    "cpu_health": metrics.cpu_health,
                    "memory_health": metrics.memory_health,
                    "gpu_health": metrics.gpu_health,
                    "latency_health": metrics.latency_health,
                    "error_health": metrics.error_health
                }
            )
    
    def _trigger_alert(self, message: str, status: HealthStatus, context: Dict[str, Any]):
        """
        Trigger an alert.
        
        Args:
            message: Alert message
            status: Health status
            context: Alert context
        """
        # Log alert
        if status == HealthStatus.CRITICAL:
            self.logger.critical(f"ALERT: {message}")
        elif status == HealthStatus.WARNING:
            self.logger.warning(f"ALERT: {message}")
        else:
            self.logger.info(f"ALERT: {message}")
        
        # Call alert callback if provided
        if self.alert_callback:
            self.alert_callback(message, status, context)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage.
        
        Returns:
            Dictionary with memory usage information
        """
        if not self.memory_metrics:
            return {}
        
        latest = self.memory_metrics[-1]
        
        result = {
            "cpu_total_gb": latest.cpu_total_gb,
            "cpu_available_gb": latest.cpu_available_gb,
            "cpu_used_gb": latest.cpu_used_gb,
            "cpu_percent": latest.cpu_percent,
            "process_memory_gb": latest.process_memory_gb
        }
        
        if latest.gpu_total_gb is not None:
            result["gpu_total_gb"] = latest.gpu_total_gb
        
        if latest.gpu_used_gb is not None:
            result["gpu_used_gb"] = latest.gpu_used_gb
        
        if latest.gpu_percent is not None:
            result["gpu_percent"] = latest.gpu_percent
        
        return result
    
    def get_latency_stats(self) -> Dict[str, float]:
        """
        Get latency statistics.
        
        Returns:
            Dictionary with latency statistics
        """
        if not self.latency_metrics:
            return {}
        
        total_times = [m.total_time_ms for m in self.latency_metrics]
        inference_times = [m.inference_time_ms for m in self.latency_metrics]
        
        return {
            "total_time_ms_mean": np.mean(total_times),
            "total_time_ms_std": np.std(total_times),
            "total_time_ms_min": np.min(total_times),
            "total_time_ms_max": np.max(total_times),
            "total_time_ms_p50": np.percentile(total_times, 50),
            "total_time_ms_p95": np.percentile(total_times, 95),
            "total_time_ms_p99": np.percentile(total_times, 99),
            "inference_time_ms_mean": np.mean(inference_times),
            "inference_time_ms_std": np.std(inference_times),
            "inference_time_ms_min": np.min(inference_times),
            "inference_time_ms_max": np.max(inference_times),
            "inference_time_ms_p50": np.percentile(inference_times, 50),
            "inference_time_ms_p95": np.percentile(inference_times, 95),
            "inference_time_ms_p99": np.percentile(inference_times, 99)
        }
    
    def get_error_stats(self) -> Dict[str, Any]:
        """
        Get error statistics.
        
        Returns:
            Dictionary with error statistics
        """
        if self.total_inferences == 0:
            return {
                "error_rate": 0.0,
                "error_count": 0,
                "total_inferences": 0
            }
        
        error_rate = self.total_errors / self.total_inferences
        
        result = {
            "error_rate": error_rate,
            "error_count": self.total_errors,
            "total_inferences": self.total_inferences
        }
        
        # Add error type counts
        error_types = {}
        for metrics in self.error_metrics:
            if metrics.error_type not in error_types:
                error_types[metrics.error_type] = 0
            error_types[metrics.error_type] += 1
        
        if error_types:
            result["error_types"] = error_types
        
        return result
    
    def get_fallback_stats(self) -> Dict[str, Any]:
        """
        Get fallback statistics.
        
        Returns:
            Dictionary with fallback statistics
        """
        if self.total_inferences == 0:
            return {
                "fallback_rate": 0.0,
                "fallback_count": 0,
                "total_inferences": 0
            }
        
        fallback_rate = self.total_fallbacks / self.total_inferences
        
        result = {
            "fallback_rate": fallback_rate,
            "fallback_count": self.total_fallbacks,
            "total_inferences": self.total_inferences
        }
        
        # Add fallback level counts
        fallback_levels = {}
        for metrics in self.fallback_metrics:
            level = metrics.fallback_level
            if level not in fallback_levels:
                fallback_levels[level] = 0
            fallback_levels[level] += 1
        
        if fallback_levels:
            result["fallback_levels"] = fallback_levels
        
        # Add fallback success rate
        if self.fallback_metrics:
            success_count = sum(1 for m in self.fallback_metrics if m.success)
            result["fallback_success_rate"] = success_count / len(self.fallback_metrics)
        
        return result
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get current health status.
        
        Returns:
            Dictionary with health status information
        """
        if not self.health_metrics:
            return {
                "status": HealthStatus.HEALTHY.value,
                "overall_health": 1.0
            }
        
        latest = self.health_metrics[-1]
        
        result = {
            "status": latest.status.value,
            "overall_health": latest.overall_health,
            "cpu_health": latest.cpu_health,
            "memory_health": latest.memory_health
        }
        
        if latest.gpu_health is not None:
            result["gpu_health"] = latest.gpu_health
        
        if latest.latency_health is not None:
            result["latency_health"] = latest.latency_health
        
        if latest.error_health is not None:
            result["error_health"] = latest.error_health
        
        return result
    
    def get_system_summary(self) -> Dict[str, Any]:
        """
        Get system summary.
        
        Returns:
            Dictionary with system summary information
        """
        return {
            "health": self.get_health_status(),
            "memory": self.get_memory_usage(),
            "latency": self.get_latency_stats(),
            "errors": self.get_error_stats(),
            "fallbacks": self.get_fallback_stats(),
            "timestamp": time.time()
        }
    
    def save_metrics(self, filename_prefix: str = "system_health"):
        """
        Save metrics to files.
        
        Args:
            filename_prefix: Prefix for metric filenames
        """
        # Save memory metrics
        if self.memory_metrics:
            memory_filepath = os.path.join(self.run_dir, f"{filename_prefix}_memory.json")
            with open(memory_filepath, "w") as f:
                json.dump([m.to_dict() for m in self.memory_metrics], f, indent=2)
        
        # Save latency metrics
        if self.latency_metrics:
            latency_filepath = os.path.join(self.run_dir, f"{filename_prefix}_latency.json")
            with open(latency_filepath, "w") as f:
                json.dump([m.to_dict() for m in self.latency_metrics], f, indent=2)
        
        # Save error metrics
        if self.error_metrics:
            error_filepath = os.path.join(self.run_dir, f"{filename_prefix}_errors.json")
            with open(error_filepath, "w") as f:
                json.dump([m.to_dict() for m in self.error_metrics], f, indent=2)
        
        # Save fallback metrics
        if self.fallback_metrics:
            fallback_filepath = os.path.join(self.run_dir, f"{filename_prefix}_fallbacks.json")
            with open(fallback_filepath, "w") as f:
                json.dump([m.to_dict() for m in self.fallback_metrics], f, indent=2)
        
        # Save health metrics
        if self.health_metrics:
            health_filepath = os.path.join(self.run_dir, f"{filename_prefix}_health.json")
            with open(health_filepath, "w") as f:
                json.dump([m.to_dict() for m in self.health_metrics], f, indent=2)
        
        # Save system summary
        summary_filepath = os.path.join(self.run_dir, f"{filename_prefix}_summary.json")
        with open(summary_filepath, "w") as f:
            json.dump(self.get_system_summary(), f, indent=2)
        
        self.logger.info(f"System health metrics saved to {self.run_dir}")
    
    def plot_metrics(self, save_dir: Optional[str] = None):
        """
        Plot system health metrics.

        Args:
            save_dir: Directory to save plots to (defaults to run_dir)
        """
        save_dir = save_dir or self.run_dir
        os.makedirs(save_dir, exist_ok=True)

        self._plot_memory_metrics(save_dir)
        self._plot_latency_metrics(save_dir)
        self._plot_error_metrics(save_dir)
        self._plot_fallback_metrics(save_dir)
        self._plot_health_metrics(save_dir)

        self.logger.info(f"System health plots saved to {save_dir}")

    def _plot_memory_metrics(self, save_dir: str):
        if not self.memory_metrics:
            return
        timestamps = [(m.timestamp - self.memory_metrics[0].timestamp) for m in self.memory_metrics]
        cpu_percents = [m.cpu_percent for m in self.memory_metrics]
        process_memory_gbs = [m.process_memory_gb for m in self.memory_metrics]

        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, cpu_percents, label="CPU Usage (%)")
        plt.axhline(y=self.memory_warning_threshold * 100, color='orange', linestyle='--', label=f"Warning Threshold ({self.memory_warning_threshold * 100}%)")
        plt.axhline(y=self.memory_critical_threshold * 100, color='red', linestyle='--', label=f"Critical Threshold ({self.memory_critical_threshold * 100}%)")
        plt.xlabel(TIME_SECONDS_LABEL)
        plt.ylabel("CPU Usage (%)")
        plt.title("CPU Usage")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "cpu_usage.png"))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, process_memory_gbs, label="Process Memory (GB)")
        plt.xlabel(TIME_SECONDS_LABEL)
        plt.ylabel("Memory (GB)")
        plt.title("Process Memory Usage")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "process_memory.png"))
        plt.close()

        gpu_percents = [m.gpu_percent for m in self.memory_metrics if m.gpu_percent is not None]
        if gpu_percents:
            gpu_timestamps = [(m.timestamp - self.memory_metrics[0].timestamp) for m in self.memory_metrics if m.gpu_percent is not None]
            plt.figure(figsize=(10, 6))
            plt.plot(gpu_timestamps, gpu_percents, label="GPU Usage (%)")
            plt.axhline(y=self.memory_warning_threshold * 100, color='orange', linestyle='--', label=f"Warning Threshold ({self.memory_warning_threshold * 100}%)")
            plt.axhline(y=self.memory_critical_threshold * 100, color='red', linestyle='--', label=f"Critical Threshold ({self.memory_critical_threshold * 100}%)")
            plt.xlabel(TIME_SECONDS_LABEL)
            plt.ylabel("GPU Usage (%)")
            plt.title("GPU Usage")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, "gpu_usage.png"))
            plt.close()

    def _plot_latency_metrics(self, save_dir: str):
        if not self.latency_metrics:
            return
        timestamps = [(m.timestamp - self.latency_metrics[0].timestamp) for m in self.latency_metrics]
        total_times = [m.total_time_ms for m in self.latency_metrics]
        inference_times = [m.inference_time_ms for m in self.latency_metrics]

        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, total_times, label="Total Time")
        plt.plot(timestamps, inference_times, label="Inference Time")
        plt.axhline(y=self.latency_warning_threshold_ms, color='orange', linestyle='--', label=f"Warning Threshold ({self.latency_warning_threshold_ms} ms)")
        plt.axhline(y=self.latency_critical_threshold_ms, color='red', linestyle='--', label=f"Critical Threshold ({self.latency_critical_threshold_ms} ms)")
        plt.xlabel(TIME_SECONDS_LABEL)
        plt.ylabel("Latency (ms)")
        plt.title("Inference Latency")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "latency.png"))
        plt.close()

    def _plot_error_metrics(self, save_dir: str):
        if not self.error_metrics:
            return
        timestamps = [(m.timestamp - self.error_metrics[0].timestamp) for m in self.error_metrics]
        error_rates = [m.error_rate for m in self.error_metrics]

        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, error_rates, label="Error Rate")
        plt.axhline(y=self.error_rate_warning_threshold, color='orange', linestyle='--', label=f"Warning Threshold ({self.error_rate_warning_threshold})")
        plt.axhline(y=self.error_rate_critical_threshold, color='red', linestyle='--', label=f"Critical Threshold ({self.error_rate_critical_threshold})")
        plt.xlabel(TIME_SECONDS_LABEL)
        plt.ylabel(ERROR_RATE_LABEL)
        plt.title(ERROR_RATE_LABEL)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "error_rate.png"))
        plt.close()

        error_types = {}
        for metrics in self.error_metrics:
            if metrics.error_type not in error_types:
                error_types[metrics.error_type] = 0
            error_types[metrics.error_type] += 1

        if error_types:
            plt.figure(figsize=(10, 6))
            plt.bar(error_types.keys(), error_types.values())
            plt.xlabel("Error Type")
            plt.ylabel("Count")
            plt.title("Error Types")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "error_types.png"))
            plt.close()

    def _plot_fallback_metrics(self, save_dir: str):
        if not self.fallback_metrics:
            return
        timestamps = [(m.timestamp - self.fallback_metrics[0].timestamp) for m in self.fallback_metrics]
        fallback_levels = [m.fallback_level for m in self.fallback_metrics]

        plt.figure(figsize=(10, 6))
        plt.scatter(timestamps, fallback_levels)
        plt.xlabel(TIME_SECONDS_LABEL)
        plt.ylabel("Fallback Level")
        plt.title("Fallback Activations")
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "fallback_levels.png"))
        plt.close()

        fallback_reasons = {}
        for metrics in self.fallback_metrics:
            if metrics.fallback_reason not in fallback_reasons:
                fallback_reasons[metrics.fallback_reason] = 0
            fallback_reasons[metrics.fallback_reason] += 1

        if fallback_reasons:
            plt.figure(figsize=(10, 6))
            plt.bar(fallback_reasons.keys(), fallback_reasons.values())
            plt.xlabel("Fallback Reason")
            plt.ylabel("Count")
            plt.title("Fallback Reasons")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "fallback_reasons.png"))
            plt.close()

    def _plot_health_metrics(self, save_dir: str):
        if not self.health_metrics:
            return
        timestamps = [(m.timestamp - self.health_metrics[0].timestamp) for m in self.health_metrics]
        overall_health = [m.overall_health for m in self.health_metrics]
        cpu_health = [m.cpu_health for m in self.health_metrics]
        memory_health = [m.memory_health for m in self.health_metrics]

        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, overall_health, label="Overall Health")
        plt.plot(timestamps, cpu_health, label="CPU Health")
        plt.plot(timestamps, memory_health, label="Memory Health")

        gpu_health = [m.gpu_health for m in self.health_metrics if m.gpu_health is not None]
        if gpu_health:
            gpu_timestamps = [(m.timestamp - self.health_metrics[0].timestamp) for m in self.health_metrics if m.gpu_health is not None]
            plt.plot(gpu_timestamps, gpu_health, label="GPU Health")

        latency_health = [m.latency_health for m in self.health_metrics if m.latency_health is not None]
        if latency_health:
            latency_timestamps = [(m.timestamp - self.health_metrics[0].timestamp) for m in self.health_metrics if m.latency_health is not None]
            plt.plot(latency_timestamps, latency_health, label="Latency Health")

        error_health = [m.error_health for m in self.health_metrics if m.error_health is not None]
        if error_health:
            error_timestamps = [(m.timestamp - self.health_metrics[0].timestamp) for m in self.health_metrics if m.error_health is not None]
            plt.plot(error_timestamps, error_health, label="Error Health")

        plt.axhline(y=0.7, color='orange', linestyle='--', label="Warning Threshold")
        plt.axhline(y=0.5, color='red', linestyle='--', label="Critical Threshold")
        plt.xlabel(TIME_SECONDS_LABEL)
        plt.ylabel("Health Score (0-1)")
        plt.title("System Health")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "system_health.png"))
        plt.close()
    
    def close(self):
        """Close the monitor and clean up resources."""
        # Stop monitoring thread
        if self.monitoring_thread is not None and self.monitoring_thread.is_alive():
            self.stop_monitoring.set()
            self.monitoring_thread.join(timeout=5.0)
        
        # Save metrics
        self.save_metrics()
        
        # Plot metrics
        self.plot_metrics()
        
        self.logger.info("System health monitor closed")
