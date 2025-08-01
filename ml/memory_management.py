"""
Memory management infrastructure for the RL model optimization.

This module provides comprehensive memory monitoring, bounded storage,
and device-aware tensor caching to prevent memory leaks and optimize
performance in production trading environments.
"""

import gc
import psutil
import torch
from collections import deque, OrderedDict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional, Any, Tuple
import numpy as np


class CleanupLevel(Enum):
    """Levels of cleanup intensity based on memory pressure."""
    LIGHT = "light"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    CRITICAL = "critical"


@dataclass
class MemoryStatus:
    """Memory usage status with alert levels."""
    cpu_usage_gb: float
    cpu_percent: float
    gpu_usage_gb: float
    gpu_percent: float
    warning_level: bool
    critical_level: bool
    timestamp: datetime
    
    def __str__(self) -> str:
        return (f"Memory Status - CPU: {self.cpu_usage_gb:.2f}GB ({self.cpu_percent:.1%}), "
                f"GPU: {self.gpu_usage_gb:.2f}GB ({self.gpu_percent:.1%}), "
                f"Warning: {self.warning_level}, Critical: {self.critical_level}")


class MemoryMonitor:
    """
    Monitors system memory usage and triggers cleanup actions when thresholds are exceeded.
    
    Integrates psutil for CPU memory tracking and torch.cuda.memory_allocated() for GPU memory.
    Provides configurable thresholds and cleanup callback registration.
    """
    
    def __init__(self, 
                 alert_threshold_gb: float = 8.0,
                 warning_threshold: float = 0.7,
                 critical_threshold: float = 0.9):
        """
        Initialize memory monitor with configurable thresholds.
        
        Args:
            alert_threshold_gb: Absolute memory threshold in GB for alerts
            warning_threshold: Relative threshold (0-1) for warning level
            critical_threshold: Relative threshold (0-1) for critical level
        """
        self.alert_threshold_gb = alert_threshold_gb
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.cleanup_callbacks: Dict[CleanupLevel, List[Callable]] = {
            level: [] for level in CleanupLevel
        }
        
        # Get system memory info
        self.system_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Check if CUDA is available for GPU monitoring
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            self.gpu_memory_gb = 0.0
    
    def monitor_memory(self) -> MemoryStatus:
        """
        Monitor current memory usage and return status.
        
        Returns:
            MemoryStatus with current usage metrics and alert levels
        """
        # CPU memory monitoring using psutil
        cpu_memory = psutil.virtual_memory()
        cpu_usage_gb = cpu_memory.used / (1024**3)
        cpu_percent = cpu_memory.percent / 100.0
        
        # GPU memory monitoring using torch.cuda
        gpu_usage_gb = 0.0
        gpu_percent = 0.0
        if self.cuda_available:
            gpu_usage_gb = torch.cuda.memory_allocated() / (1024**3)
            if self.gpu_memory_gb > 0:
                gpu_percent = gpu_usage_gb / self.gpu_memory_gb
        
        # Determine alert levels
        warning_level = (cpu_usage_gb >= self.alert_threshold_gb * self.warning_threshold or
                        cpu_percent >= self.warning_threshold or
                        gpu_percent >= self.warning_threshold)
        
        critical_level = (cpu_usage_gb >= self.alert_threshold_gb * self.critical_threshold or
                         cpu_percent >= self.critical_threshold or
                         gpu_percent >= self.critical_threshold)
        
        status = MemoryStatus(
            cpu_usage_gb=cpu_usage_gb,
            cpu_percent=cpu_percent,
            gpu_usage_gb=gpu_usage_gb,
            gpu_percent=gpu_percent,
            warning_level=warning_level,
            critical_level=critical_level,
            timestamp=datetime.now()
        )
        
        # Trigger cleanup if thresholds are exceeded
        if critical_level:
            self.trigger_cleanup(CleanupLevel.CRITICAL)
        elif warning_level:
            self.trigger_cleanup(CleanupLevel.MODERATE)
        
        return status
    
    def register_cleanup_callback(self, callback: Callable, level: CleanupLevel = CleanupLevel.MODERATE) -> None:
        """
        Register a cleanup callback for a specific cleanup level.
        
        Args:
            callback: Function to call during cleanup
            level: Cleanup level when this callback should be triggered
        """
        self.cleanup_callbacks[level].append(callback)
    
    def trigger_cleanup(self, level: CleanupLevel) -> None:
        """
        Trigger cleanup callbacks for the specified level and all lower levels.
        
        Args:
            level: Cleanup level to trigger
        """
        # Define cleanup level hierarchy
        level_hierarchy = [CleanupLevel.LIGHT, CleanupLevel.MODERATE, 
                          CleanupLevel.AGGRESSIVE, CleanupLevel.CRITICAL]
        
        # Trigger callbacks for current level and all lower levels
        current_index = level_hierarchy.index(level)
        for i in range(current_index + 1):
            cleanup_level = level_hierarchy[i]
            for callback in self.cleanup_callbacks[cleanup_level]:
                try:
                    callback()
                except Exception as e:
                    print(f"Error in cleanup callback for level {cleanup_level}: {e}")
        
        # Perform system-level cleanup
        if level in [CleanupLevel.AGGRESSIVE, CleanupLevel.CRITICAL]:
            gc.collect()  # Force garbage collection
            if self.cuda_available:
                torch.cuda.empty_cache()  # Clear GPU cache
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get detailed memory statistics for monitoring and debugging.
        
        Returns:
            Dictionary with detailed memory statistics
        """
        status = self.monitor_memory()
        
        stats = {
            'cpu_memory': {
                'used_gb': status.cpu_usage_gb,
                'percent': status.cpu_percent,
                'total_gb': self.system_memory_gb,
                'available_gb': self.system_memory_gb - status.cpu_usage_gb
            },
            'gpu_memory': {
                'used_gb': status.gpu_usage_gb,
                'percent': status.gpu_percent,
                'total_gb': self.gpu_memory_gb,
                'available_gb': self.gpu_memory_gb - status.gpu_usage_gb
            },
            'thresholds': {
                'alert_threshold_gb': self.alert_threshold_gb,
                'warning_threshold': self.warning_threshold,
                'critical_threshold': self.critical_threshold
            },
            'alert_status': {
                'warning': status.warning_level,
                'critical': status.critical_level
            },
            'timestamp': status.timestamp
        }
        
        return stats


class MetricsOverflowHandler:
    """Handles storage overflow scenarios for bounded metrics storage."""
    
    def __init__(self, strategy: str = "drop_oldest"):
        """
        Initialize overflow handler with specified strategy.
        
        Args:
            strategy: Strategy for handling overflow ("drop_oldest", "compress", "flush")
        """
        self.strategy = strategy
        self.overflow_count = 0
    
    def handle_overflow(self, metrics_storage: 'BoundedMetricsStorage') -> None:
        """
        Handle overflow based on configured strategy.
        
        Args:
            metrics_storage: The storage instance experiencing overflow
        """
        self.overflow_count += 1
        
        if self.strategy == "drop_oldest":
            # Default behavior - deque automatically drops oldest
            pass
        elif self.strategy == "compress":
            # Compress older metrics by keeping every nth metric
            self._compress_metrics(metrics_storage)
        elif self.strategy == "flush":
            # Flush half of the metrics to persistent storage
            self._flush_metrics(metrics_storage)
    
    def _compress_metrics(self, metrics_storage: 'BoundedMetricsStorage') -> None:
        """Compress metrics by keeping every 2nd metric from older half."""
        metrics_list = list(metrics_storage.metrics)
        half_point = len(metrics_list) // 2
        
        # Keep all recent metrics, compress older half
        compressed_older = metrics_list[:half_point:2]  # Every 2nd metric
        recent_metrics = metrics_list[half_point:]
        
        # Replace storage with compressed version
        metrics_storage.metrics.clear()
        for metric in compressed_older + recent_metrics:
            metrics_storage.metrics.append(metric)
    
    def _flush_metrics(self, metrics_storage: 'BoundedMetricsStorage') -> None:
        """Flush older half of metrics to persistent storage."""
        metrics_list = list(metrics_storage.metrics)
        half_point = len(metrics_list) // 2
        
        # In production, this would write to database or file
        # For now, just remove older half
        for _ in range(half_point):
            if metrics_storage.metrics:
                metrics_storage.metrics.popleft()


class BoundedMetricsStorage:
    """
    Bounded storage for training metrics using deque with configurable maximum length.
    
    Replaces unbounded List[TrainingMetrics] to prevent memory leaks during long training sessions.
    """
    
    def __init__(self, maxlen: int = 10000):
        """
        Initialize bounded metrics storage.
        
        Args:
            maxlen: Maximum number of metrics to store
        """
        self.maxlen = maxlen
        self.metrics = deque(maxlen=maxlen)
        self.overflow_handler = MetricsOverflowHandler()
        self.total_metrics_added = 0
        self.overflow_events = 0
    
    def add_metric(self, metric: 'TrainingMetrics') -> None:
        """
        Add a training metric to the bounded storage.
        
        Args:
            metric: TrainingMetrics instance to store
        """
        # Check if adding this metric will cause overflow
        if len(self.metrics) == self.maxlen:
            self.overflow_events += 1
            self.overflow_handler.handle_overflow(self)
        
        self.metrics.append(metric)
        self.total_metrics_added += 1
    
    def get_recent_metrics(self, n: int) -> List['TrainingMetrics']:
        """
        Get the n most recent metrics.
        
        Args:
            n: Number of recent metrics to retrieve
            
        Returns:
            List of recent TrainingMetrics instances
        """
        if n <= 0:
            return []
        
        # Convert deque to list and get last n items
        metrics_list = list(self.metrics)
        return metrics_list[-n:] if len(metrics_list) >= n else metrics_list
    
    def get_all_metrics(self) -> List['TrainingMetrics']:
        """
        Get all stored metrics.
        
        Returns:
            List of all stored TrainingMetrics instances
        """
        return list(self.metrics)
    
    def flush_to_storage(self, persistent_storage: Optional[Callable] = None) -> int:
        """
        Flush metrics to persistent storage and clear the buffer.
        
        Args:
            persistent_storage: Optional callable to handle persistent storage
            
        Returns:
            Number of metrics flushed
        """
        metrics_count = len(self.metrics)
        
        if persistent_storage:
            try:
                persistent_storage(list(self.metrics))
            except Exception as e:
                print(f"Error flushing metrics to persistent storage: {e}")
                return 0
        
        self.metrics.clear()
        return metrics_count
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics for monitoring.
        
        Returns:
            Dictionary with storage statistics
        """
        return {
            'current_size': len(self.metrics),
            'max_size': self.maxlen,
            'utilization': len(self.metrics) / self.maxlen if self.maxlen > 0 else 0,
            'total_metrics_added': self.total_metrics_added,
            'overflow_events': self.overflow_events,
            'overflow_rate': self.overflow_events / max(self.total_metrics_added, 1)
        }
    
    def clear(self) -> None:
        """Clear all stored metrics."""
        self.metrics.clear()
    
    def __len__(self) -> int:
        """Return the number of stored metrics."""
        return len(self.metrics)
    
    def __iter__(self):
        """Make the storage iterable."""
        return iter(self.metrics)

@dataclass
class CacheStats:
    """Statistics for tensor cache performance monitoring."""
    hits: int
    misses: int
    evictions: int
    current_size: int
    max_size: int
    memory_usage_mb: float
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.hits + self.misses
        return self.hits / total_requests if total_requests > 0 else 0.0
    
    def __str__(self) -> str:
        return (f"Cache Stats - Hit Rate: {self.hit_rate:.2%}, "
                f"Size: {self.current_size}/{self.max_size}, "
                f"Memory: {self.memory_usage_mb:.1f}MB")


class DeviceAwareTensorCache:
    """
    LRU cache for efficient tensor device management.
    
    Provides caching for frequently used tensors to minimize repeated CPU-GPU transfers
    and torch.from_numpy() conversions. Uses OrderedDict for LRU eviction policy.
    """
    
    def __init__(self, device: torch.device, cache_size: int = 1000):
        """
        Initialize device-aware tensor cache.
        
        Args:
            device: Target device for cached tensors
            cache_size: Maximum number of tensors to cache
        """
        self.device = device
        self.cache_size = cache_size
        self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        
        # Statistics tracking
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Memory tracking
        self.memory_usage_bytes = 0
    
    def to_tensor(self, 
                  array: np.ndarray, 
                  key: Optional[str] = None,
                  dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Convert numpy array to tensor with optional caching.
        
        Args:
            array: Numpy array to convert
            key: Optional cache key. If None, no caching is performed
            dtype: Target tensor dtype
            
        Returns:
            Tensor on the target device
        """
        # If no key provided, perform direct conversion without caching
        if key is None:
            return torch.from_numpy(array).to(self.device, dtype=dtype)
        
        # Check if tensor is already cached
        if key in self.cache:
            self.hits += 1
            # Move to end (most recently used)
            tensor = self.cache.pop(key)
            self.cache[key] = tensor
            return tensor
        
        # Cache miss - create new tensor
        self.misses += 1
        tensor = torch.from_numpy(array).to(self.device, dtype=dtype)
        
        # Add to cache with LRU eviction if necessary
        self._add_to_cache(key, tensor)
        
        return tensor
    
    def _add_to_cache(self, key: str, tensor: torch.Tensor) -> None:
        """
        Add tensor to cache with LRU eviction.
        
        Args:
            key: Cache key
            tensor: Tensor to cache
        """
        # Calculate tensor memory usage
        tensor_bytes = tensor.numel() * tensor.element_size()
        
        # Evict oldest entries if cache is full
        while len(self.cache) >= self.cache_size:
            self._evict_oldest()
        
        # Add new tensor to cache
        self.cache[key] = tensor
        self.memory_usage_bytes += tensor_bytes
    
    def _evict_oldest(self) -> None:
        """Evict the oldest (least recently used) tensor from cache."""
        if not self.cache:
            return
        
        # Remove oldest item (first in OrderedDict)
        _oldest_key, oldest_tensor = self.cache.popitem(last=False)  # oldest_key unused
        
        # Update statistics
        self.evictions += 1
        tensor_bytes = oldest_tensor.numel() * oldest_tensor.element_size()
        self.memory_usage_bytes -= tensor_bytes
    
    def clear_cache(self) -> None:
        """Clear all cached tensors."""
        self.cache.clear()
        self.memory_usage_bytes = 0
    
    def get_cache_stats(self) -> CacheStats:
        """
        Get cache performance statistics.
        
        Returns:
            CacheStats with performance metrics
        """
        return CacheStats(
            hits=self.hits,
            misses=self.misses,
            evictions=self.evictions,
            current_size=len(self.cache),
            max_size=self.cache_size,
            memory_usage_mb=self.memory_usage_bytes / (1024 * 1024)
        )
    
    def prefill_cache(self, arrays_with_keys: List[Tuple[str, np.ndarray]], 
                      dtype: torch.dtype = torch.float32) -> None:
        """
        Prefill cache with commonly used tensors.
        
        Args:
            arrays_with_keys: List of (key, array) tuples to cache
            dtype: Target tensor dtype
        """
        for key, array in arrays_with_keys:
            if key not in self.cache:
                tensor = torch.from_numpy(array).to(self.device, dtype=dtype)
                self._add_to_cache(key, tensor)
    
    def remove_from_cache(self, key: str) -> bool:
        """
        Remove specific tensor from cache.
        
        Args:
            key: Cache key to remove
            
        Returns:
            True if key was found and removed, False otherwise
        """
        if key in self.cache:
            tensor = self.cache.pop(key)
            tensor_bytes = tensor.numel() * tensor.element_size()
            self.memory_usage_bytes -= tensor_bytes
            return True
        return False
    
    def get_cached_keys(self) -> List[str]:
        """Get list of all cached keys."""
        return list(self.cache.keys())
    
    def __len__(self) -> int:
        """Return number of cached tensors."""
        return len(self.cache)