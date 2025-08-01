"""
PerformanceProfiler for identifying bottlenecks in RL model execution.

This module provides tools for profiling PyTorch models to identify performance
bottlenecks in both CPU and GPU operations, memory usage, and data transfers.
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
import psutil
import gc

from src.utils.constants import (
    BYTES_TO_MB, MILLISECONDS_TO_SECONDS, DEFAULT_METRICS_BUFFER_SIZE, 
    DEFAULT_REPEAT_COUNT
)


@dataclass
class ProfileResult:
    """Container for profiling results."""
    name: str
    total_time_ms: float
    cpu_time_ms: float
    cuda_time_ms: float
    cpu_memory_mb: float
    cuda_memory_mb: float
    self_cpu_time_ms: float
    self_cuda_time_ms: float
    input_shapes: List[List[int]]
    output_shapes: List[List[int]]
    flops: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    total_time_ms: float
    forward_time_ms: float
    backward_time_ms: float
    optimizer_time_ms: float
    cpu_memory_mb: float
    cuda_memory_mb: float
    cpu_utilization: float
    cuda_utilization: float
    samples_per_second: float
    flops: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class InferenceMetrics:
    """Container for inference performance metrics."""
    total_time_ms: float
    preprocess_time_ms: float
    inference_time_ms: float
    postprocess_time_ms: float
    cpu_memory_mb: float
    cuda_memory_mb: float
    batch_size: int
    samples_per_second: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class BottleneckInfo:
    """Information about a performance bottleneck."""
    name: str
    type: str  # 'cpu', 'cuda', 'memory', 'data_transfer'
    severity: float  # 0.0 to 1.0
    time_ms: float
    percentage: float
    recommendation: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class PerformanceProfiler:
    """
    Performance profiler for PyTorch models.
    
    This class provides tools for:
    - Profiling training and inference performance
    - Identifying bottlenecks in CPU/GPU operations
    - Tracking memory usage
    - Detecting performance regressions
    """
    
    def __init__(
        self,
        model: nn.Module,
        log_dir: str = "logs",
        model_name: str = "rl_model",
        metrics_buffer_size: int = DEFAULT_METRICS_BUFFER_SIZE,
        use_cuda: bool = torch.cuda.is_available(),
        log_frequency: int = 10
    ):
        """
        Initialize the performance profiler.
        
        Args:
            model: PyTorch model to profile
            log_dir: Directory for logs and visualizations
            model_name: Name of the model being profiled
            metrics_buffer_size: Maximum number of metrics to store in memory
            use_cuda: Whether to profile CUDA operations
            log_frequency: How often to log metrics (in steps)
        """
        self.model = model
        self.log_dir = log_dir
        self.model_name = model_name
        self.metrics_buffer_size = metrics_buffer_size
        self.use_cuda = use_cuda
        self.log_frequency = log_frequency
        
        # Create log directory
        self.run_dir = os.path.join(
            log_dir, 
            f"{model_name}_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.training_metrics = deque(maxlen=metrics_buffer_size)
        self.inference_metrics = deque(maxlen=metrics_buffer_size)
        self.profile_results = {}
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize step counter
        self.step_counter = 0
        
        # Initialize baseline metrics
        self.baseline_training_metrics = None
        self.baseline_inference_metrics = None
        
        # Check if PyTorch profiler is available
        self.profiler_available = hasattr(torch, 'profiler')
        if not self.profiler_available:
            self.logger.warning("PyTorch profiler not available. Some features will be disabled.")
        
        self.logger.info(f"Performance profiler initialized. Logs will be saved to {self.run_dir}")
    
    def profile_training_step(
        self,
        input_tensor: torch.Tensor,
        target: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        batch_size: Optional[int] = None,
        warmup: int = 3,
        repeat: int = 5,
        detailed: bool = False
    ) -> PerformanceMetrics:
        """
        Profile a complete training step.
        
        Args:
            input_tensor: Input tensor
            target: Target tensor
            optimizer: Optimizer
            loss_fn: Loss function
            batch_size: Batch size (if different from input_tensor.shape[0])
            warmup: Number of warmup iterations
            repeat: Number of iterations to average over
            detailed: Whether to collect detailed profiling information
            
        Returns:
            PerformanceMetrics object with profiling results
        """
        self.step_counter += 1
        batch_size = batch_size or input_tensor.shape[0]
        self.model.train()
        
        # Perform warmup iterations
        self._perform_warmup(input_tensor, target, optimizer, loss_fn, warmup)
        
        # Get baseline measurements
        memory_before, cpu_before = self._get_baseline_measurements()
        
        # Profile the training step timing
        timing_results = self._profile_training_timing(
            input_tensor, target, optimizer, loss_fn, repeat
        )
        
        # Get final measurements
        memory_after, cpu_after = self._get_final_measurements()
        
        # Calculate metrics
        metrics = self._calculate_training_metrics(
            timing_results, memory_before, memory_after, 
            cpu_before, cpu_after, batch_size
        )
        
        # Store and log results
        self._finalize_training_profile(metrics, detailed, input_tensor, 
                                      target, optimizer, loss_fn)
        
        return metrics

    def _perform_warmup(self, input_tensor: torch.Tensor, target: torch.Tensor,
                       optimizer: torch.optim.Optimizer, loss_fn: Callable, 
                       warmup: int) -> None:
        """Perform warmup iterations before profiling."""
        for _ in range(warmup):
            optimizer.zero_grad()
            output = self.model(input_tensor)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

    def _get_baseline_measurements(self) -> Tuple[Tuple[float, float], float]:
        """Get baseline memory and CPU measurements."""
        # Synchronize CUDA if available
        if self.use_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        cpu_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        cuda_memory = (torch.cuda.memory_allocated() / (1024 * 1024) 
                      if self.use_cuda and torch.cuda.is_available() else 0)
        cpu_percent = psutil.cpu_percent()
        
        return (cpu_memory, cuda_memory), cpu_percent

    def _get_final_measurements(self) -> Tuple[Tuple[float, float], float]:
        """Get final memory and CPU measurements."""
        cpu_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        cuda_memory = (torch.cuda.memory_allocated() / (1024 * 1024) 
                      if self.use_cuda and torch.cuda.is_available() else 0)
        cpu_percent = psutil.cpu_percent()
        
        return (cpu_memory, cuda_memory), cpu_percent

    def _profile_training_timing(self, input_tensor: torch.Tensor, target: torch.Tensor,
                               optimizer: torch.optim.Optimizer, loss_fn: Callable,
                               repeat: int) -> Dict[str, float]:
        """Profile timing for each component of training step."""
        forward_time = 0
        backward_time = 0
        optimizer_time = 0
        
        for _ in range(repeat):
            optimizer.zero_grad()
            
            # Forward pass timing
            start_time = time.time()
            output = self.model(input_tensor)
            if self.use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_time += (time.time() - start_time) * 1000
            
            # Loss calculation
            loss = loss_fn(output, target)
            
            # Backward pass timing
            start_time = time.time()
            loss.backward()
            if self.use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            backward_time += (time.time() - start_time) * 1000
            
            # Optimizer step timing
            start_time = time.time()
            optimizer.step()
            if self.use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            optimizer_time += (time.time() - start_time) * 1000
        
        # Return average times
        return {
            'forward': forward_time / repeat,
            'backward': backward_time / repeat,
            'optimizer': optimizer_time / repeat
        }

    def _calculate_training_metrics(self, timing_results: Dict[str, float],
                                  memory_before: Tuple[float, float],
                                  memory_after: Tuple[float, float],
                                  cpu_before: float, cpu_after: float,
                                  batch_size: int) -> PerformanceMetrics:
        """Calculate performance metrics from profiling results."""
        # Calculate timing metrics
        forward_time = timing_results['forward']
        backward_time = timing_results['backward']
        optimizer_time = timing_results['optimizer']
        total_time = forward_time + backward_time + optimizer_time
        
        # Calculate memory usage
        cpu_memory = memory_after[0] - memory_before[0]
        cuda_memory = memory_after[1] - memory_before[1]
        
        # Calculate CPU utilization
        cpu_utilization = (cpu_before + cpu_after) / 2
        
        # Calculate CUDA utilization
        cuda_utilization = self._get_cuda_utilization()
        
        # Calculate throughput
        samples_per_second = batch_size * 1000 / total_time
        
        return PerformanceMetrics(
            total_time_ms=total_time,
            forward_time_ms=forward_time,
            backward_time_ms=backward_time,
            optimizer_time_ms=optimizer_time,
            cpu_memory_mb=cpu_memory,
            cuda_memory_mb=cuda_memory,
            cpu_utilization=cpu_utilization,
            cuda_utilization=cuda_utilization,
            samples_per_second=samples_per_second
        )

    def _get_cuda_utilization(self) -> float:
        """Get CUDA utilization if available."""
        if (self.use_cuda and torch.cuda.is_available() and 
            hasattr(torch.cuda, 'utilization')):
            return torch.cuda.utilization()
        return 0

    def _finalize_training_profile(self, metrics: PerformanceMetrics, detailed: bool,
                                 input_tensor: torch.Tensor, target: torch.Tensor,
                                 optimizer: torch.optim.Optimizer, loss_fn: Callable) -> None:
        """Finalize profiling by storing metrics and running detailed profiling if needed."""
        # Store metrics
        self.training_metrics.append(metrics)
        
        # Detailed profiling if requested
        if detailed and self.profiler_available:
            self._run_detailed_profiling(input_tensor, target, optimizer, loss_fn)
        
        # Log metrics periodically
        if self.step_counter % self.log_frequency == 0:
            self._log_training_metrics(metrics)
    
    def profile_inference(
        self,
        input_tensor: torch.Tensor,
        batch_size: Optional[int] = None,
        warmup: int = 3,
        repeat: int = DEFAULT_REPEAT_COUNT,
        detailed: bool = False,
        include_preprocessing: bool = False,
        include_postprocessing: bool = False,
        preprocess_fn: Optional[Callable] = None,
        postprocess_fn: Optional[Callable] = None
    ) -> InferenceMetrics:
        """
        Profile inference performance.
        
        Args:
            input_tensor: Input tensor
            batch_size: Batch size (if different from input_tensor.shape[0])
            warmup: Number of warmup iterations
            repeat: Number of iterations to average over
            detailed: Whether to collect detailed profiling information
            include_preprocessing: Whether to include preprocessing in profiling
            include_postprocessing: Whether to include postprocessing in profiling
            preprocess_fn: Preprocessing function (if include_preprocessing is True)
            postprocess_fn: Postprocessing function (if include_postprocessing is True)
            
        Returns:
            InferenceMetrics object with profiling results
        """
        batch_size = batch_size or input_tensor.shape[0]
        self.model.eval()
        
        # Perform warmup
        self._perform_inference_warmup(input_tensor, warmup, include_preprocessing,
                                     include_postprocessing, preprocess_fn, postprocess_fn)
        
        # Get baseline measurements
        memory_before = self._get_inference_baseline_measurements()
        
        # Profile inference timing
        timing_results = self._profile_inference_timing(
            input_tensor, repeat, include_preprocessing, include_postprocessing,
            preprocess_fn, postprocess_fn
        )
        
        # Get final measurements
        memory_after = self._get_inference_final_measurements()
        
        # Calculate and return metrics
        metrics = self._calculate_inference_metrics(
            timing_results, memory_before, memory_after, batch_size
        )
        
        # Store and log results
        self._finalize_inference_profile(metrics, detailed, input_tensor)
        
        return metrics

    def _perform_inference_warmup(self, input_tensor: torch.Tensor, warmup: int,
                                include_preprocessing: bool, include_postprocessing: bool,
                                preprocess_fn: Optional[Callable], 
                                postprocess_fn: Optional[Callable]) -> None:
        """Perform warmup iterations for inference profiling."""
        with torch.no_grad():
            for _ in range(warmup):
                input_processed = self._apply_preprocessing(
                    input_tensor, include_preprocessing, preprocess_fn
                )
                output = self.model(input_processed)
                
                if include_postprocessing and postprocess_fn is not None:
                    _ = postprocess_fn(output)

    def _apply_preprocessing(self, input_tensor: torch.Tensor, 
                           include_preprocessing: bool,
                           preprocess_fn: Optional[Callable]) -> torch.Tensor:
        """Apply preprocessing if specified."""
        if include_preprocessing and preprocess_fn is not None:
            return preprocess_fn(input_tensor)
        return input_tensor

    def _get_inference_baseline_measurements(self) -> Tuple[float, float]:
        """Get baseline memory measurements for inference."""
        # Synchronize CUDA if available
        if self.use_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        cpu_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        cuda_memory = (torch.cuda.memory_allocated() / (1024 * 1024) 
                      if self.use_cuda and torch.cuda.is_available() else 0)
        
        return cpu_memory, cuda_memory

    def _get_inference_final_measurements(self) -> Tuple[float, float]:
        """Get final memory measurements for inference."""
        cpu_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        cuda_memory = (torch.cuda.memory_allocated() / (1024 * 1024) 
                      if self.use_cuda and torch.cuda.is_available() else 0)
        
        return cpu_memory, cuda_memory

    def _profile_inference_timing(self, input_tensor: torch.Tensor, repeat: int,
                                include_preprocessing: bool, include_postprocessing: bool,
                                preprocess_fn: Optional[Callable],
                                postprocess_fn: Optional[Callable]) -> Dict[str, float]:
        """Profile timing for each component of inference."""
        preprocess_time = 0
        inference_time = 0
        postprocess_time = 0

        with torch.no_grad():
            for _ in range(repeat):
                input_processed, preprocess_elapsed = self._timed_preprocessing(
                    input_tensor, include_preprocessing, preprocess_fn
                )
                preprocess_time += preprocess_elapsed

                output, inference_elapsed = self._timed_inference(input_processed)
                inference_time += inference_elapsed

                postprocess_elapsed = self._timed_postprocessing(
                    output, include_postprocessing, postprocess_fn
                )
                postprocess_time += postprocess_elapsed

        # Return average times
        return {
            'preprocess': preprocess_time / repeat,
            'inference': inference_time / repeat,
            'postprocess': postprocess_time / repeat
        }

    def _timed_preprocessing(self, input_tensor: torch.Tensor, include_preprocessing: bool,
                             preprocess_fn: Optional[Callable]) -> Tuple[torch.Tensor, float]:
        """Time the preprocessing step."""
        start_time = time.time()
        input_processed = self._apply_preprocessing(input_tensor, include_preprocessing, preprocess_fn)
        elapsed = 0
        if include_preprocessing and preprocess_fn is not None:
            if self.use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = (time.time() - start_time) * 1000
        return input_processed, elapsed

    def _timed_inference(self, input_processed: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Time the inference step."""
        start_time = time.time()
        output = self.model(input_processed)
        if self.use_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.time() - start_time) * 1000
        return output, elapsed

    def _timed_postprocessing(self, output: torch.Tensor, include_postprocessing: bool,
                              postprocess_fn: Optional[Callable]) -> float:
        """Time the postprocessing step."""
        elapsed = 0
        if include_postprocessing and postprocess_fn is not None:
            start_time = time.time()
            _ = postprocess_fn(output)
            if self.use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = (time.time() - start_time) * 1000
        return elapsed

    def _calculate_inference_metrics(self, timing_results: Dict[str, float],
                                   memory_before: Tuple[float, float],
                                   memory_after: Tuple[float, float],
                                   batch_size: int) -> InferenceMetrics:
        """Calculate inference performance metrics."""
        preprocess_time = timing_results['preprocess']
        inference_time = timing_results['inference']
        postprocess_time = timing_results['postprocess']
        total_time = preprocess_time + inference_time + postprocess_time
        
        # Calculate memory usage
        cpu_memory = memory_after[0] - memory_before[0]
        cuda_memory = memory_after[1] - memory_before[1]
        
        # Calculate throughput
        samples_per_second = batch_size * 1000 / total_time
        
        return InferenceMetrics(
            total_time_ms=total_time,
            preprocess_time_ms=preprocess_time,
            inference_time_ms=inference_time,
            postprocess_time_ms=postprocess_time,
            cpu_memory_mb=cpu_memory,
            cuda_memory_mb=cuda_memory,
            batch_size=batch_size,
            samples_per_second=samples_per_second
        )

    def _finalize_inference_profile(self, metrics: InferenceMetrics, detailed: bool,
                                  input_tensor: torch.Tensor) -> None:
        """Finalize inference profiling."""
        # Store metrics
        self.inference_metrics.append(metrics)
        
        # Detailed profiling if requested
        if detailed and self.profiler_available:
            self._run_detailed_inference_profiling(input_tensor)
        
        # Log metrics
        self._log_inference_metrics(metrics)
    
    def _run_detailed_profiling(
        self,
        input_tensor: torch.Tensor,
        target: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable
    ):
        """
        Run detailed profiling using PyTorch profiler.
        
        Args:
            input_tensor: Input tensor
            target: Target tensor
            optimizer: Optimizer
            loss_fn: Loss function
        """
        if not self.profiler_available:
            self.logger.warning("PyTorch profiler not available. Skipping detailed profiling.")
            return
        
        # Create profiler
        activities = []
        if hasattr(torch.profiler.ProfilerActivity, 'CPU'):
            activities.append(torch.profiler.ProfilerActivity.CPU)
        if self.use_cuda and torch.cuda.is_available() and hasattr(torch.profiler.ProfilerActivity, 'CUDA'):
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            # Training step
            optimizer.zero_grad()
            output = self.model(input_tensor)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
        
        # Process profiling results
        self._process_profiling_results(prof, "training")
    
    def _run_detailed_inference_profiling(self, input_tensor: torch.Tensor):
        """
        Run detailed inference profiling using PyTorch profiler.
        
        Args:
            input_tensor: Input tensor
        """
        if not self.profiler_available:
            self.logger.warning("PyTorch profiler not available. Skipping detailed profiling.")
            return
        
        # Create profiler
        activities = []
        if hasattr(torch.profiler.ProfilerActivity, 'CPU'):
            activities.append(torch.profiler.ProfilerActivity.CPU)
        if self.use_cuda and torch.cuda.is_available() and hasattr(torch.profiler.ProfilerActivity, 'CUDA'):
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            # Inference
            with torch.no_grad():
                _ = self.model(input_tensor)
        
        # Process profiling results
        self._process_profiling_results(prof, "inference")
    
    def _process_profiling_results(self, prof, mode: str):
        """
        Process profiling results.
        
        Args:
            prof: PyTorch profiler object
            mode: Profiling mode ('training' or 'inference')
        """
        events = prof.key_averages()
        results = [self._event_to_profile_result(event) for event in events if not event.key.startswith('ProfilerStep')]
        self.profile_results[mode] = results
        self._save_profile_results(results, mode)
        bottlenecks = self._identify_bottlenecks(results)
        self._log_bottlenecks(bottlenecks)

    def _event_to_profile_result(self, event) -> ProfileResult:
        """Convert a profiler event to a ProfileResult."""
        return ProfileResult(
            name=event.key,
            total_time_ms=event.cpu_time_total / 1000,  # Convert to ms
            cpu_time_ms=event.cpu_time / 1000,  # Convert to ms
            cuda_time_ms=event.cuda_time / 1000 if hasattr(event, 'cuda_time') else 0,  # Convert to ms
            cpu_memory_mb=event.cpu_memory_usage / (1024 * 1024) if hasattr(event, 'cpu_memory_usage') else 0,
            cuda_memory_mb=event.cuda_memory_usage / (1024 * 1024) if hasattr(event, 'cuda_memory_usage') else 0,
            self_cpu_time_ms=event.self_cpu_time / 1000,  # Convert to ms
            self_cuda_time_ms=event.self_cuda_time / 1000 if hasattr(event, 'self_cuda_time') else 0,  # Convert to ms
            input_shapes=[list(shape) for shape in event.input_shapes] if hasattr(event, 'input_shapes') else [],
            output_shapes=[list(shape) for shape in event.output_shapes] if hasattr(event, 'output_shapes') else [],
            flops=event.flops if hasattr(event, 'flops') else None
        )
    
    def _identify_bottlenecks(self, results: List[ProfileResult]) -> List[BottleneckInfo]:
        """
        Identify performance bottlenecks.
        
        Args:
            results: List of profile results
            
        Returns:
            List of bottleneck information
        """
        bottlenecks = []
        total_time = sum(result.total_time_ms for result in results)
        
        # Identify different types of bottlenecks
        bottlenecks.extend(self._identify_cpu_cuda_bottlenecks(results, total_time))
        bottlenecks.extend(self._identify_memory_bottlenecks(results, total_time))
        bottlenecks.extend(self._identify_data_transfer_bottlenecks(results, total_time))
        
        return bottlenecks

    def _identify_cpu_cuda_bottlenecks(self, results: List[ProfileResult], 
                                     total_time: float) -> List[BottleneckInfo]:
        """Identify CPU and CUDA bottlenecks based on execution time."""
        bottlenecks = []
        sorted_results = sorted(results, key=lambda x: x.total_time_ms, reverse=True)
        
        # Top 5 time-consuming operations
        for result in sorted_results[:5]:
            percentage = self._calculate_time_percentage(result.total_time_ms, total_time)
            
            if percentage > 10:  # Operations taking more than 10% of total time
                bottleneck_type, recommendation = self._determine_cpu_cuda_bottleneck_type(result)
                
                bottlenecks.append(BottleneckInfo(
                    name=result.name,
                    type=bottleneck_type,
                    severity=min(1.0, percentage / 100),
                    time_ms=result.total_time_ms,
                    percentage=percentage,
                    recommendation=recommendation
                ))
        
        return bottlenecks

    def _determine_cpu_cuda_bottleneck_type(self, result: ProfileResult) -> Tuple[str, str]:
        """Determine if bottleneck is CPU or CUDA based and provide recommendation."""
        if result.cuda_time_ms > result.cpu_time_ms:
            return ("cuda", "Consider optimizing CUDA kernels or reducing complexity")
        else:
            return ("cpu", "Consider vectorizing operations or moving to GPU")

    def _identify_memory_bottlenecks(self, results: List[ProfileResult], 
                                   total_time: float) -> List[BottleneckInfo]:
        """Identify memory bottlenecks based on memory usage."""
        bottlenecks = []
        memory_sorted = sorted(results, 
                             key=lambda x: x.cpu_memory_mb + x.cuda_memory_mb, 
                             reverse=True)
        
        # Top 3 memory-consuming operations
        for result in memory_sorted[:3]:
            total_memory = result.cpu_memory_mb + result.cuda_memory_mb
            
            if total_memory > 100:  # More than 100MB
                bottlenecks.append(BottleneckInfo(
                    name=result.name,
                    type="memory",
                    severity=min(1.0, total_memory / 1000),
                    time_ms=result.total_time_ms,
                    percentage=self._calculate_time_percentage(result.total_time_ms, total_time),
                    recommendation="Consider reducing batch size or optimizing memory usage"
                ))
        
        return bottlenecks

    def _identify_data_transfer_bottlenecks(self, results: List[ProfileResult], 
                                          total_time: float) -> List[BottleneckInfo]:
        """Identify data transfer bottlenecks."""
        bottlenecks = []
        
        for result in results:
            if self._is_data_transfer_operation(result.name):
                bottlenecks.append(BottleneckInfo(
                    name=result.name,
                    type="data_transfer",
                    severity=self._calculate_data_transfer_severity(result.total_time_ms, total_time),
                    time_ms=result.total_time_ms,
                    percentage=self._calculate_time_percentage(result.total_time_ms, total_time),
                    recommendation="Consider using pinned memory or reducing data transfers"
                ))
        
        return bottlenecks

    def _calculate_time_percentage(self, time_ms: float, total_time: float) -> float:
        """Calculate percentage of total time."""
        return time_ms / total_time * 100 if total_time > 0 else 0

    def _is_data_transfer_operation(self, operation_name: str) -> bool:
        """Check if operation is a data transfer operation."""
        return "to(device" in operation_name or "copy_" in operation_name

    def _calculate_data_transfer_severity(self, time_ms: float, total_time: float) -> float:
        """Calculate severity of data transfer bottleneck."""
        return min(1.0, time_ms / total_time * 10) if total_time > 0 else 0
    
    def _log_training_metrics(self, metrics: PerformanceMetrics):
        """
        Log training metrics.
        
        Args:
            metrics: Performance metrics
        """
        self.logger.info(
            f"Training performance (step {self.step_counter}): "
            f"Total={metrics.total_time_ms:.2f} ms, "
            f"Forward={metrics.forward_time_ms:.2f} ms, "
            f"Backward={metrics.backward_time_ms:.2f} ms, "
            f"Optimizer={metrics.optimizer_time_ms:.2f} ms, "
            f"Samples/sec={metrics.samples_per_second:.2f}"
        )
    
    def _log_inference_metrics(self, metrics: InferenceMetrics):
        """
        Log inference metrics.
        
        Args:
            metrics: Inference metrics
        """
        self.logger.info(
            "Inference performance: "
            f"Total={metrics.total_time_ms:.2f} ms, "
            f"Inference={metrics.inference_time_ms:.2f} ms, "
            f"Samples/sec={metrics.samples_per_second:.2f}"
        )
    
    def _log_bottlenecks(self, bottlenecks: List[BottleneckInfo]):
        """
        Log bottleneck information.
        
        Args:
            bottlenecks: List of bottleneck information
        """
        if not bottlenecks:
            return
        
        self.logger.info(f"Identified {len(bottlenecks)} performance bottlenecks:")
        
        for i, bottleneck in enumerate(bottlenecks):
            self.logger.info(
                f"{i+1}. {bottleneck.name} ({bottleneck.type}): "
                f"{bottleneck.time_ms:.2f} ms ({bottleneck.percentage:.1f}%) - "
                f"Severity: {bottleneck.severity:.2f} - "
                f"Recommendation: {bottleneck.recommendation}"
            )
    
    def _save_profile_results(self, results: List[ProfileResult], mode: str):
        """
        Save profile results to file.
        
        Args:
            results: List of profile results
            mode: Profiling mode ('training' or 'inference')
        """
        # Convert results to dictionaries
        results_dict = [result.to_dict() for result in results]
        
        # Save to file
        filepath = os.path.join(self.run_dir, f"{mode}_profile_results.json")
        with open(filepath, "w") as f:
            json.dump(results_dict, f, indent=2)
    
    def set_baseline_metrics(
        self,
        training_metrics: Optional[PerformanceMetrics] = None,
        inference_metrics: Optional[InferenceMetrics] = None
    ):
        """
        Set baseline metrics for regression detection.
        
        Args:
            training_metrics: Baseline training metrics
            inference_metrics: Baseline inference metrics
        """
        if training_metrics is not None:
            self.baseline_training_metrics = training_metrics
        
        if inference_metrics is not None:
            self.baseline_inference_metrics = inference_metrics
    
    def detect_regressions(
        self,
        threshold: float = 0.2  # 20% regression
    ) -> Dict[str, Any]:
        """
        Detect performance regressions compared to baseline.

        Args:
            threshold: Regression threshold (e.g., 0.2 for 20% regression)

        Returns:
            Dictionary with regression information
        """
        regressions = {}
        regressions.update(self._detect_training_regressions(threshold))
        regressions.update(self._detect_inference_regressions(threshold))

        if regressions:
            self.logger.warning(f"Detected {len(regressions)} performance regressions:")
            for name, info in regressions.items():
                self.logger.warning(
                    f"{name}: {info['baseline']:.2f} -> {info['current']:.2f} "
                    f"({info['regression']*100:.1f}% regression)"
                )

        return regressions

    def _detect_training_regressions(self, threshold: float) -> Dict[str, Any]:
        """Detect training regressions compared to baseline."""
        regressions = {}
        if self.baseline_training_metrics is not None and self.training_metrics:
            current = self.training_metrics[-1]
            baseline = self.baseline_training_metrics

            checks = [
                ("training_total_time", current.total_time_ms, baseline.total_time_ms, lambda c, b: c > b * (1 + threshold)),
                ("training_samples_per_second", current.samples_per_second, baseline.samples_per_second, lambda c, b: c < b * (1 - threshold)),
                ("training_cpu_memory", current.cpu_memory_mb, baseline.cpu_memory_mb, lambda c, b: c > b * (1 + threshold)),
                ("training_cuda_memory", current.cuda_memory_mb, baseline.cuda_memory_mb, lambda c, b: c > b * (1 + threshold)),
            ]

            for name, curr, base, cond in checks:
                if cond(curr, base):
                    regression = (curr - base) / base if "samples_per_second" not in name else (base - curr) / base
                    regressions[name] = {
                        "baseline": base,
                        "current": curr,
                        "regression": regression
                    }
        return regressions

    def _detect_inference_regressions(self, threshold: float) -> Dict[str, Any]:
        """Detect inference regressions compared to baseline."""
        regressions = {}
        if self.baseline_inference_metrics is not None and self.inference_metrics:
            current = self.inference_metrics[-1]
            baseline = self.baseline_inference_metrics

            checks = [
                ("inference_total_time", current.total_time_ms, baseline.total_time_ms, lambda c, b: c > b * (1 + threshold)),
                ("inference_time", current.inference_time_ms, baseline.inference_time_ms, lambda c, b: c > b * (1 + threshold)),
                ("inference_samples_per_second", current.samples_per_second, baseline.samples_per_second, lambda c, b: c < b * (1 - threshold)),
                ("inference_cpu_memory", current.cpu_memory_mb, baseline.cpu_memory_mb, lambda c, b: c > b * (1 + threshold)),
                ("inference_cuda_memory", current.cuda_memory_mb, baseline.cuda_memory_mb, lambda c, b: c > b * (1 + threshold)),
            ]

            for name, curr, base, cond in checks:
                if cond(curr, base):
                    regression = (curr - base) / base if "samples_per_second" not in name else (base - curr) / base
                    regressions[name] = {
                        "baseline": base,
                        "current": curr,
                        "regression": regression
                    }
        return regressions
    
    def get_training_metrics_percentiles(self) -> Dict[str, Dict[str, float]]:
        """
        Get percentiles of training metrics.
        
        Returns:
            Dictionary with percentiles for each metric
        """
        if not self.training_metrics:
            return {}
        
        # Extract metrics
        total_times = [m.total_time_ms for m in self.training_metrics]
        forward_times = [m.forward_time_ms for m in self.training_metrics]
        backward_times = [m.backward_time_ms for m in self.training_metrics]
        optimizer_times = [m.optimizer_time_ms for m in self.training_metrics]
        samples_per_second = [m.samples_per_second for m in self.training_metrics]
        
        # Calculate percentiles
        percentiles = {}
        
        percentiles["total_time_ms"] = {
            "p50": np.percentile(total_times, 50),
            "p90": np.percentile(total_times, 90),
            "p95": np.percentile(total_times, 95),
            "p99": np.percentile(total_times, 99),
            "min": np.min(total_times),
            "max": np.max(total_times),
            "mean": np.mean(total_times),
            "std": np.std(total_times)
        }
        
        percentiles["forward_time_ms"] = {
            "p50": np.percentile(forward_times, 50),
            "p90": np.percentile(forward_times, 90),
            "p95": np.percentile(forward_times, 95),
            "p99": np.percentile(forward_times, 99),
            "min": np.min(forward_times),
            "max": np.max(forward_times),
            "mean": np.mean(forward_times),
            "std": np.std(forward_times)
        }
        
        percentiles["backward_time_ms"] = {
            "p50": np.percentile(backward_times, 50),
            "p90": np.percentile(backward_times, 90),
            "p95": np.percentile(backward_times, 95),
            "p99": np.percentile(backward_times, 99),
            "min": np.min(backward_times),
            "max": np.max(backward_times),
            "mean": np.mean(backward_times),
            "std": np.std(backward_times)
        }
        
        percentiles["optimizer_time_ms"] = {
            "p50": np.percentile(optimizer_times, 50),
            "p90": np.percentile(optimizer_times, 90),
            "p95": np.percentile(optimizer_times, 95),
            "p99": np.percentile(optimizer_times, 99),
            "min": np.min(optimizer_times),
            "max": np.max(optimizer_times),
            "mean": np.mean(optimizer_times),
            "std": np.std(optimizer_times)
        }
        
        percentiles["samples_per_second"] = {
            "p50": np.percentile(samples_per_second, 50),
            "p90": np.percentile(samples_per_second, 90),
            "p95": np.percentile(samples_per_second, 95),
            "p99": np.percentile(samples_per_second, 99),
            "min": np.min(samples_per_second),
            "max": np.max(samples_per_second),
            "mean": np.mean(samples_per_second),
            "std": np.std(samples_per_second)
        }
        
        return percentiles
    
    def get_inference_metrics_percentiles(self) -> Dict[str, Dict[str, float]]:
        """
        Get percentiles of inference metrics.
        
        Returns:
            Dictionary with percentiles for each metric
        """
        if not self.inference_metrics:
            return {}
        
        # Extract metrics
        total_times = [m.total_time_ms for m in self.inference_metrics]
        inference_times = [m.inference_time_ms for m in self.inference_metrics]
        samples_per_second = [m.samples_per_second for m in self.inference_metrics]
        
        # Calculate percentiles
        percentiles = {}
        
        percentiles["total_time_ms"] = {
            "p50": np.percentile(total_times, 50),
            "p90": np.percentile(total_times, 90),
            "p95": np.percentile(total_times, 95),
            "p99": np.percentile(total_times, 99),
            "min": np.min(total_times),
            "max": np.max(total_times),
            "mean": np.mean(total_times),
            "std": np.std(total_times)
        }
        
        percentiles["inference_time_ms"] = {
            "p50": np.percentile(inference_times, 50),
            "p90": np.percentile(inference_times, 90),
            "p95": np.percentile(inference_times, 95),
            "p99": np.percentile(inference_times, 99),
            "min": np.min(inference_times),
            "max": np.max(inference_times),
            "mean": np.mean(inference_times),
            "std": np.std(inference_times)
        }
        
        percentiles["samples_per_second"] = {
            "p50": np.percentile(samples_per_second, 50),
            "p90": np.percentile(samples_per_second, 90),
            "p95": np.percentile(samples_per_second, 95),
            "p99": np.percentile(samples_per_second, 99),
            "min": np.min(samples_per_second),
            "max": np.max(samples_per_second),
            "mean": np.mean(samples_per_second),
            "std": np.std(samples_per_second)
        }
        
        return percentiles
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive performance report.

        Returns:
            Report as a string
        """
        report = []
        report.append(f"# Performance Report for {self.model_name}")
        report.append(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        report.extend(self._get_training_report_section())
        report.extend(self._get_inference_report_section())
        report.extend(self._get_bottleneck_report_section())
        report.extend(self._get_regression_report_section())

        return "\n".join(report)

    def _get_training_report_section(self) -> List[str]:
        section = []
        if not self.training_metrics:
            return section
        section.append("## Training Performance")
        latest = self.training_metrics[-1]
        section.append(f"- Total time: {latest.total_time_ms:.2f} ms")
        section.append(f"- Forward time: {latest.forward_time_ms:.2f} ms")
        section.append(f"- Backward time: {latest.backward_time_ms:.2f} ms")
        section.append(f"- Optimizer time: {latest.optimizer_time_ms:.2f} ms")
        section.append(f"- Samples per second: {latest.samples_per_second:.2f}")
        section.append(f"- CPU memory: {latest.cpu_memory_mb:.2f} MB")
        section.append(f"- CUDA memory: {latest.cuda_memory_mb:.2f} MB")
        section.append("")
        percentiles = self.get_training_metrics_percentiles()
        if percentiles:
            section.append("### Training Percentiles")
            for metric, stats in percentiles.items():
                section.append(f"#### {metric}")
                section.append(f"- P50: {stats['p50']:.2f}")
                section.append(f"- P90: {stats['p90']:.2f}")
                section.append(f"- P95: {stats['p95']:.2f}")
                section.append(f"- P99: {stats['p99']:.2f}")
                section.append(f"- Min: {stats['min']:.2f}")
                section.append(f"- Max: {stats['max']:.2f}")
                section.append(f"- Mean: {stats['mean']:.2f}")
                section.append(f"- Std: {stats['std']:.2f}")
                section.append("")
        return section

    def _get_inference_report_section(self) -> List[str]:
        section = []
        if not self.inference_metrics:
            return section
        section.append("## Inference Performance")
        latest = self.inference_metrics[-1]
        section.append(f"- Total time: {latest.total_time_ms:.2f} ms")
        section.append(f"- Inference time: {latest.inference_time_ms:.2f} ms")
        section.append(f"- Preprocess time: {latest.preprocess_time_ms:.2f} ms")
        section.append(f"- Postprocess time: {latest.postprocess_time_ms:.2f} ms")
        section.append(f"- Samples per second: {latest.samples_per_second:.2f}")
        section.append(f"- CPU memory: {latest.cpu_memory_mb:.2f} MB")
        section.append(f"- CUDA memory: {latest.cuda_memory_mb:.2f} MB")
        section.append("")
        percentiles = self.get_inference_metrics_percentiles()
        if percentiles:
            section.append("### Inference Percentiles")
            for metric, stats in percentiles.items():
                section.append(f"#### {metric}")
                section.append(f"- P50: {stats['p50']:.2f}")
                section.append(f"- P90: {stats['p90']:.2f}")
                section.append(f"- P95: {stats['p95']:.2f}")
                section.append(f"- P99: {stats['p99']:.2f}")
                section.append(f"- Min: {stats['min']:.2f}")
                section.append(f"- Max: {stats['max']:.2f}")
                section.append(f"- Mean: {stats['mean']:.2f}")
                section.append(f"- Std: {stats['std']:.2f}")
                section.append("")
        return section

    def _get_bottleneck_report_section(self) -> List[str]:
        section = []
        for mode in ["training", "inference"]:
            if mode in self.profile_results:
                section.append(f"## {mode.capitalize()} Bottlenecks")
                bottlenecks = self._identify_bottlenecks(self.profile_results[mode])
                if bottlenecks:
                    for i, bottleneck in enumerate(bottlenecks):
                        section.append(f"### Bottleneck {i+1}: {bottleneck.name}")
                        section.append(f"- Type: {bottleneck.type}")
                        section.append(f"- Severity: {bottleneck.severity:.2f}")
                        section.append(f"- Time: {bottleneck.time_ms:.2f} ms ({bottleneck.percentage:.1f}%)")
                        section.append(f"- Recommendation: {bottleneck.recommendation}")
                        section.append("")
                else:
                    section.append("No significant bottlenecks detected.")
                    section.append("")
        return section

    def _get_regression_report_section(self) -> List[str]:
        section = []
        regressions = self.detect_regressions()
        if regressions:
            section.append("## Performance Regressions")
            for name, info in regressions.items():
                section.append(f"### {name}")
                section.append(f"- Baseline: {info['baseline']:.2f}")
                section.append(f"- Current: {info['current']:.2f}")
                section.append(f"- Regression: {info['regression']*100:.1f}%")
                section.append("")
        return section
    
    def save_report(self, filename: str = "performance_report.md"):
        """
        Save performance report to file.
        
        Args:
            filename: Name of the file to save report to
        """
        report = self.generate_report()
        
        filepath = os.path.join(self.run_dir, filename)
        with open(filepath, "w") as f:
            f.write(report)
        
        self.logger.info(f"Performance report saved to {filepath}")
    
    def plot_metrics(self, save_dir: Optional[str] = None):
        """
        Plot performance metrics.
        
        Args:
            save_dir: Directory to save plots to (defaults to run_dir)
        """
        save_dir = save_dir or self.run_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Plot training metrics
        if self.training_metrics:
            # Extract data
            steps = list(range(1, len(self.training_metrics) + 1))
            total_times = [m.total_time_ms for m in self.training_metrics]
            forward_times = [m.forward_time_ms for m in self.training_metrics]
            backward_times = [m.backward_time_ms for m in self.training_metrics]
            optimizer_times = [m.optimizer_time_ms for m in self.training_metrics]
            samples_per_second = [m.samples_per_second for m in self.training_metrics]
            
            # Plot training times
            plt.figure(figsize=(10, 6))
            plt.plot(steps, total_times, label="Total")
            plt.plot(steps, forward_times, label="Forward")
            plt.plot(steps, backward_times, label="Backward")
            plt.plot(steps, optimizer_times, label="Optimizer")
            plt.xlabel("Step")
            plt.ylabel("Time (ms)")
            plt.title("Training Times")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, "training_times.png"))
            plt.close()
            
            # Plot samples per second
            plt.figure(figsize=(10, 6))
            plt.plot(steps, samples_per_second)
            plt.xlabel("Step")
            plt.ylabel("Samples per Second")
            plt.title("Training Throughput")
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, "training_throughput.png"))
            plt.close()
        
        # Plot inference metrics
        if self.inference_metrics:
            # Extract data
            steps = list(range(1, len(self.inference_metrics) + 1))
            total_times = [m.total_time_ms for m in self.inference_metrics]
            inference_times = [m.inference_time_ms for m in self.inference_metrics]
            preprocess_times = [m.preprocess_time_ms for m in self.inference_metrics]
            postprocess_times = [m.postprocess_time_ms for m in self.inference_metrics]
            samples_per_second = [m.samples_per_second for m in self.inference_metrics]
            
            # Plot inference times
            plt.figure(figsize=(10, 6))
            plt.plot(steps, total_times, label="Total")
            plt.plot(steps, inference_times, label="Inference")
            plt.plot(steps, preprocess_times, label="Preprocess")
            plt.plot(steps, postprocess_times, label="Postprocess")
            plt.xlabel("Step")
            plt.ylabel("Time (ms)")
            plt.title("Inference Times")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, "inference_times.png"))
            plt.close()
            
            # Plot samples per second
            plt.figure(figsize=(10, 6))
            plt.plot(steps, samples_per_second)
            plt.xlabel("Step")
            plt.ylabel("Samples per Second")
            plt.title("Inference Throughput")
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, "inference_throughput.png"))
            plt.close()
        
        self.logger.info(f"Performance plots saved to {save_dir}")
    
    def close(self):
        """Close the profiler and clean up resources."""
        # Save report
        self.save_report()
        
        # Plot metrics
        self.plot_metrics()
        
        self.logger.info("Performance profiler closed")