"""
ML Metrics Tracker for AURA AI.
Tracks in-sample, validation, and out-of-sample metrics for ML models.
Includes TensorBoard integration for visualization.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import os
import json
from datetime import datetime
import torch

# Import TensorBoard SummaryWriter
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("TensorBoard not available. Install with: pip install tensorboard")

@dataclass
class MLMetrics:
    """Container for ML-specific metrics."""
    # Loss metrics
    loss: float = 0.0
    
    # Accuracy metrics
    accuracy: float = 0.0
    
    # Sample count
    sample_count: int = 0
    
    # Timestamp
    timestamp: str = ""

class MLMetricsTracker:
    """
    ML metrics tracker for AURA AI.
    Tracks in-sample, validation, and out-of-sample metrics.
    """
    
    def __init__(self, output_dir: str = "ml_metrics"):
        """
        Initialize the ML metrics tracker.
        
        Args:
            output_dir: Directory to save metrics and plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.in_sample_metrics: List[MLMetrics] = []
        self.validation_metrics: List[MLMetrics] = []
        self.out_of_sample_metrics: List[MLMetrics] = []
        
        # Best metrics tracking
        self.best_validation_loss = float('inf')
        self.best_validation_accuracy = 0.0
        self.epochs_without_improvement = 0
        
        # Initialize TensorBoard writer if available
        self.writer = None
        if TENSORBOARD_AVAILABLE:
            tensorboard_dir = os.path.join(output_dir, 'tensorboard')
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tensorboard_dir)
            print(f"TensorBoard logs will be saved to {tensorboard_dir}")
            print(f"View TensorBoard with: tensorboard --logdir={tensorboard_dir}")
    
    def add_metrics(self, 
                   split: str, 
                   loss: float, 
                   accuracy: float, 
                   sample_count: int,
                   global_step: int = None) -> None:
        """
        Add metrics for a specific data split.
        
        Args:
            split: Data split ('in_sample', 'validation', 'out_of_sample')
            loss: Loss value
            accuracy: Accuracy value (0-100)
            sample_count: Number of samples
            global_step: Global step for TensorBoard (optional)
        """
        metrics = MLMetrics(
            loss=loss,
            accuracy=accuracy,
            sample_count=sample_count,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        if split == 'in_sample':
            self.in_sample_metrics.append(metrics)
        elif split == 'validation':
            self.validation_metrics.append(metrics)
            
            # Check for improvement
            if loss < self.best_validation_loss:
                self.best_validation_loss = loss
                self.epochs_without_improvement = 0
            elif accuracy > self.best_validation_accuracy:
                self.best_validation_accuracy = accuracy
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                
        elif split == 'out_of_sample':
            self.out_of_sample_metrics.append(metrics)
        else:
            raise ValueError(f"Invalid split: {split}")
        
        # Log to TensorBoard if available
        if self.writer is not None and global_step is not None:
            self.writer.add_scalar(f'{split}/loss', loss, global_step)
            self.writer.add_scalar(f'{split}/accuracy', accuracy, global_step)
            
            # Add overfitting metrics if we have both in-sample and out-of-sample data
            if split == 'out_of_sample' and self.in_sample_metrics:
                in_sample = self.in_sample_metrics[-1]
                out_of_sample = self.out_of_sample_metrics[-1]
                
                # Calculate accuracy gap and loss ratio
                accuracy_gap = in_sample.accuracy - out_of_sample.accuracy
                loss_ratio = out_of_sample.loss / max(in_sample.loss, 1e-8)
                
                # Log overfitting metrics
                self.writer.add_scalar('overfitting/accuracy_gap', accuracy_gap, global_step)
                self.writer.add_scalar('overfitting/loss_ratio', loss_ratio, global_step)
                self.writer.add_scalar('overfitting/score', self.get_overfitting_score(), global_step)
    
    def add_histogram(self, name: str, values: torch.Tensor, global_step: int) -> None:
        """
        Add histogram to TensorBoard.
        
        Args:
            name: Histogram name
            values: Tensor of values
            global_step: Global step
        """
        if self.writer is not None:
            self.writer.add_histogram(name, values, global_step)
    
    def add_model_graph(self, model: torch.nn.Module, input_tensor: torch.Tensor) -> None:
        """
        Add model graph to TensorBoard.
        
        Args:
            model: PyTorch model
            input_tensor: Example input tensor
        """
        if self.writer is not None:
            try:
                self.writer.add_graph(model, input_tensor)
            except Exception as e:
                print(f"Failed to add model graph to TensorBoard: {e}")
    
    def get_latest_metrics(self) -> Dict[str, MLMetrics]:
        """Get the latest metrics for all splits."""
        return {
            'in_sample': self.in_sample_metrics[-1] if self.in_sample_metrics else MLMetrics(),
            'validation': self.validation_metrics[-1] if self.validation_metrics else MLMetrics(),
            'out_of_sample': self.out_of_sample_metrics[-1] if self.out_of_sample_metrics else MLMetrics()
        }
    
    def get_overfitting_score(self) -> float:
        """
        Calculate overfitting score based on in-sample vs out-of-sample performance.
        Higher score indicates more overfitting.
        """
        if not self.in_sample_metrics or not self.out_of_sample_metrics:
            return 0.0
        
        in_sample = self.in_sample_metrics[-1]
        out_of_sample = self.out_of_sample_metrics[-1]
        
        # Calculate accuracy gap
        accuracy_gap = in_sample.accuracy - out_of_sample.accuracy
        
        # Calculate loss ratio (out-of-sample loss / in-sample loss)
        loss_ratio = out_of_sample.loss / max(in_sample.loss, 1e-8)
        
        # Combine into overfitting score
        return (accuracy_gap * 0.5) + ((loss_ratio - 1.0) * 0.5)
    
    def get_overfitting_status(self) -> Dict[str, Any]:
        """Get overfitting status and recommendations."""
        if not self.in_sample_metrics or not self.out_of_sample_metrics:
            return {"status": "insufficient_data"}
        
        overfitting_score = self.get_overfitting_score()
        
        # Determine overfitting status
        if overfitting_score > 15:
            status = "severe"
            recommendation = "Stop training and reduce model complexity significantly."
        elif overfitting_score > 10:
            status = "moderate"
            recommendation = "Implement stronger regularization or reduce model complexity."
        elif overfitting_score > 5:
            status = "mild"
            recommendation = "Consider early stopping or regularization techniques."
        else:
            status = "none"
            recommendation = "Model is generalizing well. Continue training."
        
        # Get latest metrics
        latest = self.get_latest_metrics()
        
        return {
            "status": status,
            "overfitting_score": overfitting_score,
            "accuracy_gap": latest['in_sample'].accuracy - latest['out_of_sample'].accuracy,
            "loss_ratio": latest['out_of_sample'].loss / max(latest['in_sample'].loss, 1e-8),
            "in_sample_accuracy": latest['in_sample'].accuracy,
            "out_of_sample_accuracy": latest['out_of_sample'].accuracy,
            "in_sample_loss": latest['in_sample'].loss,
            "out_of_sample_loss": latest['out_of_sample'].loss,
            "epochs_without_improvement": self.epochs_without_improvement,
            "recommendation": recommendation
        }
    
    def plot_metrics(self, save_path: Optional[str] = None) -> None:
        """
        Plot training metrics for all data splits.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.in_sample_metrics:
            print("No metrics available for plotting")
            return
        
        # Create figure with subplots
        _fig, axes = plt.subplots(2, 1, figsize=(12, 10))  # fig unused
        
        # Prepare data for plotting
        epochs = range(1, len(self.in_sample_metrics) + 1)
        
        # Plot loss curves
        axes[0].plot(
            epochs, 
            [m.loss for m in self.in_sample_metrics], 
            'b-', 
            label='In-Sample'
        )
        
        if self.validation_metrics:
            val_epochs = range(1, len(self.validation_metrics) + 1)
            axes[0].plot(
                val_epochs, 
                [m.loss for m in self.validation_metrics], 
                'g-', 
                label='Validation'
            )
        
        if self.out_of_sample_metrics:
            test_epochs = range(1, len(self.out_of_sample_metrics) + 1)
            axes[0].plot(
                test_epochs, 
                [m.loss for m in self.out_of_sample_metrics], 
                'r-', 
                label='Out-of-Sample'
            )
        
        axes[0].set_title('Loss Curves')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot accuracy curves
        axes[1].plot(
            epochs, 
            [m.accuracy for m in self.in_sample_metrics], 
            'b-', 
            label='In-Sample'
        )
        
        if self.validation_metrics:
            val_epochs = range(1, len(self.validation_metrics) + 1)
            axes[1].plot(
                val_epochs, 
                [m.accuracy for m in self.validation_metrics], 
                'g-', 
                label='Validation'
            )
        
        if self.out_of_sample_metrics:
            test_epochs = range(1, len(self.out_of_sample_metrics) + 1)
            axes[1].plot(
                test_epochs, 
                [m.accuracy for m in self.out_of_sample_metrics], 
                'r-', 
                label='Out-of-Sample'
            )
        
        axes[1].set_title('Accuracy Curves')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Metrics plot saved to {save_path}")
        else:
            plt.show()
    
    def save_metrics(self, filename: Optional[str] = None) -> None:
        """
        Save metrics to JSON file.
        
        Args:
            filename: Output filename (optional)
        """
        if filename is None:
            filename = os.path.join(self.output_dir, f"ml_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # Convert metrics to serializable format
        data = {
            'in_sample': [vars(m) for m in self.in_sample_metrics],
            'validation': [vars(m) for m in self.validation_metrics],
            'out_of_sample': [vars(m) for m in self.out_of_sample_metrics],
            'best_validation_loss': self.best_validation_loss,
            'best_validation_accuracy': self.best_validation_accuracy,
            'epochs_without_improvement': self.epochs_without_improvement,
            'overfitting_status': self.get_overfitting_status()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Metrics saved to {filename}")
    
    def print_summary(self) -> None:
        """Print a summary of current metrics."""
        latest = self.get_latest_metrics()
        overfitting = self.get_overfitting_status()
        
        print("\n" + "="*50)
        print("ML METRICS SUMMARY")
        print("="*50)
        
        print("\nLatest Metrics:")
        print("-"*30)
        
        for split_name, metrics in latest.items():
            if metrics.sample_count > 0:
                print(f"\n{split_name.upper()}:")
                print(f"  Loss: {metrics.loss:.4f}")
                print(f"  Accuracy: {metrics.accuracy:.2f}%")
                print(f"  Sample Count: {metrics.sample_count}")
        
        print("\nOverfitting Analysis:")
        print("-"*30)
        print(f"Status: {overfitting['status'].upper()}")
        print(f"Overfitting Score: {overfitting.get('overfitting_score', 0):.2f}")
        print(f"Accuracy Gap: {overfitting.get('accuracy_gap', 0):.2f}%")
        print(f"Loss Ratio: {overfitting.get('loss_ratio', 0):.2f}")
        print(f"Recommendation: {overfitting.get('recommendation', 'N/A')}")
        
        print("\nTraining Status:")
        print("-"*30)
        print(f"Best Validation Loss: {self.best_validation_loss:.4f}")
        print(f"Best Validation Accuracy: {self.best_validation_accuracy:.2f}%")
        print(f"Epochs Without Improvement: {self.epochs_without_improvement}")
        print("="*50)
    
    def close(self) -> None:
        """Close TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()