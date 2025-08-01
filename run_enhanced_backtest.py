"""
Enhanced backtest runner for the integrated SelfImprovingRLModel.

This script runs backtests using the enhanced SelfImprovingRLModel with
all integrated components for memory management, numerical stability,
error handling, and performance optimizations.
"""

import argparse
import json
import logging
import os
import sys
import time
import numpy as np
import torch
from datetime import datetime

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ml.rl_model import SelfImprovingRLModel
from src.backtesting.backtester import Backtester, BacktestEnvironment
from src.utils.config_manager import ConfigManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/enhanced_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        raise


def create_model_from_config(config):
    """Create SelfImprovingRLModel from configuration."""
    model_config = config["model_config"]
    memory_config = config["memory_management"]
    
    # Create model with basic parameters
    model = SelfImprovingRLModel(
        input_size=model_config["input_size"],
        hidden_size=model_config["hidden_size"],
        action_size=model_config["action_size"],
        config_size=model_config["config_size"],
        learning_rate=model_config["learning_rate"],
        gamma=model_config["gamma"],
        tau=model_config["tau"],
        buffer_size=model_config["buffer_size"],
        batch_size=model_config["batch_size"],
        n_step=model_config["n_step"],
        use_meta_learning=model_config["use_meta_learning"],
        use_curiosity=model_config["use_curiosity"],
        use_uncertainty=model_config["use_uncertainty"],
        use_auxiliary=model_config["use_auxiliary"],
        use_prioritized_replay=model_config["use_prioritized_replay"],
        curiosity_weight=model_config["curiosity_weight"],
        auxiliary_weight=model_config["auxiliary_weight"],
        device=model_config["device"]
    )
    
    # Configure memory management
    model.memory_monitor.alert_threshold_gb = memory_config["alert_threshold_gb"]
    model.memory_monitor.warning_threshold = memory_config["warning_threshold"]
    model.memory_monitor.critical_threshold = memory_config["critical_threshold"]
    
    # Configure numerical stability
    model.stability_manager.epsilon = config["numerical_stability"]["epsilon"]
    model.gradient_validator.max_grad_norm = config["numerical_stability"]["max_grad_norm"]
    
    # Configure error handling
    input_validation_config = config["error_handling"]["input_validation"]
    model.input_validator.allow_sanitization = input_validation_config["allow_sanitization"]
    model.input_validator.strict_mode = input_validation_config["strict_mode"]
    model.input_validator.value_range = tuple(input_validation_config["value_range"])
    
    # Configure fallback manager
    fallback_config = config["error_handling"]["fallback"]
    model.fallback_manager.confidence_threshold = fallback_config["confidence_threshold"]
    model.fallback_manager.fallback_strategies[-1].liquidate_positions = fallback_config["liquidate_on_halt"]
    
    # Configure health monitoring
    health_config = config["error_handling"]["health_monitoring"]
    model.health_monitor.thresholds.error_rate_warning = health_config["error_rate_warning"]
    model.health_monitor.thresholds.error_rate_critical = health_config["error_rate_critical"]
    model.health_monitor.thresholds.q_value_range_warning = health_config["q_value_range_warning"]
    model.health_monitor.thresholds.q_value_range_critical = health_config["q_value_range_critical"]
    model.health_monitor.thresholds.gradient_norm_warning = health_config["gradient_norm_warning"]
    model.health_monitor.thresholds.gradient_norm_critical = health_config["gradient_norm_critical"]
    
    return model


def run_backtest(model, data_path, episodes=100, steps_per_episode=1000):
    """Run backtest with the enhanced model."""
    logger.info(f"Starting backtest with {episodes} episodes, {steps_per_episode} steps per episode")
    
    # Create backtest environment
    env = BacktestEnvironment(data_path=data_path)
    
    # Track metrics
    total_rewards = []
    episode_lengths = []
    inference_times = []
    update_times = []
    
    # Run episodes
    for episode in range(episodes):
        logger.info(f"Starting episode {episode+1}/{episodes}")
        
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(steps_per_episode):
            # Select action
            start_time = time.time()
            action, _, _ = model.act(state, epsilon=0.1)
            inference_time = (time.time() - start_time) * 1000
            inference_times.append(inference_time)
            
            # Take action in environment
            next_state, reward, done, _ = env.step(action)
            
            # Store experience
            model.add_experience(state, action, reward, next_state, done)
            
            # Update model
            if step % 4 == 0:  # Update every 4 steps
                start_time = time.time()
                model.update()
                update_time = (time.time() - start_time) * 1000
                update_times.append(update_time)
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            # Check health status
            if step % 100 == 0:
                health_report = model.health_monitor.get_health_report()
                logger.info(f"Health status: {health_report.current_metrics.status}, "
                           f"Score: {health_report.current_metrics.overall_health:.2f}")
                
                # Check memory usage
                memory_stats = model.memory_monitor.get_memory_stats()
                logger.info(f"Memory usage - CPU: {memory_stats['cpu_memory']['used_gb']:.2f}GB, "
                           f"GPU: {memory_stats['gpu_memory']['used_gb']:.2f}GB")
            
            if done:
                break
        
        # Record episode metrics
        total_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        logger.info(f"Episode {episode+1} - Reward: {episode_reward:.2f}, Length: {episode_length}")
        logger.info(f"Avg inference time: {np.mean(inference_times):.2f}ms, "
                   f"Avg update time: {np.mean(update_times):.2f}ms")
    
    # Log final results
    logger.info("Backtest complete")
    logger.info(f"Average reward: {np.mean(total_rewards):.2f}")
    logger.info(f"Average episode length: {np.mean(episode_lengths):.2f}")
    logger.info(f"Average inference time: {np.mean(inference_times):.2f}ms")
    logger.info(f"Average update time: {np.mean(update_times):.2f}ms")
    
    # Return results
    return {
        "total_rewards": total_rewards,
        "episode_lengths": episode_lengths,
        "inference_times": inference_times,
        "update_times": update_times
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run enhanced backtest with integrated RL model")
    parser.add_argument("--config", type=str, default="config/enhanced_config.json",
                       help="Path to configuration file")
    parser.add_argument("--data", type=str, default="data/BTCUSDT_5m_5years.csv",
                       help="Path to backtest data")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of episodes to run")
    parser.add_argument("--steps", type=int, default=1000,
                       help="Maximum steps per episode")
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Create model
    logger.info("Creating model from configuration")
    model = create_model_from_config(config)
    
    # Run backtest
    results = run_backtest(
        model=model,
        data_path=args.data,
        episodes=args.episodes,
        steps_per_episode=args.steps
    )
    
    # Save results
    results_path = f"results/enhanced_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("results", exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump({
            "config": config,
            "results": {
                "total_rewards": [float(r) for r in results["total_rewards"]],
                "episode_lengths": [int(l) for l in results["episode_lengths"]],
                "avg_inference_time_ms": float(np.mean(results["inference_times"])),
                "avg_update_time_ms": float(np.mean(results["update_times"]))
            }
        }, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()