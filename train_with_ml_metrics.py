"""
Training script with ML metrics tracking for AURA AI.
Demonstrates how to use the enhanced trainer with ML metrics tracking.
Includes TensorBoard integration for visualization.
"""
import argparse
import logging
import os
from datetime import datetime

from src.utils.config_manager import ConfigManager
from src.ml.rl_model import SelfImprovingRLModel
from src.ml.config_adapter import TradingConfigAdapter
from src.ml.enhanced_trainer import EnhancedTrainer

# Configure centralized logging
from src.utils.logging_config import configure_logging
configure_logging(log_file="logs/training.log")
logger = logging.getLogger(__name__)

def main():
    """Main function to run training with ML metrics tracking."""
    # Use centralized CLI utilities
    from src.utils.cli_utils import create_training_parser
    parser = create_training_parser()
    parser.set_defaults(output_dir="ml_metrics")  # Override default for training
    
    args = parser.parse_args()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.symbol}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration
    try:
        config_manager = ConfigManager(args.config_path, args.schema_path)
    except Exception as e:
        logger.critical(f"Error loading config file: {e}")
        return
    
    # Initialize RL model
    input_size = config_manager.get("reinforcement_learning.input_size", 20)
    action_size = 3  # HOLD, BUY, SELL
    config_size = 4  # stop_loss_pct, profit_target_pct, risk_per_trade_pct, confidence_threshold
    
    model = SelfImprovingRLModel(
        input_size=input_size,
        hidden_size=256,
        action_size=action_size,
        config_size=config_size,
        learning_rate=config_manager.get("reinforcement_learning.learning_rate", 0.001),
        gamma=config_manager.get("reinforcement_learning.gamma", 0.99),
        tau=config_manager.get("reinforcement_learning.tau", 0.005),
        buffer_size=config_manager.get("reinforcement_learning.buffer_size", 100000),
        batch_size=config_manager.get("reinforcement_learning.batch_size", 32),
        n_step=config_manager.get("reinforcement_learning.n_step", 3),
        use_meta_learning=config_manager.get("reinforcement_learning.use_meta_learning", True),
        use_curiosity=config_manager.get("reinforcement_learning.use_curiosity", True),
        use_uncertainty=config_manager.get("reinforcement_learning.use_uncertainty", True),
        use_auxiliary=config_manager.get("reinforcement_learning.use_auxiliary", True),
        use_prioritized_replay=config_manager.get("reinforcement_learning.use_prioritized_replay", True),
        curiosity_weight=config_manager.get("reinforcement_learning.curiosity_weight", 0.01),
        auxiliary_weight=config_manager.get("reinforcement_learning.auxiliary_weight", 0.1),
        device="cpu"  # Use CPU for training
    )
    
    # Load existing model if provided
    if args.model_path and os.path.exists(args.model_path):
        try:
            model.load_model(args.model_path)
            logger.info(f"Successfully loaded model from {args.model_path}")
        except Exception as e:
            logger.error(f"Could not load model from {args.model_path}: {e}")
    
    # Initialize config adapter
    config_adapter = TradingConfigAdapter()
    
    # Initialize enhanced trainer
    trainer = EnhancedTrainer(
        model=model,
        config_adapter=config_adapter,
        config_manager=config_manager,
        output_dir=output_dir
    )
    
    # Train model with ML metrics tracking
    logger.info(f"Starting training for {args.symbol} from {args.start_date} to {args.end_date}")
    logger.info(f"TensorBoard logs will be saved to {os.path.join(output_dir, 'tensorboard')}")
    logger.info(f"View TensorBoard with: tensorboard --logdir={os.path.join(output_dir, 'tensorboard')}")
    
    results = trainer.train(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        num_episodes=args.episodes,
        validation_split=args.validation_split,
        test_split=args.test_split,
        initial_epsilon=args.initial_epsilon,
        final_epsilon=args.final_epsilon
    )
    
    # Print final results
    logger.info("Training completed!")
    logger.info(f"Episodes trained: {results['episodes_trained']}")
    logger.info(f"Best validation reward: {results['best_validation_reward']:.2f}")
    
    if results['test_reward'] is not None:
        logger.info(f"Test reward: {results['test_reward']:.2f}")
        logger.info(f"Test accuracy: {results['test_accuracy']:.2f}%")
    
    logger.info(f"Overfitting status: {results['overfitting_status']['status']}")
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"To view training metrics in TensorBoard, run: tensorboard --logdir={os.path.join(output_dir, 'tensorboard')}")

if __name__ == "__main__":
    main()