"""
Enhanced trainer with ML metrics tracking for AURA AI.
Implements proper train/validation/test splits and metrics tracking.
Includes TensorBoard integration for visualization.
"""
import numpy as np
import pandas as pd
import torch
import logging
import os
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from src.ml.rl_model import SelfImprovingRLModel
from src.ml.config_adapter import TradingConfigAdapter
from src.utils.constants import FAILED_TO_PREFIX
from src.ml.market_env import MarketEnv
from src.ml.ml_metrics_tracker import MLMetricsTracker
from src.data.unified_data_loader import EnhancedDataLoader
from src.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class EnhancedTrainer:
    """
    Enhanced trainer with ML metrics tracking.
    Implements proper train/validation/test splits for RL models.
    """
    
    def __init__(self, 
                 model: SelfImprovingRLModel,
                 config_adapter: TradingConfigAdapter,
                 config_manager: ConfigManager,
                 output_dir: str = "ml_metrics"):
        """
        Initialize the enhanced trainer.
        
        Args:
            model: The RL model to train
            config_adapter: Adapter for trading parameters
            config_manager: Configuration manager
            output_dir: Directory to save metrics
        """
        self.model = model
        self.config_adapter = config_adapter
        self.config_manager = config_manager
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize metrics tracker
        self.metrics_tracker = MLMetricsTracker(output_dir)
        
        # Training parameters
        self.validation_frequency = 5  # Validate every N episodes
        self.early_stopping_patience = 10  # Stop after N episodes without improvement
        
        # Data environments
        self.train_env = None
        self.val_env = None
        self.test_env = None
        
        # Global step counter for TensorBoard
        self.global_step = 0
        
        # Add model graph to TensorBoard
        self._add_model_graph()
    
    def _add_model_graph(self):
        """Add model graph to TensorBoard."""
        try:
            # Create a dummy input tensor for the model
            dummy_input = torch.zeros(1, self.model.input_size, dtype=torch.float32)
            dummy_input = dummy_input.to(self.model.device)
            
            # Add model graph to TensorBoard
            self.metrics_tracker.add_model_graph(self.model, dummy_input)
        except Exception as e:
            logger.warning(f"{FAILED_TO_PREFIX} add model graph to TensorBoard: {e}")
    
    def prepare_data_splits(self, 
                           symbol: str, 
                           start_date: str, 
                           end_date: str,
                           validation_split: float = 0.2,
                           test_split: float = 0.2) -> None:
        """
        Prepare train/validation/test data splits.
        
        Args:
            symbol: Trading symbol
            start_date: Start date for data
            end_date: End date for data
            validation_split: Fraction of training data to use for validation
            test_split: Fraction of data to use for testing
        """
        logger.info(f"Preparing data splits for {symbol} from {start_date} to {end_date}")
        
        # Load and process data
        data_loader = EnhancedDataLoader(self.config_manager)
        df_ohlcv = data_loader.load_and_process_data(symbol)
        
        # Filter data for the specified date range
        df_ohlcv = df_ohlcv.loc[start_date:end_date]
        
        if df_ohlcv.empty:
            raise ValueError(f"No data found for {symbol} in the specified period")
        
        # Get features for RL model
        features_for_rl_model = [col for col in df_ohlcv.columns 
                               if col not in ['open', 'high', 'low', 'close', 'volume', 'label', 'timestamp']]
        
        if not features_for_rl_model:
            raise ValueError("No features found for RL model input")
        
        # Calculate split indices
        total_samples = len(df_ohlcv)
        test_size = int(total_samples * test_split)
        
        # First split into train+val and test
        train_val_data = df_ohlcv.iloc[:-test_size] if test_size > 0 else df_ohlcv
        test_data = df_ohlcv.iloc[-test_size:] if test_size > 0 else pd.DataFrame()
        
        # Then split train+val into train and val
        val_size = int(len(train_val_data) * validation_split)
        train_data = train_val_data.iloc[:-val_size] if val_size > 0 else train_val_data
        val_data = train_val_data.iloc[-val_size:] if val_size > 0 else pd.DataFrame()
        
        # Log split sizes
        logger.info(f"Data split sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Create environments
        self.train_env = MarketEnv(train_data, features_for_rl_model)
        self.val_env = MarketEnv(val_data, features_for_rl_model) if not val_data.empty else None
        self.test_env = MarketEnv(test_data, features_for_rl_model) if not test_data.empty else None
    
    def run_episode(self, 
                   env: MarketEnv, 
                   split: str, 
                   epsilon: float = 0.0, 
                   training: bool = False) -> Dict[str, Any]:
        """
        Run a single episode and collect metrics (refactored for lower complexity).
        
        Args:
            env: Environment to run episode in
            split: Data split name ('in_sample', 'validation', 'out_of_sample')
            epsilon: Exploration rate
            training: Whether to update the model during this episode
            
        Returns:
            Episode results
        """
        # Validate environment
        if not self._validate_episode_environment(env, split):
            return {}
        
        # Setup and initialize episode
        context = self._prepare_episode_context(env, training)
        
        # Execute episode
        final_info = self._execute_episode_loop(context, env, epsilon, training)
        
        # Process results
        return self._finalize_episode_results(context, split, final_info)
    
    def _validate_episode_environment(self, env: MarketEnv, split: str) -> bool:
        """Validate that environment is available for episode execution."""
        if env is None:
            logger.warning(f"No environment available for {split} split")
            return False
        return True
    
    def _prepare_episode_context(self, env: MarketEnv, training: bool) -> Dict[str, Any]:
        """Prepare episode context with initial state and metrics."""
        self._setup_episode_mode(training)
        state, metrics = self._initialize_episode_metrics(env)
        
        return {
            'state': state,
            'metrics': metrics,
            'training': training
        }
    
    def _execute_episode_loop(self, context: Dict[str, Any], env: MarketEnv, 
                             epsilon: float, training: bool) -> Dict[str, Any]:
        """Execute the main episode loop until completion."""
        state = context['state']
        metrics = context['metrics']
        final_info = {}
        
        while not metrics['done']:
            state, info = self._process_episode_step(state, env, epsilon, training, metrics)
            final_info = info
            
        return final_info
    
    def _finalize_episode_results(self, context: Dict[str, Any], split: str, 
                                 final_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process episode completion and return results."""
        metrics = context['metrics']
        
        # Record metrics
        self._record_episode_metrics(split, metrics)
        
        # Create and return results
        return self._create_episode_results(metrics, final_info)
        
    def _setup_episode_mode(self, training: bool) -> None:
        """Set the model mode based on whether we're training."""
        if training:
            self.model.train()
        else:
            self.model.eval()
            
    def _initialize_episode_metrics(self, env: MarketEnv) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Initialize episode state and metrics tracking."""
        state = env.reset()
        
        metrics = {
            'done': False,
            'total_reward': 0,
            'steps': 0,
            'correct_actions': 0,
            'total_actions': 0,
            'losses': [],
            'q_values_list': [],
            'td_errors': []
        }
        
        return state, metrics
        
    def _run_episode_steps(self, 
                          state: np.ndarray, 
                          env: MarketEnv, 
                          epsilon: float, 
                          training: bool, 
                          metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Run all steps in the episode and return final info."""
        final_info = {}
        
        while not metrics['done']:
            # Take action and update environment
            state, info = self._process_episode_step(state, env, epsilon, training, metrics)
            final_info = info
            
        return final_info
            
    def _process_episode_step(self, 
                             state: np.ndarray, 
                             env: MarketEnv, 
                             epsilon: float, 
                             training: bool, 
                             metrics: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process a single step in the episode."""
        # Select action
        action, _, action_info = self.model.act(state, epsilon=epsilon)
        
        # Take step in environment
        next_state, reward, done, info = env.step(action)
        
        # Update basic metrics
        metrics['total_reward'] += reward
        metrics['steps'] += 1
        metrics['done'] = done
        
        # Track action accuracy
        self._track_action_accuracy(action, info, metrics)
        
        # Track Q-values if available
        if 'q_values' in action_info:
            metrics['q_values_list'].append(action_info['q_values'])
        
        # Handle training-specific operations
        if training:
            self._perform_training_updates(state, action, reward, next_state, done, metrics)
        
        return next_state, info
        
    def _track_action_accuracy(self, action: int, info: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        """Track the accuracy of actions against optimal actions."""
        optimal_action = self._calculate_optimal_action(info)
        
        if action == optimal_action:
            metrics['correct_actions'] += 1
        metrics['total_actions'] += 1
        
    def _perform_training_updates(self, 
                                 state: np.ndarray, 
                                 action: int, 
                                 reward: float, 
                                 next_state: np.ndarray, 
                                 done: bool, 
                                 metrics: Dict[str, Any]) -> None:
        """Perform model updates during training."""
        # Add experience to replay buffer
        self.model.add_experience(state, action, reward, next_state, done)
        
        # Update model if enough samples
        if len(self.model.replay_buffer) > self.model.batch_size * 4:
            update_metrics = self.model.update(metrics['steps'])
            metrics['losses'].append(update_metrics.average_loss)
            
            # Track TD errors if available
            if hasattr(update_metrics, 'td_error_mean'):
                metrics['td_errors'].append(update_metrics.td_error_mean)
            
            # Increment global step
            self.global_step += 1
            
    def _record_episode_metrics(self, split: str, metrics: Dict[str, Any]) -> None:
        """Record episode metrics to tracker and TensorBoard."""
        # Calculate aggregated metrics
        accuracy = (metrics['correct_actions'] / max(1, metrics['total_actions'])) * 100
        avg_loss = np.mean(metrics['losses']) if metrics['losses'] else 0.0
        
        # Add metrics to tracker
        self.metrics_tracker.add_metrics(
            split=split,
            loss=avg_loss,
            accuracy=accuracy,
            sample_count=metrics['steps'],
            global_step=self.global_step
        )
        
        # Add additional metrics to TensorBoard
        self._record_tensorboard_histograms(split, metrics)
        
    def _record_tensorboard_histograms(self, split: str, metrics: Dict[str, Any]) -> None:
        """Record histogram metrics to TensorBoard."""
        # Record Q-values
        if metrics['q_values_list']:
            q_values_array = np.array(metrics['q_values_list'])
            q_values_tensor = torch.tensor(q_values_array)
            self.metrics_tracker.add_histogram(f'{split}/q_values', q_values_tensor, self.global_step)
        
        # Record TD errors
        if metrics['td_errors']:
            td_errors_tensor = torch.tensor(metrics['td_errors'])
            self.metrics_tracker.add_histogram(f'{split}/td_errors', td_errors_tensor, self.global_step)
            
    def _create_episode_results(self, metrics: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
        """Create the final episode results dictionary."""
        # Calculate aggregated metrics
        accuracy = (metrics['correct_actions'] / max(1, metrics['total_actions'])) * 100
        avg_loss = np.mean(metrics['losses']) if metrics['losses'] else 0.0
        
        return {
            'reward': metrics['total_reward'],
            'steps': metrics['steps'],
            'accuracy': accuracy,
            'loss': avg_loss,
            'final_portfolio_value': info.get('portfolio_value', 0)
        }
    
    def _calculate_optimal_action(self, info: Dict[str, Any]) -> int:
        """
        Calculate the optimal action based on environment info.
        This is a simplified heuristic - in practice, you might use more sophisticated methods.
        
        Args:
            info: Environment info dictionary
            
        Returns:
            Optimal action (0=HOLD, 1=BUY, 2=SELL)
        """
        # Simple heuristic: if next return is positive, BUY should have been optimal
        # if negative, SELL should have been optimal, otherwise HOLD
        next_return = info.get('next_return', 0.0)
        
        if next_return > 0.005:  # Threshold for significant positive return
            return 1  # BUY
        elif next_return < -0.005:  # Threshold for significant negative return
            return 2  # SELL
        else:
            return 0  # HOLD
    
    def train(self, 
             symbol: str, 
             start_date: str, 
             end_date: str, 
             num_episodes: int = 100,
             validation_split: float = 0.2,
             test_split: float = 0.2,
             initial_epsilon: float = 0.5,
             final_epsilon: float = 0.01) -> Dict[str, Any]:
        """
        Train the model with proper validation and testing.
        
        Args:
            symbol: Trading symbol
            start_date: Start date for data
            end_date: End date for data
            num_episodes: Number of episodes to train
            validation_split: Fraction of training data to use for validation
            test_split: Fraction of data to use for testing
            initial_epsilon: Initial exploration rate
            final_epsilon: Final exploration rate
            
        Returns:
            Training results
        """
        logger.info(f"Starting enhanced training for {symbol} from {start_date} to {end_date}")
        
        # Initialize training
        self._initialize_training(symbol, start_date, end_date, validation_split, test_split)
        
        # Run training loop
        training_state = self._run_training_loop(num_episodes, initial_epsilon, final_epsilon)
        
        # Final evaluation
        test_results = self._run_final_evaluation()
        
        # Generate report and cleanup
        self._finalize_training()
        
        return self._create_training_results(training_state, test_results)

    def _initialize_training(self, symbol: str, start_date: str, end_date: str, 
                           validation_split: float, test_split: float) -> None:
        """Initialize training data and environments."""
        self.prepare_data_splits(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            validation_split=validation_split,
            test_split=test_split
        )

    def _run_training_loop(self, num_episodes: int, initial_epsilon: float, 
                          final_epsilon: float) -> Dict[str, Any]:
        """Run the main training loop with validation and early stopping."""
        best_val_reward = float('-inf')
        episodes_without_improvement = 0
        final_episode = 0
        
        for episode in range(num_episodes):
            final_episode = episode
            
            # Calculate epsilon and run training episode
            epsilon = self._calculate_epsilon(episode, num_episodes, initial_epsilon, final_epsilon)
            train_results = self._run_training_episode(epsilon)
            
            # Validation and early stopping
            should_stop, best_val_reward, episodes_without_improvement = self._handle_validation(
                episode, best_val_reward, episodes_without_improvement
            )
            
            # Progress logging
            self._log_training_progress(episode, num_episodes, train_results)
            
            if should_stop:
                logger.info(f"Early stopping triggered after {episode} episodes")
                break
        
        return {
            'episodes_trained': final_episode + 1,
            'best_validation_reward': best_val_reward
        }

    def _calculate_epsilon(self, episode: int, num_episodes: int, 
                          initial_epsilon: float, final_epsilon: float) -> float:
        """Calculate epsilon for exploration (linear decay)."""
        return max(final_epsilon, initial_epsilon - (initial_epsilon - final_epsilon) * episode / num_episodes)

    def _run_training_episode(self, epsilon: float) -> Dict[str, Any]:
        """Run a single training episode."""
        return self.run_episode(
            env=self.train_env,
            split='in_sample',
            epsilon=epsilon,
            training=True
        )

    def _handle_validation(self, episode: int, best_val_reward: float, 
                          episodes_without_improvement: int) -> Tuple[bool, float, int]:
        """Handle validation and early stopping logic."""
        should_stop = False
        
        if episode % self.validation_frequency == 0 and self.val_env is not None:
            val_results = self.run_episode(
                env=self.val_env,
                split='validation',
                epsilon=0.0,
                training=False
            )
            
            # Check for improvement
            if val_results.get('reward', float('-inf')) > best_val_reward:
                best_val_reward = val_results['reward']
                episodes_without_improvement = 0
                self._save_best_model(episode)
            else:
                episodes_without_improvement += 1
            
            # Check early stopping
            if episodes_without_improvement >= self.early_stopping_patience:
                should_stop = True
            
            # Check overfitting
            self._check_overfitting(episode)
        
        return should_stop, best_val_reward, episodes_without_improvement

    def _save_best_model(self, episode: int) -> None:
        """Save the best model."""
        model_path = os.path.join(self.output_dir, 'best_model.pth')
        self.model.save_model(model_path)
        logger.info(f"New best model saved at episode {episode}")

    def _check_overfitting(self, episode: int) -> None:
        """Check and log overfitting status."""
        if episode % self.validation_frequency == 0:
            overfitting = self.metrics_tracker.get_overfitting_status()
            if overfitting['status'] != 'insufficient_data':
                logger.info(f"Overfitting status: {overfitting['status']}")

    def _log_training_progress(self, episode: int, num_episodes: int, 
                              train_results: Dict[str, Any]) -> None:
        """Log training progress."""
        if episode % 10 == 0:
            logger.info(f"Episode {episode}/{num_episodes}, "
                       f"Reward: {train_results.get('reward', 0):.2f}, "
                       f"Loss: {train_results.get('loss', 0):.4f}, "
                       f"Accuracy: {train_results.get('accuracy', 0):.2f}%")

    def _run_final_evaluation(self) -> Optional[Dict[str, Any]]:
        """Run final evaluation on test set."""
        if self.test_env is None:
            return None
            
        logger.info("Running final evaluation on test set...")
        
        # Load best model if available
        self._load_best_model_for_testing()
        
        test_results = self.run_episode(
            env=self.test_env,
            split='out_of_sample',
            epsilon=0.0,
            training=False
        )
        
        logger.info("Test results - "
                   f"Reward: {test_results.get('reward', 0):.2f}, "
                   f"Accuracy: {test_results.get('accuracy', 0):.2f}%")
        
        return test_results

    def _load_best_model_for_testing(self) -> None:
        """Load best model for final testing."""
        best_model_path = os.path.join(self.output_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            self.model.load_model(best_model_path)
            logger.info("Loaded best model for testing")

    def _finalize_training(self) -> None:
        """Generate final report and cleanup."""
        self._generate_report()
        self.metrics_tracker.close()

    def _create_training_results(self, training_state: Dict[str, Any], 
                               test_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Create final training results dictionary."""
        return {
            'episodes_trained': training_state['episodes_trained'],
            'best_validation_reward': training_state['best_validation_reward'],
            'test_reward': test_results.get('reward', 0) if test_results else None,
            'test_accuracy': test_results.get('accuracy', 0) if test_results else None,
            'overfitting_status': self.metrics_tracker.get_overfitting_status()
        }
    
    def _generate_report(self) -> None:
        """Generate training report with metrics and plots."""
        # Save metrics
        self.metrics_tracker.save_metrics(os.path.join(self.output_dir, 'ml_metrics.json'))
        
        # Plot metrics
        self.metrics_tracker.plot_metrics(os.path.join(self.output_dir, 'ml_metrics_plot.png'))
        
        # Print summary
        self.metrics_tracker.print_summary()
        
        logger.info(f"Training report saved to {self.output_dir}")
    
    def evaluate(self, 
                symbol: str, 
                start_date: str, 
                end_date: str,
                model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate a trained model on new data.
        
        Args:
            symbol: Trading symbol
            start_date: Start date for evaluation
            end_date: End date for evaluation
            model_path: Path to saved model (optional)
            
        Returns:
            Evaluation results
        """
        # Load model if provided
        self._load_evaluation_model(model_path)
        
        # Prepare evaluation environment
        eval_env = self._prepare_evaluation_environment(symbol, start_date, end_date)
        
        # Run evaluation and log results
        return self._execute_evaluation(eval_env)
    
    def _load_evaluation_model(self, model_path: Optional[str]) -> None:
        """Load model for evaluation if path is provided."""
        if model_path and os.path.exists(model_path):
            self.model.load_model(model_path)
            logger.info(f"Loaded model from {model_path}")
    
    def _prepare_evaluation_environment(self, symbol: str, start_date: str, end_date: str) -> MarketEnv:
        """Prepare and return evaluation environment with processed data."""
        # Load and process evaluation data
        evaluation_data = self._load_evaluation_data(symbol, start_date, end_date)
        
        # Extract features for RL model
        features_for_rl_model = self._extract_rl_features(evaluation_data)
        
        # Create and return evaluation environment
        return MarketEnv(evaluation_data, features_for_rl_model)
    
    def _load_evaluation_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load and validate evaluation data for specified period."""
        data_loader = EnhancedDataLoader(self.config_manager)
        df_ohlcv = data_loader.load_and_process_data(symbol)
        df_ohlcv = df_ohlcv.loc[start_date:end_date]
        
        if df_ohlcv.empty:
            raise ValueError(f"No evaluation data found for {symbol} in the specified period")
        
        return df_ohlcv
    
    def _extract_rl_features(self, df_ohlcv: pd.DataFrame) -> List[str]:
        """Extract feature columns for RL model from processed data."""
        excluded_columns = {'open', 'high', 'low', 'close', 'volume', 'label', 'timestamp'}
        return [col for col in df_ohlcv.columns if col not in excluded_columns]
    
    def _execute_evaluation(self, eval_env: MarketEnv) -> Dict[str, Any]:
        """Execute evaluation episode and log results."""
        eval_results = self.run_episode(
            env=eval_env,
            split='out_of_sample',
            epsilon=0.0,
            training=False
        )
        
        # Log evaluation results
        logger.info("Evaluation results - "
                   f"Reward: {eval_results.get('reward', 0):.2f}, "
                   f"Accuracy: {eval_results.get('accuracy', 0):.2f}%")
        
        return eval_results