"""
Automated hyperparameter optimization for AURA AI.
Uses Optuna for efficient hyperparameter search.
"""
import optuna
import logging
from typing import Dict, Any, Optional
import numpy as np
from src.ml.rl_model import SelfImprovingRLModel
from src.ml.market_env import MarketEnv
from src.ml.rl_model import SelfImprovingRLModel
from src.ml.enhanced_trainer import SelfImprovingRLModel  # Use relative import if trainer.py is in the same package
from src.ml.enhanced_trainer import EnhancedTrainer  # Import the trainer class

logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """Optimizes hyperparameters for AURA AI using Bayesian optimization."""
    
    def __init__(self, env: MarketEnv, n_trials: int = 100, timeout: Optional[int] = None):
        self.env = env
        self.n_trials = n_trials
        self.timeout = timeout
        self.best_params = None
        self.best_score = -np.inf
        
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for hyperparameter optimization."""
        
        # Suggest hyperparameters
        params = {
            # Network architecture
            'hidden_size': trial.suggest_categorical('hidden_size', [128, 256, 512, 1024]),
            
            # Learning parameters
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'gamma': trial.suggest_float('gamma', 0.9, 0.999),
            'tau': trial.suggest_float('tau', 0.001, 0.01),
            
            # Experience replay
            'buffer_size': trial.suggest_categorical('buffer_size', [50000, 100000, 200000]),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'n_step': trial.suggest_int('n_step', 1, 5),
            
            # Advanced features
            'curiosity_weight': trial.suggest_float('curiosity_weight', 0.001, 0.1, log=True),
            'auxiliary_weight': trial.suggest_float('auxiliary_weight', 0.01, 0.5),
            
            # Training parameters
            'epsilon_start': trial.suggest_float('epsilon_start', 0.5, 1.0),
            'epsilon_end': trial.suggest_float('epsilon_end', 0.01, 0.1),
            'epsilon_decay': trial.suggest_float('epsilon_decay', 0.995, 0.9995),
        }
        
        try:
            # Create model with suggested parameters
            input_size = self.env.observation_space.shape[0]
            action_size = self.env.action_space.n
            config_size = 3
            
            model = SelfImprovingRLModel(
                input_size=input_size,
                hidden_size=params['hidden_size'],
                action_size=action_size,
                config_size=config_size,
                learning_rate=params['learning_rate'],
                gamma=params['gamma'],
                tau=params['tau'],
                buffer_size=params['buffer_size'],
                batch_size=params['batch_size'],
                n_step=params['n_step'],
                curiosity_weight=params['curiosity_weight'],
                auxiliary_weight=params['auxiliary_weight'],
                device='cpu'  # Use CPU for optimization to avoid memory issues
            )
            
            # Train for a limited number of episodes
            trainer = EnhancedTrainer(model, self.env, None)
            
            # Quick training evaluation (fewer episodes for speed)
            total_reward = 0
            n_eval_episodes = 10
            
            for _ in range(n_eval_episodes):
                metrics = trainer.train_episode()
                total_reward += metrics.total_reward
                
            avg_reward = total_reward / n_eval_episodes
            
            # Penalize for instability (high variance in rewards)
            recent_rewards = [m.total_reward for m in model.performance_history]
            if len(recent_rewards) > 5:
                reward_std = np.std(recent_rewards[-5:])
                stability_penalty = reward_std * 0.1  # Adjust weight as needed
                score = avg_reward - stability_penalty
            else:
                score = avg_reward
                
            logger.info(f"Trial {trial.number}: Score = {score:.4f}, Params = {params}")
            
            return score
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return -np.inf
            
    def optimize(self, study_name: str = "aura_ai_optimization") -> Dict[str, Any]:
        """Run hyperparameter optimization."""
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42),  # Reproducible results
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials")
        
        # Optimize
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        # Store best parameters
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        logger.info("Optimization completed!")
        logger.info(f"Best score: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        # Return optimization results
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': len(study.trials),
            'study': study
        }
        
    def get_optimized_model_config(self) -> Dict[str, Any]:
        """Get configuration for creating optimized model."""
        if self.best_params is None:
            raise ValueError("No optimization has been run yet. Call optimize() first.")
            
        return self.best_params