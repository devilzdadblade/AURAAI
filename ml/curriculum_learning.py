"""
Curriculum Learning Manager for AURA AI - Progressive difficulty scaling for day trading.
Implements adaptive learning stages based on market conditions and performance.
"""
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class DifficultyLevel(Enum):
    """Difficulty levels for curriculum learning."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

@dataclass
class CurriculumStage:
    """Represents a curriculum learning stage."""
    level: DifficultyLevel
    market_conditions: List[str]
    success_threshold: float
    min_episodes: int
    max_episodes: int
    risk_multiplier: float
    complexity_features: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'level': self.level.value,
            'market_conditions': self.market_conditions,
            'success_threshold': self.success_threshold,
            'min_episodes': self.min_episodes,
            'max_episodes': self.max_episodes,
            'risk_multiplier': self.risk_multiplier,
            'complexity_features': self.complexity_features
        }

class CurriculumLearningManager:
    """
    Manages curriculum learning progression for day trading ML models.
    Adapts training difficulty based on performance and market conditions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize curriculum learning manager.
        
        Args:
            config: Configuration dictionary containing curriculum settings
        """
        self.config = config
        self.current_stage_index = 0
        self.episodes_in_current_stage = 0
        self.stage_performance_history = []
        self.advancement_history = []
        
        # Define curriculum stages for day trading
        self.stages = self._define_curriculum_stages()
        
        # Performance tracking
        self.recent_rewards = []
        self.recent_drawdowns = []
        self.success_streak = 0
        self.failure_streak = 0
        
        logger.info(f"CurriculumLearningManager initialized with {len(self.stages)} stages")
        logger.info(f"Starting at {self.get_current_stage().level.value} level")
    
    def _define_curriculum_stages(self) -> List[CurriculumStage]:
        """Define the curriculum stages for day trading."""
        return [
            # Stage 1: Beginner - Simple trending markets
            CurriculumStage(
                level=DifficultyLevel.BEGINNER,
                market_conditions=["trending", "low_volatility"],
                success_threshold=0.6,  # 60% win rate
                min_episodes=100,
                max_episodes=500,
                risk_multiplier=0.5,  # Lower risk
                complexity_features=["sma", "rsi", "volume"]
            ),
            
            # Stage 2: Intermediate - Mixed market conditions
            CurriculumStage(
                level=DifficultyLevel.INTERMEDIATE,
                market_conditions=["trending", "ranging", "medium_volatility"],
                success_threshold=0.55,  # 55% win rate
                min_episodes=200,
                max_episodes=800,
                risk_multiplier=0.75,
                complexity_features=["sma", "ema", "rsi", "macd", "bollinger", "volume", "atr"]
            ),
            
            # Stage 3: Advanced - Complex market conditions
            CurriculumStage(
                level=DifficultyLevel.ADVANCED,
                market_conditions=["trending", "ranging", "high_volatility", "news_events"],
                success_threshold=0.52,  # 52% win rate
                min_episodes=300,
                max_episodes=1000,
                risk_multiplier=1.0,
                complexity_features=[
                    "sma", "ema", "rsi", "macd", "bollinger", "volume", "atr",
                    "stochastic", "cci", "williams_r", "momentum", "roc"
                ]
            ),
            
            # Stage 4: Expert - All market conditions with microstructure
            CurriculumStage(
                level=DifficultyLevel.EXPERT,
                market_conditions=[
                    "trending", "ranging", "high_volatility", "news_events",
                    "low_liquidity", "market_open", "market_close"
                ],
                success_threshold=0.51,  # 51% win rate (realistic for expert level)
                min_episodes=500,
                max_episodes=2000,
                risk_multiplier=1.2,
                complexity_features=[
                    "sma", "ema", "rsi", "macd", "bollinger", "volume", "atr",
                    "stochastic", "cci", "williams_r", "momentum", "roc",
                    "order_book_imbalance", "bid_ask_spread", "funding_rate",
                    "volume_profile", "market_microstructure"
                ]
            )
        ]
    
    def get_current_stage(self) -> CurriculumStage:
        """Get the current curriculum stage."""
        return self.stages[min(self.current_stage_index, len(self.stages) - 1)]
    
    def update_progress(self, episode_reward: float, episode_drawdown: float = 0.0) -> bool:
        """
        Update curriculum progress with episode results.
        
        Args:
            episode_reward: Reward achieved in the episode
            episode_drawdown: Maximum drawdown in the episode
            
        Returns:
            bool: True if advanced to next stage
        """
        self.episodes_in_current_stage += 1
        self.recent_rewards.append(episode_reward)
        self.recent_drawdowns.append(episode_drawdown)
        
        # Keep only recent history (last 50 episodes)
        if len(self.recent_rewards) > 50:
            self.recent_rewards.pop(0)
            self.recent_drawdowns.pop(0)
        
        # Update success/failure streaks
        if episode_reward > 0:
            self.success_streak += 1
            self.failure_streak = 0
        else:
            self.failure_streak += 1
            self.success_streak = 0
        
        # Check for stage advancement
        current_stage = self.get_current_stage()
        
        # Calculate performance metrics
        win_rate = self._calculate_win_rate()
        avg_reward = np.mean(self.recent_rewards) if self.recent_rewards else 0.0
        avg_drawdown = np.mean(self.recent_drawdowns) if self.recent_drawdowns else 0.0
        
        # Record stage performance
        stage_performance = {
            'episode': self.episodes_in_current_stage,
            'win_rate': win_rate,
            'avg_reward': avg_reward,
            'avg_drawdown': avg_drawdown,
            'success_streak': self.success_streak
        }
        self.stage_performance_history.append(stage_performance)
        
        # Check advancement criteria
        should_advance = self._should_advance_stage(current_stage, win_rate, avg_reward)
        
        if should_advance and self.current_stage_index < len(self.stages) - 1:
            self._advance_to_next_stage()
            return True
        
        # Check for regression (if performance is very poor)
        should_regress = self._should_regress_stage(current_stage, win_rate, avg_reward)
        if should_regress and self.current_stage_index > 0:
            self._regress_to_previous_stage()
        
        return False
    
    def _calculate_win_rate(self) -> float:
        """Calculate current win rate."""
        if not self.recent_rewards:
            return 0.0
        wins = sum(1 for r in self.recent_rewards if r > 0)
        return wins / len(self.recent_rewards)
    
    def _should_advance_stage(self, stage: CurriculumStage, win_rate: float, avg_reward: float) -> bool:
        """
        Determine if should advance to next stage.
        
        Args:
            stage: Current curriculum stage
            win_rate: Current win rate
            avg_reward: Average reward
            
        Returns:
            bool: True if should advance
        """
        # Must meet minimum episodes requirement
        if self.episodes_in_current_stage < stage.min_episodes:
            return False
        
        # Must exceed success threshold
        if win_rate < stage.success_threshold:
            return False
        
        # Must have positive average reward
        if avg_reward <= 0:
            return False
        
        # Must have consistent performance (low drawdown)
        avg_drawdown = np.mean(self.recent_drawdowns) if self.recent_drawdowns else 0.0
        if avg_drawdown > 0.1:  # 10% max drawdown threshold
            return False
        
        # Must have recent success streak
        if self.success_streak < 5:
            return False
        
        logger.info(f"Stage advancement criteria met: WR={win_rate:.2%}, "
                   f"Reward={avg_reward:.3f}, Episodes={self.episodes_in_current_stage}")
        
        return True
    
    def _should_regress_stage(self, stage: CurriculumStage, win_rate: float, avg_reward: float) -> bool:
        """
        Determine if should regress to previous stage due to poor performance.
        
        Args:
            stage: Current curriculum stage
            win_rate: Current win rate
            avg_reward: Average reward
            
        Returns:
            bool: True if should regress
        """
        # Only consider regression after sufficient episodes
        if self.episodes_in_current_stage < 50:
            return False
        
        # Regress if performance is significantly below threshold
        performance_gap = stage.success_threshold - win_rate
        if performance_gap > 0.15:  # 15% below threshold
            return True
        
        # Regress if consistent negative rewards
        if avg_reward < -0.1 and self.failure_streak > 10:
            return True
        
        return False
    
    def _advance_to_next_stage(self):
        """Advance to the next curriculum stage."""
        old_stage = self.get_current_stage()
        self.current_stage_index += 1
        new_stage = self.get_current_stage()
        
        # Record advancement
        advancement_record = {
            'timestamp': datetime.now().isoformat(),
            'from_stage': old_stage.level.value,
            'to_stage': new_stage.level.value,
            'episodes_completed': self.episodes_in_current_stage,
            'final_win_rate': self._calculate_win_rate(),
            'final_avg_reward': np.mean(self.recent_rewards) if self.recent_rewards else 0.0
        }
        self.advancement_history.append(advancement_record)
        
        # Reset stage tracking
        self.episodes_in_current_stage = 0
        self.stage_performance_history.clear()
        self.success_streak = 0
        self.failure_streak = 0
        
        logger.info(f"🎓 Advanced from {old_stage.level.value} to {new_stage.level.value} level!")
        logger.info(f"New stage features: {', '.join(new_stage.complexity_features[:5])}...")
    
    def _regress_to_previous_stage(self):
        """Regress to the previous curriculum stage."""
        old_stage = self.get_current_stage()
        self.current_stage_index -= 1
        new_stage = self.get_current_stage()
        
        # Reset stage tracking
        self.episodes_in_current_stage = 0
        self.stage_performance_history.clear()
        self.success_streak = 0
        self.failure_streak = 0
        
        logger.warning(f"📉 Regressed from {old_stage.level.value} to {new_stage.level.value} level")
        logger.info("Performance will be monitored for re-advancement")
    
    def get_training_config_adjustments(self) -> Dict[str, Any]:
        """
        Get training configuration adjustments for current stage.
        
        Returns:
            Dict containing configuration adjustments
        """
        current_stage = self.get_current_stage()
        
        # Base adjustments
        adjustments = {
            'risk_multiplier': current_stage.risk_multiplier,
            'complexity_features': current_stage.complexity_features,
            'market_conditions': current_stage.market_conditions
        }
        
        # Stage-specific learning rate adjustments
        if current_stage.level == DifficultyLevel.BEGINNER:
            adjustments['learning_rate_multiplier'] = 1.2  # Faster learning
            adjustments['exploration_rate'] = 0.3  # Higher exploration
        elif current_stage.level == DifficultyLevel.INTERMEDIATE:
            adjustments['learning_rate_multiplier'] = 1.0  # Normal learning
            adjustments['exploration_rate'] = 0.2
        elif current_stage.level == DifficultyLevel.ADVANCED:
            adjustments['learning_rate_multiplier'] = 0.8  # Slower, more careful learning
            adjustments['exploration_rate'] = 0.15
        else:  # EXPERT
            adjustments['learning_rate_multiplier'] = 0.6  # Very careful learning
            adjustments['exploration_rate'] = 0.1  # Low exploration
        
        return adjustments
    
    def get_curriculum_summary(self) -> Dict[str, Any]:
        """Get comprehensive curriculum learning summary."""
        current_stage = self.get_current_stage()
        
        # Calculate progress percentage within current stage
        progress_in_stage = min(
            self.episodes_in_current_stage / current_stage.max_episodes,
            1.0
        )
        
        # Calculate overall curriculum progress
        overall_progress = (
            self.current_stage_index + progress_in_stage
        ) / len(self.stages)
        
        return {
            'current_level': current_stage.level.value,
            'current_stage_index': self.current_stage_index,
            'total_stages': len(self.stages),
            'episodes_in_current_stage': self.episodes_in_current_stage,
            'progress_percentage': overall_progress * 100,
            'stage_progress_percentage': progress_in_stage * 100,
            'success_threshold': current_stage.success_threshold,
            'current_win_rate': self._calculate_win_rate(),
            'success_streak': self.success_streak,
            'failure_streak': self.failure_streak,
            'market_conditions': current_stage.market_conditions,
            'complexity_level': len(current_stage.complexity_features),
            'advancements_count': len(self.advancement_history),
            'time_in_current_stage': self.episodes_in_current_stage
        }
    
    def save_progress(self, filepath: str):
        """Save curriculum learning progress to file."""
        try:
            progress_data = {
                'current_stage_index': self.current_stage_index,
                'episodes_in_current_stage': self.episodes_in_current_stage,
                'advancement_history': self.advancement_history,
                'recent_rewards': self.recent_rewards[-20:],  # Save last 20
                'success_streak': self.success_streak,
                'failure_streak': self.failure_streak,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(progress_data, f, indent=2)
            
            logger.info(f"Curriculum progress saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save curriculum progress: {e}")
    
    def load_progress(self, filepath: str):
        """Load curriculum learning progress from file."""
        try:
            with open(filepath, 'r') as f:
                progress_data = json.load(f)
            
            self.current_stage_index = progress_data.get('current_stage_index', 0)
            self.episodes_in_current_stage = progress_data.get('episodes_in_current_stage', 0)
            self.advancement_history = progress_data.get('advancement_history', [])
            self.recent_rewards = progress_data.get('recent_rewards', [])
            self.success_streak = progress_data.get('success_streak', 0)
            self.failure_streak = progress_data.get('failure_streak', 0)
            
            logger.info(f"Curriculum progress loaded from {filepath}")
            logger.info(f"Resumed at {self.get_current_stage().level.value} level")
        except Exception as e:
            logger.error(f"Failed to load curriculum progress: {e}")
            logger.info("Starting with fresh curriculum progress")