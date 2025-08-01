"""
Backtest module for AURA AI trading system.
This file serves as a bridge to the enhanced backtesting functionality.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime

from src.utils.config_manager import ConfigManager
from src.data.unified_data_loader import EnhancedDataLoader
from src.core.portfolio_manager import PortfolioManager
from src.ml.rl_model import SelfImprovingRLModel
from src.ml.config_adapter import TradingConfigAdapter
from src.core.trading_agent import TradingAgent
from src.backtesting.enhanced_backtest import BacktestResult

logger = logging.getLogger(__name__)

def run_backtest_loop(
    config_manager: ConfigManager,
    symbol: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 100000.0,
    model_path: Optional[str] = None
) -> BacktestResult:
    """
    Run a backtest for the specified symbol and date range.
    
    Args:
        config_manager: Configuration manager
        symbol: Trading symbol
        start_date: Start date for backtest (YYYY-MM-DD)
        end_date: End date for backtest (YYYY-MM-DD)
        initial_capital: Initial capital for the backtest
        model_path: Path to RL model (optional)
        
    Returns:
        BacktestResult object with backtest results
    """
    logger.info(f"Starting backtest for {symbol} from {start_date} to {end_date}")
    
    # Import here to avoid circular imports
    from src.backtesting.enhanced_backtest import run_backtest
    
    # Run the backtest using the enhanced backtest module
    result = run_backtest(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        config_manager=config_manager,
        model_path=model_path,
        use_hybrid_agent=True  # Default to hybrid agent for better performance
    )
    
    # Print summary
    result.print_summary()
    
    return result