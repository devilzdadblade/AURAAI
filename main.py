# FILE: main.py
# PURPOSE: Entry point for AURA AI - Advanced Unified Risk-Aware Artificial Intelligence Trading System
# ENHANCED: Day trading optimized with ML-driven decisions, TimescaleDB, and Prometheus monitoring

import logging
import sys
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Generator, Tuple
from contextlib import contextmanager
import pandas as pd
import numpy as np
import argparse
import asyncio
from threading import Thread

from dotenv import load_dotenv

# Core imports
from src.core.secure_config_manager import SecureConfigManager
from src.core.trading_constants import TradingConstants
from src.core.trading_exceptions import (
    TradingSystemError, BrokerConnectionError, BrokerInitializationError,
    DataValidationError, ConfigurationError, ModelLoadError
)
from src.utils.constants import DEFAULT_LEARNING_RATE, DEFAULT_BATCH_SIZE
from src.data.unified_data_loader import EnhancedDataLoader
from src.core.portfolio_manager import PortfolioManager
from src.execution.enhanced_binance_broker import EnhancedBinanceBroker
from src.execution.order_executor import OrderExecutor
from src.utils.notifications import TelegramNotifier
from src.core.trading_agent import TradingAgent
from src.ml.rl_model import SelfImprovingRLModel
from src.ml.config_adapter import TradingConfigAdapter
from src.ml.data_validator import TrainingDataValidator
from src.ml.training_monitor import TrainingMonitor
from src.ml.curriculum_learning import CurriculumLearningManager
from src.data.trade_logger import TradeLogger

# Enhanced components
from src.database.timescale_manager import TimescaleManager
from src.monitoring.prometheus_exporter import AuraPrometheusExporter
from src.rules.enhanced_risk_management import EnhancedRiskManager
from src.core.decision_processor import DecisionProcessor, TradingContext
from src.core.trading_engine_factory import create_and_run_trading_engine
from src.core.graceful_shutdown import GracefulShutdown, get_shutdown_manager
from src.core.shutdown_integrations import (
    TradingSystemShutdownCoordinator,
    create_trading_system_shutdown_manager
)
from src.core.market_data_processor import MarketDataProcessor
from src.core.event_bus import EventBus

from backtest import run_backtest_loop as run_backtest_from_file
load_dotenv()

# Configure centralized logging
from src.utils.logging_config import configure_logging
configure_logging(log_file="logs/aura_ai.log")
logger = logging.getLogger(__name__)

# -----------------------------------------------

# ---------------------------------------------------------------------------
#                       ACTION PROCESSING
# ---------------------------------------------------------------------------
def process_decision_action(
    action: int, 
    trade_params: Dict[str, Any], 
    symbol: str,
    order_executor: OrderExecutor,
    portfolio_manager: PortfolioManager,
    telegram_notifier: TelegramNotifier,
    current_time: datetime
) -> None:
    """
    Process and execute a trading decision action from the AI agent.
    
    This function handles the execution of trading decisions by mapping RL actions
    to actual trading operations, validating parameters, and coordinating with
    the order executor and portfolio manager.
    
    Args:
        action: Integer representing the trading action (0=HOLD, 1=BUY, 2=SELL)
        trade_params: Dictionary containing trading parameters including:
            - quantity: Position size to trade
            - entry_price: Entry price for the trade
            - stop_loss_price: Stop loss price for risk management
            - confidence: AI model confidence in the decision
        symbol: Trading symbol (e.g., 'BTCUSDT')
        order_executor: Order execution interface for placing trades
        portfolio_manager: Portfolio management system for position tracking
        telegram_notifier: Notification system for trade alerts
        current_time: Current timestamp for trade recording
        
    Returns:
        None
        
    Raises:
        Exception: Logs any errors during action processing but doesn't re-raise
        
    Example:
        >>> trade_params = {
        ...     'quantity': 0.1,
        ...     'entry_price': 50000.0,
        ...     'stop_loss_price': 49000.0,
        ...     'confidence': 0.85
        ... }
        >>> process_decision_action(
        ...     action=1,  # BUY
        ...     trade_params=trade_params,
        ...     symbol='BTCUSDT',
        ...     order_executor=executor,
        ...     portfolio_manager=pm,
        ...     telegram_notifier=notifier,
        ...     current_time=datetime.now()
        ... )
    """
    try:
        context = _prepare_action_context(
            action, trade_params, symbol, order_executor, 
            portfolio_manager, telegram_notifier, current_time
        )
        _execute_action_by_type(context)
            
    except Exception as e:
        logger.error(f"Error processing action: {e}", exc_info=True)

def _prepare_action_context(
    action: int,
    trade_params: Dict[str, Any],
    symbol: str,
    order_executor: OrderExecutor,
    portfolio_manager: PortfolioManager,
    telegram_notifier: TelegramNotifier,
    current_time: datetime
) -> Dict[str, Any]:
    """Prepare unified context dictionary for action execution."""
    action_str = TradingConstants.get_action_name(action)
    
    return {
        'action_str': action_str,
        'trade_params': trade_params,
        'symbol': symbol,
        'order_executor': order_executor,
        'portfolio_manager': portfolio_manager,
        'telegram_notifier': telegram_notifier,
        'current_time': current_time
    }

def _execute_action_by_type(context: Dict[str, Any]) -> None:
    """Route action execution based on action type."""
    action_str = context['action_str']
    
    if action_str == TradingConstants.ACTION_NAMES[TradingConstants.ACTION_BUY]:
        _execute_buy_action(context)
    elif action_str == TradingConstants.ACTION_NAMES[TradingConstants.ACTION_SELL]:
        _execute_sell_action(context)
    elif action_str == TradingConstants.ACTION_NAMES[TradingConstants.ACTION_HOLD]:
        logger.info(f"{context['symbol']}: {TradingConstants.ACTION_NAMES[TradingConstants.ACTION_HOLD]}")

def _execute_buy_action(context: Dict[str, Any]) -> None:
    """Execute a BUY trading action using context."""
    symbol = context['symbol']
    trade_params = context['trade_params']
    order_executor = context['order_executor']
    portfolio_manager = context['portfolio_manager']
    telegram_notifier = context['telegram_notifier']
    current_time = context['current_time']
    
    qty = trade_params.get('quantity')
    entry = trade_params.get('entry_price')
    sl = trade_params.get('stop_loss_price')
    
    if not _validate_trade_parameters(qty, entry, TradingConstants.ACTION_NAMES[TradingConstants.ACTION_BUY]):
        return
        
    ctx = _create_trade_context(TradingConstants.STRATEGY_RL_AGENT_BUY, entry, sl, current_time, trade_params)
    order_id = order_executor.place_market_order(symbol, qty, TradingConstants.ORDER_SIDE_BUY, ctx)
    
    if order_id:
        _handle_successful_order(symbol, qty, TradingConstants.ORDER_SIDE_BUY, entry, sl, ctx, portfolio_manager, telegram_notifier)
    else:
        logger.error(TradingConstants.ERROR_MESSAGES['failed_buy'].format(symbol=symbol))


def _execute_sell_action(context: Dict[str, Any]) -> None:
    """Execute a SELL trading action using context."""
    symbol = context['symbol']
    trade_params = context['trade_params']
    order_executor = context['order_executor']
    portfolio_manager = context['portfolio_manager']
    telegram_notifier = context['telegram_notifier']
    current_time = context['current_time']
    
    qty = trade_params.get('quantity')
    entry = trade_params.get('entry_price')
    sl = trade_params.get('stop_loss_price')
    
    if not _validate_trade_parameters(qty, entry, TradingConstants.ACTION_NAMES[TradingConstants.ACTION_SELL]):
        return
        
    ctx = _create_trade_context(TradingConstants.STRATEGY_RL_AGENT_SELL, entry, sl, current_time, trade_params)
    order_id = order_executor.place_market_order(symbol, qty, TradingConstants.ORDER_SIDE_SELL, ctx)
    
    if order_id:
        _handle_successful_order(symbol, qty, TradingConstants.ORDER_SIDE_SELL, entry, sl, ctx, portfolio_manager, telegram_notifier)
    else:
        logger.error(TradingConstants.ERROR_MESSAGES['failed_sell'].format(symbol=symbol))


def _validate_trade_parameters(qty: float, entry: float, action: str) -> bool:
    """Validate trading parameters before execution."""
    if not qty or qty <= 0 or entry is None:
        logger.warning(TradingConstants.ERROR_MESSAGES['invalid_' + action.lower()].format(qty=qty, price=entry))
        return False
    return True


def _create_trade_context(strategy: str, entry: float, sl: float, 
                         current_time: datetime, trade_params: Dict[str, Any]) -> Dict[str, Any]:
    """Create trade context for order execution."""
    return {
        'strategy': strategy,
        'entry_price': entry,
        'stop_loss_price': sl,
        'entry_time': current_time,
        'parameters': trade_params
    }


def _handle_successful_order(symbol: str, qty: float, side: str, entry: float, sl: float,
                           ctx: Dict[str, Any], portfolio_manager: PortfolioManager,
                           telegram_notifier: TelegramNotifier) -> None:
    """Handle successful order placement."""
    portfolio_manager.add_position(
        symbol, qty, side, entry, ctx['entry_time'], sl, ctx['strategy'], ctx['parameters']
    )
    
    if telegram_notifier and telegram_notifier.enabled:
        message_key = 'buy_success' if side == TradingConstants.ORDER_SIDE_BUY else 'sell_success'
        telegram_notifier.send_message(
            TradingConstants.TELEGRAM_MESSAGES[message_key].format(symbol=symbol, qty=qty, price=entry, sl=sl)
        )

# ---------------------------------------------------------------------------
#                         BROKER CONTEXT MANAGER
# ---------------------------------------------------------------------------
@contextmanager
def get_broker(
    config_manager: SecureConfigManager, 
    symbols: List[str]
) -> Generator[Any, None, None]:
    """
    Context manager to initialize and close broker connection safely.
    
    This context manager handles the complete lifecycle of broker connections,
    including initialization, WebSocket stream setup, and proper cleanup.
    It supports multiple broker types and ensures resources are properly
    released even if errors occur.
    
    Args:
        config_manager: Secure configuration manager containing broker settings
        symbols: List of trading symbols to initialize WebSocket streams for
        
    Yields:
        Broker instance (EnhancedBinanceBroker, MockBroker, etc.)
        
    Raises:
        BrokerInitializationError: If broker initialization fails
        
    Example:
        >>> config_manager = SecureConfigManager('config.json', 'schema.json')
        >>> symbols = ['BTCUSDT', 'ETHUSDT']
        >>> with get_broker(config_manager, symbols) as broker:
        ...     # Use broker for trading operations
        ...     ohlcv_data = broker.get_latest_ohlcv('BTCUSDT')
    """
    broker_type = config_manager.get('broker_config.broker_type')
    broker = None
    try:
        if broker_type == "EnhancedBinanceBroker":
            testnet = config_manager.get('broker_config.testnet', False)
            broker = EnhancedBinanceBroker(testnet=testnet)
            # Start WebSocket streams for all symbols
            broker.start_websocket_streams(symbols)
        elif broker_type == "MockBroker":
            from src.execution.brokers import MockBroker
            broker = MockBroker()
        else:
            raise BrokerInitializationError(f"Unknown broker type: {broker_type}")
        yield broker
    except Exception as e:
        logger.error(f"Broker initialization error: {e}")
        raise BrokerInitializationError(f"Broker initialization failed: {e}")
    finally:
        if broker and hasattr(broker, 'close') and callable(getattr(broker, 'close')):
            try:
                broker.close()
            except Exception as e:
                logger.error(f"Broker close error: {e}")

# ---------------------------------------------------------------------------
#                         ENHANCED LIVE TRADING LOOP
# ---------------------------------------------------------------------------
async def run_live_trading_loop(config_manager: SecureConfigManager) -> None:
    """
    Execute the main live trading loop with enhanced AI decision making.
    
    This is the core trading loop that orchestrates all trading operations including:
    - Market data fetching and processing
    - AI-driven decision making with risk management
    - Trade execution and portfolio management
    - Performance monitoring and curriculum learning
    - Real-time notifications and logging
    
    The loop runs continuously until interrupted, processing trading cycles at
    configured intervals while maintaining comprehensive error handling and
    graceful degradation capabilities.
    
    Args:
        config_manager: Secure configuration manager containing all system settings
        
    Returns:
        None
        
    Raises:
        KeyboardInterrupt: When user interrupts the trading loop
        Exception: Any critical errors that cannot be handled gracefully
        
    Example:
        >>> config_manager = SecureConfigManager('config.json', 'schema.json')
        >>> await run_live_trading_loop(config_manager)
    """
    symbols = config_manager.get(TradingConstants.DATA_SOURCE_SYMBOLS_CONFIG)
    primary_symbol = symbols[0]  # Primary trading symbol
    interval_seconds = config_manager.get(TradingConstants.CONFIG_KEYS['cycle_interval'], TradingConstants.DEFAULT_CYCLE_INTERVAL_SECONDS)
    
    logger.info("🚀 Starting AURA AI Enhanced Day Trading")
    logger.info(f"   Primary Symbol: {primary_symbol}")
    logger.info(f"   All Symbols: {', '.join(symbols)}")
    logger.info(f"   Cycle Interval: {interval_seconds}s")

    # Initialize infrastructure components
    _, _ = _initialize_infrastructure(config_manager)
    
    # Initialize core trading components
    trade_logger, portfolio_manager, risk_manager, telegram_notifier = _initialize_core_components(config_manager)
    
    # Initialize AI and data processing components
    data_components = _initialize_ai_components(config_manager)
    
    # Initialize RL model and trading agent
    _, _, decision_processor = _initialize_trading_models(
        config_manager, data_components, risk_manager
    )

    # Main trading loop
    with get_broker(config_manager, symbols) as broker:
        order_executor = OrderExecutor(broker, trade_logger, portfolio_manager, config_manager.config)
        
        try:
            portfolio_manager.sync_with_broker()
        except Exception as e:
            logger.error(f"Initial broker sync failed: {e}")

        logger.info("✅ Real-time data streams active")
        
        await _execute_trading_loop(
            broker=broker,
            order_executor=order_executor,
            portfolio_manager=portfolio_manager,
            decision_processor=decision_processor,
            data_components=data_components,
            trade_logger=trade_logger,
            telegram_notifier=telegram_notifier,
            primary_symbol=primary_symbol,
            interval_seconds=interval_seconds,
            risk_manager=risk_manager
        )

# ---------------------------------------------------------------------------
#                    LIVE TRADING WITH SHUTDOWN SUPPORT
# ---------------------------------------------------------------------------
async def run_live_trading_with_shutdown(
    config_manager: SecureConfigManager, 
    shutdown_manager: GracefulShutdown
) -> None:
    """
    Run live trading with graceful shutdown support.
    
    This function wraps the main trading loop with comprehensive shutdown management,
    ensuring all system components are properly cleaned up when the system is
    terminated. It registers all trading system components with the shutdown
    coordinator for orderly cleanup.
    
    Args:
        config_manager: Secure configuration manager containing system settings
        shutdown_manager: Graceful shutdown manager for coordinating cleanup
        
    Returns:
        None
        
    Raises:
        Exception: Any errors during trading execution or shutdown coordination
        
    Example:
        >>> config_manager = SecureConfigManager('config.json', 'schema.json')
        >>> shutdown_manager = create_trading_system_shutdown_manager()
        >>> await run_live_trading_with_shutdown(config_manager, shutdown_manager)
    """
    logger.info("🚀 Starting AURA AI with graceful shutdown support")
    
    # Initialize components
    symbols = config_manager.get(TradingConstants.DATA_SOURCE_SYMBOLS_CONFIG)
    # Primary symbol not used in this shutdown-focused function
    
    # Initialize infrastructure components
    timescale_manager = None
    prometheus_exporter = None
    trade_logger = None
    broker = None
    
    try:
        # Initialize TimescaleDB
        try:
            timescale_manager = TimescaleManager()
            logger.info("✅ TimescaleDB connected")
        except Exception as e:
            logger.error(f"TimescaleDB connection failed: {e}")
        
        # Initialize Prometheus monitoring
        try:
            prometheus_exporter = AuraPrometheusExporter(
                port=config_manager.get(TradingConstants.CONFIG_KEYS['prometheus_port'], 
                                      TradingConstants.DEFAULT_PROMETHEUS_PORT),
                update_interval=config_manager.get(TradingConstants.CONFIG_KEYS['metrics_update_interval'], 
                                                 TradingConstants.DEFAULT_METRICS_UPDATE_INTERVAL)
            )
            prometheus_exporter.start()
            logger.info("✅ Prometheus monitoring started")
        except Exception as e:
            logger.error(f"Prometheus setup failed: {e}")
        
        # Initialize trade logger
        trade_logger = TradeLogger(
            db_path=config_manager.get('system.persistence.trade_log.db_path'),
            excel_path=config_manager.get('system.persistence.trade_log.excel_path')
        )
        
        # Initialize broker
        broker_type = config_manager.get('broker_config.broker_type')
        if broker_type == "EnhancedBinanceBroker":
            testnet = config_manager.get('broker_config.testnet', False)
            broker = EnhancedBinanceBroker(testnet=testnet)
            broker.start_websocket_streams(symbols)
        elif broker_type == "MockBroker":
            from src.execution.brokers import MockBroker
            broker = MockBroker()
        else:
            raise BrokerInitializationError(f"Unknown broker type: {broker_type}")
        
        # Register all components for shutdown
        shutdown_coordinator = TradingSystemShutdownCoordinator(shutdown_manager)
        
        # Collect SQLite connections
        sqlite_connections = {}
        if trade_logger and hasattr(trade_logger, 'db_path'):
            # Note: TradeLogger manages its own connections, but we can register cleanup
            pass
        
        shutdown_coordinator.register_trading_system_components(
            broker=broker,
            timescale_manager=timescale_manager,
            prometheus_exporter=prometheus_exporter,
            sqlite_connections=sqlite_connections
        )
        
        # Add custom cleanup for trade logger
        if trade_logger:
            def cleanup_trade_logger():
                try:
                    trade_logger.export_to_excel()
                    logger.info("✅ Final trade logs exported")
                except Exception as e:
                    logger.error(f"Error exporting final trade logs: {e}")
            
            shutdown_coordinator.add_custom_cleanup(
                name="trade_logger_export",
                cleanup_func=cleanup_trade_logger,
                priority=200,
                timeout=15.0
            )
        
        logger.info("✅ All components registered for graceful shutdown")
        
        # Run the original trading loop
        await run_live_trading_loop(config_manager)
        
    except Exception as e:
        logger.error(f"Error in live trading with shutdown: {e}", exc_info=True)
        raise
    finally:
        # Cleanup will be handled by shutdown manager
        logger.info("Live trading session ended")

# ---------------------------------------------------------------------------
#                         BACKTESTING LOOP
# ---------------------------------------------------------------------------


def run_backtest_loop(config_manager: SecureConfigManager) -> None:
    """
    Execute backtesting loop using historical data.
    
    This function runs the trading system in backtesting mode using historical
    market data to evaluate strategy performance. It delegates to the dedicated
    backtesting module for comprehensive historical analysis.
    
    Args:
        config_manager: Secure configuration manager containing backtest settings
        
    Returns:
        None
        
    Example:
        >>> config_manager = SecureConfigManager('config.json', 'schema.json')
        >>> run_backtest_loop(config_manager)
    """
    logger.info("Starting backtesting mode.")
    # For backtesting, we need to provide symbol, start_date, end_date, initial_capital
    # These should ideally come from command line arguments or a specific backtest config.
    # For now, using placeholders or values from main config if available.
    symbol = config_manager.get(TradingConstants.CONFIG_KEYS['data_source_symbols'])[0]
    # These dates should be configurable for backtesting
    start_date = TradingConstants.DEFAULT_BACKTEST_START_DATE
    end_date = TradingConstants.DEFAULT_BACKTEST_END_DATE
    initial_capital = config_manager.get(TradingConstants.CONFIG_KEYS['initial_cash'], TradingConstants.DEFAULT_BACKTEST_INITIAL_CAPITAL)

    run_backtest_from_file(
        config_manager=config_manager,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital
    )

# ---------------------------------------------------------------------------
#                              ENTRY POINT
# ---------------------------------------------------------------------------
def main() -> None:
    """
    Main entry point for the AURA AI trading system.
    
    This function serves as the primary entry point that:
    - Parses command line arguments for different execution modes
    - Initializes the secure configuration manager with validation
    - Sets up graceful shutdown handling for all system components
    - Routes execution to the appropriate mode (live, backtest, train)
    - Handles system-level errors and cleanup
    
    The function supports three main execution modes:
    - live: Execute live trading with real market data
    - backtest: Run historical backtesting analysis
    - train: Display training instructions (actual training via train.py)
    
    Returns:
        None
        
    Raises:
        SystemExit: On configuration errors or unknown execution modes
        
    Example:
        Command line usage:
        >>> python main.py --mode live     # Start live trading
        >>> python main.py --mode backtest # Run backtesting
        >>> python main.py --mode train   # Show training info
    """
    # Initialize system components
    config_manager, shutdown_manager = _initialize_main_components()
    
    # Parse command line arguments
    args = _parse_command_line_arguments()
    
    # Execute the selected mode with proper error handling
    _execute_trading_mode(args, config_manager, shutdown_manager)

def _initialize_main_components() -> Tuple[SecureConfigManager, Any]:
    """Initialize main system components with proper error handling."""
    # Use relative paths or environment variables for portability
    config_path = os.getenv(TradingConstants.ENV_VARS['config_path'], TradingConstants.DEFAULT_PATHS['config'])
    schema_path = os.getenv(TradingConstants.ENV_VARS['schema_path'], TradingConstants.DEFAULT_PATHS['schema'])

    # Initialize graceful shutdown system
    shutdown_manager = create_trading_system_shutdown_manager(
        shutdown_timeout=60.0,
        enable_signal_handlers=True
    )
    
    # Initialize configuration manager
    try:
        config_manager = SecureConfigManager(config_path, schema_path)
        return config_manager, shutdown_manager
    except ConfigurationError as e:
        logger.critical(f"Failed to load or validate configuration: {e}")
        sys.exit(TradingConstants.EXIT_CODE_ERROR)

def _parse_command_line_arguments():
    """Parse and return command line arguments."""
    from src.utils.cli_utils import create_main_parser
    parser = create_main_parser()
    return parser.parse_args()

def _execute_trading_mode(args, config_manager: SecureConfigManager, shutdown_manager: Any) -> None:
    """Execute the selected trading mode with proper error handling."""
    try:
        _run_selected_mode(args, config_manager, shutdown_manager)
    except KeyboardInterrupt:
        _handle_keyboard_interrupt(shutdown_manager)
    except Exception as e:
        _handle_fatal_error(e, shutdown_manager)

def _run_selected_mode(args, config_manager: SecureConfigManager, shutdown_manager: Any) -> None:
    """Run the selected trading mode."""
    if args.mode == "live":
        # Run live trading with graceful shutdown support
        asyncio.run(run_live_trading_with_shutdown(config_manager, shutdown_manager))
    elif args.mode == "train":
        # The actual training is handled by train.py, this just provides a message
        logger.info("For training, please run: python train.py")
    elif args.mode == "backtest":
        # Run backtest with basic shutdown support
        with shutdown_manager:
            run_backtest_loop(config_manager)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)

def _handle_keyboard_interrupt(shutdown_manager: Any) -> None:
    """Handle keyboard interrupt gracefully."""
    logger.info("Received keyboard interrupt, initiating graceful shutdown")
    if not shutdown_manager.is_shutdown_complete():
        shutdown_manager.shutdown()

def _handle_fatal_error(error: Exception, shutdown_manager: Any) -> None:
    """Handle fatal errors with proper cleanup."""
    logger.critical(f"Fatal error in main: {error}", exc_info=True)
    if not shutdown_manager.is_shutdown_complete():
        shutdown_manager.shutdown()
    sys.exit(1)


# ---------------------------------------------------------------------------
#                           HELPER FUNCTIONS FOR REFACTORING
# ---------------------------------------------------------------------------

def _initialize_infrastructure(config_manager: SecureConfigManager) -> Tuple[Optional[Any], Optional[Any]]:
    """Initialize infrastructure components (TimescaleDB and Prometheus)."""
    timescale_manager = None
    prometheus_exporter = None
    
    # TimescaleDB for high-performance data storage
    try:
        timescale_manager = TimescaleManager()
        logger.info("✅ TimescaleDB connected")
    except Exception as e:
        logger.error(f"TimescaleDB connection failed: {e}")
    
    # Prometheus monitoring
    try:
        prometheus_exporter = AuraPrometheusExporter(
            port=config_manager.get(TradingConstants.CONFIG_KEYS['prometheus_port'], TradingConstants.DEFAULT_PROMETHEUS_PORT),
            update_interval=config_manager.get(TradingConstants.CONFIG_KEYS['metrics_update_interval'], TradingConstants.DEFAULT_METRICS_UPDATE_INTERVAL)
        )
        prometheus_exporter.start()
        logger.info("✅ Prometheus monitoring started")
    except Exception as e:
        logger.error(f"Prometheus setup failed: {e}")
    
    return timescale_manager, prometheus_exporter


def _initialize_core_components(config_manager: SecureConfigManager) -> Tuple[Any, Any, Any, Any]:
    """Initialize core trading components."""
    trade_logger = TradeLogger(
        db_path=config_manager.get('system.persistence.trade_log.db_path'),
        excel_path=config_manager.get('system.persistence.trade_log.excel_path')
    )
    
    portfolio_manager = PortfolioManager(
        initial_capital=config_manager.get('portfolio_config.initial_cash'),
        trade_logger=trade_logger
    )
    
    # Enhanced Risk Management
    risk_manager = EnhancedRiskManager(config_manager.get('enhanced_risk_management', {}))
    logger.info("✅ Enhanced Risk Management initialized")
    
    # Telegram notifications
    telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
    telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
    telegram_notifier = TelegramNotifier(token=telegram_token, chat_id=telegram_chat_id)
    
    return trade_logger, portfolio_manager, risk_manager, telegram_notifier


def _initialize_ai_components(config_manager: SecureConfigManager) -> Dict[str, Any]:
    """Initialize AI and data processing components."""
    data_pipeline = EnhancedDataLoader(config_manager)
    trading_config_adapter = TradingConfigAdapter()
    
    # Initialize data validator for real-time data quality checks
    data_validator = TrainingDataValidator(config_manager.config)
    
    # Initialize event bus for decoupled communication
    event_bus = EventBus()
    
    # Initialize market data processor for async data handling
    market_data_processor = MarketDataProcessor(
        config_manager=config_manager,
        data_validator=data_validator,
        event_bus=event_bus,
        max_workers=4
    )
    logger.info("✅ Market Data Processor initialized with async capabilities")
    
    # Initialize training monitor for live performance tracking
    training_monitor = TrainingMonitor(
        patience=TradingConstants.TRAINING_MONITOR_PATIENCE,
        min_delta=TradingConstants.TRAINING_MONITOR_MIN_DELTA,
        save_best=True,
        model_save_path=config_manager.get(TradingConstants.CONFIG_KEYS['ml_model_path']) + TradingConstants.DEFAULT_PATHS['live_best_model']
    )
    
    # Initialize curriculum learning manager for adaptive difficulty
    curriculum_manager = CurriculumLearningManager(config_manager.config)
    logger.info(f"🎓 Curriculum Learning initialized at {curriculum_manager.get_current_stage().level.name} level")
    
    # Log AURA AI system status
    logger.info("🤖 AURA AI Enhanced Components Loaded:")
    logger.info("   ✅ Advanced RL Model with Meta-Learning")
    logger.info("   ✅ Real-time Data Validation")
    logger.info("   ✅ Performance Monitoring & Early Stopping")
    logger.info("   ✅ Curriculum Learning System")
    logger.info("   ✅ Risk-Aware Decision Making")
    logger.info("   ✅ Multi-broker Support")
    logger.info("   ✅ Real-time Feature Engineering")
    
    return {
        'data_pipeline': data_pipeline,
        'trading_config_adapter': trading_config_adapter,
        'data_validator': data_validator,
        'event_bus': event_bus,
        'market_data_processor': market_data_processor,
        'training_monitor': training_monitor,
        'curriculum_manager': curriculum_manager
    }


def _initialize_trading_models(config_manager: SecureConfigManager, data_components: Dict[str, Any], 
                              risk_manager: Any) -> Tuple[Any, Any, Any]:
    """Initialize RL model, trading agent, and decision processor."""
    # Dynamically determine input_size for RL model
    data_pipeline = data_components['data_pipeline']
    trading_config_adapter = data_components['trading_config_adapter']
    curriculum_manager = data_components['curriculum_manager']
    
    # Load a small sample of data to get feature count
    symbols = config_manager.get(TradingConstants.DATA_SOURCE_SYMBOLS_CONFIG)
    primary_symbol = symbols[0]
    sample_df = data_pipeline.load_and_process_data(primary_symbol)
    
    if sample_df.empty:
        logger.critical(f"Could not load sample data for {primary_symbol} to determine RL model input size. Aborting.")
        sys.exit(1)
    
    # Exclude non-feature columns and the 'label' column
    features_for_rl_model = [col for col in sample_df.columns if col not in TradingConstants.EXCLUDED_FEATURE_COLUMNS]
    input_size = len(features_for_rl_model)
    if input_size == 0:
        logger.critical("No features found for RL model input. Aborting.")
        sys.exit(1)
    
    logger.info(f"✅ RL Model Input Size: {input_size} features")

    # Initialize RL model parameters
    action_size = TradingConstants.DEFAULT_ACTION_SIZE
    config_size = TradingConstants.DEFAULT_CONFIG_SIZE

    rl_model = SelfImprovingRLModel(
        input_size=input_size,
        hidden_size=TradingConstants.DEFAULT_HIDDEN_SIZE,
        action_size=action_size,
        config_size=config_size,
        learning_rate=config_manager.get(TradingConstants.CONFIG_KEYS['rl_learning_rate'], DEFAULT_LEARNING_RATE),
        gamma=config_manager.get(TradingConstants.CONFIG_KEYS['rl_gamma'], TradingConstants.DEFAULT_RL_GAMMA),
        tau=config_manager.get(TradingConstants.CONFIG_KEYS['rl_tau'], TradingConstants.DEFAULT_RL_TAU),
        buffer_size=config_manager.get(TradingConstants.CONFIG_KEYS['rl_buffer_size'], TradingConstants.DEFAULT_RL_BUFFER_SIZE),
        batch_size=config_manager.get(TradingConstants.CONFIG_KEYS['rl_batch_size'], DEFAULT_BATCH_SIZE),
        n_step=config_manager.get(TradingConstants.CONFIG_KEYS['rl_n_step'], TradingConstants.DEFAULT_RL_N_STEP),
        use_meta_learning=config_manager.get(TradingConstants.CONFIG_KEYS['rl_use_meta_learning'], True),
        use_curiosity=config_manager.get(TradingConstants.CONFIG_KEYS['rl_use_curiosity'], True),
        use_uncertainty=config_manager.get(TradingConstants.CONFIG_KEYS['rl_use_uncertainty'], True),
        use_auxiliary=config_manager.get(TradingConstants.CONFIG_KEYS['rl_use_auxiliary'], True),
        use_prioritized_replay=config_manager.get(TradingConstants.CONFIG_KEYS['rl_use_prioritized_replay'], True),
        curiosity_weight=config_manager.get(TradingConstants.CONFIG_KEYS['rl_curiosity_weight'], TradingConstants.DEFAULT_RL_CURIOSITY_WEIGHT),
        auxiliary_weight=config_manager.get(TradingConstants.CONFIG_KEYS['rl_auxiliary_weight'], TradingConstants.DEFAULT_RL_AUXILIARY_WEIGHT),
        device="cpu"
    )
    
    rl_model_path = config_manager.get('ml_model_config.model_path') + "/rl_model.pth"
    try:
        rl_model.load_model(rl_model_path)
        logger.info(f"Successfully loaded RL model from {rl_model_path}")
    except Exception as e:
        logger.error(f"Could not load RL model from {rl_model_path}: {e}. Starting with untrained model.", exc_info=True)

    trading_agent = TradingAgent(rl_model, trading_config_adapter)

    # Initialize DecisionProcessor for enhanced decision processing
    decision_processor = DecisionProcessor(
        trading_agent=trading_agent,
        risk_manager=risk_manager,
        curriculum_manager=curriculum_manager
    )
    logger.info("✅ DecisionProcessor initialized with comprehensive validation")
    
    return rl_model, trading_agent, decision_processor


async def _execute_trading_loop(broker: Any, order_executor: Any, portfolio_manager: Any, 
                               decision_processor: Any, data_components: Dict[str, Any],
                               trade_logger: Any, telegram_notifier: Any, primary_symbol: str,
                               interval_seconds: int, risk_manager: Any) -> None:
    """Execute the main trading loop with error handling."""
    market_data_processor = data_components['market_data_processor']
    training_monitor = data_components['training_monitor']
    curriculum_manager = data_components['curriculum_manager']
    
    # Initialize performance tracking variables
    cycle_count = 0
    episode_rewards = []
    current_episode_reward = 0.0
    last_portfolio_value = portfolio_manager.get_current_capital()
    
    try:
        while True:
            ts = datetime.now()
            cycle_count += 1
            logger.info(f"🔄 AURA AI Cycle {cycle_count} - {ts}")

            # Process market data and make trading decisions
            await _process_trading_cycle(
                market_data_processor=market_data_processor,
                broker=broker,
                primary_symbol=primary_symbol,
                decision_processor=decision_processor,
                portfolio_manager=portfolio_manager,
                order_executor=order_executor,
                telegram_notifier=telegram_notifier,
                risk_manager=risk_manager,
                interval_seconds=interval_seconds
            )
            
            # Update performance metrics
            current_portfolio_value = portfolio_manager.get_current_capital()
            cycle_reward = (current_portfolio_value - last_portfolio_value) / last_portfolio_value
            current_episode_reward += cycle_reward
            last_portfolio_value = current_portfolio_value
            
            # Update training monitor and curriculum every 10 cycles
            if cycle_count % 10 == 0:
                _update_performance_tracking(
                    episode_rewards=episode_rewards,
                    current_episode_reward=current_episode_reward,
                    training_monitor=training_monitor,
                    curriculum_manager=curriculum_manager,
                    portfolio_manager=portfolio_manager,
                    cycle_count=cycle_count,
                    telegram_notifier=telegram_notifier
                )
                current_episode_reward = 0.0

            await asyncio.sleep(interval_seconds)

    except KeyboardInterrupt:
        logger.info("User interrupt")
    except Exception as e:
        logger.critical(f"Fatal loop error: {e}", exc_info=True)
        if telegram_notifier.enabled:
            telegram_notifier.send_message(f"💥 CRITICAL: {e}")
    finally:
        try:
            trade_logger.export_to_excel()
        except Exception as e:
            logger.error(f"Excel export error: {e}")


async def _process_trading_cycle(market_data_processor: Any, broker: Any, primary_symbol: str,
                                decision_processor: Any, portfolio_manager: Any, order_executor: Any,
                                telegram_notifier: Any, risk_manager: Any,
                                interval_seconds: int) -> None:
    """Process a single trading cycle including data processing and decision making."""
    try:
        # Process market data
        market_data_result = await market_data_processor.process_market_data_pipeline(broker, primary_symbol)
        
        # Extract processed data
        ohlcv_df = market_data_result['ohlcv_data']
        current_state = market_data_result['current_state']
        is_valid = market_data_result['data_quality']['is_valid']
        issues = market_data_result['data_quality']['issues']
        
        logger.info(f"📊 Market data processed: {len(market_data_result['feature_columns'])} features, "
                   f"Quality: {' OK' if is_valid else ' Issues'}")
        
        # Handle data quality issues
        if not is_valid and issues:
            logger.warning(f"Data quality issues detected: {issues}")
            if telegram_notifier.enabled:
                telegram_notifier.send_message(f" Data Quality Alert: {issues[0][:100]}...")
        
        # Create trading context and process decision
        context = _create_trading_context(
            primary_symbol=primary_symbol,
            ohlcv_df=ohlcv_df,
            portfolio_manager=portfolio_manager,
            risk_manager=risk_manager,
            current_state=current_state,
            is_valid=is_valid
        )
        
        # Process decision with comprehensive validation
        decision = await decision_processor.process_decision(context)
        
        # Execute decision if valid
        _execute_trading_decision(
            decision=decision,
            order_executor=order_executor,
            portfolio_manager=portfolio_manager,
            telegram_notifier=telegram_notifier,
            ts=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Market data processing failed: {e}")
        if telegram_notifier.enabled:
            telegram_notifier.send_message(f" Market Data Error: {str(e)[:100]}")
        await asyncio.sleep(interval_seconds)


def _create_trading_context(primary_symbol: str, ohlcv_df: pd.DataFrame, portfolio_manager: Any,
                           risk_manager: Any, current_state: Any, is_valid: bool) -> Any:
    """Create comprehensive trading context for decision making."""
    current_portfolio_value = portfolio_manager.get_current_capital()
    
    # Calculate market conditions
    volatility = np.std(ohlcv_df['close'].pct_change().dropna()) if len(ohlcv_df) > 1 else 0.02
    volume = ohlcv_df.iloc[-1]['volume'] if 'volume' in ohlcv_df.columns else 1000.0
    
    # Detect market trend
    if len(ohlcv_df) >= 10:
        trend = 'bullish' if ohlcv_df['close'].iloc[-1] > ohlcv_df['close'].iloc[-10] else 'bearish'
    else:
        trend = 'sideways'
    
    # Get available capital and positions
    available_capital = current_portfolio_value * 0.5  # Assume 50% available for new positions
    open_positions = []  # portfolio_manager.get_open_positions() if method exists
    
    context = TradingContext(
        symbol=primary_symbol,
        current_price=ohlcv_df.iloc[-1]['close'],
        portfolio_value=current_portfolio_value,
        available_capital=available_capital,
        open_positions=open_positions,
        market_conditions={
            'volatility': volatility,
            'volume': volume,
            'trend': trend,
            'data_quality_valid': is_valid
        },
        risk_metrics=risk_manager.get_risk_summary(),
        features=current_state,
        market_data=ohlcv_df
    )
    
    return context


def _execute_trading_decision(decision: Any, order_executor: Any, portfolio_manager: Any,
                             telegram_notifier: Any, ts: datetime) -> None:
    """Execute trading decision with proper logging and error handling."""
    current_portfolio_value = portfolio_manager.get_current_capital()
    action_str = TradingConstants.get_action_name(decision.action)
    
    logger.info(f" Enhanced AI Decision: {action_str} | "
               f"Confidence: {decision.confidence:.3f} | "
               f"Risk Validated: {decision.risk_validated} | "
               f"Portfolio: ${current_portfolio_value:.2f}")
    
    if decision.reasoning:
        logger.info(f" Decision Reasoning: {decision.reasoning}")
    
    # Execute decision only if risk validated
    if decision.risk_validated and decision.action != TradingConstants.ACTION_HOLD:
        logger.info(f" Executing validated decision: {action_str} "
                   f"Quantity: {decision.quantity:.4f} @ ${decision.entry_price:.2f}")
        
        process_decision_action(
            decision.action, 
            decision.parameters, 
            decision.symbol,
            order_executor,
            portfolio_manager,
            telegram_notifier,
            ts
        )
    elif not decision.risk_validated:
        logger.warning(f" Decision rejected by risk management: {decision.risk_validation_message}")
        if telegram_notifier.enabled:
            telegram_notifier.send_message(
                f" Trade Rejected: {decision.risk_validation_message}"
            )
    else:
        logger.info("💤 Holding position - no action required")


def _update_performance_tracking(episode_rewards: List[float], current_episode_reward: float,
                                training_monitor: Any, curriculum_manager: Any, portfolio_manager: Any,
                                cycle_count: int, telegram_notifier: Any) -> None:
    """Update performance tracking metrics and curriculum learning."""
    episode_rewards.append(current_episode_reward)
    
    # Calculate max drawdown for this episode
    if len(episode_rewards) > 1:
        cumulative = np.cumsum(episode_rewards[-10:])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
    else:
        max_drawdown = 0
    
    # Create metrics for monitoring
    current_portfolio_value = portfolio_manager.get_current_capital()
    metrics = {
        'total_reward': current_episode_reward,
        'average_loss': 0.0,
        'learning_rate': 0.001,  # Default value
        'portfolio_value': current_portfolio_value,
        'cycle_count': cycle_count
    }
    
    # Update monitor and curriculum
    should_stop = training_monitor.update(metrics)
    if should_stop:
        logger.info(" Training monitor suggests stopping due to lack of improvement")
    
    # Update curriculum learning progress
    advanced = curriculum_manager.update_progress(
        episode_reward=current_episode_reward,
        episode_drawdown=max_drawdown
    )
    
    if advanced:
        _handle_curriculum_advancement(curriculum_manager, telegram_notifier)
    
    # Log performance summary
    _log_performance_summary(training_monitor, curriculum_manager, cycle_count, 
                            current_portfolio_value, telegram_notifier)


def _handle_curriculum_advancement(curriculum_manager: Any, telegram_notifier: Any) -> None:
    """Handle curriculum learning stage advancement."""
    stage_info = curriculum_manager.get_current_stage()
    logger.info(f"🎓 Advanced to curriculum stage: {stage_info.level.name}")
    
    # Get adjusted parameters for new difficulty level
    adjustments = curriculum_manager.get_training_config_adjustments()
    
    # Apply adjustments to trading agent if needed
    if 'learning_rate_multiplier' in adjustments:
        new_lr = 0.001 * adjustments['learning_rate_multiplier']  # Default base learning rate
        logger.info(f"Adjusting learning rate to {new_lr:.6f} for new curriculum stage")
    
    # Notify about curriculum advancement
    if telegram_notifier.enabled:
        telegram_notifier.send_message(
            f"🎓 AURA AI Advanced to {stage_info.level.name} level!\n"
            f"New challenges: {', '.join(stage_info.market_conditions)}\n"
            f"Target reward: {stage_info.success_threshold:.4f}"
        )


def _log_performance_summary(training_monitor: Any, curriculum_manager: Any, cycle_count: int,
                            current_portfolio_value: float, telegram_notifier: Any) -> None:
    """Log performance summary and send periodic updates."""
    summary = training_monitor.get_training_summary()
    curriculum_summary = curriculum_manager.get_curriculum_summary()
    
    logger.info(f" Performance Summary: Avg Reward: {summary.get('recent_avg_reward', 0):.4f}, "
               f"Best Score: {summary.get('best_validation_score', 0):.4f}, "
               f"Curriculum Level: {curriculum_summary.get('current_level', 'BEGINNER')}")
    
    # Send periodic performance update
    if telegram_notifier.enabled and cycle_count % 50 == 0:  # Every 50 cycles
        telegram_notifier.send_message(
            f" AURA AI Performance Update\n"
            f"Cycles: {cycle_count}\n"
            f"Portfolio: ${current_portfolio_value:.2f}\n"
            f"Recent Avg Reward: {summary.get('recent_avg_reward', 0):.4f}\n"
            f"Win Rate: {summary.get('win_rate', 0):.2%}\n"
            f"Curriculum: {curriculum_summary.get('current_level', 'BEGINNER')} "
            f"({curriculum_summary.get('progress_percentage', 0):.1f}% complete)"
        )


if __name__ == "__main__":
    main()





