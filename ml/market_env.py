import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from src.strategy.decision_maker import DecisionMaker
from src.trading.trade_manager import TradeManager
from src.rules import risk_management # Import risk management module

class MarketEnv(gym.Env):
    """
    A custom trading environment for reinforcement learning, compliant with the Gymnasium API.
    This environment simulates trading on historical market data, incorporating realistic
    PnL calculation, transaction costs, and basic risk management (SL/TP).
    """
    def __init__(self, df: pd.DataFrame, features: list, initial_capital=100000, transaction_cost_pct=0.001, decision_maker: DecisionMaker = None, teacher_model = None):
        """
        Initializes the MarketEnv.

        Args:
            df (pd.DataFrame): The DataFrame containing historical market data and pre-calculated features.
            features (list): A list of column names to be used as the observation space.
            initial_capital (int, optional): The starting capital for the simulation. Defaults to 100000.
            transaction_cost_pct (float, optional): Percentage cost per trade (e.g., 0.001 for 0.1%). Defaults to 0.001.
            decision_maker (DecisionMaker, optional): An instance of the DecisionMaker for reward shaping. Defaults to None.
        """
        super().__init__()
        self.df = df
        self.features = features
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.decision_maker = decision_maker
        self.teacher_model = teacher_model
        
        # Initialize TradeManager
        self.trade_manager = TradeManager(initial_capital=initial_capital, commission_rate=transaction_cost_pct)
        
        # Risk management parameters (can be configured or learned by RL agent)
        self.risk_params = {
            'max_dd_pct': 5.0, # Max drawdown percentage
            'max_loss_pct': 2.0, # Max daily loss percentage
            'max_streak': 3, # Max consecutive losses
            'max_risk_per_trade_pct': 1.0, # Max risk per trade as % of capital
            'max_exposure_pct': 20.0, # Max exposure per trade as % of capital
            'kill_switch_loss_pct': 4.0, # Kill switch daily loss percentage
            'kill_switch_consecutive_losses_streak': 5 # Kill switch consecutive losses
        }

        # These will now be managed by TradeManager
        self.capital = self.trade_manager.current_capital
        self.shares_held = self.trade_manager.position
        self.position_type = 0 # 0: None, 1: Long, -1: Short (derived from trade_manager.position)
        self.entry_price = self.trade_manager.entry_price
        self.current_step = 0
        self.consecutive_losses = 0

        # ACTION SPACE: 0 = HOLD, 1 = BUY, 2 = SELL
        self.action_space = spaces.Discrete(3)
        
        # OBSERVATION SPACE: The financial features at each timestep
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(features),), dtype=np.float32
        )

    def _get_state(self) -> np.ndarray:
        """Retrieves the state (features) for the current timestep."""
        # Ensure we don't go out of bounds
        if self.current_step >= len(self.df):
            return np.zeros(len(self.features), dtype=np.float32) # Return a zero state if episode is done
        return self.df.iloc[self.current_step][self.features].values.astype(np.float32)

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """
        Resets the environment to its initial state.
        Returns the initial observation and an info dictionary.
        """
        super().reset(seed=seed)
        self.trade_manager.reset()
        self.capital = self.trade_manager.current_capital
        self.shares_held = self.trade_manager.position
        self.position_type = 0
        self.entry_price = 0
        self.consecutive_losses = 0
        self.current_step = 0
        return self._get_state(), {"capital": self.capital, "shares_held": self.shares_held, "current_portfolio_value": self.initial_capital}

    def step(self, action: int, config_params: np.ndarray = None) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Executes one step in the environment.

        Args:
            action (int): The action to take (0: HOLD, 1: BUY, 2: SELL).
            config_params (np.ndarray): Array of self-tuned trading parameters from RL model
                                        (e.g., stop_loss_pct, profit_target_pct, risk_per_trade_pct).

        Returns:
            tuple: A tuple containing the next state, reward, done flag, truncated flag, and info dict.
        """
        # Check if we've reached the end of data
        if self.current_step >= len(self.df):
            return self._get_state(), 0.0, True, False, {
                "capital": self.trade_manager.current_capital, 
                "shares_held": self.trade_manager.position, 
                "current_portfolio_value": self.trade_manager.get_net_worth(self.df.iloc[self.current_step-1]['close']), 
                "reason": "END_OF_DATA"
            }

        # Get current market data
        current_timestamp, current_price, high_price, low_price = self._get_current_market_data()
        
        # Parse trading parameters from RL model
        trade_params = self._parse_config_params(config_params)
        
        # Initialize step variables
        reward, info = self._initialize_step_variables()
        previous_net_worth = self.trade_manager.get_net_worth(current_price)
        pnl_before_action = self.trade_manager.realized_pnl

        # Check risk management kill switch
        kill_switch_triggered, kill_switch_result = self._check_kill_switch()
        if kill_switch_triggered:
            return kill_switch_result

        # Apply reward shaping if teacher model is available
        if self.decision_maker and self.teacher_model:
            reward += self._apply_reward_shaping(action)

        # Handle position management (stop loss, take profit, explicit closes)
        position_closed_this_step = self._handle_position_management(
            current_timestamp, current_price, high_price, low_price, 
            trade_params, action, info
        )

        # Execute new actions if no position or position was closed
        if not position_closed_this_step and self.trade_manager.position == 0:
            self._execute_new_action(action, trade_params, current_timestamp, 
                                   current_price, reward, info)

        # Finalize step
        return self._finalize_step(current_price, previous_net_worth, 
                                 pnl_before_action, reward, info)

    def _get_current_market_data(self) -> tuple:
        """Get current market data for the current step."""
        current_timestamp = self.df.index[self.current_step]
        current_price = self.df.iloc[self.current_step]['close']
        high_price = self.df.iloc[self.current_step]['high']
        low_price = self.df.iloc[self.current_step]['low']
        return current_timestamp, current_price, high_price, low_price

    def _parse_config_params(self, config_params: np.ndarray) -> dict:
        """Parse and scale config parameters from RL model."""
        if config_params is not None:
            return {
                'stop_loss_pct': 0.005 + (config_params[0] + 1) * 0.0225,  # 0.5% to 5%
                'profit_target_pct': 0.01 + (config_params[1] + 1) * 0.045, # 1% to 10%
                'risk_per_trade_pct': 0.001 + (config_params[2] + 1) * 0.0095 # 0.1% to 2%
            }
        else:
            return {
                'stop_loss_pct': 0.02,
                'profit_target_pct': 0.04,
                'risk_per_trade_pct': 0.01
            }

    def _initialize_step_variables(self) -> tuple[float, dict]:
        """Initialize variables for the current step."""
        reward = 0.0
        info = {
            "capital": self.trade_manager.current_capital, 
            "shares_held": self.trade_manager.position, 
            "realized_pnl": 0.0, 
            "action_taken": "HOLD"
        }
        return reward, info

    def _check_kill_switch(self) -> tuple[bool, tuple]:
        """Check if risk management kill switch should be triggered."""
        current_pnl_today = self.trade_manager.realized_pnl
        
        if not risk_management.risk_portfolio_kill_switch(
            current_pnl_today, 
            self.trade_manager.initial_capital,
            self.consecutive_losses, 
            self.risk_params
        ):
            reward = -100.0  # Large penalty for hitting kill switch
            info = {
                "capital": self.trade_manager.current_capital,
                "shares_held": self.trade_manager.position,
                "reason": "KILL_SWITCH_HIT",
                "action_taken": "KILL_SWITCH"
            }
            return True, (self._get_state(), reward, True, False, info)
        
        return False, None

    def _apply_reward_shaping(self, action: int) -> float:
        """Apply reward shaping based on teacher model alignment."""
        # Get teacher's action
        state_df = self.df.iloc[[self.current_step]]
        features = self.teacher_model.predict(state_df[self.features])
        ml_signal = 1 if features[0] > 0.5 else -1
        ml_confidence = abs(features[0] - 0.5) * 2
        
        teacher_action, _, _ = self.decision_maker.make_decision(
            ml_signal=ml_signal,
            ml_confidence=ml_confidence,
            current_position={},
            ohlcv_df=self.df,
            processed_features_df=self.df,
            symbol="BTCUSDT",
            capital=self.trade_manager.current_capital,
            pnl_today=0,
            consecutive_losses=0,
            open_positions_count=0,
            total_exposure_value=0,
            last_trade_details={},
            portfolio_risk_state={}
        )

        # Provide bonus for aligning with teacher
        if ((action == 1 and teacher_action == "BUY") or 
            (action == 2 and teacher_action == "SELL") or 
            (action == 0 and teacher_action == "HOLD")):
            return 0.1  # Alignment bonus
        
        return 0.0

    def _handle_position_management(self, current_timestamp, current_price: float, 
                                  high_price: float, low_price: float, 
                                  trade_params: dict, action: int, info: dict) -> bool:
        """Handle existing position management (SL/TP/explicit closes)."""
        if self.trade_manager.position == 0:
            return False

        position_closed = False
        current_position_type = 1 if self.trade_manager.position > 0 else -1
        
        # Calculate stop loss and take profit prices
        sl_price, tp_price = self._calculate_sl_tp_prices(current_position_type, trade_params)
        
        # Check for SL/TP hits
        position_closed = self._check_sl_tp_hits(
            current_timestamp, current_position_type, sl_price, tp_price, 
            high_price, low_price, info
        )
        
        # Check for explicit close
        if not position_closed:
            position_closed = self._check_explicit_close(
                current_timestamp, current_price, current_position_type, action, info
            )
        
        return position_closed

    def _calculate_sl_tp_prices(self, position_type: int, trade_params: dict) -> tuple[float, float]:
        """Calculate stop loss and take profit prices."""
        if position_type == 1:  # Long position
            sl_price = self.trade_manager.entry_price * (1 - trade_params['stop_loss_pct'])
            tp_price = self.trade_manager.entry_price * (1 + trade_params['profit_target_pct'])
        else:  # Short position
            sl_price = self.trade_manager.entry_price * (1 + trade_params['stop_loss_pct'])
            tp_price = self.trade_manager.entry_price * (1 - trade_params['profit_target_pct'])
        
        return sl_price, tp_price

    def _check_sl_tp_hits(self, current_timestamp, position_type: int, sl_price: float, 
                         tp_price: float, high_price: float, low_price: float, 
                         info: dict) -> bool:
        """Check if stop loss or take profit was hit."""
        if position_type == 1:  # Long
            if low_price <= sl_price:
                self.trade_manager.close_position(current_timestamp, sl_price)
                info["reason"] = "SL_HIT"
                return True
            elif high_price >= tp_price:
                self.trade_manager.close_position(current_timestamp, tp_price)
                info["reason"] = "TP_HIT"
                return True
        else:  # Short
            if high_price >= sl_price:
                self.trade_manager.close_position(current_timestamp, sl_price)
                info["reason"] = "SL_HIT"
                return True
            elif low_price <= tp_price:
                self.trade_manager.close_position(current_timestamp, tp_price)
                info["reason"] = "TP_HIT"
                return True
        
        return False

    def _check_explicit_close(self, current_timestamp, current_price: float, 
                            position_type: int, action: int, info: dict) -> bool:
        """Check if agent explicitly closes the position."""
        if ((position_type == 1 and action == 2) or 
            (position_type == -1 and action == 1)):
            self.trade_manager.close_position(current_timestamp, current_price)
            info["reason"] = "EXPLICIT_CLOSE"
            return True
        
        return False

    def _execute_new_action(self, action: int, trade_params: dict, current_timestamp, 
                          current_price: float, reward: float, info: dict) -> None:
        """Execute new trading actions (BUY/SELL)."""
        if action == 1:  # BUY
            self._execute_buy_action(trade_params, current_timestamp, current_price, 
                                   reward, info)
        elif action == 2:  # SELL
            self._execute_sell_action(trade_params, current_timestamp, current_price, 
                                    reward, info)

    def _execute_buy_action(self, trade_params: dict, current_timestamp, 
                          current_price: float, reward: float, info: dict) -> None:
        """Execute a BUY action with risk checks."""
        info["action_taken"] = "BUY"
        
        # Calculate position size
        amount_to_buy_usdt = self._calculate_position_size(trade_params, 'buy')
        
        # Risk checks
        if self._validate_trade_risk(amount_to_buy_usdt, current_price, trade_params, 'buy'):
            if amount_to_buy_usdt > 0:
                self.trade_manager.buy(current_timestamp, current_price, amount_to_buy_usdt)
        else:
            # Note: reward penalty would be applied here if reward was mutable
            info["reason"] = "RISK_VIOLATION_BUY"
            info["action_taken"] = "BLOCKED_BUY"

    def _execute_sell_action(self, trade_params: dict, current_timestamp, 
                           current_price: float, reward: float, info: dict) -> None:
        """Execute a SELL action with risk checks."""
        info["action_taken"] = "SELL"
        
        # Calculate position size
        amount_to_sell_usdt = self._calculate_position_size(trade_params, 'sell')
        
        # Risk checks
        if self._validate_trade_risk(amount_to_sell_usdt, current_price, trade_params, 'sell'):
            if amount_to_sell_usdt > 0:
                self.trade_manager.sell(current_timestamp, current_price, 
                                      amount_to_sell_usdt / current_price)
        else:
            # Note: reward penalty would be applied here if reward was mutable
            info["reason"] = "RISK_VIOLATION_SELL"
            info["action_taken"] = "BLOCKED_SELL"

    def _calculate_position_size(self, trade_params: dict, action_type: str) -> float:
        """Calculate position size based on risk parameters."""
        risk_amount = self.initial_capital * trade_params['risk_per_trade_pct']
        
        if trade_params['stop_loss_pct'] > 0:
            amount_usdt = risk_amount / trade_params['stop_loss_pct']
        else:
            amount_usdt = self.trade_manager.current_capital * 0.1
        
        # Ensure we don't exceed available capital
        return min(amount_usdt, 
                  self.trade_manager.current_capital * (1 - self.transaction_cost_pct))

    def _validate_trade_risk(self, amount_usdt: float, current_price: float, 
                           trade_params: dict, action_type: str) -> bool:
        """Validate trade against risk management rules."""
        potential_quantity = amount_usdt / current_price
        
        if action_type == 'buy':
            potential_stop_loss_price = current_price * (1 - trade_params['stop_loss_pct'])
        else:  # sell
            potential_stop_loss_price = current_price * (1 + trade_params['stop_loss_pct'])
        
        order_value = potential_quantity * current_price
        
        # Check risk per trade
        risk_per_trade_ok = risk_management.risk_per_trade_pct(
            self.trade_manager.current_capital,
            current_price,
            potential_stop_loss_price,
            potential_quantity,
            {'max_risk_pct': self.risk_params['max_risk_per_trade_pct']}
        )
        
        # Check max exposure
        exposure_ok = risk_management.risk_max_exposure_per_trade_pct(
            order_value,
            self.trade_manager.current_capital,
            {'max_exposure_pct': self.risk_params['max_exposure_pct']}
        )
        
        return risk_per_trade_ok and exposure_ok

    def _finalize_step(self, current_price: float, previous_net_worth: float, 
                      pnl_before_action: float, reward: float, info: dict) -> tuple:
        """Finalize the step and return results."""
        # Advance step
        self.current_step += 1
        done = self.current_step >= len(self.df)
        
        # Calculate reward based on net worth change
        current_net_worth = self.trade_manager.get_net_worth(current_price)
        reward += (current_net_worth - previous_net_worth) / self.initial_capital

        # Update consecutive losses tracking
        self._update_consecutive_losses(pnl_before_action)
        
        # Update info dictionary
        self._update_info_dict(info, current_net_worth, current_price)
        
        return self._get_state(), reward, done, False, info

    def _update_consecutive_losses(self, pnl_before_action: float) -> None:
        """Update consecutive losses counter."""
        pnl_after_action = self.trade_manager.realized_pnl
        if pnl_after_action != pnl_before_action:  # A trade was closed
            if pnl_after_action < pnl_before_action:  # Loss realized
                self.consecutive_losses += 1
            else:  # Profit realized
                self.consecutive_losses = 0

    def _update_info_dict(self, info: dict, current_net_worth: float, 
                         current_price: float) -> None:
        """Update the info dictionary with current state."""
        info.update({
            "capital": self.trade_manager.current_capital,
            "shares_held": self.trade_manager.position,
            "current_portfolio_value": current_net_worth,
            "current_price": current_price,
            "realized_pnl": self.trade_manager.realized_pnl
        })
