"""
Label generation module for AURA AI.
Generates trading signals based on future price movements.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def generate_labels(df, lookahead_minutes=30):
    """
    Generates labels (Buy, Sell, Hold) based on future price movements,
    incorporating ATR-based dynamic thresholds and an intraday trading constraint.
    - Buy (1): If price increases by profit_target_pct within lookahead_minutes without hitting stop_loss_pct.
    - Sell (-1): If price decreases by profit_target_pct within lookahead_minutes without hitting stop_loss_pct.
    - Hold (0): Otherwise.
    """
    logger.info("Starting label generation...")
    if df.empty:
        logger.warning("Input DataFrame is empty. Returning with no labels generated.")
        df['label'] = 0
        return df

    try:
        df = df.copy()
        df['label'] = 0 # Default to Hold

        # Calculate future max/min prices within the lookahead window
        for i in range(1, lookahead_minutes // 5 + 1): # 5-minute bars
            df[f'future_high_{i}'] = df['high'].shift(-i)
            df[f'future_low_{i}'] = df['low'].shift(-i)
        logger.debug("Future high/low columns created.")

        # Combine future high/low into a single series for easier comparison
        future_high = df[[f'future_high_{i}' for i in range(1, lookahead_minutes // 5 + 1)]].max(axis=1)
        future_low = df[[f'future_low_{i}' for i in range(1, lookahead_minutes // 5 + 1)]].min(axis=1)
        logger.debug("Combined future high/low series.")

        # Define ATR-based dynamic profit and stop-loss thresholds
        ATR_MULTIPLIER_PROFIT = 1.2
        ATR_MULTIPLIER_STOP_LOSS = 0.8

        if 'ATR_14' in df.columns:
            profit_threshold_buy = df['close'] + (df['ATR_14'] * ATR_MULTIPLIER_PROFIT)
            stop_loss_threshold_buy = df['close'] - (df['ATR_14'] * ATR_MULTIPLIER_STOP_LOSS)

            profit_threshold_sell = df['close'] - (df['ATR_14'] * ATR_MULTIPLIER_PROFIT)
            stop_loss_threshold_sell = df['close'] + (df['ATR_14'] * ATR_MULTIPLIER_STOP_LOSS)
            logger.debug("ATR-based profit/stop-loss thresholds calculated.")
        else:
            logger.warning("ATR_14 not found in DataFrame. Falling back to fixed profit/stop-loss percentages.")
            profit_threshold_buy = df['close'] * (1 + 0.001)
            stop_loss_threshold_buy = df['close'] * (1 - 0.0005)

            profit_threshold_sell = df['close'] * (1 - 0.001)
            stop_loss_threshold_sell = df['close'] * (1 + 0.0005)

        # Determine Buy/Sell/Hold labels
        buy_condition = (future_high >= profit_threshold_buy) & (future_low > stop_loss_threshold_buy)
        df.loc[buy_condition, 'label'] = 1

        sell_condition = (future_low <= profit_threshold_sell) & (future_high < stop_loss_threshold_sell)
        df.loc[sell_condition, 'label'] = -1
        logger.debug("Buy/Sell labels assigned.")

        # Apply intraday trading constraints
        df.loc[(df.index.hour == 15) & (df.index.minute >= 15), 'label'] = 0 # Force hold/exit
        logger.debug("Intraday closing time constraint applied.")

        # Clean up temporary columns
        df = df.drop(columns=[f'future_high_{i}' for i in range(1, lookahead_minutes // 5 + 1)] +
                             [f'future_low_{i}' for i in range(1, lookahead_minutes // 5 + 1)], errors='ignore')
        logger.info("Label generation completed successfully.")
        return df

    except Exception as e:
        logger.error(f"Error during label generation: {e}", exc_info=True)
        df['label'] = 0 # Ensure label column exists even on error
        return df