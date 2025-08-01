import logging
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class StrategyRecommender:
    def __init__(self):
        self.mock_recommendations = {
            "BTCUSDT": 1,
            "ETHUSDT": 0,
            "SOLUSDT": 5,
            "XRPUSDT": 0,
            "BNBUSDT": 1
        }
        logger.info("StrategyRecommender initialized with mock recommendations.")

    def recommend_action(self, symbol: str, _features_df: pd.DataFrame) -> int:
        return self.mock_recommendations.get(symbol, 0)