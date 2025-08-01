"""
Data validation and preprocessing for AURA AI training.
Ensures high-quality, consistent data for optimal AI performance.
"""
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler, RobustScaler

logger = logging.getLogger(__name__)

class TrainingDataValidator:
    """Validates and preprocesses data for AI training."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.feature_stats = {}
        
    def validate_data_quality(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Comprehensive data quality validation."""
        issues = []
        
        # Check for minimum data requirements
        min_samples = self.config.get('training', {}).get('min_samples', 1000)
        if len(df) < min_samples:
            issues.append(f"Insufficient data: {len(df)} < {min_samples} required")
            
        # Check for missing values
        missing_pct = df.isnull().sum() / len(df) * 100
        critical_missing = missing_pct[missing_pct > 5]  # More than 5% missing
        if not critical_missing.empty:
            issues.append(f"High missing values: {critical_missing.to_dict()}")
            
        # Check for data consistency
        if 'close' in df.columns and (df['close'] <= 0).any():
            issues.append("Invalid price data: non-positive close prices found")
                
        # Check for extreme outliers (beyond 6 standard deviations)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            extreme_outliers = (z_scores > 6).sum()
            if extreme_outliers > len(df) * 0.01:  # More than 1% extreme outliers
                issues.append(f"Excessive outliers in {col}: {extreme_outliers} samples")
                
        return len(issues) == 0, issues
        
    def preprocess_features(self, df: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """Advanced feature preprocessing for AI training."""
        df_processed = df.copy()
        
        # Handle infinite values
        df_processed = df_processed.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill then backward fill for missing values
        df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')
        
        # Identify feature columns (exclude OHLCV and metadata)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'label']
        feature_cols = [col for col in df_processed.columns if col not in exclude_cols]
        
        if feature_cols and fit_scaler:
            # Fit scaler on training data
            scaled_features = self.scaler.fit_transform(df_processed[feature_cols])
            # Store feature statistics for monitoring
            self.feature_stats = {
                'mean': df_processed[feature_cols].mean().to_dict(),
                'std': df_processed[feature_cols].std().to_dict(),
                'min': df_processed[feature_cols].min().to_dict(),
                'max': df_processed[feature_cols].max().to_dict()
            }
        elif feature_cols:
            # Transform using existing scaler
            scaled_features = self.scaler.transform(df_processed[feature_cols])
            
        # Replace original features with scaled versions
        if feature_cols:
            df_processed[feature_cols] = scaled_features
            
        return df_processed
        
    def create_training_splits(self, df: pd.DataFrame, 
                             train_ratio: float = 0.7, 
                             val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create time-aware train/validation/test splits."""
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        logger.info(f"Data splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df