import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load data from CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the loaded data
    """
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the data by handling missing values and packet loss events.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Add is_packet_loss column
    df_clean['is_packet_loss'] = (df['delay_ms'] == -1).astype(int)
    
    # For feature engineering, we'll keep the -1 values
    # This way we can properly identify packet loss events in the future window
    return df_clean

def create_sliding_windows(df: pd.DataFrame, 
                         lookback_window: int = 10,
                         prediction_window: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows for feature engineering.
    
    Args:
        df: Input DataFrame
        lookback_window: Number of past values to use for prediction
        prediction_window: Number of future values to check for packet loss
        
    Returns:
        Tuple of (feature windows, labels)
    """
    delay_values = df['delay_ms'].values
    n_samples = len(delay_values) - lookback_window - prediction_window + 1
    
    # Initialize arrays
    feature_windows = np.zeros((n_samples, lookback_window))
    labels = np.zeros(n_samples)
    
    # Create windows
    for i in range(n_samples):
        # Get window of past values
        feature_windows[i] = delay_values[i:i+lookback_window]
        
        # Check if there's a packet loss in the prediction window
        future_window = delay_values[i+lookback_window:i+lookback_window+prediction_window]
        labels[i] = 1 if np.any(future_window == -1) else 0
    
    return feature_windows, labels

def extract_statistical_features(features: np.ndarray) -> np.ndarray:
    """
    Extract statistical features from the sliding windows.
    
    Args:
        features (np.ndarray): Raw features from sliding windows
        
    Returns:
        np.ndarray: Statistical features
    """
    statistical_features = []
    
    for window in features:
        stats = [
            np.mean(window),
            np.std(window),
            np.min(window),
            np.max(window),
            np.percentile(window, 25),
            np.percentile(window, 75),
            np.median(window)
        ]
        statistical_features.append(stats)
    
    return np.array(statistical_features) 