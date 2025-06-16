import numpy as np
from typing import List, Dict
from scipy import stats
from scipy.signal import find_peaks

def extract_time_domain_features(window: np.ndarray) -> Dict[str, float]:
    """Extract time domain features from a window of delay values."""
    features = {}
    
    # Handle packet loss in window
    valid_values = window[window != -1]
    if len(valid_values) == 0:
        # If all values are packet loss, use default values
        features.update({
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'range': 0.0,
            'slope': 0.0,
            'median': 0.0,
            'q1': 0.0,
            'q3': 0.0,
            'iqr': 0.0,
            'peak_count': 0
        })
        return features
    
    # Basic statistics
    features['mean'] = np.mean(valid_values)
    features['std'] = np.std(valid_values)
    features['min'] = np.min(valid_values)
    features['max'] = np.max(valid_values)
    features['range'] = features['max'] - features['min']
    
    # Trend features
    if len(valid_values) > 1:
        features['slope'] = np.polyfit(np.arange(len(valid_values)), valid_values, 1)[0]
    else:
        features['slope'] = 0.0
    
    # Additional statistics
    features['median'] = np.median(valid_values)
    features['q1'] = np.percentile(valid_values, 25)
    features['q3'] = np.percentile(valid_values, 75)
    features['iqr'] = features['q3'] - features['q1']
    
    # Peak features
    peaks, _ = find_peaks(valid_values, height=np.mean(valid_values))
    features['peak_count'] = len(peaks)
    
    return features

def extract_advanced_features(window: np.ndarray) -> Dict[str, float]:
    """Extract advanced features from a window of delay values."""
    features = {}
    
    # Handle packet loss in window
    valid_values = window[window != -1]
    if len(valid_values) < 5:
        features.update({
            'ma_diff': 0.0,
            'rate_of_change': 0.0,
            'diff_variance': 0.0
        })
        return features
    
    # Moving average features
    ma_short = np.convolve(valid_values, np.ones(3)/3, mode='valid')
    ma_long = np.convolve(valid_values, np.ones(5)/5, mode='valid')
    min_len = min(len(ma_short), len(ma_long))
    features['ma_diff'] = np.mean(ma_short[:min_len] - ma_long[:min_len])
    
    # Rate of change
    if len(valid_values) > 1:
        features['rate_of_change'] = np.mean(np.diff(valid_values))
    else:
        features['rate_of_change'] = 0.0
    
    # Variance of differences
    if len(valid_values) > 1:
        features['diff_variance'] = np.var(np.diff(valid_values))
    else:
        features['diff_variance'] = 0.0
    
    return features

def create_feature_vector(window: np.ndarray) -> np.ndarray:
    """Create a feature vector from a window of delay values."""
    # Extract features
    time_features = extract_time_domain_features(window)
    advanced_features = extract_advanced_features(window)
    
    # Combine all features
    all_features = {**time_features, **advanced_features}
    
    # Convert to array
    feature_vector = np.array(list(all_features.values()))
    
    return feature_vector

def create_feature_matrix(delay_values: np.ndarray, 
                         lookback_window: int = 10,
                         prediction_window: int = 5) -> Dict[str, np.ndarray]:
    """
    Create feature matrix and labels from delay values.
    
    Args:
        delay_values: Array of delay values
        lookback_window: Number of past values to use for prediction
        prediction_window: Number of future values to check for packet loss
        
    Returns:
        Dictionary containing feature matrix and labels
    """
    # Ensure delay_values is a numpy array
    delay_values = np.array(delay_values)
    
    # Calculate number of samples
    n_samples = len(delay_values) - lookback_window - prediction_window + 1
    if n_samples <= 0:
        raise ValueError(f"Not enough data points. Need at least {lookback_window + prediction_window} points.")
    
    # Get number of features from a sample window
    sample_window = delay_values[:lookback_window]
    n_features = len(create_feature_vector(sample_window))
    
    # Initialize arrays
    feature_matrix = np.zeros((n_samples, n_features))
    labels = np.zeros(n_samples)
    
    # Create feature matrix and labels
    for i in range(n_samples):
        # Get window of past values
        window = delay_values[i:i+lookback_window]
        
        # Create feature vector
        feature_matrix[i] = create_feature_vector(window)
        
        # Check if there's a packet loss in the prediction window
        future_window = delay_values[i+lookback_window:i+lookback_window+prediction_window]
        labels[i] = 1 if np.any(future_window == -1) else 0
    
    # Print label statistics
    print(f"\nLabel Statistics:")
    print(f"Total samples: {len(labels)}")
    print(f"Positive samples: {sum(labels)}")
    print(f"Positive ratio: {sum(labels)/len(labels)*100:.2f}%")
    
    return {
        'feature_matrix': feature_matrix,
        'labels': labels
    }