import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing import load_data, clean_data
from src.feature_engineering import create_feature_matrix

def test_random_forest():
    """Test Random Forest model on a single dataset."""
    # Load config
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Load and process a single file
    file_path = Path(config['paths']['dataset']['first_capture']) / 'cpe_a-cpe_b-fiber.csv'
    print(f"\nProcessing {file_path.name}...")
    
    # Load and clean data
    df = load_data(file_path)
    print("\nRaw data sample:")
    print(df.head())
    print("\nRaw data info:")
    print(df.info())
    
    # Check for packet loss in raw data
    raw_packet_loss = (df['delay_ms'] == -1).sum()
    print(f"\nPacket loss in raw data: {raw_packet_loss}")
    
    df_clean = clean_data(df)
    print("\nCleaned data sample:")
    print(df_clean.head())
    print("\nCleaned data info:")
    print(df_clean.info())
    
    # Check for packet loss in cleaned data
    clean_packet_loss = (df_clean['delay_ms'] == -1).sum()
    print(f"\nPacket loss in cleaned data: {clean_packet_loss}")
    
    # Create features
    print("\nCreating features...")
    features = create_feature_matrix(df_clean['delay_ms'].values)
    X = features['feature_matrix']
    y = features['labels']
    
    # Print data statistics
    print("\nData Statistics:")
    print(f"Total samples: {len(X)}")
    print(f"Positive samples: {sum(y)}")
    print(f"Positive sample ratio: {sum(y)/len(y)*100:.2f}%")
    
    # Print feature matrix info
    print("\nFeature Matrix Info:")
    print(f"Shape: {X.shape}")
    print("Sample of first row:")
    print(X[0])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\nTraining set statistics:")
    print(f"Training samples: {len(X_train)}")
    print(f"Training positive samples: {sum(y_train)}")
    print(f"Training positive ratio: {sum(y_train)/len(y_train)*100:.2f}%")
    
    # Train Random Forest
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }
    
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    # Print prediction distribution
    print("\nPrediction Distribution:")
    print(f"Predicted positive: {sum(y_pred)}")
    print(f"Predicted negative: {len(y_pred) - sum(y_pred)}")
    
    # Feature importance
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("--------------  rf  -----------------")
    print("\nTop 10 Most Important Features:")
    for i in range(10):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances')
    plt.bar(range(10), importances[indices[:10]])
    plt.xticks(range(10), [feature_names[i] for i in indices[:10]], rotation=45)
    plt.tight_layout()
    plt.savefig('results/rf_feature_importance.png')
    plt.close()
    
    return metrics

if __name__ == "__main__":
    test_random_forest() 