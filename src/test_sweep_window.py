import os
from pathlib import Path
import pandas as pd
import numpy as np
from src.preprocessing import load_data, clean_data
from src.feature_engineering import create_feature_matrix
from src.model_training import ModelTrainer

def get_all_delay_values():
    """只做一次数据清洗和合并，返回所有delay_ms序列"""
    dataset_dir = Path("dataset")
    all_delay_series = []
    for capture_dir in ["1st_capture", "2nd_capture"]:
        capture_path = dataset_dir / capture_dir
        if not capture_path.exists():
            continue
        for csv_file in capture_path.glob("*.csv"):
            try:
                df = load_data(csv_file)
                df_cleaned = clean_data(df)
                all_delay_series.append(df_cleaned['delay_ms'].values)
            except Exception as e:
                print(f"Error processing {csv_file.name}: {str(e)}")
                continue
    if all_delay_series:
        delay_values = np.concatenate(all_delay_series)
        return delay_values
    else:
        raise RuntimeError("No valid data found in any files")

def sweep_windows(N_list, X_list, delay_values):
    results = []
    for N in N_list:
        for X in X_list:
            print(f"\n==== Testing N={N}, X={X} ====")
            try:
                features_dict = create_feature_matrix(delay_values, lookback_window=N, prediction_window=X)
                X_data = features_dict['feature_matrix']
                y_data = features_dict['labels']
                if len(np.unique(y_data)) < 2:
                    print("Not enough positive/negative samples, skipping...")
                    continue
                trainer = ModelTrainer()
                metrics = trainer.train_random_forest(X_data, y_data)["metrics"]
                results.append({
                    'N': N,
                    'X': X,
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1'],
                    'roc_auc': metrics['roc_auc']
                })
            except Exception as e:
                print(f"Error for N={N}, X={X}: {str(e)}")
                continue
    # 输出结果表格
    df_results = pd.DataFrame(results)
    print("\n===== Sweep Results =====")
    print(df_results)
    df_results.to_csv("results/sweep_window_results.csv", index=False)

if __name__ == "__main__":
    N_list = [5, 10, 15, 20]
    X_list = [1, 3, 5, 10]
    delay_values = get_all_delay_values()
    sweep_windows(N_list, X_list, delay_values) 