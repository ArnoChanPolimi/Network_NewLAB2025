import numpy as np
from src.model_training import train_neural_network
from src.preprocessing import load_data, clean_data
from src.feature_engineering import create_feature_matrix
from sklearn.model_selection import train_test_split
from pathlib import Path

# 只做一次数据清洗和合并
all_delay_series = []
dataset_dir = Path("dataset")
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
if not all_delay_series:
    raise RuntimeError("No valid data found in any files")
delay_values = np.concatenate(all_delay_series)

# 选定窗口参数（可根据sweep结果调整）
N = 10
X = 3
features_dict = create_feature_matrix(delay_values, lookback_window=N, prediction_window=X)
X = features_dict['feature_matrix']
y = features_dict['labels']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
train_neural_network(X_train, y_train, X_test, y_test) 