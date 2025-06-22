import pandas as pd
import numpy as np
import glob
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# === CONFIGURATION ===
LOOKBACK = 10
PRED_WINDOW = 1
# ROOT_DIRS = ["dataset/1st_capture", "dataset/2nd_capture"]
# FILENAME_PATTERN = "*-mobile.csv"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIRS = [
    os.path.join(PROJECT_ROOT, "dataset", "1st_capture"),
    os.path.join(PROJECT_ROOT, "dataset", "2nd_capture")
]
FILENAME_PATTERN = "*-mobile.csv"


# === Load and Combine All Matching Files ===
all_files = []
for root in ROOT_DIRS:
    pattern = os.path.join(root, FILENAME_PATTERN)
    all_files.extend(glob.glob(pattern))

combined_df = pd.DataFrame()
for file in all_files:
    df = pd.read_csv(file)
    df['source_file'] = os.path.basename(file)  # Track file origin if needed
    combined_df = pd.concat([combined_df, df], ignore_index=True)

print("\n========= Combined DF Columns =========")
print(combined_df.columns.tolist())

print("\n========= Combined DF Head =========")
print(combined_df.head(100))

# === Preprocess ===
combined_df['delay_ms'] = pd.to_numeric(combined_df['delay_ms'], errors='coerce')
combined_df['is_packet_loss'] = (combined_df['delay_ms'] == -1).astype(int)

# === Feature Engineering ===
def generate_features(df, lookback, pred_window):
    X, y, indices = [], [], []
    delay = df['delay_ms'].values
    labels = df['is_packet_loss'].values

    for i in range(lookback, len(df) - pred_window):
        if np.any(np.isnan(delay[i:i + pred_window])):
            continue

        valid_window = []
        j = i - 1
        while len(valid_window) < lookback and j >= 0:
            if not np.isnan(delay[j]):
                valid_window.insert(0, delay[j])
            j -= 1

        if len(valid_window) < lookback:
            continue

        label = 1 if labels[i:i + pred_window].sum() > 0 else 0

        mean = np.mean(valid_window)
        std = np.std(valid_window)
        min_ = np.min(valid_window)
        max_ = np.max(valid_window)
        range_ = max_ - min_
        median = np.median(valid_window)
        slope = np.polyfit(np.arange(lookback), valid_window, 1)[0]
        last_delta = valid_window[-1] - np.mean(valid_window[:-1])
        coef_var = std / (mean + 1e-5)

        rolling_std_last_5 = np.std(valid_window[-5:])
        rolling_mean_diff_last_3 = (valid_window[-1] - valid_window[-2]) + (valid_window[-2] - valid_window[-3])
        tail_repeated = [rolling_std_last_5, rolling_mean_diff_last_3] * 3

        features = [
            mean, std, min_, max_, range_, median,
            slope, last_delta, coef_var
        ] + tail_repeated

        X.append(features)
        y.append(label)
        indices.append(i)

    return np.array(X), np.array(y), np.array(indices)

# === Generate Dataset ===
X, y, indices = generate_features(combined_df, LOOKBACK, PRED_WINDOW)

# === Train/Test Split ===
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, indices, test_size=0.2, random_state=42
)

# === Train Random Forest ===
model = RandomForestClassifier(
    n_estimators=50,#300,
    max_depth=8,#12,
    min_samples_leaf=5,#3,
    class_weight='balanced_subsample',
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# === Evaluation ===
print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, digits=4))

# === False Negatives Mapping ===
fn_indices = np.where((y_test == 1) & (y_pred == 0))[0]
print("\n=== False Negative 样本原始 CSV 行号(窗口起点)===")
for i in fn_indices:
    print(f"测试集样本 {i} → 原始窗口起点：{idx_test[i]}")

# === Feature Importance Visualization ===
importances = model.feature_importances_
feature_names = [
    "mean", "std", "min", "max", "range", "median",
    "slope", "last_delta", "coef_var",
    "rolling_std_last_5", "rolling_mean_diff_last_3",
    "rolling_std_last_5", "rolling_mean_diff_last_3",
    "rolling_std_last_5", "rolling_mean_diff_last_3"
]

indices = np.argsort(importances)[::-1]
print("\nAll Features Importance:")
for i in range(len(importances)):
    print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# 创建文件夹，如果不存在
script_dir = os.path.dirname(os.path.abspath(__file__))  # 脚本所在目录（就是你说的同级目录）
# 输出文件夹：脚本同级目录下的 output_figure
output_dir = os.path.join(script_dir, "output_figure")

# 如果文件夹不存在就创建
os.makedirs(output_dir, exist_ok=True)
# 例如保存图片：
output_path = os.path.join(output_dir, "rf_Mobile_feature_importance.png")



plt.figure(figsize=(13, 5))
plt.bar(range(len(importances)), importances)
plt.xticks(ticks=range(len(importances)), labels=feature_names, rotation=45)
plt.title("Random Forest Feature Importance (All Mobile CSVs)")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.grid(True, axis='y', linestyle='--', alpha=0.5)

# 保存图像
plt.savefig(output_path)
plt.close()

print(f"特征重要性图已保存到：{output_path}")

# plt.show()
# plt.close()