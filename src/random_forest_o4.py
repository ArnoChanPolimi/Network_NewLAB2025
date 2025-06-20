import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# === CONFIGURATION ===
CSV_PATH = "dataset/1st_capture/cpe_a-cpe_b-mobile.csv"
LOOKBACK = 50
PRED_WINDOW = 10

# === Load Data ===
df = pd.read_csv(CSV_PATH)
df['delay_ms'] = pd.to_numeric(df['delay_ms'], errors='coerce')
df['is_packet_loss'] = (df['delay_ms'] == -1).astype(int)
df['delay_ms'] = df['delay_ms'].replace(-1, np.nan)
df['delay_ms'] = df['delay_ms'].fillna(method='ffill')

# === Feature Engineering with Tail-Sensitive Features and Weight Boost ===
def generate_dataset_with_emphasized_tail(df, lookback, pred_window):
    X, y, indices = [], [], []
    delay_series = df['delay_ms'].values
    label_series = df['is_packet_loss'].values

    for i in range(lookback, len(df) - pred_window):
        window = delay_series[i - lookback:i]
        label = 1 if label_series[i:i + pred_window].sum() > 0 else 0

        # 基础统计特征
        mean = np.mean(window)
        std = np.std(window)
        min_ = np.min(window)
        max_ = np.max(window)
        range_ = max_ - min_
        median = np.median(window)

        # 趋势特征
        slope = np.polyfit(np.arange(lookback), window, 1)[0]
        last_delta = window[-1] - np.mean(window[:-1])
        coef_var = std / (mean + 1e-5)

        # ✅ 新增尾部特征
        rolling_std_last_5 = np.std(window[-5:])
        rolling_mean_diff_last_3 = (window[-1] - window[-2]) + (window[-2] - window[-3])

        # ✅ 尾部特征加权（通过重复）
        tail_features = [
            rolling_std_last_5, rolling_mean_diff_last_3
        ]
        tail_repeated = tail_features * 3  # 你可以调节重复次数（比如 *2 或 *3）

        # 合并所有特征
        features = list(window) + [
            mean, std, min_, max_, range_, median,
            slope, last_delta, coef_var
        ] + tail_repeated

        X.append(features)
        y.append(label)
        indices.append(i)

    return np.array(X), np.array(y), np.array(indices)

# === Feature Generation ===
X, y, indices = generate_dataset_with_emphasized_tail(df, lookback=LOOKBACK, pred_window=PRED_WINDOW)

# === Train/Test Split ===
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, indices, test_size=0.2, random_state=42)

# === Skip normalization
X_train_scaled = X_train
X_test_scaled = X_test

# === Train Random Forest ===
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# === Evaluate ===
print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, digits=4))

# === False Negatives Mapping to CSV Rows ===
fn_indices = np.where((y_test == 1) & (y_pred == 0))[0]
print("\n=== False Negative 样本原始 CSV 行号（窗口起点）===")
for i in fn_indices:
    print(f"测试集样本 {i} → 原始窗口起点：{idx_test[i]}")

# === Feature Importance Visualization ===
importances = model.feature_importances_
feature_names = (
    [f"-{LOOKBACK - i}s" for i in range(LOOKBACK)] +
    ["mean", "std", "min", "max", "range", "median",
     "slope", "last_delta", "coef_var"] +
    ["rolling_std_last_5", "rolling_mean_diff_last_3"] * 3  # 对应重复
)

plt.figure(figsize=(13, 5))
plt.bar(range(len(importances)), importances)
plt.xticks(ticks=range(len(importances)), labels=feature_names, rotation=45)
plt.title("Random Forest Feature Importance (Tail Repetition)")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.show()
