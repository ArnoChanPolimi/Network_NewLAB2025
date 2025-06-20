import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# === CONFIGURATION ===
CSV_SRC = "dataset/1st_capture/cpe_a-cpe_b-mobile.csv"   # 源任务 A→B
CSV_TGT = "dataset/1st_capture/cpe_b-cpe_a-mobile.csv"   # 目标任务 B→A
LOOKBACK = 10
PRED_WINDOW = 1

# === Load and preprocess ===
def preprocess_dataframe(df):
    df['delay_ms'] = pd.to_numeric(df['delay_ms'], errors='coerce')
    df['is_packet_loss'] = (df['delay_ms'] == -1).astype(int)
    df['delay_ms'] = df['delay_ms'].replace(-1, np.nan).fillna(method='ffill')
    return df

def generate_features(df, lookback=10, pred_window=1):
    X, y = [], []
    delay = df['delay_ms'].values
    label = df['is_packet_loss'].values
    for i in range(lookback, len(df) - pred_window):
        window = delay[i - lookback:i]
        y_label = 1 if label[i:i + pred_window].sum() > 0 else 0

        # 核心特征
        mean = np.mean(window)
        std = np.std(window)
        min_ = np.min(window)
        max_ = np.max(window)
        range_ = max_ - min_
        median = np.median(window)
        slope = np.polyfit(np.arange(lookback), window, 1)[0]
        last_delta = window[-1] - np.mean(window[:-1])
        coef_var = std / (mean + 1e-5)

        # 尾部特征
        rolling_std_last_5 = np.std(window[-5:])
        rolling_mean_diff_last_3 = (window[-1] - window[-2]) + (window[-2] - window[-3])
        tail_repeated = [rolling_std_last_5, rolling_mean_diff_last_3] * 3

        features = [mean, std, min_, max_, range_, median, slope, last_delta, coef_var] + tail_repeated
        X.append(features)
        y.append(y_label)

    return np.array(X), np.array(y)

# Load data
df_src = preprocess_dataframe(pd.read_csv(CSV_SRC))
df_tgt = preprocess_dataframe(pd.read_csv(CSV_TGT))

X_src, y_src = generate_features(df_src, LOOKBACK, PRED_WINDOW)
X_tgt, y_tgt = generate_features(df_tgt, LOOKBACK, PRED_WINDOW)

# Split target task: 80% train, 20% test
split_index = int(len(X_tgt) * 0.8)
X_tgt_train, X_tgt_test = X_tgt[:split_index], X_tgt[split_index:]
y_tgt_train, y_tgt_test = y_tgt[:split_index], y_tgt[split_index:]

# Normalize
scaler = StandardScaler()
X_src_scaled = scaler.fit_transform(X_src)
X_tgt_train_scaled = scaler.transform(X_tgt_train)
X_tgt_test_scaled = scaler.transform(X_tgt_test)

# === Build base model ===
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_src.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

# === Train on source domain ===
print("\nTraining on source (A→C)...")
model.fit(
    X_src_scaled, y_src,
    validation_split=0.2,
    epochs=50,
    batch_size=64,
    callbacks=[EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)],
    verbose=1
)

# === Freeze first two layers for transfer learning ===
for layer in model.layers[:-1]:
    layer.trainable = False

# Re-compile (important!)
model.compile(optimizer=Adam(0.0005), loss='binary_crossentropy', metrics=['accuracy'])

# === Fine-tune on target domain ===
print("\nFine-tuning on target (C→A)...")
model.fit(
    X_tgt_train_scaled, y_tgt_train,
    validation_split=0.2,
    epochs=50,
    batch_size=64,
    callbacks=[EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)],
    verbose=1
)

# === Evaluate ===
y_pred_prob = model.predict(X_tgt_test_scaled).flatten()
y_pred = (y_pred_prob >= 0.5).astype(int)

print("\n=== Confusion Matrix (Transfer Learned) ===")
print(confusion_matrix(y_tgt_test, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_tgt_test, y_pred, digits=4))
