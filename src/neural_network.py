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
CSV_PATH = "dataset/1st_capture/cpe_b-cpe_a-mobile.csv"
LOOKBACK = 10
PRED_WINDOW = 1

# === Load Data ===
df = pd.read_csv(CSV_PATH)
df['delay_ms'] = pd.to_numeric(df['delay_ms'], errors='coerce')
df['is_packet_loss'] = (df['delay_ms'] == -1).astype(int)
df['delay_ms'] = df['delay_ms'].replace(-1, np.nan).fillna(method='ffill')

# === Feature Engineering ===
def generate_features(df, lookback, pred_window):
    X, y = [], []
    delay = df['delay_ms'].values
    label = df['is_packet_loss'].values

    for i in range(lookback, len(df) - pred_window):
        window = delay[i - lookback:i]
        y_label = 1 if label[i:i + pred_window].sum() > 0 else 0

        # 核心统计特征
        mean = np.mean(window)
        std = np.std(window)
        min_ = np.min(window)
        max_ = np.max(window)
        range_ = max_ - min_
        median = np.median(window)
        slope = np.polyfit(np.arange(lookback), window, 1)[0]
        last_delta = window[-1] - np.mean(window[:-1])
        coef_var = std / (mean + 1e-5)

        # 关键尾部特征
        rolling_std_last_5 = np.std(window[-5:])
        rolling_mean_diff_last_3 = (window[-1] - window[-2]) + (window[-2] - window[-3])
        tail_repeated = [rolling_std_last_5, rolling_mean_diff_last_3] * 3

        features = [
            mean, std, min_, max_, range_, median,
            slope, last_delta, coef_var
        ] + tail_repeated

        X.append(features)
        y.append(y_label)

    return np.array(X), np.array(y)

# === Feature Generation ===
X, y = generate_features(df, LOOKBACK, PRED_WINDOW)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Normalize ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Build Neural Network ===
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

# === Train Model ===
model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=64,
    callbacks=[EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)],
    verbose=1
)

# === Evaluate Model ===
y_pred_prob = model.predict(X_test_scaled).flatten()
y_pred = (y_pred_prob >= 0.5).astype(int)

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, digits=4))
