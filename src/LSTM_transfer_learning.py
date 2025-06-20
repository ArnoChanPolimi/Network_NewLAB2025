import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# === CONFIGURATION ===
CSV_SRC = "dataset/1st_capture/cpe_a-cpe_b-mobile.csv"  # 源任务 A→B
CSV_TGT = "dataset/1st_capture/cpe_b-cpe_a-mobile.csv"  # 目标任务 B→A
LOOKBACK = 40
PRED_WINDOW = 10

# === Functions ===
def preprocess_dataframe(df):
    df['delay_ms'] = pd.to_numeric(df['delay_ms'], errors='coerce')
    df['is_packet_loss'] = (df['delay_ms'] == -1).astype(int)
    df['delay_ms'] = df['delay_ms'].replace(-1, np.nan).fillna(method='ffill')
    return df

def generate_sequence_data(df, lookback, pred_window):
    X, y = [], []
    delay = df['delay_ms'].values
    label = df['is_packet_loss'].values

    for i in range(lookback, len(df) - pred_window):
        window = delay[i - lookback:i]
        y_label = 1 if label[i:i + pred_window].sum() > 0 else 0
        X.append(window)
        y.append(y_label)
    return np.array(X), np.array(y)

# === Load and preprocess data ===
df_src = preprocess_dataframe(pd.read_csv(CSV_SRC))
df_tgt = preprocess_dataframe(pd.read_csv(CSV_TGT))

X_src, y_src = generate_sequence_data(df_src, LOOKBACK, PRED_WINDOW)
X_tgt, y_tgt = generate_sequence_data(df_tgt, LOOKBACK, PRED_WINDOW)

# Normalize each sample
scaler = StandardScaler()
X_src_scaled = scaler.fit_transform(X_src).reshape((-1, LOOKBACK, 1))
X_tgt_scaled = scaler.transform(X_tgt).reshape((-1, LOOKBACK, 1))

# Split target into train/test
split_idx = int(0.8 * len(X_tgt_scaled))
X_tgt_train, X_tgt_test = X_tgt_scaled[:split_idx], X_tgt_scaled[split_idx:]
y_tgt_train, y_tgt_test = y_tgt[:split_idx], y_tgt[split_idx:]

# Class weights (target domain)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_tgt_train), y=y_tgt_train)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# === Step 1: Build and Train Source LSTM Model ===
base_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 1), name='lstm_1'),
    LSTM(32, return_sequences=False, name='lstm_2'),
    Dropout(0.3),
    Dense(16, activation='relu', name='dense_1'),
    Dropout(0.2),
    Dense(1, activation='sigmoid', name='output')
])
base_model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

print("\n=== Training on source domain (A→C) ===")
base_model.fit(
    X_src_scaled, y_src,
    validation_split=0.2,
    epochs=50,
    batch_size=64,
    class_weight=class_weights_dict,
    callbacks=[EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)],
    verbose=1
)

# === Step 2: Freeze front layers, fine-tune on target ===
for layer in base_model.layers[:3]:  # Freeze LSTM + Dropout
    layer.trainable = False

base_model.compile(optimizer=Adam(0.0005), loss='binary_crossentropy', metrics=['accuracy'])

print("\n=== Fine-tuning on target domain (C→A) ===")
base_model.fit(
    X_tgt_train, y_tgt_train,
    validation_split=0.2,
    epochs=50,
    batch_size=64,
    class_weight=class_weights_dict,
    callbacks=[EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)],
    verbose=1
)

# === Step 3: Evaluate ===
y_pred_prob = base_model.predict(X_tgt_test).flatten()
y_pred = (y_pred_prob >= 0.5).astype(int)

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_tgt_test, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_tgt_test, y_pred, digits=4))
