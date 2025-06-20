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
CSV_PATH = "dataset/1st_capture/cpe_a-cpe_c-mobile.csv"
LOOKBACK = 40
PRED_WINDOW = 10

# === Load Data ===
df = pd.read_csv(CSV_PATH)
df['delay_ms'] = pd.to_numeric(df['delay_ms'], errors='coerce')
df['is_packet_loss'] = (df['delay_ms'] == -1).astype(int)
df['delay_ms'] = df['delay_ms'].replace(-1, np.nan).fillna(method='ffill')

# === Sequence Generation ===
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

X, y = generate_sequence_data(df, LOOKBACK, PRED_WINDOW)

# === Normalize ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = X_scaled.reshape((-1, LOOKBACK, 1))  # Reshape for LSTM input

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === Class Weights ===
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# === Build Improved LSTM Model ===
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 1)),
    LSTM(32, return_sequences=False),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

# === Train with EarlyStopping ===
model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=64,
    class_weight=class_weights_dict,
    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
    verbose=1
)

# === Predict & Evaluate ===
y_pred_prob = model.predict(X_test).flatten()
y_pred = (y_pred_prob >= 0.5).astype(int)  # 若你更关注 Recall 可调为 0.3

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, digits=4))
