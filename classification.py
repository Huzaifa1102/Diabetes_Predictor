import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# --- 1. Data Setup ---
housing = fetch_california_housing()
X, y = housing.data, housing.target
X_temp, X_val, y_temp, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, _, y_train, _ = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

# Convert regression target to binary: High (>2.5) vs Low
y_class_train = (y_train > 2.5).astype(int)
y_class_val = (y_val > 2.5).astype(int)

# --- 2. Build Classification Model ---
# Notice the two major changes for classification: 'sigmoid' and 'binary_crossentropy'
model_class = keras.Sequential([
    keras.Input(shape=(8,)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid') # Sigmoid squashes output between 0 and 1
])

model_class.compile(
    optimizer='adam',
    loss='binary_crossentropy', # Loss function specifically for Yes/No categories
    metrics=['accuracy']
)

# --- 3. Train Model ---
print("Training Classification Model in progress...\n")
history_class = model_class.fit(
    X_train, y_class_train,
    validation_data=(X_val, y_class_val),
    epochs=100,
    batch_size=64,
    verbose=0
)

# --- 4. Manual Metric Functions (From your images) ---
def calc_confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])

def calc_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def calc_precision(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def calc_recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def calc_f1(y_true, y_pred):
    precision = calc_precision(y_true, y_pred)
    recall = calc_recall(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# --- 5. Evaluate ---
y_val_pred_prob = model_class.predict(X_val, verbose=0)
y_val_pred_bin = (y_val_pred_prob.flatten() > 0.5).astype(int)

acc_val = calc_accuracy(y_class_val, y_val_pred_bin)
prec_val = calc_precision(y_class_val, y_val_pred_bin)
rec_val = calc_recall(y_class_val, y_val_pred_bin)
f1_val = calc_f1(y_class_val, y_val_pred_bin)

print("--- Manual Metrics ---")
print(f"Accuracy:  {acc_val:.4f}")
print(f"Precision: {prec_val:.4f}")
print(f"Recall:    {rec_val:.4f}")
print(f"F1-Score:  {f1_val:.4f}\n")

print("--- Scikit-Learn Verification ---")
print(classification_report(y_class_val, y_val_pred_bin))