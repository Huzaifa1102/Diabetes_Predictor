import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers, regularizers
from keras.callbacks import EarlyStopping
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- 1. Data Setup (Same as before) ---
housing = fetch_california_housing()
X, y = housing.data, housing.target
X_temp, X_val, y_temp, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, _, y_train, _ = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

# --- 2. Build Model v2 with Regularization (From your images) ---
model_v2 = keras.Sequential([
    keras.Input(shape=(8,)),
    
    # Layer 1: L2 Regularization + 20% Dropout
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.2), 
    
    # Layer 2: L2 Regularization + 20% Dropout
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.2),
    
    # Layer 3: Standard Dense
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.2),
    
    # Output Layer for Regression
    layers.Dense(1)
])

# --- 3. Compile and Early Stopping ---
model_v2.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# --- 4. Train the Regularized Model ---
print("Training Model v2 with Regularization...")
history_v2 = model_v2.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=64,
    callbacks=[early_stopping],
    verbose=1
)

print(f"\nTraining stopped at epoch: {len(history_v2.history['loss'])}")

# --- 5. Calculate Metrics (From your image) ---
# Get predictions
y_train_pred_v2 = model_v2.predict(X_train, verbose=0)
y_val_pred_v2 = model_v2.predict(X_val, verbose=0)

# Calculate Train Metrics
train_mse_v2 = mean_squared_error(y_train, y_train_pred_v2)
train_rmse_v2 = np.sqrt(train_mse_v2)
train_mae_v2 = mean_absolute_error(y_train, y_train_pred_v2)
train_r2_v2 = r2_score(y_train, y_train_pred_v2)

# Calculate Val Metrics
val_mse_v2 = mean_squared_error(y_val, y_val_pred_v2)
val_rmse_v2 = np.sqrt(val_mse_v2)
val_mae_v2 = mean_absolute_error(y_val, y_val_pred_v2)
val_r2_v2 = r2_score(y_val, y_val_pred_v2)

print("\n--- Model v2 Final Evaluation ---")
print(f"Train MSE:  {train_mse_v2:.4f}  |  Val MSE:  {val_mse_v2:.4f}")
print(f"Train RMSE: {train_rmse_v2:.4f}  |  Val RMSE: {val_rmse_v2:.4f}")
print(f"Train MAE:  {train_mae_v2:.4f}  |  Val MAE:  {val_mae_v2:.4f}")
print(f"Train R2:   {train_r2_v2:.4f}  |  Val R2:   {val_r2_v2:.4f}")

# --- 6. Plotting the Cured Curves ---
plt.figure(figsize=(8, 5))
plt.plot(history_v2.history['loss'], label='Train Loss')
plt.plot(history_v2.history['val_loss'], label='Val Loss')
plt.title('Model v2 - Regularized')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()