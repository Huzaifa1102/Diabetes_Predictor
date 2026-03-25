import numpy as np
import pandas as pd
import tensorflow as tf
# Using the direct tf.keras path to resolve the "not defined" errors
from keras import Sequential
from keras import Dense
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load data
housing = fetch_california_housing()
X, y = housing.data, housing.target

# 2. Split: 80% train, 20% validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# 4. Build MLP
# Now using Sequential and Dense directly after the fixed imports
model_v1 = Sequential([
    Dense(64, activation='relu', input_shape=(8,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

# 5. Compile
model_v1.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# 6. Train
print("Training starting...")
history_v1 = model_v1.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=64,
    verbose=1
)