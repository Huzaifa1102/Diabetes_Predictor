import kagglehub
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

# Download latest version
path = kagglehub.dataset_download("alexteboul/diabetes-health-indicators-dataset")

print("Path to dataset files:", path)

file_path = os.path.join(path, "diabetes_012_health_indicators_BRFSS2015.csv")
df = pd.read_csv(file_path)

#converting 0 and 1s to 0 and 2s to 1
df["Diabetes_012"] = df["Diabetes_012"].replace({1:0,2:1})
print("New distribution according to 0,1 only:\n", df["Diabetes_012"].value_counts(normalize = True))
df = df.rename(columns={"Diabetes_012": "Diabetes_binary"})

#removing duplicate entries
print("Total number of duplicates:\n", df.duplicated().sum())
df = df.drop_duplicates()
print(f"After removing duplicates:\n {df.shape}")

#X and y variables
X = df.drop("Diabetes_binary", axis = 1)
y = df["Diabetes_binary"]

#training testing split
X_train, X_validtest, y_train, y_validtest = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify = y)

#splitting into validation and testing
X_val, X_test, y_val, y_test = train_test_split(X_validtest, y_validtest, test_size=0.5, random_state=42, stratify=y_validtest)

print(f"Training data size: {X_train.shape[0]} & Validation data size: {X_val.shape[0]} & Testing data size: {X_test.shape[0]}")
