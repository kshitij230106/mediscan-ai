import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
diabetes = pd.read_csv("diabetes.csv")

# Features and target
X = diabetes.drop(columns='Outcome')
Y = diabetes['Outcome']

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    stratify=Y,
    random_state=2
)

# Scale data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42
)

model.fit(X_train, Y_train)

# Accuracy
X_train_prediction = model.predict(X_train)

training_data_accuracy = accuracy_score(
    X_train_prediction,
    Y_train
)

print("Training Accuracy:", training_data_accuracy)

# Test accuracy

X_test_prediction = model.predict(X_test)

test_data_accuracy = accuracy_score(
    X_test_prediction,
    Y_test
)

print("Test Accuracy:", test_data_accuracy)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

# Save scaler
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model and scaler saved successfully!")