import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
diabetes = pd.read_csv("diabetes.csv")

# First 5 rows
print(diabetes.head())

# Separate features and labels
X = diabetes.drop(columns='Outcome')
Y = diabetes['Outcome']

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    stratify=Y,
    random_state=2
)

# Scale the data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, Y_train)

# Test accuracy
X_test_prediction = model.predict(X_test)

accuracy = accuracy_score(X_test_prediction, Y_test)

print("Model Accuracy:", accuracy)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model and scaler saved successfully!")