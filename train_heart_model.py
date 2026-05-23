import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
heart_data = pd.read_csv("datasets/heart.csv")

# Features and target
X = heart_data.drop(columns='target')
Y = heart_data['target']

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    stratify=Y,
    random_state=42
)

# Scale data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42
)

# Train
model.fit(X_train, Y_train)

# Training accuracy
train_prediction = model.predict(X_train)

train_accuracy = accuracy_score(
    train_prediction,
    Y_train
)

print("Training Accuracy:", train_accuracy)

# Test accuracy
test_prediction = model.predict(X_test)

test_accuracy = accuracy_score(
    test_prediction,
    Y_test
)

print("Test Accuracy:", test_accuracy)

# Save model
pickle.dump(
    model,
    open("models/heart_model.pkl", "wb")
)

# Save scaler
pickle.dump(
    scaler,
    open("models/heart_scaler.pkl", "wb")
)

print("Heart model saved successfully!")