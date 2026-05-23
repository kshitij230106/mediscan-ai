import pandas as pd

from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

import joblib

# LOAD DATASET

data = pd.read_csv(
    "datasets/heart.csv"
)

# INPUT FEATURES

X = data.drop("target", axis=1)

# TARGET

y = data["target"]

# SPLIT DATA

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42

)

# SCALE DATA

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# CATBOOST MODEL

model = CatBoostClassifier(

    iterations=300,

    learning_rate=0.05,

    depth=6,

    verbose=0

)

# TRAIN MODEL

model.fit(X_train, y_train)

# PREDICTION

y_pred = model.predict(X_test)

# ACCURACY

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(

    "Heart CatBoost Accuracy:",

    accuracy

)

# SAVE MODEL

joblib.dump(

    model,

    "models/heart_model.pkl"

)

joblib.dump(

    scaler,

    "models/heart_scaler.pkl"

)

print(
    "Heart Model Saved!"
)