import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

from catboost import CatBoostClassifier

import joblib

# LOAD DATASET

data = pd.read_csv("datasets/diabetes.csv")

# CONVERT TEXT COLUMNS TO NUMBERS

data["gender"] = data["gender"].map({

    "Male": 1,
    "Female": 0,
    "Other": 2

})

data["smoking_history"] = data["smoking_history"].astype("category").cat.codes

# INPUT FEATURES

X = data.drop("diabetes", axis=1)

# TARGET

y = data["diabetes"]
# INPUT FEATURES



# SPLIT DATA

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42

)

# FEATURE SCALING

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# CREATE CATBOOST MODEL

model = CatBoostClassifier(

    iterations=500,

    learning_rate=0.03,

    depth=6,

    loss_function='Logloss',

    verbose=0

)

# TRAIN MODEL

model.fit(

    X_train,
    y_train

)

# TEST MODEL

predictions = model.predict(X_test)

# CALCULATE ACCURACY

accuracy = accuracy_score(

    y_test,
    predictions

)

print()

print("CatBoost Accuracy:", accuracy)

print()

# SAVE MODEL

joblib.dump(

    model,
    "diabetes_model.pkl"

)

# SAVE SCALER

joblib.dump(

    scaler,
    "diabetes_scaler.pkl"

)

print("Model Saved Successfully!")

print("Scaler Saved Successfully!")