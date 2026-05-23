import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

# LOAD DATASET
data = pd.read_csv("datasets/diabetes.csv")

# FEATURES
X = data.drop("diabetes", axis=1)

# TARGET
y = data["diabetes"]

# CATEGORICAL FEATURES
categorical_features = [
    "gender",
    "smoking_history"
]

# NUMERICAL FEATURES
numerical_features = [
    "age",
    "hypertension",
    "heart_disease",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level"
]

# PREPROCESSOR
preprocessor = ColumnTransformer([
    (
        "num",
        StandardScaler(),
        numerical_features
    ),
    (
        "cat",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features
    )
])

# PIPELINE
pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        CatBoostClassifier(verbose=0)
    )
])

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# TRAIN
pipeline.fit(X_train, y_train)

# PREDICT
predictions = pipeline.predict(X_test)

# ACCURACY
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

# SAVE PIPELINE
joblib.dump(
    pipeline,
    "models/diabetes_pipeline.pkl"
)

print("Pipeline model saved successfully!")