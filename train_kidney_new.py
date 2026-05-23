import pandas as pd
import joblib

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =========================
# LOAD DATASET
# =========================

data = pd.read_csv(
    "datasets/kidney_disease.csv"
)

# =========================
# KEEP REQUIRED COLUMNS
# =========================

data = data[[
    "age",
    "bp",
    "sg",
    "al",
    "su",
    "classification"
]]

# =========================
# CLEAN TARGET COLUMN
# =========================

data["classification"] = (
    data["classification"]
    .astype(str)
    .str.strip()
    .str.replace("\t", "")
)

# =========================
# KEEP VALID LABELS ONLY
# =========================

data = data[
    data["classification"].isin([
        "ckd",
        "notckd"
    ])
]

# =========================
# CONVERT LABELS
# =========================

data["classification"] = data["classification"].map({
    "ckd": 0,
    "notckd": 1
})

# =========================
# CONVERT TO INTEGER
# =========================

data["classification"] = (
    data["classification"]
    .astype(int)
)

# =========================
# REMOVE MISSING VALUES
# =========================

data = data.dropna()

# =========================
# FEATURES
# =========================

X = data[[
    "age",
    "bp",
    "sg",
    "al",
    "su"
]]

# =========================
# TARGET
# =========================

y = data["classification"]

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# MODEL
# =========================

model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.05,
    depth=6,
    verbose=0
)

# =========================
# TRAIN MODEL
# =========================

model.fit(
    X_train,
    y_train
)

# =========================
# PREDICTIONS
# =========================

predictions = model.predict(X_test)

# =========================
# ACCURACY
# =========================

accuracy = accuracy_score(
    y_test,
    predictions
)

print(
    "Accuracy:",
    accuracy
)

# =========================
# SAVE MODEL
# =========================

joblib.dump(
    model,
    "models/kidney_model.pkl"
)

print(
    "Kidney model saved successfully!"
)