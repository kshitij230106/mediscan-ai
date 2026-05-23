import pandas as pd

from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder


from sklearn.metrics import accuracy_score

import joblib

# LOAD DATASET

data = pd.read_csv(
    "datasets/kidney_disease.csv"
)

# REMOVE ID COLUMN IF EXISTS

if "id" in data.columns:

    data = data.drop("id", axis=1)

# HANDLE MISSING VALUES

data = data.ffill()

# ENCODE TEXT COLUMNS

label_encoder = LabelEncoder()

# CONVERT ALL TEXT VALUES TO NUMBERS

# ENCODE TEXT COLUMNS

for column in data.columns:

    if data[column].dtype == object:

        data[column] = data[column].str.strip()

        encoder = LabelEncoder()

        data[column] = encoder.fit_transform(
            data[column]
        )

# CONVERT EVERYTHING TO NUMERIC


# FEATURES

X = data[[
    "age",
    "bp",
    "sg",
    "al",
    "su"
]]

# TARGET

y = data["classification"]

# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42

)

# SCALE DATA

# CATBOOST MODEL

model = CatBoostClassifier(

    iterations=300,

    learning_rate=0.05,

    depth=6,

    verbose=0

)

# TRAIN MODEL
X_train = X_train.apply(
    pd.to_numeric,
    errors="coerce"
)

X_test = X_test.apply(
    pd.to_numeric,
    errors="coerce"
)

X_train = X_train.fillna(0)

X_test = X_test.fillna(0)


model.fit(X_train, y_train)

# PREDICT

y_pred = model.predict(X_test)

# ACCURACY

accuracy = accuracy_score(

    y_test,
    y_pred

)

print(

    "Kidney CatBoost Accuracy:",

    accuracy

)

import pandas as pd

from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder


from sklearn.metrics import accuracy_score

import joblib

# LOAD DATASET

data = pd.read_csv(
    "datasets/kidney_disease.csv"
)

# REMOVE ID COLUMN IF EXISTS

if "id" in data.columns:

    data = data.drop("id", axis=1)

# HANDLE MISSING VALUES

data = data.ffill()

# ENCODE TEXT COLUMNS

label_encoder = LabelEncoder()

# CONVERT ALL TEXT VALUES TO NUMBERS

# ENCODE TEXT COLUMNS

for column in data.columns:

    if data[column].dtype == object:

        data[column] = data[column].str.strip()

        encoder = LabelEncoder()

        data[column] = encoder.fit_transform(
            data[column]
        )

# CONVERT EVERYTHING TO NUMERIC


# FEATURES

X = data[[
    "age",
    "bp",
    "sg",
    "al",
    "su"
]]

# TARGET

y = data["classification"]

# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42

)

# SCALE DATA

# CATBOOST MODEL

model = CatBoostClassifier(

    iterations=300,

    learning_rate=0.05,

    depth=6,

    verbose=0

)

# TRAIN MODEL
X_train = X_train.apply(
    pd.to_numeric,
    errors="coerce"
)

X_test = X_test.apply(
    pd.to_numeric,
    errors="coerce"
)

X_train = X_train.fillna(0)

X_test = X_test.fillna(0)


model.fit(X_train, y_train)

# PREDICT

y_pred = model.predict(X_test)

# ACCURACY

accuracy = accuracy_score(

    y_test,
    y_pred

)

print(

    "Kidney CatBoost Accuracy:",

    accuracy

)

# SAVE MODEL


joblib.dump(
    model,
    "models/kidney_model.pkl"
)

# SAVE MODEL



print(

    "Kidney Model Saved!"

)