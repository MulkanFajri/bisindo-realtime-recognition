import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =====================
# LOAD DATA
# =====================

df = pd.read_csv(
    "kata_landmarks.csv"
)

X = df.drop(
    "label",
    axis=1
)

y = df["label"]

# =====================
# SPLIT DATA
# =====================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================
# MODEL RF
# =====================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# =====================
# TRAIN
# =====================

model.fit(
    X_train,
    y_train
)

# =====================
# EVALUASI
# =====================

y_pred = model.predict(
    X_test
)

acc = accuracy_score(
    y_test,
    y_pred
)

print(
    "\nAccuracy:",
    acc
)

# =====================
# SAVE MODEL
# =====================

joblib.dump(
    model,
    "kata_model.pkl"
)

print(
    "\nModel tersimpan:"
)

print(
    "kata_model.pkl"
)