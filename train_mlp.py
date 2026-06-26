import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# ======================
# LOAD DATA
# ======================

df = pd.read_csv(
    "bisindo_landmarks.csv",
    low_memory=False
)

X = df.drop("label", axis=1)
y = df["label"].astype(str)


# ======================
# ENCODE LABEL
# ======================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

# ======================
# TRAIN TEST SPLIT
# ======================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ======================
# MODEL MLP
# ======================

model = MLPClassifier(
    hidden_layer_sizes=(256,128),
    activation="relu",
    solver="adam",
    max_iter=100,
    random_state=42,
    verbose=True
)

# ======================
# TRAIN
# ======================

model.fit(X_train, y_train)

# ======================
# EVALUASI
# ======================

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("\nAccuracy :", acc)

# ======================
# SIMPAN MODEL
# ======================

joblib.dump(model, "bisindo_mlp.pkl")
joblib.dump(encoder, "label_encoder.pkl")

print("\nModel tersimpan:")
print("bisindo_mlp.pkl")
print("label_encoder.pkl")