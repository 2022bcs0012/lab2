import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# ---------------------------
# Metadata
# ---------------------------
NAME = "Shanik Hubert"
ROLL_NO = "2022BCS0012"

# ---------------------------
# Paths
# ---------------------------
DATA_PATH = "data/winequality-red.csv"
OUTPUT_DIR = "output"
MODEL_PATH = os.path.join(OUTPUT_DIR, "decision_tree_model.pkl")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# Load dataset
# ---------------------------
df = pd.read_csv(DATA_PATH, sep=";")  # Wine dataset uses semicolon

# ---------------------------
# Features / target
# ---------------------------
y = df["quality"]
X = df.drop(columns=["quality"])

# Optional: drop less important features (radical choice)
X = X.drop(columns=[
    "free sulfur dioxide",
    "total sulfur dioxide"
])

# ---------------------------
# Train / test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# ---------------------------
# Decision Tree Regressor
# ---------------------------
model = DecisionTreeRegressor(
    max_depth=5,        # limit depth to prevent overfitting
    min_samples_leaf=5, # minimum samples per leaf
    random_state=42
)

# ---------------------------
# Train model
# ---------------------------
model.fit(X_train, y_train)

# ---------------------------
# Evaluate
# ---------------------------
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# ---------------------------
# Save model
# ---------------------------
joblib.dump(model, MODEL_PATH)

# ---------------------------
# Save metrics
# ---------------------------
metrics = {
    "name": NAME,
    "roll_no": ROLL_NO,
    "model": "Decision Tree Regressor",
    "max_depth": 5,
    "min_samples_leaf": 5,
    "test_size": 0.3,
    "mse": mse,
    "r2_score": r2
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

# ---------------------------
# Print metrics
# ---------------------------
print("Name:", NAME)
print("Roll No:", ROLL_NO)
print("Model: Decision Tree Regressor (max_depth=5, min_samples_leaf=5)")
print("Test size: 0.3")
print("MSE:", mse)
print("R2 Score:", r2)
