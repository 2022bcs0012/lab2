import pandas as pd
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# EXPERIMENT: Ridge alpha=0.1

# Load dataset
df = pd.read_csv("data/winequality-red.csv", sep=";")

X = df.drop("quality", axis=1)
y = df["quality"]

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature selection (example: use all features)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Save model
with open("output/model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save metrics
metrics = {
    "MSE": mse,
    "R2": r2
}

with open("output/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Print metrics
print("MSE:", mse)
print("R2 Score:", r2)
