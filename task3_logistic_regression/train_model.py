
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "cleaned" / "cleaned_customer_churn.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", df.shape)

# 2. Separate features and target
X = df.drop(columns=["churn_flag", "customer_id"])
y = df["churn_flag"]

# Identify numerical and categorical columns
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print("Numerical features:", numeric_features)
print("Categorical features:", categorical_features)

# 3. Preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 4. Create Logistic Regression pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

# 5. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 6. Train model
model.fit(X_train, y_train)

print("\nModel training completed!")

# 7. Predict probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# Predict class labels
y_pred = model.predict(X_test)

# 8. Evaluate model
auc = roc_auc_score(y_test, y_prob)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n========== MODEL PERFORMANCE ==========")
print(f"ROC-AUC:  {auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# 9. Save metrics
metrics = pd.DataFrame({
    "Metric": ["ROC-AUC", "Accuracy", "Precision", "Recall", "F1 Score"],
    "Score": [auc, accuracy, precision, recall, f1]
})

metrics_path = OUTPUT_DIR / "metrics.csv"
metrics.to_csv(metrics_path, index=False)

# 10. Save trained model
model_path = MODEL_DIR / "logistic_regression_model.pkl"
joblib.dump(model, model_path)

print("\nSaved model:", model_path)
print("Saved metrics:", metrics_path)

if auc >= 0.70:
    print("\nSUCCESS: ROC-AUC target of 0.70 achieved!")
else:
    print("\nROC-AUC is below 0.70. We will investigate improvements.")
