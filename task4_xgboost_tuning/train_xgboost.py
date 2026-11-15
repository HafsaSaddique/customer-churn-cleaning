
from pathlib import Path
import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cleaned" / "cleaned_customer_churn.csv"

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
MODEL_DIR = Path(__file__).resolve().parent / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Load cleaned dataset
df = pd.read_csv(DATA_PATH)

# Separate features and target
X = df.drop(columns=["churn_flag", "customer_id"])
y = df["churn_flag"].astype(int)

# Identify numerical and categorical columns
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

# Preprocessing
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

# XGBoost classifier
xgb_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42,
    n_jobs=1
)

# Complete pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", xgb_model)
])

# Hold out test data for final evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# Stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Hyperparameter grid
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [3, 5],
    "classifier__learning_rate": [0.05, 0.1],
    "classifier__subsample": [0.8],
    "classifier__colsample_bytree": [0.8]
}

# Grid Search using cross-validation
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

print("Starting XGBoost Grid Search...")
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Save cross-validation results
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results.to_csv(OUTPUT_DIR / "cv_results.csv", index=False)

# Save best model
joblib.dump(
    best_model,
    MODEL_DIR / "best_xgboost_model.pkl"
)

# Evaluate on untouched test set
test_probabilities = best_model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, test_probabilities)

# Save summary
summary = pd.DataFrame([{
    "best_cv_auc": grid_search.best_score_,
    "test_auc": test_auc,
    "best_parameters": str(grid_search.best_params_)
}])

summary.to_csv(OUTPUT_DIR / "model_summary.csv", index=False)

print("\nGrid Search completed!")
print("Best CV AUC:", round(grid_search.best_score_, 4))
print("Test AUC:", round(test_auc, 4))
print("Best parameters:", grid_search.best_params_)
print("Model saved successfully.")
