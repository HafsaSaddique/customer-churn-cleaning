
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    ConfusionMatrixDisplay,
    confusion_matrix
)

# Define project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "cleaned" / "cleaned_customer_churn.csv"
MODEL_PATH = BASE_DIR / "models" / "logistic_regression_model.pkl"
OUTPUT_DIR = BASE_DIR / "outputs"

# Load dataset and saved model
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

# Separate features and target
X = df.drop(columns=["churn_flag", "customer_id"])
y = df["churn_flag"]

# Recreate the same test split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Predict churn probabilities
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# Calculate AUC and ROC values
auc = roc_auc_score(y_test, y_prob)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {auc:.4f})",
    linewidth=2
)
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Customer Churn Logistic Regression")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()

roc_path = OUTPUT_DIR / "roc_curve.png"
plt.savefig(roc_path, dpi=300)
plt.close()

# Save confusion matrix plot
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Churn", "Churn"]
)

disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()

cm_path = OUTPUT_DIR / "confusion_matrix.png"
plt.savefig(cm_path, dpi=300)
plt.close()

print("========== EVALUATION RESULTS ==========")
print(f"ROC-AUC: {auc:.4f}")
print("\nConfusion Matrix:")
print(cm)

print("\nROC curve saved:", roc_path)
print("Confusion matrix saved:", cm_path)
