
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score, accuracy_score


# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

DATA_PATH = "data/cleaned/cleaned_customer_churn.csv"

MODEL_PATH = (
    "task4_xgboost_tuning/models/best_xgboost_model.pkl"
)

OUTPUT_DIR = "task4_xgboost_tuning/outputs"
REPORT_DIR = "task4_xgboost_tuning/reports"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# --------------------------------------------------
# 2. Load dataset and trained model
# --------------------------------------------------

print("Loading dataset and trained XGBoost model...")

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

X = df.drop(columns=["churn_flag", "customer_id"])
y = df["churn_flag"].astype(int)


# --------------------------------------------------
# 3. Recreate the original test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Test dataset prepared.")


# --------------------------------------------------
# 4. Generate predictions
# --------------------------------------------------

probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

overall_auc = roc_auc_score(y_test, probabilities)
overall_accuracy = accuracy_score(y_test, predictions)
overall_recall = recall_score(y_test, predictions)

print("\nOverall Model Performance")
print("-------------------------")
print(f"Test ROC-AUC: {overall_auc:.4f}")
print(f"Accuracy:     {overall_accuracy:.4f}")
print(f"Recall:       {overall_recall:.4f}")


# --------------------------------------------------
# 5. Check subgroup performance
# --------------------------------------------------

print("\nPerforming subgroup bias checks...")

subgroup_results = []

for column in ["gender", "senior_citizen"]:

    for group in sorted(X_test[column].dropna().unique()):

        mask = X_test[column] == group

        group_actual = y_test.loc[mask]
        group_probabilities = probabilities[mask.to_numpy()]
        group_predictions = predictions[mask.to_numpy()]

        if len(group_actual) == 0:
            continue

        if group_actual.nunique() == 2:
            group_auc = roc_auc_score(
                group_actual,
                group_probabilities
            )
        else:
            group_auc = float("nan")

        group_recall = recall_score(
            group_actual,
            group_predictions,
            zero_division=0
        )

        subgroup_results.append({
            "attribute": column,
            "group": group,
            "sample_count": len(group_actual),
            "actual_churn_rate": group_actual.mean(),
            "roc_auc": group_auc,
            "recall": group_recall
        })


subgroup_df = pd.DataFrame(subgroup_results)

subgroup_path = os.path.join(
    OUTPUT_DIR,
    "subgroup_bias_checks.csv"
)

subgroup_df.to_csv(subgroup_path, index=False)

print("\nSubgroup Results:")
print(subgroup_df.to_string(index=False))


# --------------------------------------------------
# 6. Generate quality review PDF
# --------------------------------------------------

pdf_path = os.path.join(
    REPORT_DIR,
    "xgboost_quality_review.pdf"
)

with PdfPages(pdf_path) as pdf:

    # PAGE 1: Model performance

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle(
        "XGBoost Model Quality Review",
        fontsize=18,
        fontweight="bold",
        y=0.96
    )

    performance_text = f"""
PROJECT: Customer Churn Prediction

MODEL:
Tuned XGBoost Classifier

TEST SET PERFORMANCE:

ROC-AUC: {overall_auc:.4f}
Accuracy: {overall_accuracy:.4f}
Recall: {overall_recall:.4f}

CROSS-VALIDATION:

Best mean CV ROC-AUC: 0.8502
Baseline Logistic Regression AUC: 0.8421

The tuned XGBoost model achieved a higher
cross-validation ROC-AUC than the baseline.

DATASET:
IBM Telco Customer Churn dataset.
Used as a simulated customer-retention
case study, not actual boutique transaction data.

EVALUATION:
The dataset was split using stratified
80/20 train-test splitting.
The training process used 5-fold
StratifiedKFold cross-validation.
"""

    fig.text(
        0.10, 0.85,
        performance_text,
        fontsize=12,
        va="top",
        linespacing=1.7
    )

    plt.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


    # PAGE 2: Subgroup bias checks

    fig = plt.figure(figsize=(8.27, 11.69))

    fig.suptitle(
        "Fairness and Subgroup Checks",
        fontsize=18,
        fontweight="bold",
        y=0.96
    )

    table_data = subgroup_df.round(4)

    plt.axis("off")

    plt.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc="center",
        loc="center"
    )

    fig.text(
        0.10, 0.18,
        "The model was evaluated across gender and "
        "senior citizen groups.\n"
        "Differences in performance require further "
        "investigation and do not\n"
        "automatically prove discrimination or fairness.\n\n"
        "These checks are exploratory and do not cover "
        "all possible protected\n"
        "attributes or intersectional groups.",
        fontsize=10,
        va="top"
    )

    pdf.savefig(fig)
    plt.close(fig)


    # PAGE 3: Security, assumptions and trade-offs

    fig = plt.figure(figsize=(8.27, 11.69))

    fig.suptitle(
        "Security, Assumptions and Limitations",
        fontsize=17,
        fontweight="bold",
        y=0.96
    )

    review_text = """
MODEL ASSUMPTIONS

1. Historical customer information is useful
   for predicting future churn.
2. Input columns have the same meaning and
   format as the training dataset.
3. The target variable is correctly labelled.
4. The training data is sufficiently
   representative of future customers.

BIAS AND QUALITY

1. Gender and senior citizen subgroup
   performance was examined.
2. Subgroup differences may indicate areas
   requiring further investigation.
3. Additional fairness tests are needed
   before real-world deployment.

TRADE-OFFS

1. XGBoost can capture complex relationships
   but is less interpretable than Logistic
   Regression.
2. Hyperparameter tuning improves model
   selection but requires additional
   computation.
3. The prediction threshold of 0.5 may not
   be optimal for every business situation.

SECURITY RECOMMENDATIONS

1. Load pickle/model files only from trusted
   sources because malicious pickle files
   can execute code.
2. Restrict access to model files and
   customer data.
3. Never store passwords, API keys or other
   secrets in source code.
4. Validate incoming data before predictions.
5. Protect customer information and follow
   applicable privacy requirements.

LIMITATIONS

1. The dataset is a public IBM telecom dataset.
2. Results may not generalize to boutique
   retail customers.
3. Subgroup checks are limited to two
   available attributes.
4. Model performance may change over time.
5. Human review and ongoing monitoring
   are recommended before deployment.
"""

    fig.text(
        0.09, 0.87,
        review_text,
        fontsize=10.5,
        va="top",
        linespacing=1.5
    )

    plt.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


# --------------------------------------------------
# 7. Completion message
# --------------------------------------------------

print("\nQuality review completed successfully!")

print(f"Subgroup results saved: {subgroup_path}")
print(f"PDF report saved: {pdf_path}")