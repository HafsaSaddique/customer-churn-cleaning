
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = BASE_DIR / "outputs" / "cv_results.csv"
OUTPUT_PATH = BASE_DIR / "outputs" / "cv_results_plot.png"

# Load cross-validation results
results = pd.read_csv(RESULTS_PATH)

# Sort parameter combinations by mean CV AUC
results = results.sort_values(
    by="mean_test_score",
    ascending=False
).reset_index(drop=True)

# Plot mean CV AUC with standard deviation
plt.figure(figsize=(12, 6))

plt.errorbar(
    range(1, len(results) + 1),
    results["mean_test_score"],
    yerr=results["std_test_score"],
    fmt="o-",
    capsize=5
)

plt.axhline(
    y=0.8421,
    linestyle="--",
    label="Logistic Regression baseline (0.8421)"
)

plt.xlabel("Hyperparameter Combination (ranked)")
plt.ylabel("Mean Cross-Validation ROC-AUC")
plt.title("XGBoost Grid Search: Cross-Validation Results")
plt.xticks(range(1, len(results) + 1))
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(OUTPUT_PATH, dpi=300)
plt.close()

print("CV results plot saved:", OUTPUT_PATH)
print("\nTop 5 parameter combinations:")
print(
    results[["rank_test_score", "mean_test_score", "std_test_score"]]
    .head()
    .to_string(index=False)
)
