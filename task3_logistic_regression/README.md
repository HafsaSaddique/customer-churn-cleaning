\# Task 3: Baseline Logistic Regression Churn Model



\## Objective

Build a baseline Logistic Regression model to predict customer churn using the cleaned IBM Telco Customer Churn dataset.



\## Tools and Libraries

\- Python

\- Pandas

\- Scikit-learn

\- Matplotlib

\- Joblib



\## Data Preparation

\- Loaded 7,043 customer records.

\- Used `churn\_flag` as the target variable.

\- Removed `customer\_id` from the prediction features.

\- Applied One-Hot Encoding to categorical features.

\- Applied Standard Scaling to numerical features.

\- Split the data into 80% training and 20% testing using stratification.



\## Model

A Logistic Regression model was trained using a Scikit-learn Pipeline. The pipeline includes preprocessing and classification.



\## Baseline Performance



| Metric | Score |

|---|---:|

| ROC-AUC | 0.8421 |

| Accuracy | 0.8055 |

| Precision | 0.6572 |

| Recall | 0.5588 |

| F1 Score | 0.6040 |



The model achieved an ROC-AUC of 0.8421, exceeding the required minimum of 0.70.



\## Evaluation

The model was evaluated using ROC-AUC and a confusion matrix.



Confusion Matrix:



| Actual / Predicted | No Churn | Churn |

|---|---:|---:|

| No Churn | 926 | 109 |

| Churn | 165 | 209 |



\## Output Files

\- `models/logistic\_regression\_model.pkl` - saved trained pipeline.

\- `outputs/metrics.csv` - evaluation metrics.

\- `outputs/roc\_curve.png` - ROC curve.

\- `outputs/confusion\_matrix.png` - confusion matrix.

\- `train\_model.py` - training and evaluation script.

\- `evaluate\_model.py` - evaluation visualization script.

\- `inspect\_data.py` - dataset inspection script.



\## How to Run



From the project root directory:



```bash

python task3\_logistic\_regression/inspect\_data.py

python task3\_logistic\_regression/train\_model.py

python task3\_logistic\_regression/evaluate\_model.py

```



Install dependencies from the project root:



```bash

python -m pip install -r requirements.txt

```



\## Limitations

The dataset has an imbalanced target distribution, with 26.54% churn and 73.46% non-churn customers. The model's recall is 55.88%, meaning it misses some actual churners. This is a baseline model, and future work could explore class weighting, threshold tuning, and other algorithms.



The dataset represents telecommunications customers, not actual boutique transactions.



\## Conclusion

The baseline Logistic Regression model achieved an ROC-AUC of 0.8421. It provides a starting point for comparing future customer churn prediction models.

