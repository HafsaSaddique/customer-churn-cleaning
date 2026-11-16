# Task 4: XGBoost Hyperparameter Tuning and Quality Review

## Objective

Improve the baseline customer churn prediction model using XGBoost, hyperparameter tuning, and stratified cross-validation. Evaluate the final model and document its quality, fairness considerations, security risks, assumptions, and limitations.

## Dataset

- Input: `data/cleaned/cleaned_customer_churn.csv`
- Dataset: IBM Telco Customer Churn
- Target variable: `churn_flag`
- 0 = Customer stays
- 1 = Customer churns
- `customer_id` was excluded from model training.

The IBM telecom dataset is used as a simulated customer-retention case study. It is not actual boutique transaction data.

## Methodology

### Data Preprocessing

- Numerical features: Median imputation and StandardScaler.
- Categorical features: Most-frequent imputation and OneHotEncoder.
- Used Scikit-learn Pipeline and ColumnTransformer.
- Used an 80/20 stratified train-test split.
- Random state: 42.

### Model Training and Hyperparameter Tuning

- Algorithm: XGBoost Classifier.
- Cross-validation: StratifiedKFold with 5 folds.
- Hyperparameter tuning: GridSearchCV.
- Scoring metric: ROC-AUC.
- Tested 8 parameter combinations across 5 folds (40 fits).

### Best Hyperparameters

- `colsample_bytree`: 0.8
- `learning_rate`: 0.05
- `max_depth`: 3
- `n_estimators`: 100
- `subsample`: 0.8

## Model Performance

| Metric | Result |
|---|---:|
| Best mean CV ROC-AUC | 0.8502 |
| Test ROC-AUC | 0.8481 |
| Test Accuracy | 80.77% |
| Test Recall | 52.41% |
| Logistic Regression baseline ROC-AUC | 0.8421 |

The tuned XGBoost model achieved a higher mean cross-validation ROC-AUC than the baseline Logistic Regression model.

## Cross-Validation Results

The mean ROC-AUC and standard deviation for the tested parameter combinations are saved in `outputs/cv_results.csv`.

The comparison plot is saved as `outputs/cv_results_plot.png`. It displays the mean cross-validation ROC-AUC with standard deviation and the Logistic Regression baseline.

## Fairness and Subgroup Checks

The model was evaluated across gender and senior citizen groups using the held-out test data.

| Group | ROC-AUC | Recall |
|---|---:|---:|
| Female | 0.8476 | 0.5233 |
| Male | 0.8497 | 0.5249 |
| Non-senior citizens | 0.8494 | 0.4964 |
| Senior citizens | 0.7905 | 0.6020 |

The senior citizen subgroup showed lower ROC-AUC than the non-senior citizen subgroup. This difference should be investigated further and does not, by itself, establish discrimination.

These checks are exploratory and do not establish that the model is fair. Additional subgroup and intersectional fairness evaluations are required.

Detailed subgroup results are saved in `outputs/subgroup_bias_checks.csv`.

## Security Review

- Load pickle model files only from trusted sources because malicious pickle files may execute code.
- Restrict access to model files and customer data.
- Do not store passwords, API keys, or other secrets in source code.
- Validate incoming data before making predictions.
- Protect customer information and follow applicable privacy requirements.

## Model Assumptions

- Historical customer information can help predict future churn.
- Input features have the same meaning and format as the training data.
- Target labels are correctly recorded.
- Training data is sufficiently representative of future customers.

## Trade-offs and Limitations

- XGBoost can capture complex patterns but is less interpretable than Logistic Regression.
- Hyperparameter tuning improves model selection but requires additional computation.
- A prediction threshold of 0.5 may not be optimal for every business situation.
- Recall of 52.41% means the model misses some customers who actually churn.
- Results may not generalize to boutique retail customers because the dataset represents telecommunications customers.
- Model performance may change over time and requires monitoring.

## Output Files

- `models/best_xgboost_model.pkl` — Saved tuned XGBoost pipeline.
- `outputs/cv_results.csv` — GridSearchCV results.
- `outputs/model_summary.csv` — Model performance summary.
- `outputs/cv_results_plot.png` — Cross-validation comparison plot.
- `outputs/subgroup_bias_checks.csv` — Subgroup evaluation results.
- `reports/xgboost_quality_review.pdf` — Quality and security review.

## How to Run

Install project dependencies:

```bash
python -m pip install -r requirements.txt
```

Train and tune the XGBoost model:

```bash
python task4_xgboost_tuning/train_xgboost.py
```

Generate the cross-validation results plot:

```bash
python task4_xgboost_tuning/plot_cv_results.py
```

Generate subgroup checks and the quality review PDF:

```bash
python task4_xgboost_tuning/quality_review.py
```

## Skills

XGBoost, GridSearchCV, StratifiedKFold, Scikit-learn, Machine Learning pipelines, hyperparameter tuning, ROC-AUC, subgroup evaluation, fairness considerations, model persistence, and security review.