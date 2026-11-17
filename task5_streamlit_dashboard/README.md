# Customer Churn Intelligence Dashboard

## Project Overview

The Customer Churn Intelligence Dashboard is an interactive web application built using Streamlit and Python. It uses a trained XGBoost machine learning pipeline to analyze customer churn risk and help businesses explore customer retention opportunities.

The project uses the public IBM Telco Customer Churn dataset as a simulated telecom retention case study. It does not contain actual boutique transaction data.

## Dashboard Features

### 1. Overview
- Total customer count and historical churn rate.
- High-risk customer count and average predicted churn probability.
- Actual churn and predicted risk visualizations.
- Top 10 high-risk customers table.
- Interactive filters for churn probability, risk category, and contract type.
- CSV downloads for customer records.

### 2. Churn Drivers
- Top 15 XGBoost feature importance visualization.
- Historical churn analysis by contract, internet service, payment method, gender, and senior citizen status.
- Customer group comparison table.

### 3. Recommendations
- High-risk and medium-risk customer summaries.
- Business recommendations for customer retention.
- Suggested actions for customer support, contracts, payments, and onboarding.

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- Joblib

## Project Structure

```text
customer-churn-cleaning/
│
├── data/
│   └── cleaned/
│       └── cleaned_customer_churn.csv
│
├── task4_xgboost_tuning/
│   └── models/
│       └── best_xgboost_model.pkl
│
├── task5_streamlit_dashboard/
│   ├── app.py
│   ├── README.md
│   ├── outputs/
│   └── reports/
│
└── requirements.txt
```

## Run Locally

### Step 1: Clone the repository

```bash
git clone https://github.com/HafsaSaddique/customer-churn-cleaning.git
```

### Step 2: Open the project folder

```bash
cd customer-churn-cleaning
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the dashboard

```bash
python -m streamlit run task5_streamlit_dashboard/app.py
```

The dashboard will open in your browser.

## Model Information

The application reuses the tuned XGBoost pipeline developed in Task 4. The pipeline includes preprocessing and classification, and generates churn probabilities using `predict_proba()`.

Risk categories used in the dashboard:

- Low Risk: Below 30%
- Medium Risk: 30% to 60%
- High Risk: Above 60%

These are dashboard display thresholds, not independently validated business decision thresholds.

## Important Limitations

- The dataset is a public IBM Telco Customer Churn dataset.
- Predictions shown for historical records are demonstration scores, not validated forecasts of future customer behaviour.
- Feature importance indicates how the model uses features; it does not establish causation.
- Retention recommendations are general business suggestions and should be evaluated before implementation.

## Deployment

The application is designed for deployment through Streamlit Community Cloud.

**Main application file:**

`task5_streamlit_dashboard/app.py`

**Requirements file:**

`requirements.txt`

**Live Dashboard:** To be added after successful deployment.

---

Developed as part of the Data Science Internship at Ezitech Software House.