
# Customer Churn - Exploratory Data Analysis (EDA) Report

## 1. Project Overview

This report presents an exploratory data analysis of the cleaned IBM Telco
Customer Churn dataset. The purpose is to understand customer behavior,
identify patterns associated with churn, and provide insights that may
help improve customer retention.

## 2. Dataset Overview

- Total customers: 7,043
- Total columns: 21
- Missing values: 0
- Duplicate records: 0
- Churned customers: 1,869
- Customers who stayed: 5,174

## 3. Key Statistics

- Overall churn rate: 26.54%
- Customer retention rate: 73.46%
- Average customer tenure: 32.37 months
- Median customer tenure: 29 months
- Average monthly charge: 64.76

## 4. Customer Churn Analysis

### Contract Type

- Month-to-month contracts: 42.71% churn rate.
- One-year contracts: 11.27% churn rate.
- Two-year contracts: 2.83% churn rate.

Month-to-month customers have the highest churn rate among the contract
groups.

### Internet Service

- Fiber optic: 41.89% churn rate.
- DSL: 18.96% churn rate.
- No internet service: 7.40% churn rate.

Fiber optic customers show a higher churn rate than DSL customers.

### Payment Method

- Electronic check: 45.29% churn rate.
- Mailed check: 19.11% churn rate.
- Bank transfer (automatic): 16.71% churn rate.
- Credit card (automatic): 15.24% churn rate.

Electronic check users have the highest churn rate among payment methods.

### Tech Support

- No tech support: 41.64% churn rate.
- Tech support: 15.17% churn rate.
- No internet service: 7.40% churn rate.

Customers without tech support show a higher churn rate.

### Online Security

- No online security: 41.77% churn rate.
- Online security: 14.61% churn rate.
- No internet service: 7.40% churn rate.

Customers without online security show a higher churn rate.

### Customer Tenure

- 0-12 months: 47.44% churn rate.
- 13-24 months: 28.71% churn rate.
- 25-48 months: 20.39% churn rate.
- 49-72 months: 9.51% churn rate.

Customers with shorter tenure show higher churn rates. The first year
appears to be an important period for customer retention.

## 5. Key Business Insights

1. Month-to-month contract customers are more likely to leave than
   customers with longer contracts.
2. Electronic check users have the highest churn rate among payment
   method groups.
3. Fiber optic customers show higher churn than DSL customers.
4. Customers without tech support or online security show higher
   churn rates.
5. New customers, especially those with less than one year of tenure,
   have a higher churn rate.

## 6. Recommendations

1. Improve onboarding and customer engagement during the first year.
2. Consider retention offers for month-to-month customers.
3. Investigate customer satisfaction among fiber optic users.
4. Review the payment experience for electronic check users.
5. Promote the availability and benefits of technical support and
   online security services.

## 7. Visualizations

The following visualizations were created during the analysis:

1. Customer tenure histogram
2. Customer churn distribution
3. Monthly charges by churn status
4. Numerical correlation heatmap
5. Customer tenure versus total charges

All plots are saved in the `outputs/plots/` directory.

## 8. Conclusion

The analysis identifies several customer groups with higher churn rates,
particularly month-to-month customers, electronic check users, and
customers with shorter tenure. These patterns can help the business
identify areas for further investigation and develop customer retention
strategies.

Note: These findings describe associations in the dataset. They do not
establish that any individual feature directly causes customer churn.
The dataset is the IBM Telco Customer Churn dataset and represents
telecommunications customers, not actual boutique transactions.
