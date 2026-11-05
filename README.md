\# Customer Churn Data Cleaning



\## Project Overview



This project cleans and prepares the IBM Telco Customer Churn dataset for customer-retention analysis. The telecom dataset is used as a simulated boutique customer-retention case study. It is not actual boutique transaction data.



\## Dataset Source



Original dataset: https://github.com/IBM/telco-customer-churn-on-icp4d



The dataset contains 7,043 customer records and 21 original columns.



\## Cleaning Steps



1\. Loaded the original CSV using pandas.

2\. Checked column names, data types, missing values, and duplicates.

3\. Removed duplicate records.

4\. Converted `TotalCharges` to numeric format.

5\. Identified 11 blank `TotalCharges` values and handled them using monthly charges multiplied by tenure.

6\. Renamed columns to lowercase, descriptive names.

7\. Converted `Churn` into binary `churn\_flag`: Yes = 1 and No = 0.

8\. Exported the cleaned dataset and data dictionary.



\## Output Files



\- `data/cleaned/cleaned\_customer\_churn.csv` — cleaned customer dataset.

\- `data/cleaned/data\_dictionary.csv` — column names, data types, and descriptions.

\- `clean\_data.py` — data cleaning script.

\- `inspect\_dataset.py` — initial dataset inspection script.



\## Validation Results



\- Total records: 7,043

\- Total columns: 21

\- Missing values after cleaning: 0

\- Duplicate rows: 0

\- Churn values: 0 and 1 only



\## How to Run



Install dependencies:



```bash

python -m pip install pandas numpy

```



Inspect the original dataset:



```bash

python inspect\_dataset.py

```



Run the cleaning process:



```bash

python clean\_data.py

```



\## Limitations



This project uses a telecom customer churn dataset to demonstrate customer-retention analysis in a simulated boutique context. It does not contain real boutique purchases, products, or sales transactions. Missing total charges were estimated for records with blank values using monthly charges multiplied by tenure.



\## Skills



Python, pandas, NumPy, data cleaning, CSV processing, and documentation.

## Task 2: Exploratory Data Analysis (EDA)

### Objective
Performed exploratory data analysis on the cleaned IBM Telco Customer Churn dataset to identify customer behavior patterns and factors associated with churn.

### Tools and Libraries
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook

### Analysis Performed
- Dataset overview and summary statistics
- Customer churn distribution
- Customer tenure analysis
- Monthly charges analysis
- Correlation analysis
- Churn analysis by contract type
- Churn analysis by internet service
- Churn analysis by payment method
- Churn analysis by technical support
- Churn analysis by online security

### Visualizations
All five visualizations are available in `outputs/plots/`:

1. `01_customer_tenure_histogram.png`
2. `02_churn_distribution.png`
3. `03_monthly_charges_boxplot.png`
4. `04_correlation_heatmap.png`
5. `05_tenure_vs_total_charges.png`

### Reports
- EDA Notebook: `notebooks/customer_churn_eda.ipynb`
- Insights Report: `reports/eda_insights.md`
- PDF Report: `reports/customer_churn_eda_report.pdf`

### Key Findings
- Overall customer churn rate: 26.54%.
- Month-to-month contract customers have a 42.71% churn rate.
- Electronic check users have a 45.29% churn rate.
- Customers with 0-12 months of tenure have a 47.44% churn rate.
- Customers with 49-72 months of tenure have a 9.51% churn rate.

### Dataset Note
The analysis uses the IBM Telco Customer Churn dataset. It represents telecommunications customers and is used as a simulated customer-retention case study, not actual boutique transaction data.

## Limitations

This project uses a telecommunications customer churn dataset to demonstrate customer-retention analysis in a simulated boutique context. It does not contain real boutique purchases, products, or sales transactions. The 11 blank `TotalCharges` values were estimated using monthly charges multiplied by tenure. EDA findings describe associations and should not be interpreted as proof of causation.

## Skills

Python, pandas, NumPy, data cleaning, exploratory data analysis, data visualization, Jupyter Notebook, CSV processing, and documentation.