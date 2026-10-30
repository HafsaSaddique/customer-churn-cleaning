
import pandas as pd

file_path = "data/raw/telco_customer_churn.csv"

df = pd.read_csv(file_path)

print("=== DATASET OVERVIEW ===")
print("Rows and columns:", df.shape)

print("\n=== COLUMN NAMES ===")
print(df.columns.tolist())

print("\n=== FIRST 5 ROWS ===")
print(df.head())

print("\n=== DATA TYPES ===")
print(df.dtypes)

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

print("\n=== DUPLICATE ROWS ===")
print(df.duplicated().sum())

print("\n=== CHURN VALUES ===")
if "Churn" in df.columns:
    print(df["Churn"].value_counts(dropna=False))
else:
    print("Churn column not found. Check the column names above.")
