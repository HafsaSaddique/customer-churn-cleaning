
import pandas as pd
from pathlib import Path

# Load cleaned dataset
DATA_PATH = Path("data/cleaned/cleaned_customer_churn.csv")
df = pd.read_csv(DATA_PATH)

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== TARGET DISTRIBUTION ==========")
print(df["churn_flag"].value_counts())

print("\n========== TARGET PERCENTAGES ==========")
print(df["churn_flag"].value_counts(normalize=True) * 100)

print("\n========== FIRST FIVE RECORDS ==========")
print(df.head())
