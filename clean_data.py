
import pandas as pd
from pathlib import Path

# File paths
input_file = Path("data/raw/telco_customer_churn.csv")
output_folder = Path("data/cleaned")
output_folder.mkdir(parents=True, exist_ok=True)

output_file = output_folder / "cleaned_customer_churn.csv"
dictionary_file = output_folder / "data_dictionary.csv"

# 1. Load original dataset
df = pd.read_csv(input_file)

print("Original dataset shape:", df.shape)

# 2. Remove duplicate records
df = df.drop_duplicates()

# 3. Clean TotalCharges
# Convert blank spaces into missing values, then numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].astype(str).str.strip(),
    errors="coerce"
)

# 4. Handle missing TotalCharges
# For new customers with zero tenure, total charges are zero.
missing_before = df["TotalCharges"].isna().sum()

df["TotalCharges"] = df["TotalCharges"].fillna(
    df["MonthlyCharges"] * df["tenure"]
)

# 5. Rename columns
column_mapping = {
    "customerID": "customer_id",
    "gender": "gender",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "tenure": "tenure",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charge",
    "TotalCharges": "total_charge",
    "Churn": "churn_flag"
}

df = df.rename(columns=column_mapping)

# 6. Convert churn into binary values
df["churn_flag"] = df["churn_flag"].map({
    "No": 0,
    "Yes": 1
})

# 7. Remove leading/trailing spaces from text columns
for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].str.strip()

# 8. Validate data
assert df["customer_id"].notna().all()
assert df["churn_flag"].isin([0, 1]).all()
assert df.isnull().sum().sum() == 0
assert df["customer_id"].is_unique

# 9. Save cleaned CSV
df.to_csv(output_file, index=False)

# 10. Create data dictionary
descriptions = {
    "customer_id": "Unique customer identifier",
    "gender": "Customer gender",
    "senior_citizen": "Whether customer is a senior citizen (0 or 1)",
    "partner": "Whether customer has a partner",
    "dependents": "Whether customer has dependents",
    "tenure": "Number of months the customer has stayed",
    "phone_service": "Whether customer has phone service",
    "multiple_lines": "Whether customer has multiple phone lines",
    "internet_service": "Customer internet service type",
    "online_security": "Whether customer has online security",
    "online_backup": "Whether customer has online backup",
    "device_protection": "Whether customer has device protection",
    "tech_support": "Whether customer has technical support",
    "streaming_tv": "Whether customer has streaming TV",
    "streaming_movies": "Whether customer has streaming movies",
    "contract": "Customer contract type",
    "paperless_billing": "Whether customer uses paperless billing",
    "payment_method": "Customer payment method",
    "monthly_charge": "Monthly customer charge",
    "total_charge": "Total customer charges",
    "churn_flag": "Customer churn indicator: 1 = Yes, 0 = No"
}

dictionary = pd.DataFrame({
    "column_name": df.columns,
    "data_type": [str(df[col].dtype) for col in df.columns],
    "description": [descriptions[col] for col in df.columns]
})

dictionary.to_csv(dictionary_file, index=False)

# 11. Print final report
print("\n=== CLEANING REPORT ===")
print("Cleaned dataset shape:", df.shape)
print("Missing TotalCharges before handling:", missing_before)
print("Remaining missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())
print("\nChurn flag counts:")
print(df["churn_flag"].value_counts().sort_index())
print("\nCleaned CSV saved:", output_file)
print("Data dictionary saved:", dictionary_file)
print("\nCleaning completed successfully!")
