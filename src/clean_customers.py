import pandas as pd
import os

os.makedirs("data/clean", exist_ok=True)

customers = pd.read_csv(
    "data/raw/customers.csv",
    keep_default_na=False
)

# City cleaning
customers["city"] = (
    customers["city"]
    .str.strip()
    .str.title()
)

# Gender cleaning
customers["gender"] = (
    customers["gender"]
    .str.strip()
    .str.lower()
)

gender_map = {
    "m": "male",
    "male": "male",
    "f": "female",
    "female": "female",
    "": "unknown"
}

customers["gender"] = (
    customers["gender"]
    .map(gender_map)
    .fillna("unknown")
)

# Age cleaning
customers["age"] = pd.to_numeric(
    customers["age"],
    errors="coerce"
)

customers.loc[
    (customers["age"] < 10) |
    (customers["age"] > 100),
    "age"
] = pd.NA

customers["age"] = customers.groupby(
    "city"
)["age"].transform(
    lambda x: x.fillna(x.median())
)

# Datetime conversion
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"]
)

# Save
customers.to_csv(
    "data/clean/customers.csv",
    index=False
)

print("\nCITY COUNTS")
print(customers["city"].value_counts())

print("\nGENDER COUNTS")
print(customers["gender"].value_counts())

print("\nMissing Ages:")
print(customers["age"].isna().sum())