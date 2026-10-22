import pandas as pd

# Load normal data
customers = pd.read_csv("data/raw/customers.csv")
products = pd.read_csv("data/raw/products.csv")
orders = pd.read_csv("data/raw/orders.csv")

# Load raw files preserving blanks
orders_raw = pd.read_csv(
    "data/raw/orders.csv",
    keep_default_na=False
)

customers_raw = pd.read_csv(
    "data/raw/customers.csv",
    keep_default_na=False
)

print("\n===== ORDERS AUDIT =====")

print("Total Rows:", len(orders))

print("Exact Duplicates:",
      orders.duplicated().sum())

print("Pending Orders:",
      (orders["status"] == "pending").sum())

print("Distinct Order IDs:",
      orders["order_id"].nunique())

invalid_amounts = pd.to_numeric(
    orders["order_amount"],
    errors="coerce"
).isna().sum()

print("Invalid Amounts:",
      invalid_amounts)

negative_delivery = (
    pd.to_numeric(
        orders["delivery_minutes"],
        errors="coerce"
    ) < 0
).sum()

print("Negative Delivery Times:",
      negative_delivery)

print(
    "Blank Delivery Times:",
    orders_raw["delivery_minutes"]
    .astype(str)
    .str.strip()
    .eq("")
    .sum()
)

print("\n===== CUSTOMERS AUDIT =====")

age_numeric = pd.to_numeric(
    customers["age"],
    errors="coerce"
)

print("Blank Ages:",
      age_numeric.isna().sum())

print(
    "Invalid Ages:",
    (
        (age_numeric < 10) |
        (age_numeric > 100)
    ).sum()
)

print(
    "Distinct Raw Cities:",
    customers["city"].nunique()
)

print(
    "Distinct Gender Spellings:",
    customers["gender"]
    .replace("", pd.NA)
    .dropna()
    .nunique()
)

print(
    "Blank Gender:",
    customers_raw["gender"]
    .astype(str)
    .str.strip()
    .eq("")
    .sum()
)