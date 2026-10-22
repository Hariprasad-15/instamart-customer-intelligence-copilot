import pandas as pd
import numpy as np
import os

os.makedirs("data/clean", exist_ok=True)

orders = pd.read_csv(
    "data/raw/orders.csv",
    keep_default_na=False
)

products = pd.read_csv(
    "data/raw/products.csv"
)

# -------------------------------------------------
# 1. Convert timestamp
# -------------------------------------------------

orders["order_ts"] = pd.to_datetime(
    orders["order_ts"]
)

# -------------------------------------------------
# 2. Remove exact duplicates
# -------------------------------------------------

orders = orders.drop_duplicates()

# -------------------------------------------------
# 3. Keep latest record per order_id
# -------------------------------------------------

orders = (
    orders
    .sort_values("order_ts")
    .groupby("order_id", as_index=False)
    .tail(1)
)

# -------------------------------------------------
# 4. Fix order_amount
# -------------------------------------------------

orders["order_amount"] = pd.to_numeric(
    orders["order_amount"],
    errors="coerce"
)

price_lookup = (
    products
    .set_index("product_id")["price_inr"]
)

missing_amount = orders["order_amount"].isna()

orders.loc[
    missing_amount,
    "order_amount"
] = (
    orders.loc[missing_amount, "product_id"]
    .map(price_lookup)
    * orders.loc[missing_amount, "quantity"]
    * (
        1 -
        orders.loc[
            missing_amount,
            "discount_pct"
        ] / 100
    )
).round(2)

# -------------------------------------------------
# 5. Fix delivery minutes
# -------------------------------------------------

orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"],
    errors="coerce"
)

orders.loc[
    orders["delivery_minutes"] < 0,
    "delivery_minutes"
] = np.nan

# merge category
orders = orders.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

completed_mask = (
    orders["status"] == "completed"
)

category_medians = (
    orders[completed_mask]
    .groupby("category")["delivery_minutes"]
    .transform("median")
)

orders.loc[
    completed_mask,
    "delivery_minutes"
] = (
    orders.loc[completed_mask, "delivery_minutes"]
    .fillna(category_medians)
)

orders = orders.drop(columns=["category"])

# -------------------------------------------------
# save clean file
# -------------------------------------------------

orders.to_csv(
    "data/clean/orders.csv",
    index=False
)

print("Rows:", len(orders))

print(
    "Completed:",
    (orders["status"] == "completed").sum()
)

print(
    "Cancelled:",
    (orders["status"] == "cancelled").sum()
)

print(
    "Pending:",
    (orders["status"] == "pending").sum()
)

print(
    "Missing Amount:",
    orders["order_amount"].isna().sum()
)

print(
    "Missing Delivery(Completed):",
    orders.loc[
        orders["status"] == "completed",
        "delivery_minutes"
    ].isna().sum()
)

print(
    "Missing Delivery(Cancelled):",
    orders.loc[
        orders["status"] == "cancelled",
        "delivery_minutes"
    ].isna().sum()
)

print(
    "Revenue:",
    round(
        orders.loc[
            orders["status"] == "completed",
            "order_amount"
        ].sum(),
        2
    )
)