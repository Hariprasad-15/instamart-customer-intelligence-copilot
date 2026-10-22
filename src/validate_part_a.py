import pandas as pd

customers = pd.read_csv("data/clean/customers.csv")
orders = pd.read_csv("data/clean/orders.csv")

assert len(customers) == 1200
assert len(orders) == 12309

assert (orders["status"] == "completed").sum() == 11110
assert (orders["status"] == "cancelled").sum() == 1199
assert (orders["status"] == "pending").sum() == 0

assert customers["city"].nunique() == 6
assert customers["age"].isna().sum() == 0

assert customers["gender"].value_counts()["female"] == 610
assert customers["gender"].value_counts()["male"] == 560
assert customers["gender"].value_counts()["unknown"] == 30

revenue = round(
    orders.loc[
        orders["status"] == "completed",
        "order_amount"
    ].sum(),
    2
)

assert revenue == 2998668.80

print("ALL CHECKPOINTS PASSED")