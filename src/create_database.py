import pandas as pd
import sqlite3

customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/raw/products.csv")
orders = pd.read_csv("data/clean/orders.csv")

conn = sqlite3.connect("data/instamart.db")

customers.to_sql(
    "customers",
    conn,
    index=False,
    if_exists="replace"
)

products.to_sql(
    "products",
    conn,
    index=False,
    if_exists="replace"
)

orders.to_sql(
    "orders",
    conn,
    index=False,
    if_exists="replace"
)

print("Database Created Successfully")

for table in ["customers", "products", "orders"]:
    count = pd.read_sql_query(
        f"SELECT COUNT(*) cnt FROM {table}",
        conn
    )
    print(table, ":", count["cnt"][0])

conn.close()