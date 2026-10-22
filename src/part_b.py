import pandas as pd

orders = pd.read_csv(
    "data/clean/orders.csv"
)

orders["order_ts"] = pd.to_datetime(
    orders["order_ts"]
)

# Only completed orders
orders = orders[
    orders["status"] == "completed"
].copy()

snapshot_date = pd.Timestamp(
    "2025-10-01 00:00:00"
)

rfm = orders.groupby("customer_id").agg(
    recency=(
        "order_ts",
        lambda x: (
            snapshot_date - x.max()
        ).days
    ),
    frequency=(
        "order_id",
        "nunique"
    ),
    monetary=(
        "order_amount",
        "sum"
    )
).reset_index()

print("RFM Shape:", rfm.shape)

print("\nRecency")
print(rfm["recency"].describe())

print("\nFrequency")
print(rfm["frequency"].describe())

print("\nMonetary")
print(rfm["monetary"].describe())

# Save temporarily
rfm.to_csv(
    "outputs/rfm.csv",
    index=False
)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -----------------------------
# Scaling
# -----------------------------

X = rfm[[
    "recency",
    "frequency",
    "monetary"
]]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\n===== K-MEANS RESULTS =====")

for k in range(2, 9):

    kmeans = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels = kmeans.fit_predict(
        X_scaled
    )

    inertia = kmeans.inertia_

    silhouette = silhouette_score(
        X_scaled,
        labels
    )

    print(
        f"K={k} | "
        f"Inertia={inertia:.1f} | "
        f"Silhouette={silhouette:.3f}"
    )

# -----------------------------
# FINAL MODEL K=4
# -----------------------------

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

rfm["segment_id"] = kmeans.fit_predict(
    X_scaled
)

profile = rfm.groupby(
    "segment_id"
).agg(
    size=("customer_id","count"),
    recency=("recency","mean"),
    frequency=("frequency","mean"),
    monetary=("monetary","mean")
).round(2)

print("\n===== SEGMENT PROFILE =====")
print(profile)

# -----------------------------
# Segment Names
# -----------------------------

segment_names = {
    0: "Occasional Shoppers",
    1: "Premium Customers",
    2: "Loyal Customers",
    3: "At Risk Customers"
}

rfm["segment_name"] = rfm["segment_id"].map(segment_names)

# -----------------------------
# Premium Flag
# -----------------------------

premium_cluster = profile["monetary"].idxmax()

rfm["is_premium"] = (
    rfm["segment_id"] == premium_cluster
)

# -----------------------------
# PCA
# -----------------------------

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca.fit(X_scaled)

print("\n===== PCA =====")

print(
    "PC1:",
    round(pca.explained_variance_ratio_[0], 3)
)

print(
    "PC2:",
    round(pca.explained_variance_ratio_[1], 3)
)

print(
    "Total:",
    round(
        pca.explained_variance_ratio_.sum(),
        3
    )
)

# -----------------------------
# Save Output
# -----------------------------

import os

os.makedirs(
    "outputs",
    exist_ok=True
)

segment_output = rfm[
    [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium"
    ]
]

segment_output.to_csv(
    "outputs/customer_segments.csv",
    index=False
)

print(
    "\ncustomer_segments.csv created successfully"
)