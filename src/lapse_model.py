import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =====================================================
# LOAD DATA
# =====================================================

orders = pd.read_csv("data/clean/orders.csv")
customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/raw/products.csv")

orders["order_ts"] = pd.to_datetime(
    orders["order_ts"]
)

customers["signup_date"] = pd.to_datetime(
    customers["signup_date"]
)

cutoff = pd.Timestamp("2025-09-01")

# =====================================================
# BUILD LABELS
# =====================================================

completed = orders[
    orders["status"] == "completed"
].copy()

cohort_customers = (
    completed[
        completed["order_ts"] < cutoff
    ]["customer_id"]
    .unique()
)

sept_orders = completed[
    (completed["order_ts"] >= cutoff)
    &
    (completed["order_ts"] < "2025-10-01")
]

active_in_sep = set(
    sept_orders["customer_id"]
)

dataset = pd.DataFrame(
    {
        "customer_id": cohort_customers
    }
)

dataset["lapsed"] = (
    ~dataset["customer_id"]
    .isin(active_in_sep)
).astype(int)

print("\n==== COHORT ====")

print(
    "Cohort Size:",
    len(dataset)
)

print(
    "Lapsed:",
    dataset["lapsed"].sum()
)

print(
    "Lapse %:",
    round(
        dataset["lapsed"].mean() * 100,
        2
    )
)

# =====================================================
# HISTORY BEFORE CUTOFF
# =====================================================

history = orders[
    orders["order_ts"] < cutoff
].copy()

history_completed = history[
    history["status"] == "completed"
].copy()

# =====================================================
# RECENCY
# =====================================================

recency = (
    cutoff -
    history_completed.groupby(
        "customer_id"
    )["order_ts"].max()
).dt.days.rename("recency")

# =====================================================
# FREQUENCY
# =====================================================

frequency = (
    history_completed.groupby(
        "customer_id"
    )["order_id"]
    .nunique()
    .rename("frequency")
)

# =====================================================
# MONETARY
# =====================================================

monetary = (
    history_completed.groupby(
        "customer_id"
    )["order_amount"]
    .sum()
    .rename("monetary")
)

# =====================================================
# AVG DISCOUNT
# =====================================================

avg_discount = (
    history_completed.groupby(
        "customer_id"
    )["discount_pct"]
    .mean()
    .rename("avg_discount")
)

# =====================================================
# AVG DELIVERY
# =====================================================

avg_delivery = (
    history_completed.groupby(
        "customer_id"
    )["delivery_minutes"]
    .mean()
    .rename("avg_delivery")
)

# =====================================================
# CANCEL RATE
# =====================================================

cancel_rate = (
    history.groupby("customer_id")
    .apply(
        lambda x:
        (
            x["status"] == "cancelled"
        ).mean()
    )
    .rename("cancel_rate")
)

# =====================================================
# FAVORITE CATEGORY
# =====================================================

history_completed = history_completed.merge(
    products[
        ["product_id", "category"]
    ],
    on="product_id",
    how="left"
)

favorite_category = (
    history_completed
    .groupby(
        ["customer_id", "category"]
    )
    .size()
    .reset_index(name="cnt")
    .sort_values(
        ["customer_id", "cnt"],
        ascending=[True, False]
    )
    .drop_duplicates("customer_id")
    [
        [
            "customer_id",
            "category"
        ]
    ]
    .rename(
        columns={
            "category":
            "favorite_category"
        }
    )
)

# =====================================================
# MERGE FEATURES
# =====================================================

dataset = dataset.merge(
    recency,
    on="customer_id"
)

dataset = dataset.merge(
    frequency,
    on="customer_id"
)

dataset = dataset.merge(
    monetary,
    on="customer_id"
)

dataset = dataset.merge(
    avg_discount,
    on="customer_id"
)

dataset = dataset.merge(
    avg_delivery,
    on="customer_id"
)

dataset = dataset.merge(
    cancel_rate,
    on="customer_id"
)

dataset = dataset.merge(
    favorite_category,
    on="customer_id",
    how="left"
)

dataset = dataset.merge(
    customers,
    on="customer_id",
    how="left"
)

# =====================================================
# TENURE
# =====================================================

dataset["tenure"] = (
    cutoff -
    dataset["signup_date"]
).dt.days

print("\nDataset Shape:")
print(dataset.shape)

print("\nMissing Values:")
print(dataset.isna().sum())

# =====================================================
# FEATURES / TARGET
# =====================================================

X = dataset.drop(
    columns=[
        "customer_id",
        "lapsed",
        "signup_date"
    ]
)

y = dataset["lapsed"]

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("\nTrain Rows:", len(X_train))
print("Test Rows:", len(X_test))

# =====================================================
# PREPROCESSING
# =====================================================

categorical_cols = [
    "city",
    "gender",
    "favorite_category"
]

numeric_cols = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount",
    "avg_delivery",
    "cancel_rate",
    "age",
    "tenure"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_cols
        ),
        (
            "cat",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            categorical_cols
        )
    ]
)

# =====================================================
# LOGISTIC REGRESSION
# =====================================================

log_pipeline = Pipeline(
    [
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)

log_grid = GridSearchCV(
    log_pipeline,
    {
        "model__C": [
            0.01,
            0.1,
            1,
            10,
            100
        ]
    },
    cv=5,
    scoring="f1",
    n_jobs=1
)

log_grid.fit(
    X_train,
    y_train
)

print("\n==== LOGISTIC ====")

print(
    "Best Params:",
    log_grid.best_params_
)

print(
    "Best CV F1:",
    round(
        log_grid.best_score_,
        3
    )
)

# =====================================================
# RANDOM FOREST
# =====================================================

rf_pipeline = Pipeline(
    [
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)

rf_grid = GridSearchCV(
    rf_pipeline,
    {
        "model__max_depth": [
            3,
            5,
            8
        ],
        "model__min_samples_leaf": [
            5,
            10,
            20
        ]
    },
    cv=5,
    scoring="f1",
    n_jobs=1
)

rf_grid.fit(
    X_train,
    y_train
)

print("\n==== RANDOM FOREST ====")

print(
    "Best Params:",
    rf_grid.best_params_
)

print(
    "Best CV F1:",
    round(
        rf_grid.best_score_,
        3
    )
)

# =====================================================
# EVALUATION
# =====================================================

models = {
    "Logistic":
    log_grid.best_estimator_,

    "RandomForest":
    rf_grid.best_estimator_
}

best_f1 = 0
best_model = None

for name, model in models.items():

    pred = model.predict(X_test)

    f1 = f1_score(
        y_test,
        pred,
        zero_division=0
    )

    print("\n" + "=" * 50)

    print(name)

    print(
        confusion_matrix(
            y_test,
            pred
        )
    )

    print(
        "Precision:",
        round(
            precision_score(
                y_test,
                pred,
                zero_division=0
            ),
            3
        )
    )

    print(
        "Recall:",
        round(
            recall_score(
                y_test,
                pred,
                zero_division=0
            ),
            3
        )
    )

    print(
        "F1:",
        round(
            f1,
            3
        )
    )

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
from sklearn.utils import resample

train_df = X_train.copy()
train_df["lapsed"] = y_train.values

majority = train_df[
    train_df["lapsed"] == 0
]

minority = train_df[
    train_df["lapsed"] == 1
]

minority_upsampled = resample(
    minority,
    replace=True,
    n_samples=len(majority),
    random_state=42
)

balanced_train = pd.concat(
    [
        majority,
        minority_upsampled
    ]
)

X_train_bal = balanced_train.drop(
    columns=["lapsed"]
)

y_train_bal = balanced_train["lapsed"]

print(
    "Balanced Counts:\n",
    y_train_bal.value_counts()
)

# =====================================================
# SAVE MODEL
# =====================================================

os.makedirs(
    "outputs",
    exist_ok=True
)

joblib.dump(
    best_model,
    "outputs/lapse_model.joblib"
)

print("\nModel Saved")

# =====================================================
# RELOAD CHECK
# =====================================================

loaded_model = joblib.load(
    "outputs/lapse_model.joblib"
)

original_pred = best_model.predict(X_test)

loaded_pred = loaded_model.predict(
    X_test
)

print(
    "Predictions Match:",
    (original_pred == loaded_pred).all()
)

# =====================================================
# OVERSAMPLING EXPERIMENT
# =====================================================

from sklearn.utils import resample

train_df = X_train.copy()
train_df["lapsed"] = y_train.values

majority = train_df[
    train_df["lapsed"] == 0
]

minority = train_df[
    train_df["lapsed"] == 1
]

minority_upsampled = resample(
    minority,
    replace=True,
    n_samples=len(majority),
    random_state=42
)

balanced_train = pd.concat(
    [majority, minority_upsampled]
)

X_train_bal = balanced_train.drop(
    columns=["lapsed"]
)

y_train_bal = balanced_train["lapsed"]

print("\n===== BALANCED DATASET =====")
print(y_train_bal.value_counts())

balanced_logistic = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        LogisticRegression(
            C=log_grid.best_params_["model__C"],
            max_iter=1000
        )
    )
])

balanced_logistic.fit(
    X_train_bal,
    y_train_bal
)

balanced_pred = balanced_logistic.predict(
    X_test
)

print("\n===== BALANCED LOGISTIC =====")

print(
    "Precision:",
    round(
        precision_score(
            y_test,
            balanced_pred,
            zero_division=0
        ),
        3
    )
)

print(
    "Recall:",
    round(
        recall_score(
            y_test,
            balanced_pred,
            zero_division=0
        ),
        3
    )
)

print(
    "F1:",
    round(
        f1_score(
            y_test,
            balanced_pred,
            zero_division=0
        ),
        3
    )
)

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

rf_model = rf_grid.best_estimator_

feature_names = (
    rf_model
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

importances = (
    rf_model
    .named_steps["model"]
    .feature_importances_
)

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

print("\n===== TOP 10 FEATURES =====")
print(
    importance_df.head(10)
)

# =====================================================
# SELECT BEST MODEL
# =====================================================

log_f1 = f1_score(
    y_test,
    log_grid.best_estimator_.predict(X_test),
    zero_division=0
)

rf_f1 = f1_score(
    y_test,
    rf_grid.best_estimator_.predict(X_test),
    zero_division=0
)

if log_f1 >= rf_f1:
    best_model = log_grid.best_estimator_
    selected_model_name = "LogisticRegression"
else:
    best_model = rf_grid.best_estimator_
    selected_model_name = "RandomForest"

print("\nSelected Model:")
print(selected_model_name)

# =====================================================
# SAVE MODEL
# =====================================================

import os
import joblib

os.makedirs(
    "outputs",
    exist_ok=True
)

joblib.dump(
    best_model,
    "outputs/lapse_model.joblib"
)

loaded_model = joblib.load(
    "outputs/lapse_model.joblib"
)

original_predictions = best_model.predict(
    X_test
)

loaded_predictions = loaded_model.predict(
    X_test
)

print(
    "\nPredictions Match:",
    (
        original_predictions ==
        loaded_predictions
    ).all()
)

# =====================================================
# RETENTION LIST
# =====================================================

segments = pd.read_csv(
    "outputs/customer_segments.csv"
)

premium_customers = segments[
    segments["is_premium"] == True
][["customer_id"]]

test_customer_ids = (
    dataset.loc[
        X_test.index,
        ["customer_id"]
    ]
)

retention_list = pd.DataFrame({
    "customer_id":
        test_customer_ids["customer_id"].values,

    "lapse_probability":
        best_model.predict_proba(X_test)[:, 1],

    "actually_lapsed":
        y_test.values
})

retention_list = retention_list.merge(
    premium_customers,
    on="customer_id"
)

retention_list = retention_list.sort_values(
    "lapse_probability",
    ascending=False
)

retention_list = retention_list.head(10)

retention_list.to_csv(
    "outputs/retention_list.csv",
    index=False
)

print("\n===== RETENTION LIST =====")
print(retention_list)

print(
    "\nActually Lapsed Count:",
    retention_list["actually_lapsed"].sum()
)