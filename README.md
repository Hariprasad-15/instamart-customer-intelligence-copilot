# Instamart Customer Intelligence Copilot

## Overview

Instamart Customer Intelligence Copilot is an end-to-end analytics and AI solution built for customer intelligence, support automation, and refund decision support.

The project combines:

- Data engineering and data quality remediation
- SQL analytics and reporting
- Customer segmentation using K-Means clustering
- Customer lapse prediction using machine learning
- Ticket classification with LLM guardrails
- Retrieval-Augmented Generation (RAG) for policy lookup
- Agent-based refund processing workflows

---

## Repository Structure

```text
data/
├── clean/
├── policies/
├── raw/
└── instamart.db

outputs/
├── customer_segments.csv
├── lapse_model.joblib
├── retention_list.csv
├── triage_results.csv
├── triage_budget_demo.csv
├── agent_runs.jsonl
├── refund_ledger.jsonl
└── rfm.csv

src/
├── part_a.py
├── part_b.py
├── lapse_model.py
├── part_c.py
└── part_d.py

sql/
└── queries.sql

state/
└── .gitkeep
```

---

# Reproduce

## 1. Create Virtual Environment

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Generate Dataset

```bash
python generate_data.py
```

---

## 4. Run Part A

```bash
python src/part_a.py
python src/clean_customers.py
python src/clean_orders.py
python src/create_database.py
python src/validate_part_a.py
```

---

## 5. Run Part B

```bash
python src/part_b.py
python src/lapse_model.py
```

---

## 6. Run Part C

```bash
python src/part_c.py
```

---

## 7. Run Part D

```bash
python src/part_d.py
```

---

# Part A – Data Engineering and SQL

## Objectives

- Audit raw datasets
- Clean customer records
- Clean order records
- Create SQLite database
- Validate outputs
- Write analytical SQL queries

## Results

### Customer Dataset

```text
Rows: 1200
Cities Standardized: 6
Missing Ages After Cleaning: 0
```

### Orders Dataset

```text
Rows After Cleaning: 12309

Completed Orders: 11110
Cancelled Orders: 1199
Pending Orders: 0
```

### Revenue

```text
INR 2,998,668.80
```

### Database

```text
data/instamart.db
```

Contains:

```text
customers
products
orders
```

---

# Part B – Customer Segmentation & Lapse Prediction

## RFM Analysis

Built customer-level RFM features:

- Recency
- Frequency
- Monetary

### RFM Summary

```text
Customers: 1175

Recency Median: 19
Recency Max: 180

Frequency Mean: 9.46
Frequency Max: 33

Monetary Mean: 2552.06
Monetary Max: 14766.55
```

---

## K-Means Clustering

### Cluster Selection

```text
K = 4
```

### Best Silhouette Score

```text
0.526
```

### Customer Segments

| Segment | Customers |
|----------|----------:|
| Occasional Shoppers | 436 |
| Premium Customers | 138 |
| Loyal Customers | 419 |
| At Risk Customers | 182 |

---

## PCA

```text
PC1 = 0.634
PC2 = 0.218

Total Explained Variance = 0.852
```

---

## Lapse Prediction

### Cohort

```text
Customers: 1156
Lapsed Customers: 443

Lapse Rate: 38.32%
```

### Train/Test Split

```text
Train: 924
Test: 232
```

---

## Logistic Regression

### Best Hyperparameter

```text
C = 0.1
```

### Results

```text
Precision = 0.655
Recall = 0.809
F1 Score = 0.724
```

---

## Random Forest

### Best Parameters

```text
max_depth = 3
min_samples_leaf = 5
```

### Results

```text
Precision = 0.680
Recall = 0.742
F1 Score = 0.710
```

---

## Selected Model

```text
Logistic Regression
```

Reason:

- Higher F1 score
- Higher recall
- Better suited for customer retention use cases

---

## Deliverables

```text
outputs/customer_segments.csv
outputs/lapse_model.joblib
outputs/retention_list.csv
```

---

# Part C – LLM Ticket Triage

## Input Guardrails

### Empty Tickets

```text
4
```

### Long Tickets

```text
4
```

---

## PII Redaction

### Emails

```text
55
```

### Phone Numbers

```text
91
```

### Order IDs

```text
132
```

### Total Redactions

```text
278
```

---

## Classification Output Validation

### JSON Classifications

```text
135
```

### Keyword Classifications

```text
12
```

### Invalid Outputs

```text
9
```

---

## Retry Handling

```text
19 transient failures recovered
0 permanent failures
```

---

## Deliverables

```text
outputs/triage_results.csv
outputs/triage_budget_demo.csv
```

---

# Part D – Retrieval-Augmented Generation & Refund Agent

## Policy Chunking

Chunk Size:

```text
40 words
```

Overlap:

```text
10 words
```

### Generated Chunks

```text
Total Chunks: 15
```

Distribution:

```text
refund_policy.txt      : 4
delivery_policy.txt    : 3
payment_policy.txt     : 3
escalation_policy.txt  : 3
privacy_policy.txt     : 2
```

---

## Retrieval Evaluation

### Answerable Questions

```text
12
```

### Retrieval Success

```text
12 / 12
```

### Grounded Answers

```text
12 / 12
```

---

## Unanswerable Questions

```text
2
```

### Successful Fallback Responses

```text
2
```

---

## Refund Agent

Implemented Tools:

```text
get_order()
search_policy()
calculate_refund()
issue_refund()
escalate_to_human()
```

---

## Agent Execution

### Scenarios Executed

```text
7
```

### Refunds Issued

```text
3
```

### Deliverables

```text
outputs/agent_runs.jsonl
outputs/refund_ledger.jsonl
```

---

# Limitations and Next Steps

## Current Limitations

- TF-IDF retrieval relies on keyword overlap and may miss semantic matches.
- Rule-driven refund calculations use simplified business logic.
- Agent workflow is deterministic and limited to predefined tools.
- Ticket classification uses a mock LLM environment.

---

## Future Enhancements

- Replace TF-IDF with vector embeddings.
- Add semantic search and reranking.
- Introduce real LLM API integration.
- Add workflow orchestration and approval routing.
- Deploy models and agents as production APIs.
- Implement monitoring and observability dashboards.

---

# Author

**Hariprasad K**
