# =====================================================
# IMPORTS
# =====================================================

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import os
import pandas as pd
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mock_llm import chat

# =====================================================
# TASK 1 - CHUNKING
# =====================================================

POLICY_DIR = "data/policies"

all_chunks = []

for filename in os.listdir(POLICY_DIR):

    if not filename.endswith(".txt"):
        continue

    path = os.path.join(
        POLICY_DIR,
        filename
    )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        text = f.read()

    words = text.split()

    start = 0

    while True:

        chunk_words = words[
            start:start + 40
        ]

        all_chunks.append(
            {
                "source": filename,
                "text": " ".join(chunk_words)
            }
        )

        if start + 40 >= len(words):
            break

        start += 30

print(
    "\nTotal Chunks:",
    len(all_chunks)
)

counts = Counter(
    chunk["source"]
    for chunk in all_chunks
)

print("\nChunk Counts")

for source, count in sorted(counts.items()):

    print(
        f"{source}: {count}"
    )

# =====================================================
# TASK 2 - RETRIEVAL
# =====================================================

chunk_texts = [
    chunk["text"]
    for chunk in all_chunks
]

vectorizer = TfidfVectorizer()

chunk_vectors = vectorizer.fit_transform(
    chunk_texts
)

def retrieve(query, k=3):

    query_vector = vectorizer.transform(
        [query]
    )

    similarities = cosine_similarity(
        query_vector,
        chunk_vectors
    )[0]

    top_indices = (
        similarities
        .argsort()[::-1][:k]
    )

    results = []

    for idx in top_indices:

        results.append(
            {
                "source":
                    all_chunks[idx]["source"],

                "score":
                    float(similarities[idx]),

                "text":
                    all_chunks[idx]["text"]
            }
        )

    return results

print("\n===== RETRIEVAL TEST =====")

results = retrieve(
    "refund policy late delivery",
    3
)

for i, r in enumerate(results, 1):

    print(f"\nResult {i}")

    print(
        "Source:",
        r["source"]
    )

    print(
        "Score:",
        round(
            r["score"],
            4
        )
    )

# =====================================================
# TASK 3 - POLICY ASSISTANT
# =====================================================

def answer(question):

    retrieved = retrieve(
        question,
        k=3
    )

    context = []

    for chunk in retrieved:

        context.append(
            f"[{chunk['source']}]\n"
            f"{chunk['text']}"
        )

    user_content = (
        "CONTEXT:\n"
        + "\n\n".join(context)
        + "\n\nQUESTION:\n"
        + question
    )

    response = chat(
        [
            {
                "role": "system",
                "content":
                """
POLICY ASSISTANT

Answer only from the supplied context.

If information is unavailable,
begin the answer with:

I could not find
"""
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
    )

    return {
        "answer":
            response
            .choices[0]
            .message
            .content,

        "sources":
            [
                x["source"]
                for x in retrieved
            ]
    }

print(
    "\n===== POLICY ASSISTANT TEST ====="
)

sample_response = answer(
    "What is the refund policy for late deliveries?"
)

print(
    "\nANSWER:"
)

print(
    sample_response["answer"]
)

print(
    "\nSOURCES:"
)

print(
    sample_response["sources"]
)
# =====================================================
# POLICY EVALUATION
# =====================================================

print("\n===== POLICY EVALUATION =====")

eval_df = pd.read_csv(
    "data/raw/policy_eval.csv"
)

print(
    eval_df[
        ["question", "expected_doc", "expected_fact"]
    ].head()
)

retrieval_hits = 0
grounded_hits = 0
fallback_hits = 0

answerable_count = 0
unanswerable_count = 0

for _, row in eval_df.iterrows():

    expected_doc = str(
        row["expected_doc"]
    ).strip()

    expected_fact = str(
        row["expected_fact"]
    ).strip()

    question = row["question"]

    result = answer(question)

    answer_text = result["answer"]

    sources = result["sources"]

    is_answerable = (
        expected_doc != ""
        and expected_doc.lower() != "nan"
        and expected_doc.lower() != "none"
    )

    if is_answerable:

        answerable_count += 1

        if expected_doc in sources:
            retrieval_hits += 1

        if (
            expected_fact != ""
            and expected_fact.lower() != "nan"
            and expected_fact in answer_text
        ):
            grounded_hits += 1

    else:

        unanswerable_count += 1

        if answer_text.startswith(
            "I could not find"
        ):
            fallback_hits += 1

print("\nAnswerable Questions:",
      answerable_count)

print("Retrieval Hits:",
      retrieval_hits)

print("Grounded Hits:",
      grounded_hits)

print("Unanswerable Questions:",
      unanswerable_count)

print("Fallback Hits:",
      fallback_hits)

def get_order(order_id):
    pass

def search_policy(query):
    return answer(query)

def calculate_refund(order):
    pass

def issue_refund(order_id, amount):
    pass

def escalate_to_human(order_id, reason):
    pass

TOOLS_SPEC = {
    "get_order": "...",
    "search_policy": "...",
    "calculate_refund": "...",
    "issue_refund": "...",
    "escalate_to_human": "..."
}

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}

#run_agent()
#resume_agent()

# =====================================================
# AGENT TOOLS
# =====================================================

import sqlite3
import json
import os

conn = sqlite3.connect("data/instamart.db")


def get_order(order_id):

    query = """
    SELECT *
    FROM orders
    WHERE order_id = ?
    """

    result = pd.read_sql_query(
        query,
        conn,
        params=[order_id]
    )

    if len(result) == 0:
        return {
            "found": False
        }

    return result.iloc[0].to_dict()


def search_policy(query):

    result = answer(query)

    return {
        "answer": result["answer"],
        "sources": result["sources"]
    }


def calculate_refund(order):

    if not order:
        return {
            "refund_amount": 0,
            "needs_approval": False
        }

    if order["status"] != "completed":

        return {
            "refund_amount": 0,
            "needs_approval": False
        }

    if order["delivery_minutes"] > 45:

        refund_amount = float(
            order["order_amount"]
        )

        return {
            "refund_amount": refund_amount,
            "needs_approval":
                refund_amount > 1000
        }

    return {
        "refund_amount": 0,
        "needs_approval": False
    }


def issue_refund(order_id, amount):

    return {
        "action": "REFUND_ISSUED",
        "order_id": order_id,
        "amount": round(amount, 2)
    }


def escalate_to_human(
    order_id,
    reason
):

    return {
        "action": "ESCALATED",
        "order_id": order_id,
        "reason": reason
    }
TOOLS_SPEC = {
    "get_order":
        "Retrieve order information",

    "search_policy":
        "Search policy library",

    "calculate_refund":
        "Calculate refund amount",

    "issue_refund":
        "Issue refund",

    "escalate_to_human":
        "Escalate case"
}

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}
print("\n===== TOOL TEST =====")

sample_order = get_order("ORD-100001")

print(sample_order)

print(
    calculate_refund(
        sample_order
    )
)
# =====================================================
# AGENT
# =====================================================

import re
import json

os.makedirs(
    "outputs",
    exist_ok=True
)

agent_runs = []
refund_ledger = []

def extract_order_id(text):

    match = re.search(
        r"ORD-\d{6}",
        str(text)
    )

    if match:
        return match.group(0)

    return None


def run_agent():

    scenarios = pd.read_csv(
        "data/raw/agent_scenarios.csv"
    )

    for _, row in scenarios.iterrows():

        scenario_id = row["scenario_id"]
        customer_id = row["customer_id"]
        message = row["message"]

        order_id = extract_order_id(
            message
        )

        print(
            f"\nProcessing {scenario_id}"
        )

        print(
            f"Customer: {customer_id}"
        )

        print(
            f"Order: {order_id}"
        )

        if not order_id:

            result = {
                "scenario_id": scenario_id,
                "customer_id": customer_id,
                "action": "ESCALATED",
                "reason": "Order ID not found"
            }

            agent_runs.append(result)

            continue

        order = get_order(order_id)

        if not order.get("order_id"):

            result = {
                "scenario_id": scenario_id,
                "customer_id": customer_id,
                "order_id": order_id,
                "action": "ESCALATED",
                "reason": "Order not found"
            }

            agent_runs.append(result)

            continue

        refund = calculate_refund(order)

        if refund["refund_amount"] == 0:

            result = {
                "scenario_id": scenario_id,
                "customer_id": customer_id,
                "order_id": order_id,
                "action": "DENIED",
                "refund_amount": 0
            }

        elif refund["needs_approval"]:

            result = {
                "scenario_id": scenario_id,
                "customer_id": customer_id,
                "order_id": order_id,
                "action": "ESCALATED",
                "refund_amount":
                    refund["refund_amount"],
                "reason":
                    "Refund exceeds INR 1000"
            }

        else:

            result = {
                "scenario_id": scenario_id,
                "customer_id": customer_id,
                "order_id": order_id,
                "action": "REFUND",
                "refund_amount":
                    refund["refund_amount"]
            }

            refund_ledger.append(result)

        agent_runs.append(result)

    with open(
        "outputs/agent_runs.jsonl",
        "w",
        encoding="utf-8"
    ) as f:

        for item in agent_runs:

            f.write(
                json.dumps(item)
                + "\n"
            )

    with open(
        "outputs/refund_ledger.jsonl",
        "w",
        encoding="utf-8"
    ) as f:

        for item in refund_ledger:

            f.write(
                json.dumps(item)
                + "\n"
            )

    print(
        "\nagent_runs.jsonl created"
    )

    print(
        "refund_ledger.jsonl created"
    )

    print(
        "\nAgent Runs:",
        len(agent_runs)
    )

    print(
        "Refund Ledger:",
        len(refund_ledger)
    )


print("\n===== AGENT RUN =====")

run_agent()