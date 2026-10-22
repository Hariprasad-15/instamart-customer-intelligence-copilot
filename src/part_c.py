# =====================================================
# IMPORTS
# =====================================================

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import re
import json
import random
import time

from mock_llm import (
    chat,
    est_tokens,
    reset
)

# =====================================================
# CONSTANTS
# =====================================================

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3

ALLOWED_CATEGORIES = {
    "billing",
    "delivery",
    "technical",
    "fraud"
}

# =====================================================
# PROMPTS
# =====================================================

ZERO_SHOT_PROMPT = """
CLASSIFY

Allowed categories:
billing
delivery
technical
fraud

Return ONLY JSON

{"category":"<category>","reason":"<reason>"}
"""

FEW_SHOT_PROMPT = """
CLASSIFY

Allowed categories:
billing
delivery
technical
fraud

Return ONLY JSON

{"category":"<category>","reason":"<reason>"}

EXAMPLES:

Ticket:
I was charged twice.

Answer:
{"category":"billing","reason":"duplicate charge"}

Ticket:
Someone used my account and placed an order.

Answer:
{"category":"fraud","reason":"unauthorized activity"}

Ticket:
The app crashes on checkout.

Answer:
{"category":"technical","reason":"application failure"}
"""

# =====================================================
# LOAD DATA
# =====================================================

tickets = pd.read_csv(
    "data/raw/tickets.csv",
    keep_default_na=False
)

print("Total Tickets:", len(tickets))

# =====================================================
# TASK 1:
# INPUT GUARDRAILS + REDACTION
# =====================================================

EMAIL_PATTERN = r'[\w.+-]+@[\w-]+\.[\w.]+'
PHONE_PATTERN = r'\b[6-9]\d{9}\b'
ORDER_PATTERN = r'ORD-\d{6}'

email_count = 0
phone_count = 0
order_count = 0

redacted_texts = []
statuses = []

for _, row in tickets.iterrows():

    text = row["text"]

    if not text.strip():
        statuses.append("skipped_empty")
        redacted_texts.append(text)
        continue

    if est_tokens(text) > MAX_INPUT_TOKENS:
        statuses.append("flagged_long")
        redacted_texts.append(text)
        continue

    text, e = re.subn(
        EMAIL_PATTERN,
        "[EMAIL]",
        text
    )

    text, p = re.subn(
        PHONE_PATTERN,
        "[PHONE]",
        text
    )

    text, o = re.subn(
        ORDER_PATTERN,
        "[ORDER_ID]",
        text
    )

    email_count += e
    phone_count += p
    order_count += o

    statuses.append("eligible")
    redacted_texts.append(text)

print("\nEmails:", email_count)
print("Phones:", phone_count)
print("Order IDs:", order_count)
print("Total:", email_count + phone_count + order_count)

# =====================================================
# VERIFY PII REMOVED
# =====================================================

combined_text = "\n".join(redacted_texts)

assert len(re.findall(
    EMAIL_PATTERN,
    combined_text
)) == 0

assert len(re.findall(
    PHONE_PATTERN,
    combined_text
)) == 0

assert len(re.findall(
    ORDER_PATTERN,
    combined_text
)) == 0

# =====================================================
# RETRY FUNCTION
# =====================================================

def call_with_retry(
    ticket_id,
    messages
):

    retries = 0

    while True:

        try:
            return chat(messages)

        except Exception:

            if retries >= MAX_RETRIES:
                raise

            wait = (
                BASE_DELAY
                * (2 ** retries)
                * random.uniform(
                    0.9,
                    1.15
                )
            )

            print(
                f"{ticket_id}"
                f" retry={retries}"
                f" wait={wait:.4f}"
            )

            time.sleep(wait)

            retries += 1

# =====================================================
# FEW SHOT CLASSIFICATION
# =====================================================

reset()

results = []

classified_json = 0
classified_keyword = 0
flagged_invalid_output = 0

for idx, row in tickets.iterrows():

    text = redacted_texts[idx]

    if statuses[idx] != "eligible":

        results.append({
            "ticket_id": row["ticket_id"],
            "status": statuses[idx],
            "category": ""
        })

        continue

    response = call_with_retry(
        row["ticket_id"],
        [
            {
                "role": "system",
                "content": FEW_SHOT_PROMPT
            },
            {
                "role": "user",
                "content": text
            }
        ]
    )

    answer = (
        response
        .choices[0]
        .message
        .content
    )

    tokens = response.usage.total_tokens

    final_status =  None
    category = ""

    try:

        parsed = json.loads(answer)

        if (
            parsed.get("category")
            in ALLOWED_CATEGORIES
        ):
            category = parsed["category"]
            final_status = "classified_json"
            classified_json += 1

    except:

        words = set(
            re.findall(
                r"[a-z]+",
                answer.lower()
            )
        )

        matches = (
            ALLOWED_CATEGORIES
            .intersection(words)
        )

        if len(matches) == 1:

            category = list(matches)[0]

            final_status = (
                "classified_keyword"
            )

            classified_keyword += 1

    if final_status is None:

        final_status = (
            "flagged_invalid_output"
        )

        flagged_invalid_output += 1

    results.append({
        "ticket_id":
            row["ticket_id"],

        "customer_id":
            row["customer_id"],

        "status":
            final_status,

        "category":
            category,

        "tokens_used":
            tokens
    })

# =====================================================
# SUMMARY
# =====================================================

print("\n===== FINAL COUNTS =====")

print(
    "classified_json:",
    classified_json
)

print(
    "classified_keyword:",
    classified_keyword
)

print(
    "flagged_invalid_output:",
    flagged_invalid_output
)

# =====================================================
# SAVE
# =====================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "outputs/triage_results.csv",
    index=False
)

print(
    "\ntriage_results.csv created"
)