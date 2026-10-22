Instamart Customer Intelligence & Support Copilot — Capstone Project
Swiggy Instamart delivers groceries, dairy, snacks, personal-care items and small electronics in six metro cities within minutes. As the order base grows, three problems keep surfacing in the weekly leadership review: nobody trusts the raw order exports (duplicate rows, mixed spellings, impossible values), the retention team cannot tell which customers are about to stop ordering, and the support desk is drowning in thousands of free-text tickets that mix refunds, fraud reports, app crashes and late deliveries. The Head of Customer Experience has asked a small data and AI team to build one internal tool that fixes all three: a Customer Intelligence & Support Copilot.

You are that team. Over 21 days you will build the Copilot as one repository, in four connected Parts. Part A turns the messy exports into a trusted analytical database. Part B uses that database to segment customers and to predict who is likely to lapse. Part C reads support tickets, protects customer privacy, and routes each ticket using the segments from Part B. Part D builds a policy question-answering assistant and a refund agent that queries the database from Part A and reuses the policy retrieval built in the same Part. Every Part consumes what the previous Parts produced, so earlier mistakes travel forward — and the numeric checkpoints in this brief let you catch them early.

All data in this project is synthetic and fictional, produced by a script given below. The brand name is used only to frame the scenario; nothing here is real Instamart data, policy or pricing. All monetary values are in Indian Rupees (INR).

As K increases, inertia always decreases because each cluster contains fewer points and therefore less within-cluster variance. However, inertia alone cannot determine the optimal K because it will continue decreasing even when additional clusters add little business value. K=4 was selected because it provides the highest silhouette score (0.526), a clear elbow in the inertia curve, and meaningful customer segments that can be acted upon by the business.

Cluster numbers are labels only and do not imply any ranking or ordering.

Premium Customers:
High monetary value customers. Recommended action:
exclusive offers and priority support.

Loyal Customers:
Frequent recent purchasers.
Recommended action:
loyalty rewards and subscriptions.

Occasional Shoppers:
Moderate activity and spend.
Recommended action:
re-engagement campaigns and coupons.

At Risk Customers:
Long recency and low frequency.
Recommended action:
win-back promotions.

The selected model is Logistic Regression.

Although both models exceeded the required F1 score of 0.55, Logistic Regression achieved the better test F1 score (0.724 vs 0.710) and higher recall (0.809 vs 0.742). Missing a customer who is likely to lapse is more costly than contacting a loyal customer unnecessarily, therefore the higher-recall model was selected.