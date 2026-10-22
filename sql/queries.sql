-- Q1
SELECT
    customer_id,
    SUM(order_amount) AS total_spend
FROM orders
WHERE status='completed'
  AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000;

-- Q2
SELECT
    c.city,
    p.category,
    ROUND(SUM(o.order_amount),2) AS revenue
FROM orders o
JOIN customers c
    ON o.customer_id = c.customer_id
JOIN products p
    ON o.product_id = p.product_id
WHERE o.status='completed'
GROUP BY c.city, p.category
ORDER BY revenue DESC;

-- Q3
SELECT *
FROM products
WHERE price_inr >
(
    SELECT AVG(price_inr)
    FROM products
);

-- Q4
SELECT c.customer_id
FROM customers c
LEFT JOIN
(
    SELECT DISTINCT customer_id
    FROM orders
    WHERE status='completed'
) o
ON c.customer_id=o.customer_id
WHERE o.customer_id IS NULL;

-- Q5
WITH customer_spend AS
(
    SELECT
        customer_id,
        SUM(order_amount) total_spend
    FROM orders
    WHERE status='completed'
    GROUP BY customer_id
),
avg_spend AS
(
    SELECT AVG(total_spend) avg_total
    FROM customer_spend
)
SELECT *
FROM customer_spend
WHERE total_spend >
(
    SELECT avg_total
    FROM avg_spend
);

-- Q6
WITH customer_totals AS
(
    SELECT
        c.city,
        o.customer_id,
        ROUND(SUM(o.order_amount),2) total_spend
    FROM orders o
    JOIN customers c
      ON o.customer_id=c.customer_id
    WHERE o.status='completed'
    GROUP BY c.city,o.customer_id
)

SELECT *,
RANK() OVER
(
PARTITION BY city
ORDER BY total_spend DESC
) rank_num,

DENSE_RANK() OVER
(
PARTITION BY city
ORDER BY total_spend DESC
) dense_rank_num,

ROW_NUMBER() OVER
(
PARTITION BY city
ORDER BY total_spend DESC
) row_num

FROM customer_totals;

-- Q7
WITH daily_revenue AS
(
    SELECT
        substr(order_ts,1,10) AS day,
        SUM(order_amount) revenue
    FROM orders
    WHERE status='completed'
    GROUP BY day
)

SELECT
    day,
    revenue,
    SUM(revenue)
    OVER
    (
        ORDER BY day
    ) cumulative_revenue
FROM daily_revenue;
