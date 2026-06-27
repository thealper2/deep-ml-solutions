SELECT
    order_id,
    customer_id,
    amount
FROM orders
WHERE (customer_id, amount) IN (
    SELECT customer_id, MIN(amount) AS amount
    FROM orders
    GROUP BY customer_id
)
ORDER BY customer_id ASC
