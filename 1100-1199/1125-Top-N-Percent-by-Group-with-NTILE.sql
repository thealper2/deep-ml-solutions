WITH ranked AS (
    SELECT
        region,
        amount,
        NTILE(4) OVER (PARTITION BY region ORDER BY amount DESC) AS q
    FROM sales
)
SELECT region, amount
FROM ranked
WHERE q = 1
ORDER BY region ASC
