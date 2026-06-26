WITH monthly AS (
    SELECT
        DATE_TRUNC('month', sale_date) AS month,
        SUM(amount) AS total
    FROM sales
    GROUP BY DATE_TRUNC('month', sale_date)
)
SELECT
    month,
    total,
    ROUND((total - LAG(total) OVER (ORDER BY month)) / LAG(total) OVER (ORDER BY month) * 100, 2) AS pct_damage
FROM monthly
ORDER BY month ASC
