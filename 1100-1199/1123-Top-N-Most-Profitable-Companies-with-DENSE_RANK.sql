WITH company_totals AS (
    SELECT
        company,
        SUM(profit) AS total_profit
    FROM sales
    GROUP BY company
),
ranked AS (
    SELECT
        company,
        total_profit,
        DENSE_RANK() OVER (ORDER BY total_profit DESC) AS drnk
    FROM company_totals
)
SELECT company, total_profit
FROM ranked
WHERE drnk <= 3
ORDER BY total_profit DESC, company ASC
