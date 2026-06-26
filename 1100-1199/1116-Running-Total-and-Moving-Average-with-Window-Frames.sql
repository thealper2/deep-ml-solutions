SELECT
    day,
    amount,
    SUM(amount) OVER (ORDER BY day ROWS UNBOUNDED PRECEDING) AS running_total,
    AVG(amount) OVER (ORDER BY day ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS moving_avg
FROM sales
ORDER BY day ASC
