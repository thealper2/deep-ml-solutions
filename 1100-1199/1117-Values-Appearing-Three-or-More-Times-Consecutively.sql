WITH consecutive_groups AS (
    SELECT
        num,
        id - ROW_NUMBER() OVER (PARTITION BY num ORDER BY id) AS grp
    FROM logs
)
SELECT DISTINCT num
FROM consecutive_groups
GROUP BY num, grp
HAVING COUNT(*) >= 3
ORDER BY num ASC
