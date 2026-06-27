WITH max_salary AS (
    SELECT MAX(salary) AS salary FROM workers
)
SELECT
    w.name,
    w.salary,
    t.title
FROM workers w
JOIN titles t ON w.title_id  = t.title_id
CROSS JOIN max_salary ms
WHERE w.salary = ms.salary
ORDER BY w.name ASC
