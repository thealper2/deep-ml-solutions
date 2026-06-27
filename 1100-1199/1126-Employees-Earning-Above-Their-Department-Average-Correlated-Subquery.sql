SELECT
    id,
    name,
    department,
    salary
FROM employees e
WHERE salary > (
    SELECT AVG(salary)
    FROM employees
    WHERE department = e.department
)
ORDER BY id ASC
