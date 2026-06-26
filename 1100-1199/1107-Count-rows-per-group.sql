SELECT department_id, COUNT(*) AS headcount
FROM employees 
GROUP BY department_id
