SELECT e.name AS employee
FROM employees e
JOIN employees m ON e.manager_id = m.id
WHERE e.salary > m.salary
