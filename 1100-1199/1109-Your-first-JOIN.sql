SELECT e.name, d.name AS department
FROM employees e
JOIN departments d ON e.department_id=d.id
