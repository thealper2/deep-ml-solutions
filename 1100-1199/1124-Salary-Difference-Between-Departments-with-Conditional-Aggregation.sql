SELECT
    SUM(CASE WHEN department = 'Engineering' THEN salary ELSE 0 END) AS engineering_total,
    SUM(CASE WHEN department = 'Sales' THEN salary ELSE 0 END) AS sales_total,
    SUM(CASE WHEN department = 'Engineering' THEN salary ELSE 0 END) - 
    SUM(CASE WHEN department = 'Sales' THEN salary ELSE 0 END) AS salary_difference
FROM employees
