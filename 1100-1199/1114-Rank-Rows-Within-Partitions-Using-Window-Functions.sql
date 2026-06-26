SELECT 
    department,
    employee,
    score,
    ROW_NUMBER() OVER (PARTITION BY department ORDER BY score DESC, employee ASC) AS rn,
    RANK() OVER (PARTITION BY department ORDER BY score DESC) AS rnk,
    DENSE_RANK() OVER (PARTITION BY department ORDER BY score DESC) AS dense_rnk
FROM scores
ORDER BY department ASC, score DESC, employee ASC
