SELECT MIN(id) AS id, email
FROM person
GROUP BY email
ORDER BY id
