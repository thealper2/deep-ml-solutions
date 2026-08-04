SELECT 
    DATE_TRUNC('month', review_date) AS month,
    product_id,
    ROUND(AVG(stars), 2) AS avg_stars
FROM reviews
GROUP BY 
    DATE_TRUNC('month', review_date),
    product_id
ORDER BY month, product_id;
