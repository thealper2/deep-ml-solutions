SELECT recorded_on
FROM (
    SELECT
        recorded_on,
        temperature,
        LAG(temperature) OVER (ORDER BY recorded_on) AS prev_temp
    FROM weather
) t
WHERE temperature > prev_temp
ORDER BY recorded_on ASC
