SELECT AVG(thread_count) AS avg_threads
FROM (
    SELECT p.id, COUNT(DISTINCT c.thread_id) AS thread_count
    FROM posts p
    LEFT JOIN comments c ON p.id = c.post_id
    GROUP BY p.id
) post_thread_counts;
