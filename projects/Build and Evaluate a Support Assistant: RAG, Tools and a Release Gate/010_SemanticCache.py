class SemanticCache:
    """Reuse answers for questions with near-identical embeddings, per version, dropped when a source page changes."""

    def __init__(self, threshold=0.92):
        self.threshold = threshold
        self.entries = []

    def put(self, vector, version, value, doc_ids):
        self.entries.append({
            "vector": np.asarray(vector, dtype=float),
            "version": version,
            "value": value,
            "doc_ids": set(doc_ids),
        })

    def get(self, vector, version):
        v = np.asarray(vector, dtype=float)
        best = None
        best_sim = None
        for entry in self.entries:
            if entry["version"] != version:
                continue
            sim = float(np.dot(v, entry["vector"]))
            if best_sim is None or sim > best_sim:
                best_sim = sim
                best = entry
        if best is not None and best_sim >= self.threshold:
            return best["value"]
        return None

    def invalidate(self, doc_id):
        kept = []
        removed = 0
        for entry in self.entries:
            if doc_id in entry["doc_ids"]:
                removed += 1
            else:
                kept.append(entry)
        self.entries = kept
        return removed