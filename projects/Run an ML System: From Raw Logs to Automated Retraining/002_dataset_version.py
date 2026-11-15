import hashlib
import json


def dataset_version(rows, key="event_id"):
    """First 12 hex chars of sha256 over the rows sorted by key, canonical JSON."""
    ordered = sorted(rows, key=lambda r: r.get(key))
    serialized = json.dumps(ordered, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return digest[:12]
