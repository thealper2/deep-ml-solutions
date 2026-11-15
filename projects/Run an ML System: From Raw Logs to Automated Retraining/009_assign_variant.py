import hashlib


def assign_variant(user_id, canary_percent, salt="release-1"):
    """'canary' if sha256('salt:user_id') bucket (first 8 hex mod 100) < canary_percent, else 'stable'."""
    key = f"{salt}:{user_id}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return "canary" if bucket < canary_percent else "stable"
