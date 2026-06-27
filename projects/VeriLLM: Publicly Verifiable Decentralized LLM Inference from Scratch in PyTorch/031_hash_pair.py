import hashlib

def hash_pair(left_digest, right_digest):
    """Hash two child digests into a single parent digest."""
    combined = left_digest + right_digest
    return hashlib.sha256(combined).digest()
