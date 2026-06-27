import hashlib
import numpy as np

def hash_tensor(tensor):
    """Return a 32-byte SHA-256 digest of the tensor's shape, dtype, and contents."""
    shape_str = str(tensor.shape).encode('utf-8')
    dtype_str = str(tensor.dtype).encode('utf-8')
    bytes_data = tensor.tobytes()
    combined = shape_str + b'|' + dtype_str + b'|' + bytes_data
    return hashlib.sha256(combined).digest()
