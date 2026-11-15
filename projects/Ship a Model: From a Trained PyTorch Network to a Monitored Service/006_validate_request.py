_FIELDS = ["distance_km", "items", "hour", "driver_rating", "traffic_index", "rain"]
_RANGES = {
    "distance_km": (0, 100),
    "items": (1, 50),
    "hour": (0, 23),
    "driver_rating": (1, 5),
    "traffic_index": (0, 2),
    "rain": (0, 1),
}
_WHOLE = {"items", "hour", "rain"}

def validate_request(payload):
    """(row, []) for a valid request, (None, errors) otherwise."""
    errors = []
    row = []

    for field in _FIELDS:
        if field not in payload or payload[field] is None:
            errors.append(f"missing: {field}")
            continue
        v = payload[field]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            errors.append(f"not a number: {field}")
            continue
        if field in _WHOLE and float(v) != int(v):
            errors.append(f"not a whole number: {field}")
            continue
        lo, hi = _RANGES[field]
        if not (lo <= v <= hi):
            errors.append(f"out of range: {field}")
            continue
        row.append(float(v))

    for name in sorted(k for k in payload if k not in _RANGES):
        errors.append(f"unknown field: {name}")

    if errors:
        return None, errors
    return row, []