import bisect
import numpy as np

FEATURE_NAMES = ["sessions_7d", "opens_7d", "morning", "evening", "android", "web",
                 "android_morning", "android_evening", "android_opens", "has_history"]


def point_in_time_features(events, feature_log):
    """(n_events, 10) array from the latest log row at or before each event."""
    rows = []
    for ev in events:
        uid = ev["user_id"]
        ets = ev["ts"]
        best = None
        for entry in feature_log:
            if entry["user_id"] == uid and entry["ts"] <= ets:
                if best is None or entry["ts"] > best["ts"]:
                    best = entry

        if best is not None:
            sessions = float(best["sessions_7d"])
            opens = float(best["opens_7d"])
            has_history = 1.0
        else:
            sessions = 0.0
            opens = 0.0
            has_history = 0.0

        hour = ev["hour"]
        device = ev["device"]
        morning = 1.0 if 7 <= hour <= 10 else 0.0
        evening = 1.0 if 18 <= hour <= 22 else 0.0
        android = 1.0 if device == "android" else 0.0
        web = 1.0 if device == "web" else 0.0
        android_morning = android * morning
        android_evening = android * evening
        android_opens = android * opens

        rows.append([
            sessions, opens, morning, evening, android, web,
            android_morning, android_evening, android_opens, has_history,
        ])

    if not rows:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=float)
    return np.array(rows, dtype=float)