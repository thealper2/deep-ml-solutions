# ---- provided: the notification platform's logs (do not edit)
import numpy as np

DAY = 1440  # minutes


def load_history(days=28, users=400, per_day=500, seed=0, update_day=14):
    """Four weeks of push notifications, as three logs.

    events:          {"event_id", "user_id", "ts", "hour", "device"}, one per notification sent
    feature_log:     {"user_id", "ts", "sessions_7d", "opens_7d"}, a user's activity counts, updated now and then
    outcomes:        {"event_id", "opened", "label_ts"}, whether it was opened, known only at label_ts
    Times are minutes since the start. From update_day on, an app update changes how Android users respond.
    A few events are malformed, the way real logs are.
    """
    rs = np.random.RandomState(seed)
    devices = np.array(["ios", "android", "web"])
    user_device = devices[rs.choice(3, size=users, p=[0.45, 0.4, 0.15])]
    engagement = rs.gamma(2.0, 1.5, size=users)
    feature_log, events, outcomes = [], [], []
    for u in range(users):
        t = int(rs.randint(0, DAY))
        while t < days * DAY:
            sessions = int(rs.poisson(2 + 3 * engagement[u]))
            opens = int(rs.binomial(sessions, min(0.9, 0.15 + 0.1 * engagement[u])))
            feature_log.append({"user_id": u, "ts": t, "sessions_7d": sessions, "opens_7d": opens})
            t += int(rs.randint(DAY // 2, 3 * DAY))
    latest = {}
    for row in sorted(feature_log, key=lambda r: r["ts"]):
        latest.setdefault(row["user_id"], []).append(row)
    eid = 0
    for d in range(days):
        for _ in range(per_day):
            u = int(rs.randint(users))
            ts = d * DAY + int(rs.randint(DAY))
            hour = (ts % DAY) // 60
            history = [r for r in latest.get(u, []) if r["ts"] <= ts]
            opens = history[-1]["opens_7d"] if history else 0
            sessions = history[-1]["sessions_7d"] if history else 0
            android = user_device[u] == "android"
            if android and d >= update_day:
                # After the update, Android users open in the morning and past opens matter much less.
                z = -1.4 + 0.1 * opens - 0.04 * sessions + 2.0 * (7 <= hour <= 10) - 1.0 * (18 <= hour <= 22)
            else:
                z = (-2.2 + 0.35 * opens - 0.04 * sessions + 0.6 * (18 <= hour <= 22)
                     - 0.5 * (user_device[u] == "web") + 0.3 * android)
            opened = int(rs.rand() < 1.0 / (1.0 + np.exp(-z)))
            events.append({"event_id": eid, "user_id": u, "ts": ts, "hour": int(hour), "device": str(user_device[u])})
            outcomes.append({"event_id": eid, "opened": opened, "label_ts": ts + int(rs.randint(DAY // 4, 2 * DAY))})
            eid += 1
    for i in rs.choice(len(events), size=len(events) // 50, replace=False):
        bad = dict(events[i])
        kind = i % 3
        if kind == 0:
            bad["device"] = "fridge"
        elif kind == 1:
            bad["hour"] = 25
        else:
            bad["user_id"] = None
        events[i] = bad
    rs.shuffle(feature_log)
    return events, feature_log, outcomes
# ---- end provided


DEVICES = ("ios", "android", "web")


def validate_events(events):
    """(clean events, [(event_id, reason), ...]) with the first failing check as the reason."""
    clean = []
    rejected = []

    for ev in events:
        event_id = ev.get("event_id")
        user_id = ev.get("user_id")
        ts = ev.get("ts")
        hour = ev.get("hour")
        device = ev.get("device")

        if isinstance(user_id, bool) or not isinstance(user_id, int):
            rejected.append((event_id, "bad user_id"))
            continue
        if isinstance(ts, bool) or not isinstance(ts, int) or ts < 0:
            rejected.append((event_id, "bad ts"))
            continue
        if isinstance(hour, bool) or not isinstance(hour, int) or not (0 <= hour <= 23):
            rejected.append((event_id, "bad hour"))
            continue
        if device not in ("ios", "android", "web"):
            rejected.append((event_id, "bad device"))
            continue

        clean.append(ev)

    return clean, rejected
