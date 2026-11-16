"""
Run an ML System: From Raw Logs to Automated Retraining — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  validate_events ──
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

# ── Step 002  dataset_version ──
import hashlib
import json


def dataset_version(rows, key="event_id"):
    """First 12 hex chars of sha256 over the rows sorted by key, canonical JSON."""
    ordered = sorted(rows, key=lambda r: r.get(key))
    serialized = json.dumps(ordered, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return digest[:12]

# ── Step 003  point_in_time_features ──
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

# ── Step 004  matured_labels ──
def matured_labels(outcomes, as_of):
    """{event_id: opened} for outcomes with label_ts <= as_of."""
    result = {}
    for o in outcomes:
        if o["label_ts"] <= as_of:
            result[o["event_id"]] = o["opened"]
    return result

# ── Step 005  train_model ──
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_model(X, y, C=1.0):
    """Fitted StandardScaler + LogisticRegression(C, max_iter=2000, tol=1e-8) pipeline."""
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=2000, tol=1e-8)
    )
    model.fit(X, y)
    return model

# ── Step 006  ExperimentTracker ──
class ExperimentTracker:
    """Logs runs (params, data version, metrics) and finds the best run per data version."""

    def __init__(self):
        self.runs = []
        self._counter = 0

    def log_run(self, params, data_version, metrics):
        self._counter += 1
        run_id = f"run-{self._counter:03d}"
        self.runs.append({
            "run_id": run_id,
            "params": dict(params),
            "data_version": data_version,
            "metrics": dict(metrics),
        })
        return run_id

    def best_run(self, metric, data_version, higher_is_better=True):
        best_id = None
        best_val = None
        for run in self.runs:
            if run["data_version"] != data_version:
                continue
            if metric not in run["metrics"]:
                continue
            val = run["metrics"][metric]
            if best_val is None:
                best_val = val
                best_id = run["run_id"]
            else:
                better = val > best_val if higher_is_better else val < best_val
                if better:
                    best_val = val
                    best_id = run["run_id"]

        return best_id

# ── Step 007  slice_report ──
import numpy as np
from sklearn.metrics import roc_auc_score


def _auc(y, p):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return None
    return round(float(roc_auc_score(y, p)), 4)

def slice_report(y_true, p, slices):
    """Overall and per-slice counts and AUC (rounded to 4, None for a one-class group)."""
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    slices = np.asarray(slices)

    overall = {
        "n": int(len(y_true)),
        "auc": _auc(y_true, p),
    }

    result = {}
    for name in sorted(set(slices.tolist())):
        mask = slices == name
        result[name] = {
            "n": int(mask.sum()),
            "auc": _auc(y_true[mask], p[mask]),
        }

    return {"overall": overall, "slices": result}

# ── Step 008  ci_gate ──
def ci_gate(candidate, baseline, tolerance=0.015, min_count=100):
    """(passed, reasons) from overall and per-slice AUC rules."""
    reasons = []

    cand_overall = candidate["overall"]["auc"]
    base_overall = baseline["overall"]["auc"]
    if cand_overall is not None and base_overall is not None and cand_overall < base_overall:
        reasons.append("overall auc dropped")

    for name in sorted(baseline["slices"].keys()):
        base_slice = baseline["slices"][name]
        if name not in candidate["slices"]:
            reasons.append(f"slice missing: {name}")
            continue
        cand_slice = candidate["slices"][name]
        if cand_slice["n"] < min_count:
            continue
        if cand_slice["auc"] is None or base_slice["auc"] is None:
            continue
        if cand_slice["auc"] < base_slice["auc"] - tolerance:
            reasons.append(f"slice regressed: {name}")

    return (len(reasons) == 0, reasons)

# ── Step 009  assign_variant ──
import hashlib


def assign_variant(user_id, canary_percent, salt="release-1"):
    """'canary' if sha256('salt:user_id') bucket (first 8 hex mod 100) < canary_percent, else 'stable'."""
    key = f"{salt}:{user_id}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return "canary" if bucket < canary_percent else "stable"

# ── Step 010  canary_check ──
def canary_check(stable, canary, max_ratio=1.2, min_requests=200):
    """'wait', 'rollback' or 'promote' from the two groups' request and error counts."""
    if stable["requests"] < min_requests or canary["requests"] < min_requests:
        return "wait"

    stable_rate = stable["errors"] / stable["requests"]
    canary_rate = canary["errors"] / canary["requests"]

    if canary_rate > max_ratio * stable_rate:
        return "rollback"
    
    return "promote"

# ── Step 011  daily_auc ──
import numpy as np
from sklearn.metrics import roc_auc_score


def daily_auc(events, probs, labels, min_labels=100):
    """{day: AUC} on matured labels for days with enough labels and both classes."""
    groups = {}
    for ev, p in zip(events, probs):
        eid = ev["event_id"]
        if eid not in labels:
            continue
        day = ev["ts"] // 1440
        groups.setdefault(day, []).append((float(p), int(labels[eid])))

    result = {}
    for day in sorted(groups.keys()):
        items = groups[day]
        if len(items) < min_labels:
            continue
        y = [label for _, label in items]
        if len(set(y)) < 2:
            continue
        p = [prob for prob, _ in items]
        result[day] = round(float(roc_auc_score(y, p)), 4)
    
    return result

# ── Step 012  retrain_decision ──
def retrain_decision(daily, floor=0.75, patience=2, cooldown=3, last_retrain=None):
    """First day the AUC has been below floor for `patience` entries in a row, or None."""
    streak = 0
    for day in sorted(daily.keys()):
        if last_retrain is not None and day < last_retrain + cooldown:
            streak = 0
            continue
        if daily[day] < floor:
            streak += 1
        else:
            streak = 0
        if streak >= patience:
            return day
    return None

# ── Scaffold (runner) ──
"""Run an ML system: four weeks of a notification-open model, from raw logs to an automated retrain.

Story: clean and version the data, build features as they were known at each moment, train only on
labels that have arrived, track the runs, gate the release on every device slice, roll it out to a
canary, then watch it decay after an app update and retrain when the matured labels say so.
"""
import numpy as np


def subset(events, first_day, last_day):
    return [e for e in events if first_day * 1440 <= e["ts"] < last_day * 1440]


def main():
    raw, feature_log, outcomes = load_history()
    events, rejected = validate_events(raw)
    print("events:", len(raw), "| rejected:", len(rejected), "| first rejections:", rejected[:3])

    # ---- Train v1 on the first week, with the labels known on day 8 ----
    week1 = subset(events, 0, 7)
    labels = matured_labels(outcomes, as_of=8 * 1440)
    train = [e for e in week1 if e["event_id"] in labels]
    version1 = dataset_version(train)
    X = point_in_time_features(train, feature_log)
    y = np.array([labels[e["event_id"]] for e in train])
    tracker = ExperimentTracker()
    models = {}
    holdout = subset(events, 7, 9)
    hold_labels = matured_labels(outcomes, as_of=11 * 1440)
    holdout = [e for e in holdout if e["event_id"] in hold_labels]
    Xh = point_in_time_features(holdout, feature_log)
    yh = np.array([hold_labels[e["event_id"]] for e in holdout])
    for C in (0.01, 0.1, 1.0):
        model = train_model(X, y, C=C)
        report = slice_report(yh, model.predict_proba(Xh)[:, 1], [e["device"] for e in holdout])
        run = tracker.log_run({"C": C}, version1, {"auc": report["overall"]["auc"]})
        models[run] = (model, report)
    best = tracker.best_run("auc", version1)
    v1, v1_report = models[best]
    print("data version", version1, "| runs:", [(r["run_id"], r["params"]["C"], r["metrics"]["auc"]) for r in tracker.runs], "| best:", best)
    print("v1 on held-out days:", v1_report)

    # ---- Watch it in production, judging each day only on labels that have arrived ----
    live = subset(events, 9, 28)
    probs = v1.predict_proba(point_in_time_features(live, feature_log))[:, 1]
    for today in (14, 18, 22):
        daily = daily_auc(live, probs, matured_labels(outcomes, as_of=today * 1440))
        due = retrain_decision(daily)
        print("day", today, "| matured daily AUC:", daily, "| retrain due on day:", due)
        if due is not None:
            break

    # ---- Retrain, gate every candidate on each device, canary the one that passes ----
    check_labels = matured_labels(outcomes, as_of=(due + 6) * 1440)
    check = [e for e in subset(events, due + 1, due + 4) if e["event_id"] in check_labels]
    Xc = point_in_time_features(check, feature_log)
    yc = np.array([check_labels[e["event_id"]] for e in check])
    devices = [e["device"] for e in check]
    base = slice_report(yc, v1.predict_proba(Xc)[:, 1], devices)
    print("v1 after the update:", base)
    known_now = matured_labels(outcomes, as_of=(due + 1) * 1440)
    for window in (3, 7):
        recent = [e for e in subset(events, due + 1 - window, due + 1) if e["event_id"] in known_now]
        version = dataset_version(recent)
        model = train_model(point_in_time_features(recent, feature_log), np.array([known_now[e["event_id"]] for e in recent]))
        cand = slice_report(yc, model.predict_proba(Xc)[:, 1], devices)
        run = tracker.log_run({"C": 1.0, "window_days": window}, version, {"auc": cand["overall"]["auc"]})
        passed, reasons = ci_gate(cand, base)
        print(run, "| last", window, "days, data version", version, "| overall", base["overall"]["auc"], "->", cand["overall"]["auc"],
              "| by device:", {d: (base["slices"][d]["auc"], cand["slices"][d]["auc"]) for d in base["slices"]})
        print("   CI gate passed:", passed, reasons)
        if passed:
            break
    users = sorted({e["user_id"] for e in check})
    canary_users = [u for u in users if assign_variant(u, 10) == "canary"]
    print("canary gets", len(canary_users), "of", len(users), "users")
    print("canary with 5 errors in 600 requests against 40 in 5000:", canary_check({"requests": 5000, "errors": 40}, {"requests": 600, "errors": 5}))
    print("canary with 15 errors in 600 requests against 40 in 5000:", canary_check({"requests": 5000, "errors": 40}, {"requests": 600, "errors": 15}))


if __name__ == "__main__":
    main()
