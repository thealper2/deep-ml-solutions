import numpy as np


class Autoscaler:
    def __init__(self, min_replicas, max_replicas, replica_capacity, cold_start_s):
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.replica_capacity = replica_capacity
        self.cold_start_s = cold_start_s
        self.load_history = []
        self.max_history = 600
        self.prev_desired = min_replicas
        self.smooth_window = 30
        self.smooth_loads = []
        self.headroom = 1.15
        self.slack = 1
        self.drain_violation_threshold = 1.0
        self.downscale_margin = 0.85
        self.last_t = None

    def _predict_load(self):
        """Predict the offered load a few seconds ahead based on recent history."""
        if len(self.smooth_loads) < 2:
            return self.smooth_loads[-1] if self.smooth_loads else 0.0

        recent = np.array(self.smooth_loads, dtype=float)

        short = float(np.mean(recent[-5:]))
        med = float(np.mean(recent[-min(len(recent), 30):]))

        n = min(len(recent), 10)
        if n >= 3:
            xs = np.arange(n, dtype=float)
            ys = recent[-n:]
            slope = float(np.polyfit(xs, ys, 1)[0])
        else:
            slope = 0.0

        horizon = max(self.cold_start_s, 3)
        predicted = short + slope * horizon
        predicted = max(predicted, short, med)
        return max(predicted, 0.0)

    def step(self, t, offered_load, queue, ready, pending):
        self.load_history.append(offered_load)
        if len(self.load_history) > self.max_history:
            self.load_history.pop(0)

        self.smooth_loads.append(offered_load)
        if len(self.smooth_loads) > self.smooth_window:
            self.smooth_loads.pop(0)

        capacity = max(ready, 0) * self.replica_capacity

        predicted = self._predict_load()
        need = (predicted * self.headroom) / self.replica_capacity
        base_target = need + self.slack

        if queue > 0:
            extra_need = queue / (self.replica_capacity * max(self.cold_start_s, 1))
            base_target += extra_need

        if ready > 0:
            drain_time = queue / max(capacity, 1e-9)
        else:
            drain_time = float('inf') if queue > 0 else 0.0

        if queue > 0 and drain_time > self.drain_violation_threshold:
            urgent = (queue + offered_load * self.cold_start_s) / (
                self.replica_capacity * max(self.cold_start_s, 1)
            )
            base_target = max(base_target, urgent + self.slack)

        target = int(np.ceil(base_target))

        if target >= self.prev_desired:
            desired = target
        else:
            if target <= self.prev_desired * self.downscale_margin:
                desired = target
            else:
                desired = self.prev_desired

        desired = max(self.min_replicas, min(self.max_replicas, desired))

        self.prev_desired = desired
        self.last_t = t
        return desired