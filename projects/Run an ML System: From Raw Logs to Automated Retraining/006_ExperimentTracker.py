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
