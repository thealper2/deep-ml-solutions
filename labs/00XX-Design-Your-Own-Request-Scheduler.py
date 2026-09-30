import numpy as np


class Scheduler:
    def __init__(
        self,
        max_batch,
        total_blocks,
        block_size,
        prefill_tps,
        decode_ms_base,
        decode_ms_per_seq,
    ):
        self.max_batch = max_batch
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.prefill_tps = prefill_tps

        self.decode_ms_base = decode_ms_base
        self.decode_ms_per_seq = decode_ms_per_seq

        if decode_ms_per_seq > 0:
            self.latency_decode_cap = int(
                np.floor(
                    (150.0 - decode_ms_base) / decode_ms_per_seq
                )
            )
        else:
            self.latency_decode_cap = max_batch

        self.latency_decode_cap = max(
            1,
            min(max_batch, self.latency_decode_cap),
        )

    def _decode_count(self, running):
        return sum(r["phase"] == "decode" for r in running)

    def _can_admit(self, r, free_blocks, room):
        if room <= 0:
            return False

        return r["blocks_needed"] <= free_blocks

    def plan(self, t, waiting, running, free_blocks):
        """
        Latency-aware continuous batching.

        Strategy:
        - Never admit requests that cannot fit in the KV cache.
        - Preserve already decoding requests.
        - Limit the number of simultaneous decode sequences so that
          the nominal decode step remains <= 150 ms.
        - Use chunked prefill to prevent a very long prompt from
          monopolizing a step.
        - Prioritize requests with the oldest arrival time.
        """

        running_ids = {r["id"] for r in running}

        decode_running = [
            r for r in running
            if r["phase"] == "decode"
        ]

        decode_count = len(decode_running)
        preempt_ids = []

        if decode_count > self.latency_decode_cap:
            excess = decode_count - self.latency_decode_cap

            victims = sorted(
                decode_running,
                key=lambda r: (
                    r["generated"],
                    -r.get("arrival", 0.0),
                ),
            )

            preempt_ids = [
                r["id"]
                for r in victims[:excess]
            ]

            decode_count -= excess

        effective_running = [
            r for r in running
            if r["id"] not in set(preempt_ids)
        ]

        room = self.max_batch - len(effective_running)

        if room <= 0:
            return [], preempt_ids, self._prefill_budget(
                waiting,
                effective_running,
            )

        candidates = []

        for r in waiting:
            if r["id"] in running_ids:
                continue

            if not self._can_admit(r, free_blocks, room):
                continue

            age = max(0.0, t - r["arrival"])

            deadline_pressure = age / 1.0

            candidates.append(
                (
                    -deadline_pressure,
                    r["arrival"],
                    r["id"],
                    r,
                )
            )

        candidates.sort(key=lambda x: (x[0], x[1]))

        admit = []

        remaining_blocks = free_blocks
        remaining_room = room

        for _, _, _, r in candidates:
            if remaining_room <= 0:
                break

            needed = r["blocks_needed"]

            if needed > remaining_blocks:
                continue

            admit.append(r["id"])
            remaining_blocks -= needed
            remaining_room -= 1

        effective_waiting = [
            r for r in waiting
            if r["id"] in admit
        ]

        prefill_budget = self._prefill_budget(
            effective_waiting,
            effective_running,
        )

        return admit, preempt_ids, prefill_budget

    def _prefill_budget(self, waiting, running):
        """
        Return a chunk size rather than unlimited prefill.

        Keep each prefill step roughly below 50 ms, leaving room
        for decoding and reducing the chance that prefill causes
        large inter-token gaps.
        """

        prefilling = [
            r for r in running
            if r["phase"] == "prefill"
        ]

        prefilling = list(prefilling) + list(waiting)

        if not prefilling:
            return 0

        target_ms = 50.0
        budget = int(self.prefill_tps * target_ms / 1000.0)

        if budget <= 0:
            budget = 1

        return budget
