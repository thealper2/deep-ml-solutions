import numpy as np
import math


class EvictionPolicy:
    def __init__(self, capacity_blocks):
        self.capacity = capacity_blocks
        self.resident = set()
        self.meta = {}
        self.seen = {}
        self.now = 0
        self.rng = np.random.RandomState(12345)
        self.touch_count = 0

    def admit(self, block, depth, prompt_blocks):
        self.seen[block] = self.seen.get(block, 0) + 1
        if self.seen[block] >= 2:
            return True
        if len(self.resident) >= self.capacity:
            if depth > 0 and depth >= prompt_blocks - 2:
                return False
        return True

    def touch(self, block, depth, prompt_blocks, t, hit):
        self.now = t
        self.touch_count += 1
        m = self.meta.get(block)
        if m is None:
            self.meta[block] = [1, t, depth]
        else:
            m[0] += 1
            m[1] = t
            if depth < m[2]:
                m[2] = depth
        self.resident.add(block)
        if self.touch_count % 20000 == 0:
            self._prune()

    def _prune(self):
        limit = max(self.capacity * 20, 5000)
        if len(self.meta) <= limit:
            return
        cutoff = self.now - 50000
        to_del = [b for b, m in self.meta.items()
                  if b not in self.resident and m[1] < cutoff]
        for b in to_del:
            del self.meta[b]
            self.seen.pop(b, None)

    def victim(self):
        t = self.now
        n = len(self.resident)
        if n == 0:
            return None
        blocks = list(self.resident)
        if n > 64:
            idxs = self.rng.choice(n, 64, replace=False)
            candidates = [blocks[i] for i in idxs]
        else:
            candidates = blocks
        best_block = candidates[0]
        best_score = float('inf')
        for b in candidates:
            m = self.meta.get(b)
            if m is None:
                score = 0.0
            else:
                freq, last_t, min_d = m
                age = t - last_t
                recency = 1.0 / (1.0 + age / 30.0)
                depth_factor = 1.0 + 2.0 / (1.0 + min_d)
                score = freq * recency * depth_factor
            if score < best_score:
                best_score = score
                best_block = b
        self.resident.discard(best_block)
        return best_block