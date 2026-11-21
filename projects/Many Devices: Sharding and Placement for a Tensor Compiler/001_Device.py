import numpy as np


class Device:
    """One simulated device: named float64 buffers, a memory limit and counters."""

    def __init__(self, name, mem_limit):
        self.name = name
        self.mem_limit = mem_limit
        self.buffers = {}
        self.used = 0
        self.peak = 0
        self.kernels = 0
        self.bytes_computed = 0

    def _size(self, arr):
        return arr.nbytes

    def alloc(self, name, array):
        """Store a float64 copy of array under name, or raise MemoryError."""
        arr = np.array(array, dtype=np.float64)
        new_size = arr.nbytes

        old_size = 0
        if name in self.buffers:
            old_size = self.buffers[name].nbytes

        projected = self.used - old_size + new_size
        if projected > self.mem_limit:
            raise MemoryError(f"cannot allocate {name} on {self.name}")

        self.buffers[name] = arr
        self.used = projected
        if self.used > self.peak:
            self.peak = self.used

    def free(self, name):
        """Remove a buffer and give its bytes back."""
        if name not in self.buffers:
            raise KeyError(name)
        self.used -= self.buffers[name].nbytes
        del self.buffers[name]

    def run(self, out, fn, *inputs):
        """Run one kernel on this device's buffers and store its result under out."""
        arrays = []
        read_bytes = 0
        for name in inputs:
            if name not in self.buffers:
                raise KeyError(name)
            arr = self.buffers[name]
            arrays.append(arr)
            read_bytes += arr.nbytes

        result = np.array(fn(*arrays), dtype=np.float64)
        out_bytes = result.nbytes

        old_size = 0
        if out in self.buffers:
            old_size = self.buffers[out].nbytes

        projected = self.used a- old_size + out_bytes
        if projected > self.mem_limit:
            raise MemoryError(f"cannot allocate {out} on {self.name}")

        self.buffers[out] = result
        self.used = projected
        if self.used > self.peak:
            self.peak = self.used

        self.kernels += 1
        self.bytes_computed += read_bytes + out_bytes