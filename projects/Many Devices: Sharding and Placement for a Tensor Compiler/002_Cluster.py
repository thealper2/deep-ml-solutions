import numpy as np


class Cluster:
    """n simulated devices and the counted copies between them."""

    def __init__(self, n, mem_limit):
        self.devices = [Device(f"gpu{i}", mem_limit) for i in range(n)]
        self.bytes_moved = 0
        self.copies = 0
        self.log = []

    def copy(self, src, dst, name, dst_name=None, region=None):
        """Copy a buffer (or a region of it) from device src to device dst."""
        if dst_name is None:
            dst_name = name

        src_dev = self.devices[src]
        dst_dev = self.devices[dst]

        if name not in src_dev.buffers:
            raise KeyError(name)

        buf = src_dev.buffers[name]
        data = buf[region] if region is not None else buf
        data = np.array(data, dtype=np.float64, copy=True)

        old_size = dst_dev.buffers[dst_name].nbytes if dst_name in dst_dev.buffers else 0
        projected = dst_dev.used - old_size + data.nbytes
        if projected > dst_dev.mem_limit:
            raise MemoryError(f"cannot copy {name} to {dst_dev.name}")

        dst_dev.alloc(dst_name, data)
        nbytes = data.nbytes

        if src != dst:
            self.bytes_moved += nbytes
            self.copies += 1
            self.log.append((src, dst, name, nbytes))

        return nbytes

    def stats(self):
        """Return bytes_moved, copies, bytes_computed, kernels and peak."""
        return {
            "bytes_moved": self.bytes_moved,
            "copies": self.copies,
            "bytes_computed": sum(d.bytes_computed for d in self.devices),
            "kernels": sum(d.kernels for d in self.devices),
            "peak": [d.peak for d in self.devices],
        }
