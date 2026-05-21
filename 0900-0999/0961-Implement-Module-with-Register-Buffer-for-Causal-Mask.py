import numpy as np

class Module:
    def __init__(self):
        self._buffers = {}
        self.device = 'cpu'

    def register_buffer(self, name, array):
        self._buffers[name] = array
        setattr(self, name, array)

    def state_dict(self):
        return self._buffers.copy()

    def to(self, device):
        self.device = device
        for name in self._buffers:
            self._buffers[name] = self._buffers[name]

        return self


class CausalAttentionMask(Module):
    def __init__(self, context_length):
        super().__init__()
        mask = np.triu(np.ones((context_length, context_length), dtype=bool), k=1)
        self.register_buffer('mask', mask)
        self.extra = [1, 2, 3]


def build_and_report(context_length, device):
    m = CausalAttentionMask(context_length)
    m.to(device)

    return {
        'mask': m.mask.tolist(),
        'state_dict_keys': sorted(m.state_dict().keys()),
        'buffer_device': device,
        'has_extra_in_state_dict': 'extra' in m.state_dict(),
    }