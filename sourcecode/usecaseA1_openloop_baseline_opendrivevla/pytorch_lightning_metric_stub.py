import torch

class Metric:
    def __init__(self, *args, **kwargs):
        self._state_names = []

    def add_state(self, name, default, dist_reduce_fx=None):
        setattr(self, name, default)
        self._state_names.append(name)

    def to(self, device):
        for name in self._state_names:
            val = getattr(self, name)
            if isinstance(val, torch.Tensor):
                setattr(self, name, val.to(device))
        return self

    def __call__(self, *args, **kwargs):
        if hasattr(self, 'update'):
            return self.update(*args, **kwargs)
        return None
