import copy

import torch

from .model import YamlModel


class FollowingModel:

    def __init__(self,
                 model: YamlModel,
                 momentum: float = 0.1):
        self.model = model
        self.follower = copy.deepcopy(model)
        self.momentum = momentum

    @torch.no_grad()
    def __call__(self, x):
        return self.follower(x)

    def update(self, step=None):
        step = self.momentum if step is None else step
        if step > 0:
            state_f, state_m = self.follower.state_dict(), self.model.state_dict()
            for k in self.follower.state_dict():
                state_f[k] = step * state_m[k] + (1 - step) * state_f[k]
