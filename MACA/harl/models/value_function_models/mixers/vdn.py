import torch as th
import torch.nn as nn


class VDNMixer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, agent_qs, **kwargs):
        return th.sum(agent_qs, dim=1)