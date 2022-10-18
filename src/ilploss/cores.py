from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SolverCore(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        solver: nn.Module,
        criterion: nn.Module,
        known_ab_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.solver = solver
        self.criterion = criterion
        self.known_ab_encoder = known_ab_encoder

    def forward(self, i, x, y=None, temp=None):
        a, b, c = self.encoder(x)
        if self.known_ab_encoder is not None:
            a_k, b_k = self.known_ab_encoder(x)
            a = torch.cat([a, a_k], dim=-2)
            b = torch.cat([b, b_k], dim=-1)

        if y is None:
            return a, b, c

        loss = self.criterion(yhat, y)
        return loss


class SolverFreeCore(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        known_ab_encoder: nn.Module,
        sampler: nn.Module,
        criterion: nn.Module,
    ):
        super().__init__()

        self.encoder = encoder
        self.known_ab_encoder = known_ab_encoder
        self.sampler = sampler
        self.criterion = criterion

    def forward(self, i, x, y=None, temp=None):
        a_l, b_l, c = self.encoder(x)
        a_k, b_k = self.known_ab_encoder(x)
        if y is None:
            return torch.cat([a_l, a_k], dim=-2), torch.cat([b_l, b_k], dim=-1), c

        pos = y
        with torch.no_grad():
            neg = self.sampler([a_l, a_k], [b_l, b_k], c, y)

        loss = self.criterion([a_l, a_k], [b_l, b_k], c, pos, neg, temp)
        return loss
