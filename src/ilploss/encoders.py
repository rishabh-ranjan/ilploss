from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class DisjointEncoder(nn.Module):
    def __init__(
        self,
        ab_encoder: nn.Module,
        c_encoder: nn.Module,
    ):
        super().__init__()
        self.ab_encoder = ab_encoder
        self.c_encoder = c_encoder

    def forward(self, x):
        a, b = self.ab_encoder(x)
        c = self.c_encoder(x)
        return a, b, c


class StaticABEncoder(nn.Module):
    def __init__(
        self,
        num_vars: int,
        num_constrs: int,
        init_a: Callable = nn.init.normal_,
        init_r: Callable = nn.init.zeros_,
        init_o: Callable = nn.init.zeros_,
        train_a: bool = True,
        train_r: bool = True,
        train_o: bool = True,
    ):

        super().__init__()
        self.a = nn.Parameter(
            torch.empty(num_constrs, num_vars),
            requires_grad=train_a,
        )
        self.r = nn.Parameter(
            torch.empty(num_constrs),
            requires_grad=train_r,
        )
        self.o = nn.Parameter(
            torch.empty(num_constrs, num_vars),
            requires_grad=train_o,
        )
        init_a(self.a)
        init_r(self.r)
        init_o(self.o)

    def forward(self, x):
        self.b = self.r * torch.linalg.vector_norm(self.a, dim=-1) - torch.sum(
            self.a * self.o, dim=-1
        )
        return self.a.expand(x.shape[0], -1, -1), self.b.expand(x.shape[0], -1)


class StaticCEncoder(nn.Module):
    def __init__(
        self,
        num_vars: int,
        init: Union[Callable, dict] = nn.init.ones_,
        train: bool = True,
    ):

        super().__init__()
        self.c = nn.Parameter(torch.empty(num_vars), requires_grad=train)
        init(self.c)

    def forward(self, x):
        return self.c.expand(x.shape[0], -1)


class ABEncoderList(nn.Module):
    def __init__(
        self,
        ab_encoders: List[nn.Module],
    ):
        super().__init__()
        self.ab_encoders = nn.ModuleList(ab_encoders)

    def forward(self, x):
        a_list, b_list = zip(*[enc(x) for enc in self.ab_encoders])
        a = torch.cat(a_list, dim=-2)
        b = torch.cat(b_list, dim=-1)
        return a, b


class EqualityABEncoder(nn.Module):
    def __init__(
        self,
        ab_encoder: nn.Module,
        margin: float,
    ):
        super().__init__()

        self.ab_encoder = ab_encoder
        self.margin = margin

    def forward(self, x):
        a, b = self.ab_encoder(x)
        return torch.cat([a, -a], dim=-2), torch.cat(
            [b + self.margin, -b + self.margin], dim=-1
        )


class LUToABEncoder(nn.Module):
    def __init__(
        self,
        lu_encoder: nn.Module,
    ):
        super().__init__()
        self.lu_encoder = lu_encoder

    def forward(self, x):
        l, u = self.lu_encoder(x)
        e = torch.eye(l.shape[-1], l.shape[-1], device=l.device)
        e = e.expand(l.shape[0], -1, -1)
        a = torch.cat([e, -e], dim=-2)
        b = torch.cat([-l, u], dim=-1)
        return a, b


class StaticLUEncoder(nn.Module):
    def __init__(
        self,
        num_vars: int,
        lb: float,
        ub: float,
    ):
        super().__init__()
        self.num_vars = num_vars

        self.l = nn.Parameter(torch.empty(self.num_vars).fill_(lb), requires_grad=False)
        self.u = nn.Parameter(torch.empty(self.num_vars).fill_(ub), requires_grad=False)

    def forward(self, x):
        bs = x.shape[0]
        return self.l[None, :].expand(bs, -1), self.u[None, :].expand(bs, -1)


class ZeroABEncoder(nn.Module):
    def __init__(
        self,
        num_vars: int,
    ):
        super().__init__()

        self.register_buffer("a", torch.zeros(num_vars))
        self.register_buffer("b", torch.ones(()))

    def forward(self, x):
        bs = x.shape[0]
        return self.a[None, None, :].expand(bs, 1, -1), self.b[None, None].expand(bs, 1)
