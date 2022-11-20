from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot_argmin(x, dim):
    return F.one_hot(torch.argmin(x, dim=dim), num_classes=x.shape[dim]).float()


class ILPLoss(nn.Module):
    def __init__(
        self,
        balancer: nn.Module,
        pos_margin: float,
        neg_margin: float,
    ):
        super().__init__()
        self.balancer = balancer
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin

    def forward(self, a, b, c, pos, neg, temp):
        """
        a and b are tuples of learnable and known constraints.
        Use ZeroABEncoder to avoid specifying known constraints.
        """

        a_l, a_k = a
        b_l, b_k = b
        """
        a_l, a_k: *batch x constr x var
        b_l, b_k: *batch x constr
        c: *batch x var
        pos: *batch x var
        neg: *batch x neg x var
        """

        known_c = not c.requires_grad and c.grad_fn is None

        if known_c:
            a_k = torch.cat([a_k, -c[..., None, :]], dim=-2)
            b_k = torch.cat([b_k, torch.sum(c * pos, dim=-1)[..., None]], dim=-1)
        else:
            a_l = torch.cat([a_l, -c[..., None, :]], dim=-2)
            b_l = torch.cat([b_l, torch.sum(c * pos, dim=-1)[..., None]], dim=-1)

        norm_l = torch.linalg.vector_norm(a_l, dim=-1)
        a_l = a_l / norm_l[..., :, None]
        b_l = b_l / norm_l
        dist_pos_l = torch.sum(a_l * pos[..., None, :], dim=-1) + b_l
        dist_neg_l = neg @ a_l.transpose(-1, -2) + b_l[..., None, :]
        if not known_c:
            dist_pos_l[..., -1] = 0
        """
        norm_l: *batch x constr
        dist_pos_l: *batch x constr
        dist_neg_l: *batch x neg x constr
        """

        fdist_pos_k = torch.sum(a_k * pos[..., None, :], dim=-1) + b_k
        fdist_neg_k = neg @ a_k.transpose(-1, -2) + b_k[..., None, :]
        """
        fdist_pos_k: *batch x constr
        fdist_neg_k: *batch x neg x constr
        """

        msk_pos = torch.any(neg != pos[..., None, :], dim=-1)
        msk_known = torch.all(fdist_neg_k >= 0, dim=-1)
        msk = msk_pos & msk_known
        if known_c:
            loss_pos = torch.mean(F.relu(self.pos_margin - dist_pos_l), dim=-1)
        elif dist_pos_l.shape[-1] > 1:
            loss_pos = torch.mean(
                F.relu(self.pos_margin - dist_pos_l[..., :-1]), dim=-1
            )
        else:
            loss_pos = torch.tensor(0.0, device=dist_pos_l.device)

        with torch.no_grad():
            if temp <= torch.finfo().eps:
                w = one_hot_argmin(dist_neg_l, dim=-1)
            else:
                w = F.softmin(dist_neg_l / temp, dim=-1)

        loss_neg = torch.sum(
            msk * torch.sum(w * F.relu(self.neg_margin + dist_neg_l), dim=-1),
            dim=-1,
        ) / (torch.sum(msk, dim=-1) + torch.finfo().eps)

        a_lk = torch.cat([a_l, F.normalize(a_k, dim=-1)], dim=-2)
        loss_var = torch.mean(torch.mean(a_lk, dim=-2) ** 2, dim=-1)

        return self.balancer(
            {
                "pos": torch.mean(loss_pos),
                "neg": torch.mean(loss_neg),
                "var": torch.mean(loss_var),
            },
        )


class StaticBalancer(nn.Module):
    def __init__(
        self,
        weights: dict,
    ):
        super().__init__()
        self.weights = weights

    def forward(self, loss_dict):
        ret = 0
        for k, v in self.weights.items():
            ret += v * loss_dict[k]
        return ret


class CoVBalancer(nn.Module):
    """from https://arxiv.org/abs/2009.01717"""

    def __init__(
        self,
        num_losses: int,
        decay: Optional[float] = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.decay = decay
        self.eps = eps

        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        self.register_buffer("ema_dr", torch.zeros(()))
        self.register_buffer("ema_loss", torch.zeros(num_losses))
        self.register_buffer("ema_loss_ratio", torch.zeros(num_losses))
        self.register_buffer("var_nr", torch.zeros(num_losses))

    @torch.no_grad()
    def update(self, loss):
        self.step += 1
        decay = 1 / self.step if self.decay is None else self.decay

        self.ema_dr = (1 - decay) * self.ema_dr + decay * 1
        self.ema_loss = (1 - decay) * self.ema_loss + decay * loss
        self.ema_loss /= self.ema_dr
        loss_ratio = (loss + self.eps) / (self.ema_loss + self.eps)
        ema_loss_ratio = (1 - decay) * self.ema_loss_ratio + decay * loss_ratio
        ema_loss_ratio /= self.ema_dr
        self.var_nr = self.var_nr + (loss_ratio - self.ema_loss_ratio) * (
            loss_ratio - ema_loss_ratio
        )
        self.ema_loss_ratio = ema_loss_ratio

    @torch.no_grad()
    def compute(self):
        std_loss_ratio = torch.sqrt(self.var_nr / self.step + self.eps)
        cov_loss_ratio = std_loss_ratio / self.ema_loss_ratio
        self.w = cov_loss_ratio / torch.sum(cov_loss_ratio)

    def forward(self, loss_dict):
        loss = torch.stack(list(loss_dict.values()))
        self.update(loss)
        self.compute()
        return torch.sum(self.w * loss)
