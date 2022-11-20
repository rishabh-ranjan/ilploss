#!/usr/bin/env python3

"""
Demo script to replicate the random constraints experiments in the paper.
"""

import argparse
from collections import Counter
from copy import deepcopy
from functools import partial

import torch
from torch import nn, optim

from ilploss.encoders import StaticABEncoder, LUToABEncoder, StaticLUEncoder
from ilploss.samplers import (
    SamplerList,
    NbrSampler,
    KHopSampler,
    BitNbrSampler,
    BitKHopSampler,
    RandIntNbrWrapper,
    ProjSampler,
)
from ilploss.losses import CoVBalancer, ILPLoss
from ilploss.ilp import ILPSolver, STATUS_MSG

from utils import FastTensorDataLoader


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    pt = torch.load(args.data_path, map_location=device)
    x = pt["x"].float()
    y = pt["y"].float()
    lb = pt["meta"]["lb"]
    ub = pt["meta"]["ub"]
    num_constrs = pt["meta"]["num_constrs"]

    xs = {}
    xs["train"], xs["val"], xs["test"] = torch.split(x, [1500, 100, 1000])

    ys = {}
    ys["train"], ys["val"], ys["test"] = torch.split(y, [1500, 100, 1000])

    loaders = {
        "train": FastTensorDataLoader(
            xs["train"], ys["train"], batch_size=10, shuffle=True
        ),
        "val": FastTensorDataLoader(xs["val"], ys["val"], batch_size=100),
        "test": FastTensorDataLoader(xs["test"], ys["test"], batch_size=1000),
    }

    # initialization scheme from CombOptNet
    net = StaticABEncoder(
        num_vars=16,
        num_constrs=num_constrs * 2,
        init_a=partial(nn.init.uniform_, a=-0.5, b=0.5),
        init_r=partial(nn.init.constant_, val=0.2),
        init_o=partial(nn.init.uniform_, a=(3 * lb + ub) / 4, b=(lb + 3 * ub) / 4),
    ).to(device)

    # known constraints, in this case the bounding box given by lb and ub
    known = LUToABEncoder(StaticLUEncoder(num_vars=16, lb=lb, ub=ub)).to(device)

    opt = optim.AdamW(net.parameters())

    lrs = optim.lr_scheduler.CyclicLR(
        opt, base_lr=0.0, max_lr=1e-3, step_size_up=1500, cycle_momentum=False
    )

    # use lr_scheduler on an optimizer with a dummy parameter to schedule temperature
    temp = 1.0
    temp_opt = optim.SGD([nn.Parameter(torch.zeros(1))], lr=temp)
    temp_lrs = optim.lr_scheduler.ReduceLROnPlateau(
        temp_opt,
        mode="max",
        factor=0.1,
        patience=4,
    )

    if (lb, ub) == (0, 1):
        # optimized samplers for binary domain
        sampler = SamplerList(
            [
                BitNbrSampler(),
                BitKHopSampler(num_hops=2, num_samples=16),
                BitKHopSampler(num_hops=3, num_samples=16),
                BitKHopSampler(num_hops=4, num_samples=16),
                RandIntNbrWrapper(ProjSampler()),
            ]
        )
    else:
        # generic samplers
        sampler = SamplerList(
            [
                NbrSampler(),
                KHopSampler(num_hops=2, num_samples=16),
                KHopSampler(num_hops=3, num_samples=16),
                KHopSampler(num_hops=4, num_samples=16),
                RandIntNbrWrapper(ProjSampler()),
            ]
        )

    balancer = CoVBalancer(num_losses=3).to(device)

    criterion = ILPLoss(balancer, pos_margin=0.01, neg_margin=0.01)

    solver = ILPSolver(num_workers=10)

    best_acc_idv = 0
    best_net_state = None
    best_epoch = None
    patience = 20
    epoch = -1
    while patience:
        epoch += 1

        # validate every 20 epochs
        if epoch % 20 == 0:
            net.eval()
            with torch.no_grad():
                yhats = []
                ys = []
                for x, y in loaders["val"]:
                    a_l, b_l = net(x)
                    a_k, b_k = known(x)
                    a = torch.cat([a_l, a_k], dim=-2)
                    b = torch.cat([b_l, b_k], dim=-1)
                    c = x

                    # solve using y as hint
                    # biases validation metrics but expedites solving
                    yhat, _ = solver(a, b, c, y)
                    yhats.append(yhat)
                    ys.append(y)

                yhat = torch.cat(yhats)
                y = torch.cat(ys)

                # vector accuracy
                acc_all = torch.mean(torch.all(yhat == y, dim=-1).float())

                # individual accuracy
                acc_idv = torch.mean((yhat == y).float())

                # update temperature
                temp_lrs.step(acc_idv)
                temp = temp_opt.param_groups[0]["lr"]

                # update early stopping patience
                if acc_idv >= best_acc_idv:
                    best_acc_idv = acc_idv
                    best_net_state = deepcopy(net.state_dict())
                    best_epoch = epoch
                    patience = 20
                else:
                    patience -= 1

                print(
                    f"{epoch=}"
                    f"  val/acc/all={acc_all:.4f}"
                    f"  val/acc/idv={acc_idv:.4f}"
                    f"  {temp=:.5g}"
                    f"  {patience=}"
                )

        net.train()
        for x, y in loaders["train"]:
            a_l, b_l = net(x)
            a_k, b_k = known(x)
            a = [a_l, a_k]
            b = [b_l, b_k]
            c = x

            neg = sampler(a, b, c, y)

            loss = criterion(a, b, c, y, neg, temp)

            opt.zero_grad()
            loss.backward()
            opt.step()
            lrs.step()

    # test

    print(f"using model from epoch {best_epoch}")
    net.load_state_dict(best_net_state)
    net.eval()
    with torch.no_grad():
        yhats = []
        statuses = []
        ys = []
        for x, y in loaders["test"]:
            a_l, b_l = net(x)
            a_k, b_k = known(x)
            a = torch.cat([a_l, a_k], dim=-2)
            b = torch.cat([b_l, b_k], dim=-1)
            c = x

            yhat, status = solver(a, b, c)
            yhats.append(yhat)
            statuses.append(status)
            ys.append(y)

        yhat = torch.cat(yhats)
        status = torch.cat(statuses)
        y = torch.cat(ys)

        acc_all = torch.mean(torch.all(yhat == y, dim=-1).float())
        acc_idv = torch.mean((yhat == y).float())

        status_dict = {
            STATUS_MSG[k]: v for k, v in Counter(status.cpu().tolist()).items()
        }

        print(f"test/acc/all={acc_all:.4f}  test/acc/idv={acc_idv:.4f}")
        print(f"{status_dict=}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", help="path to dataset file")
    args = parser.parse_args()
    main(args)
