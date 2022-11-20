from copy import deepcopy
from functools import partial

import torch
from torch import nn, optim

from ilploss.encoders import (
    StaticABEncoder,
    LUToABEncoder,
    StaticLUEncoder,
)
from ilploss.samplers import (
    SamplerList,
    NbrSampler,
    KHopSampler,
    RandIntNbrWrapper,
    ProjSampler,
)
from ilploss.losses import (
    CoVBalancer,
    ILPLoss,
)
from ilploss.ilp import ILPSolver

from utils import FastTensorDataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

pt = torch.load("data/random_dense_4x16.pt", map_location=device)
x = pt["x"].float()
y = pt["y"].float()

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

net = StaticABEncoder(
    num_vars=16,
    num_constrs=16,
    init_a=partial(nn.init.uniform_, a=-0.5, b=0.5),
    init_r=partial(nn.init.constant_, val=0.2),
    init_o=partial(nn.init.uniform_, a=-2.5, b=2.5),
).to(device)

opt = optim.AdamW(net.parameters())

lrs = optim.lr_scheduler.CyclicLR(
    opt, base_lr=0.0, max_lr=1.0e-3, step_size_up=1500, cycle_momentum=False
)

temp = 1.0
temp_opt = optim.SGD([nn.Parameter(torch.zeros(1))], lr=temp)
temp_lrs = optim.lr_scheduler.ReduceLROnPlateau(
    temp_opt,
    mode="max",
    factor=0.1,
    patience=4,
)

known = LUToABEncoder(StaticLUEncoder(num_vars=16, lb=-5.0, ub=5.0)).to(device)

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
patience = 20
epoch = -1
while patience:
    epoch += 1

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

                yhat, _ = solver(a, b, c, None)
                yhats.append(yhat)
                ys.append(y)

            yhat = torch.cat(yhats)
            y = torch.cat(ys)

            acc_all = torch.mean(torch.all(yhat == y, dim=-1).float())
            acc_idv = torch.mean((yhat == y).float())

            temp_lrs.step(acc_idv)
            temp = temp_opt.param_groups[0]["lr"]

            if acc_idv >= best_acc_idv:
                best_acc_idv = acc_idv
                best_net_state = deepcopy(net.state_dict())
                patience = 20
            else:
                patience -= 1

            print(f"{epoch=}\t{acc_all=:.5f}\t{acc_idv=:.5f}\t{temp=:.5g}\t{patience=}")

    net.train()
    for x, y in loaders["train"]:
        a_l, b_l = net(x)
        a_k, b_k = known(x)
        a = [a_l, a_k]
        b = [b_l, b_k]
        c = x

        pos = y
        neg = sampler(a, b, c, y)

        loss = criterion(a, b, c, pos, neg, temp)

        opt.zero_grad()
        loss.backward()
        opt.step()
        lrs.step()

net.load_state_dict(best_net_state)
net.eval()
with torch.no_grad():
    yhats = []
    ys = []
    for x, y in loaders["test"]:
        a_l, b_l = net(x)
        a_k, b_k = known(x)
        a = torch.cat([a_l, a_k], dim=-2)
        b = torch.cat([b_l, b_k], dim=-1)
        c = x

        yhat, _ = solver(a, b, c, None)
        yhats.append(yhat)
        ys.append(y)

    yhat = torch.cat(yhats)
    y = torch.cat(ys)

    acc_all = torch.mean(torch.all(yhat == y, dim=-1).float())
    acc_idv = torch.mean((yhat == y).float())

    print("=== test ===")
    print(f"{acc_all=:.5f}\t{acc_idv:.5f}")
