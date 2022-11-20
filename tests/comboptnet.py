from copy import deepcopy
from functools import partial

import torch
from torch import nn, optim

from ilploss.encoders import StaticABEncoder, LUToABEncoder, StaticLUEncoder
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

known = LUToABEncoder(StaticLUEncoder(num_vars=16, lb=-5.0, ub=5.0)).to(device)

solver = ILPSolver(num_workers=10, temp=0.5)

criterion = nn.L1Loss()

opt = optim.Adam(net.parameters(), lr=5e-4)

best_val = 0
best_state = None
patience = 20
epoch = -1
while patience:
    epoch += 1

    if epoch % 1 == 0:
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

                yhat, status = solver(a, b, c, None)
                yhats.append(yhat)
                ys.append(y)

            yhat = torch.cat(yhats)
            y = torch.cat(ys)

            acc_all = torch.mean(torch.all(yhat == y, dim=-1).float())
            acc_idv = torch.mean((yhat == y).float())

            if acc_idv >= best_val:
                best_val = acc_idv
                best_state = deepcopy(net.state_dict())
                patience = 20
            else:
                patience -= 1

            print(f"{epoch=}\t{acc_all=:.5f}\t{acc_idv=:.5f}\t{patience=}")

    net.train()
    for x, y in loaders["train"]:
        a_l, b_l = net(x)
        a_k, b_k = known(x)
        a = torch.cat([a_l, a_k], dim=-2)
        b = torch.cat([b_l, b_k], dim=-1)
        c = x

        yhat, status = solver(a, b, c, None)

        loss = criterion(yhat, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

net.load_state_dict(best_state)
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
