# ilploss: fast solver-free training of neural-ILP architectures

`ilploss` is a pytorch-based library which lets you train models with an Integer Linear Programming (ILP) output layer, without ever calling an ILP solver during training. It implements techniques from the paper [A Solver-Free Framework for Scalable Learning in Neural ILP Architectures][paper], accepted at NeurIPS 2022. Full code for the results in the paper can be found [here][dair].


# Install

In a python environment with [`torch`][torch] and [`gurobi`][gurobi] (not `gurobipy`) installed:

```
pip install git+https://github.com/rishabh-ranjan/ilploss
```

# Usage

The `ilploss` library provides the following modules:

1. `ilploss.encoders`: get ILP from input
2. `ilploss.samplers`: sample negatives
3. `ilploss.losses`: compute and balance loss terms
4. `ilploss.solvers`: solve batched ILPs at inference, in parallel

ILP instances are specified by `a`, `b`, `c` under the convention that for solution vector `z`, the cost `c^T z` is to be minimized under the constraints `a @ z + b >= 0` (`@` denotes matrix multiply).


Check out the source code for further details.


# Example

A demo script is provided [here][demo]. This replicates the ILP-Loss experiments on random constraints from Table 2 of the [paper][paper]. To run the demo:

```
git clone https://github.com/rishabh-ranjan/ilploss
cd ilploss
tests/random.py tests/data/random/dense/16_dim/8_const/0/dataset.pt
```

You can choose any file from `tests/data` as the argument.

# Citation

```
@inproceedings{ilploss,
  author = {Nandwani, Yatin and Ranjan, Rishabh and Mausam and Singla, Parag},
  title = {A Solver-Free Framework for Scalable Learning in Neural ILP Architectures},
  booktitle = {Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, November 29-Decemer 1, 2022},
  year = {2022},
}
```

[paper]: https://arxiv.org/abs/2210.09082
[torch]: https://pytorch.org/get-started/locally/
[gurobi]: https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python
[dair]: https://github.com/dair-iitd/ilploss
[demo]: https://github.com/rishabh-ranjan/ilploss/blob/main/tests/random.py
