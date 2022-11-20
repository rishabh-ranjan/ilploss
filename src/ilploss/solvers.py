from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import gurobi as grb
from gurobi import GRB
import numpy as np
import torch
from tqdm.auto import tqdm

STATUS_MSG = defaultdict(
    lambda: "unknown",
    {
        1: "loaded",
        2: "optimal",
        3: "infeasible",
        4: "inf_or_unbd",
        5: "unbounded",
        6: "cutoff",
        7: "iteration_limit",
        8: "node_limit",
        9: "time_limit",
        10: "solution_limit",
        11: "interrupted",
        12: "numeric",
        13: "suboptimal",
        14: "inprogress",
        15: "user_obj_limit",
    },
)


class ILPSolver:
    """Solve a batch of ILPs/LPs, parallelized via multi-threading.

    GIL doesn't hurt as Gurobi runs C/C++ code.
    Use vtype=GRB.CONTINUOUS to solve LPs.
    Optionally accepts solution hint to provide to solver.

    Returns the solution as well as Gurobi's status on each instance.
    Decode status using STATUS_MSG defined above.
    """

    def __init__(
        self,
        vtype: str = GRB.INTEGER,
        env: Optional[grb.Env] = None,
        num_workers: int = 1,
        show_tqdm: bool = False,
    ):
        super().__init__()
        self.vtype = vtype
        self.env = grb.Env(params={"OutputFlag": 0}) if env is None else env
        self.exe = ThreadPoolExecutor(num_workers)
        self.show_tqdm = show_tqdm

    def __call__(self, a, b, c, h=None):
        batch_size, _, num_vars = a.shape
        y = np.empty([batch_size, num_vars], dtype=np.float32)
        status = np.empty([batch_size], dtype=np.int64)

        def aux(i, a, b, c, h=None):
            num_constrs, num_vars = a.shape
            m = grb.Model(env=self.env)
            y_obj = m.addMVar(num_vars, float("-inf"), float("inf"), vtype=self.vtype)
            m.addMConstr(a, y_obj, GRB.GREATER_EQUAL, -b)
            m.setMObjective(None, c, 0.0, sense=GRB.MINIMIZE)
            if h is not None:
                y_obj.varHintVal = h
            m.optimize()
            try:
                y[i] = y_obj.x
            except grb.GurobiError:
                logger.warning("dummy value of 0 for no soln, e.g. for infeasible ILP")
                y[i] = np.zeros(num_vars)
            status[i] = m.status

        list(
            tqdm(
                self.exe.map(
                    aux,
                    np.arange(batch_size),
                    a.cpu().numpy(),
                    b.cpu().numpy(),
                    c.cpu().numpy(),
                    [None] * batch_size if h is None else h.cpu().numpy(),
                ),
                "instances",
                total=batch_size,
                disable=not self.show_tqdm,
                leave=False,
            )
        )

        return (
            torch.as_tensor(y, device=a.device),
            torch.as_tensor(status, device=a.device),
        )
