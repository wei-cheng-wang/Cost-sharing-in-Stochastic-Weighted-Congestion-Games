from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:  # pragma: no cover
    gp = None
    GRB = None

from .bernoulli import bernoulli_cumulants

@dataclass
class LoadMomentVars:
    w: List
    pow: Dict[Tuple[int,int], object]
    P: Dict[int, object]
    K: Dict[int, object]
    M: Dict[int, object]
    ub_P: Dict[int, float]
    ub_K: Dict[int, float]
    ub_M: Dict[int, float]
    name: str

def build_load_moments(model, name: str, n_players: int, d: int, p: float, w_ub: float) -> LoadMomentVars:
    if gp is None:
        raise ImportError("gurobipy is required")

    kap = bernoulli_cumulants(p, d)

    w = [model.addVar(lb=0.0, ub=w_ub, name=f"{name}_w[{i}]") for i in range(n_players)]

    # symmetry breaking: weights are exchangeable within each set.
    # enforce a nonincreasing order to shrink the search space.
    for i in range(n_players - 1):
        model.addConstr(w[i] >= w[i + 1], name=f"{name}_w_order[{i}]")

    pow_vars: Dict[Tuple[int,int], object] = {}
    for i in range(n_players):
        pow_vars[(i,1)] = w[i]
        prev = w[i]
        for r in range(2, d+1):
            ub = w_ub**r
            v = model.addVar(lb=0.0, ub=ub, name=f"{name}_w{i}_pow{r}")
            model.addQConstr(v == prev * w[i], name=f"{name}_pow_link[{i},{r}]")
            pow_vars[(i,r)] = v
            prev = v

    P: Dict[int, object] = {}
    ub_P: Dict[int, float] = {}
    for r in range(1, d+1):
        ub = n_players*(w_ub**r)
        Pr = model.addVar(lb=0.0, ub=ub, name=f"{name}_P{r}")
        model.addConstr(Pr == gp.quicksum(pow_vars[(i,r)] for i in range(n_players)), name=f"{name}_Pdef{r}")
        P[r] = Pr
        ub_P[r] = ub

    K: Dict[int, object] = {}
    ub_K: Dict[int, float] = {}
    for r in range(1, d+1):
        ub = abs(kap[r]) * n_players*(w_ub**r)
        Kr = model.addVar(lb=-ub, ub=ub, name=f"{name}_K{r}")
        model.addConstr(Kr == kap[r] * P[r], name=f"{name}_Kdef{r}")
        K[r] = Kr
        ub_K[r] = ub

    M: Dict[int, object] = {}
    ub_M: Dict[int, float] = {}
    M[0] = model.addVar(lb=1.0, ub=1.0, name=f"{name}_M0")
    model.addConstr(M[0] == 1.0, name=f"{name}_M0fix")
    ub_M[0] = 1.0

    max_load = n_players * w_ub
    for n in range(1, d+1):
        ub = max(1.0, max_load**n)
        Mn = model.addVar(lb=-ub, ub=ub, name=f"{name}_M{n}")
        expr = 0.0
        for k in range(1, n+1):
            coef = math.comb(n-1, k-1)
            # Do not query variable attributes before Model.update(); use explicit bounds instead.
            prod_ub = abs(ub_K[k]) * abs(ub_M[n-k])
            prod = model.addVar(lb=-prod_ub, ub=prod_ub, name=f"{name}_prodM[{n},{k}]")
            model.addQConstr(prod == K[k] * M[n-k], name=f"{name}_prod_link[{n},{k}]")
            expr += coef * prod
        model.addConstr(Mn == expr, name=f"{name}_Mrec{n}")
        M[n] = Mn
        ub_M[n] = ub

    return LoadMomentVars(w=w, pow=pow_vars, P=P, K=K, M=M, ub_P=ub_P, ub_K=ub_K, ub_M=ub_M, name=name)
