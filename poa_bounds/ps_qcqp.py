
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:  # pragma: no cover
    gp = None
    GRB = None

from .moments import build_load_moments


@dataclass
class SolveResult:
    status: int
    obj: float
    details: Dict[str, Any]


def _default_wub_D(p: float, d: int) -> float:
    return max(1.0, 1.5 * (1.0 / max(float(p), 1e-9)) ** (1.0 / d))

def _default_wub_I(p: float, d: int, mu_min: float) -> float:
    if mu_min <= 0:
        return 100.0
    return max(2.0, 3.0 / max(mu_min, 1e-3))


def lambda_mu_ps(
    n: int,
    p: float,
    d: int,
    mu: float,
    nD: int,
    nI: int,
    wubI: Optional[float] = None,
    time_limit: float = 10.0,
    mipgap: float = 1e-4,
    verbose: bool = False,
) -> SolveResult:
    res_map = lambda_mu_ps_multi(
        n=n,
        p=p,
        d=d,
        mu_list=[mu],
        nD=nD,
        nI=nI,
        wubI=wubI,
        time_limit=time_limit,
        mipgap=mipgap,
        verbose=verbose,
    )
    return res_map[mu]


def lambda_mu_ps_multi(
    n: int,
    p: float,
    d: int,
    mu_list: List[float],
    nD: int,
    nI: int,
    wubI: Optional[float] = None,
    time_limit: float = 10.0,
    mipgap: float = 1e-4,
    verbose: bool = False,
) -> Dict[float, SolveResult]:
    if gp is None:
        raise ImportError("gurobipy is required")
    if not mu_list:
        raise ValueError("mu_list must be non-empty")
    if not (n >= 2 and 1 <= nD <= n and 0 <= nI <= n):
        raise ValueError("invalid (n,nD,nI)")
    if not (0 < p <= 1.0):
        raise ValueError("p must be in (0,1]")
    if d < 2:
        raise ValueError("d must be >=2")

    mu_min = min(mu_list)
    wubD = _default_wub_D(p, d)
    if wubI is None:
        wubI = _default_wub_I(p, d, mu_min)

    model = gp.Model("ps_qcqp_multi")
    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.NonConvex = 2
    model.Params.TimeLimit = float(time_limit)
    model.Params.MIPGap = float(mipgap)

    D = build_load_moments(model, "D", nD, d, p, wubD)
    I = build_load_moments(model, "I", nI, d, p, wubI) if nI > 0 else None

    # Normalize denominator: C_e(D)=E[W_D^d]=1
    model.addConstr(D.M[d] == 1.0, name="norm_den")

    # Unilateral deviation for PS: \sum_{i \in D} E[ w_i (w_i + W_I)^{d-1} ]
    dev_expr = 0.0
    if nI == 0:
        cost_I = 0.0
        dev_expr = D.P[d]
    else:
        cost_I = I.M[d]
        for k in range(d):
            coef = p * math.comb(d - 1, k)
            if k == 0:
                dev_expr += coef * D.P[d]
            else:
                prod_ub = abs(D.ub_P[d - k]) * abs(I.ub_M[k])
                t = model.addVar(lb=-prod_ub, ub=prod_ub, name=f"dev_prod[{k}]")
                model.addQConstr(t == D.P[d - k] * I.M[k], name=f"dev_prod_link[{k}]")
                dev_expr += coef * t

    results: Dict[float, SolveResult] = {}
    t_start = time.time()
    for idx, mu in enumerate(mu_list):
        remaining = float(time_limit) - (time.time() - t_start)
        if remaining <= 0:
            break
        model.Params.TimeLimit = max(0.1, remaining / max(1, len(mu_list) - idx))
        if not (0 <= mu < 1):
            raise ValueError("each mu must be in [0,1)")

        obj_expr = dev_expr - (mu * cost_I if nI > 0 else 0.0)
        model.setObjective(obj_expr, GRB.MAXIMIZE)
        model.optimize()

        status = int(model.Status)
        obj = float("nan")
        if status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
            try:
                obj = float(model.ObjVal)
            except Exception:
                obj = float("nan")

        details: Dict[str, Any] = {
            "mu": mu,
            "n": n,
            "p": p,
            "d": d,
            "nD": nD,
            "nI": nI,
            "wubD": wubD,
            "wubI": wubI,
        }
        if status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
            try:
                details["wD"] = [float(v.X) for v in D.w]
                details["wI"] = [float(v.X) for v in I.w] if I is not None else []
            except Exception:
                details["wD"] = []
                details["wI"] = []

        results[mu] = SolveResult(status=status, obj=obj, details=details)

    return results
