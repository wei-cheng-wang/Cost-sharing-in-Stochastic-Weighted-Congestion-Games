"""
sv_explicit_qcqp.py

Shapley-Value (SV) smoothness maximization using EXPLICIT weight variables.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple
import math

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:  # pragma: no cover
    gp = None
    GRB = None

from .moments import build_load_moments
from .qcqp_utils import add_bilinear
from .sv_coeff import get_sv_coeffs, Partition


def _default_wub_D(p: float, d: int) -> float:
    return max(1.0, 1.5 * (1.0 / max(float(p), 1e-9)) ** (1.0 / d))

def _default_wub_I(p: float, d: int, mu_min: float) -> float:
    if mu_min <= 0:
        return 1e3
    return max(2.0, 3.0 / max(mu_min, 1e-3))

def _partition_product_var(
    model: "gp.Model",
    P: Dict[int, "gp.Var"],
    ub_P: Dict[int, float],
    part: Partition,
    prefix: str,
) -> Tuple[float | "gp.Var", float]:
    if len(part) == 0:
        return 1.0, 1.0

    r0 = part[0]
    cur = P[r0]
    cur_ub = float(ub_P[r0])

    for t, r in enumerate(part[1:], start=1):
        nxt = P[r]
        nxt_ub = float(ub_P[r])
        ub = cur_ub * nxt_ub
        cur = add_bilinear(model, cur, nxt, ub=ub, name=f"{prefix}_mul[{t}]")
        cur_ub = ub

    return cur, cur_ub


def lambda_mu_sv_explicit_multi(
    n: int,
    p: float,
    d: int,
    mu_list: List[float],
    nD: int,
    nI: int,
    wubD: Optional[float] = None,
    wubI: Optional[float] = None,
    time_limit: float = 20.0,
    mipgap: float = 1e-4,
    verbose: bool = False,
    threads: Optional[int] = None,
) -> Dict[float, Dict[str, Any]]:
    
    if gp is None:
        raise ImportError("gurobipy is required")
    if d < 2 or not (0 < p <= 1) or nD < 1 or nI < 0 or len(mu_list) == 0:
        raise ValueError("Invalid parameters")

    mu_min = float(min(mu_list))
    if wubD is None:
        wubD = _default_wub_D(p, d)
    if wubI is None:
        wubI = _default_wub_I(p, d, mu_min)

    model = gp.Model("sv_explicit_qcqp")
    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.NonConvex = 2
    model.Params.NumericFocus = 1 
    model.Params.ScaleFlag = 2
    model.Params.MIPGap = float(mipgap)
    if threads is not None and int(threads) > 0:
        model.Params.Threads = int(threads)

    D = build_load_moments(model, "D", nD, d, p, wubD)
    I = build_load_moments(model, "I", nI, d, p, wubI) if nI > 0 else None

    model.addConstr(D.M[d] >= 1.0 - 1e-7, name="normalize_MDd_lb")
    model.addConstr(D.M[d] <= 1.0 + 1e-7, name="normalize_MDd_ub")

    # Use exact SV coefficients for a single deviator to compute Unilateral Deviation.
    coeffs = get_sv_coeffs(d=d, p=p, nD=1, nI=nI)

    sv_dev = 0.0
    term_id = 0
    for (partD, partI), coef in coeffs.items():
        if nI == 0 and len(partI) > 0:
            continue

        # IMPORTANT: partD is a partition (e.g., (2,1,1)), representing the monomial
        #     \prod_{r \in partD} P_r(D)
        # and not the single power-sum P_{sum(partD)}(D). Build the product explicitly.
        vD, ubD = _partition_product_var(model, D.P, D.ub_P, partD, prefix=f"t{term_id}_D")

        if nI > 0 and len(partI) > 0:
            vI, ubI = _partition_product_var(model, I.P, I.ub_P, partI, prefix=f"t{term_id}_I")

            if isinstance(vD, (int, float)):
                term = vI
            elif isinstance(vI, (int, float)):
                term = vD
            else:
                ub = ubD * ubI
                term = add_bilinear(model, vD, vI, ub=ub, name=f"t{term_id}_DI")

            sv_dev += float(coef) * term
        else:
            sv_dev += float(coef) * vD
        term_id += 1    


    cost_I = I.M[d] if nI > 0 else 0.0

    out: Dict[float, Dict[str, Any]] = {}
    t0 = time.time()
    
    for idx, mu in enumerate(mu_list):
        remaining = max(0.1, float(time_limit) - (time.time() - t0))
        per = remaining / max(1, (len(mu_list) - idx))
        model.Params.TimeLimit = float(per)

        model.setObjective(sv_dev - float(mu) * cost_I, GRB.MAXIMIZE)
        model.optimize()
        
        status = int(model.Status)
        bound = float('nan')
        inc = float('nan')
        
        try: bound = float(model.ObjBound)
        except Exception: pass
            
        if model.SolCount > 0:
            try: inc = float(model.ObjVal)
            except Exception: pass

        val = bound if math.isfinite(bound) else inc

        arg = {
            'nD': nD,
            'nI': nI,
            'mu': float(mu),
            'wubD': float(wubD),
            'wubI': float(wubI),
            'ObjBound': bound,
            'ObjVal': inc,
            'certified': bool(math.isfinite(bound)),
        }
        
        if model.SolCount > 0:
            try:
                arg['wD'] = [float(v.X) for v in D.w]
                if nI > 0: arg['wI'] = [float(v.X) for v in I.w]
            except Exception: pass

        out[float(mu)] = {'Lambda': val, 'status': status, 'argmax': arg}

    return out