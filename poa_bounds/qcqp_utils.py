import math
from typing import Dict, List, Optional, Tuple

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:  # pragma: no cover
    gp = None
    GRB = None


def add_power_chain(
    model: "gp.Model",
    base: "gp.Var",
    max_power: int,
    base_ub: float,
    name_prefix: str,
) -> Dict[int, "gp.Var"]:
    """Create vars for base^r (r=1..max_power) using a quadratic chain.

    Returns a dict pow[r] = base^r.

    Assumes base is nonnegative and base <= base_ub.
    """
    if gp is None:
        raise ImportError("gurobipy is required")
    if max_power < 1:
        raise ValueError("max_power must be >=1")

    powv: Dict[int, gp.Var] = {1: base}
    prev = base
    for r in range(2, max_power + 1):
        ub = float(base_ub ** r) if base_ub > 0 else 0.0
        v = model.addVar(lb=0.0, ub=ub, name=f"{name_prefix}_pow[{r}]")
        model.addQConstr(v == prev * base, name=f"{name_prefix}_pow_link[{r}]")
        powv[r] = v
        prev = v
    return powv



def _interval_product_bounds(lx: float, ux: float, ly: float, uy: float) -> tuple[float, float]:
    """Return (lb, ub) for the product of intervals [lx,ux]*[ly,uy]."""
    cands = (lx * ly, lx * uy, ux * ly, ux * uy)
    lb = min(cands)
    ub = max(cands)
    return float(lb), float(ub)


def _is_finite(x: float) -> bool:
    return math.isfinite(float(x))


def add_bilinear(
    model: "gp.Model",
    x: "gp.Var",
    y: "gp.Var",
    ub: float,
    name: str,
    add_mccormick: bool = True,
    register: bool = True,
) -> "gp.Var":
    """Create z = x*y with a quadratic equality constraint.

    We also *optionally* add the 4 McCormick envelope linear inequalities as
    redundant valid constraints (keeps NonConvex=2, but tightens relaxations).

    ub should be a safe absolute upper bound on |x*y|.
    """
    if gp is None:
        raise ImportError("gurobipy is required")

    ub = float(abs(ub))

    # Use current variable bounds to tighten z bounds when possible.
    # NOTE: In Gurobi, querying attributes like LB/UB on a freshly created
    # variable (before it is integrated into the model) can raise
    # "Index out of range". Calling model.update() fixes this.
    def _safe_bounds(v: "gp.Var") -> Tuple[float, float]:
        try:
            return float(v.LB), float(v.UB)
        except AttributeError:
            model.update()
            return float(v.LB), float(v.UB)

    lx, ux = _safe_bounds(x)
    ly, uy = _safe_bounds(y)

    if _is_finite(lx) and _is_finite(ux) and _is_finite(ly) and _is_finite(uy):
        lb_prod, ub_prod = _interval_product_bounds(lx, ux, ly, uy)
        lb_z = max(-ub, lb_prod)
        ub_z = min(ub, ub_prod)
    else:
        lb_z, ub_z = -ub, ub

    z = model.addVar(lb=lb_z, ub=ub_z, name=name)
    model.addQConstr(z == x * y, name=f"{name}_link")

    mcc = None
    if add_mccormick and _is_finite(lx) and _is_finite(ux) and _is_finite(ly) and _is_finite(uy) and (ux > lx) and (uy > ly):
        # McCormick envelope constraints for z = x*y
        # z >= lx*y + ly*x - lx*ly
        c1 = model.addConstr(z - lx * y - ly * x >= -lx * ly, name=f"{name}_mcc1")
        # z >= ux*y + uy*x - ux*uy
        c2 = model.addConstr(z - ux * y - uy * x >= -ux * uy, name=f"{name}_mcc2")
        # z <= ux*y + ly*x - ux*ly
        c3 = model.addConstr(z - ux * y - ly * x <= -ux * ly, name=f"{name}_mcc3")
        # z <= lx*y + uy*x - lx*uy
        c4 = model.addConstr(z - lx * y - uy * x <= -lx * uy, name=f"{name}_mcc4")
        mcc = (c1, c2, c3, c4)

    if register:
        if not hasattr(model, "_poa_bilin_records"):
            model._poa_bilin_records = []
        model._poa_bilin_records.append(
            {
                "x": x,
                "y": y,
                "z": z,
                "abs_ub": ub,
                "mcc": mcc,
                "name": name,
            }
        )

    return z


def tighten_bilinear_records(model: "gp.Model") -> None:
    """Tighten bounds and refresh McCormick constraints after bound updates.

    Safe to call multiple times. No-op if the model has no recorded bilinears.
    """
    if gp is None:
        raise ImportError("gurobipy is required")
    recs = getattr(model, "_poa_bilin_records", None)
    if not recs:
        return

    for rec in recs:
        x: gp.Var = rec["x"]
        y: gp.Var = rec["y"]
        z: gp.Var = rec["z"]
        abs_ub = float(rec["abs_ub"])
        lx, ux = float(x.LB), float(x.UB)
        ly, uy = float(y.LB), float(y.UB)

        if _is_finite(lx) and _is_finite(ux) and _is_finite(ly) and _is_finite(uy):
            lb_prod, ub_prod = _interval_product_bounds(lx, ux, ly, uy)
            z.LB = max(float(z.LB), max(-abs_ub, lb_prod))
            z.UB = min(float(z.UB), min(abs_ub, ub_prod))

        mcc = rec.get("mcc")
        if mcc is None:
            continue
        c1, c2, c3, c4 = mcc

        # Update coefficients + RHS for each McCormick constraint.
        # c1: z - lx*y - ly*x >= -lx*ly
        model.chgCoeff(c1, y, -lx)
        model.chgCoeff(c1, x, -ly)
        c1.RHS = float(-lx * ly)

        # c2: z - ux*y - uy*x >= -ux*uy
        model.chgCoeff(c2, y, -ux)
        model.chgCoeff(c2, x, -uy)
        c2.RHS = float(-ux * uy)

        # c3: z - ux*y - ly*x <= -ux*ly
        model.chgCoeff(c3, y, -ux)
        model.chgCoeff(c3, x, -ly)
        c3.RHS = float(-ux * ly)

        # c4: z - lx*y - uy*x <= -lx*uy
        model.chgCoeff(c4, y, -lx)
        model.chgCoeff(c4, x, -uy)
        c4.RHS = float(-lx * uy)
