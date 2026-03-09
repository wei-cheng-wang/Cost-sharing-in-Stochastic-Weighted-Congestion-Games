import math
from typing import Any, Dict, List, Optional

from .ps_qcqp import lambda_mu_ps_multi
from .sv_explicit_qcqp import lambda_mu_sv_explicit_multi


def _default_mu_grid() -> List[float]:
    return [
        0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
        0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
    ]

def poa_upper_bound(
    n: int,
    p: float,
    d: int,
    rule: str = "ps",
    mu_grid: Optional[List[float]] = None,
    time_limit_per: float = 10.0,
    mipgap: float = 1e-4,
    verbose: bool = False,
    max_pairs: Optional[int] = None,
    max_workers: Optional[int] = None,
    **_: Any,
) -> Dict[str, Any]:
    
    if mu_grid is None:
        mu_grid = _default_mu_grid()
    mu_grid = [float(mu) for mu in mu_grid]
    if any(mu <= 0 or mu >= 1 for mu in mu_grid):
        raise ValueError("mu_grid must be contained in (0,1)")

    best_lambda = {float(mu): -float("inf") for mu in mu_grid}
    best_argmax = {float(mu): None for mu in mu_grid}

    pairs_seen = 0
    errors = []

    if rule == "ps":
        nD_range = range(1, n + 1)
        nI_start = 1
    elif rule == "sv":
        nD_range = range(1, n + 1)
        nI_start = 1
    else:
        raise ValueError("rule must be ps or sv")

    stop_all = False
    total_time_limit = time_limit_per * len(mu_grid)

    for nD in nD_range:
        if stop_all:
            break

        for nI in range(nI_start, n + 1):
            if max_pairs is not None and pairs_seen >= max_pairs:
                stop_all = True
                break
            pairs_seen += 1

            # NEW: solve one model per mu to avoid mu_min contaminating all mus
            for mu in mu_grid:
                mu_f = float(mu)

                if rule == "ps":
                    res_map = lambda_mu_ps_multi(
                        n=n, p=p, d=d, mu_list=[mu_f], nD=nD, nI=nI,
                        time_limit=time_limit_per,  # per-mu time limit
                        mipgap=mipgap, verbose=verbose,
                    )
                else:
                    res_map = lambda_mu_sv_explicit_multi(
                        n=n, p=p, d=d, mu_list=[mu_f], nD=nD, nI=nI,
                        time_limit=time_limit_per,  # per-mu time limit
                        mipgap=mipgap, verbose=verbose,
                    )

                # res_map has a single entry now
                rr = res_map.get(mu_f, next(iter(res_map.values())))

                lam = rr.get("Lambda", float("nan")) if isinstance(rr, dict) else rr.obj
                st  = rr.get("status", None)        if isinstance(rr, dict) else rr.status

                if not math.isfinite(lam):
                    errors.append((nD, nI, mu_f, st, "Lambda not finite"))
                    continue

                if lam > best_lambda[mu_f]:
                    best_lambda[mu_f] = float(lam)
                    best_argmax[mu_f] = rr.get("argmax", None) if isinstance(rr, dict) else rr.details

    bad = [mu for mu in mu_grid if not math.isfinite(best_lambda[mu]) or best_lambda[mu] == -math.inf]
    if bad:
        head = errors[:10]
        print(f"WARNING: {rule} failed for mu={bad[:5]}..., sample errors={head}")
        # Remove bad mus from processing
        mu_grid = [m for m in mu_grid if m not in bad]
        if not mu_grid:
            raise RuntimeError(f"All mu failed for {rule}!")

    best_poa = math.inf
    best_mu = None
    for mu in mu_grid:
        lam = best_lambda[mu]
        poa = lam / (1.0 - mu)
        if poa < best_poa:
            best_poa = poa
            best_mu = mu

    curve = {
        mu: {
            "Lambda": best_lambda[mu],
            "PoA": best_lambda[mu] / (1.0 - mu),
            "argmax": best_argmax[mu],
        }
        for mu in mu_grid
    }

    return {
        "rule": rule, "n": n, "p": p, "d": d,
        "best": {
            "mu": best_mu,
            "Lambda": best_lambda[best_mu] if best_mu is not None else math.nan,
            "PoA": best_poa,
            "argmax": best_argmax[best_mu] if best_mu is not None else None,
        },
        "curve": curve,
    }