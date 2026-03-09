from __future__ import annotations
from typing import List

def _poly_derivative(a: List[float]) -> List[float]:
    if len(a) <= 1:
        return [0.0]
    return [k*a[k] for k in range(1, len(a))]

def _poly_shift_p(a: List[float]) -> List[float]:
    return [0.0] + a

def _poly_sub(a: List[float], b: List[float]) -> List[float]:
    n = max(len(a), len(b))
    out = [0.0]*n
    for i in range(n):
        out[i] = (a[i] if i < len(a) else 0.0) - (b[i] if i < len(b) else 0.0)
    while len(out) > 1 and abs(out[-1]) < 1e-15:
        out.pop()
    return out

def _poly_eval(a: List[float], p: float) -> float:
    s = 0.0
    powp = 1.0
    for c in a:
        s += c*powp
        powp *= p
    return s

def bernoulli_cumulant_polys(max_r: int) -> List[List[float]]:
    if max_r < 1:
        return []
    polys: List[List[float]] = [None] * (max_r + 1)
    polys[1] = [0.0, 1.0]  # p
    for r in range(1, max_r):
        der = _poly_derivative(polys[r])
        term1 = _poly_shift_p(der)
        term2 = _poly_shift_p(_poly_shift_p(der))
        polys[r+1] = _poly_sub(term1, term2)
    return polys

def bernoulli_cumulants(p: float, max_r: int) -> List[float]:
    polys = bernoulli_cumulant_polys(max_r)
    out = [0.0]*(max_r+1)
    for r in range(1, max_r+1):
        out[r] = _poly_eval(polys[r], p)
    return out
