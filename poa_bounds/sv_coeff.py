
"""
sv_coeff.py

Exact coefficient generation for expected Shapley Value (SV) payment of deviators.
"""
from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple

import sympy as sp

Partition = Tuple[int, ...]
BasisTerm = Tuple[Partition, Partition]

def _integer_partitions(n: int, max_part: int | None = None) -> List[Partition]:
    if n == 0:
        return [tuple()]
    if max_part is None or max_part > n:
        max_part = n
    out: List[Partition] = []
    for first in range(max_part, 0, -1):
        for rest in _integer_partitions(n - first, first):
            out.append((first,) + rest)
    return out

def generate_basis_terms(d: int, nD: int = 1) -> List[BasisTerm]:
    if nD == 1:
        terms = set()
        for k in range(1, d + 1):
            partI_list = _integer_partitions(d - k)
            for pI in partI_list:
                terms.add(((k,), pI))
        return sorted(list(terms), key=lambda t: (t[0], t[1]))
    else:
        full_parts = _integer_partitions(d)
        terms = set()
        for part in full_parts:
            m = len(part)
            for mask in range(1 << m):
                pD = tuple(sorted([part[i] for i in range(m) if (mask >> i) & 1], reverse=True))
                pI = tuple(sorted([part[i] for i in range(m) if not ((mask >> i) & 1)], reverse=True))
                terms.add((pD, pI))
        return sorted(list(terms), key=lambda t: (t[0], t[1]))

def _power_sums(w: List[int], d: int) -> Dict[int, sp.Rational]:
    out: Dict[int, sp.Rational] = {}
    for r in range(1, d + 1):
        out[r] = sp.Integer(sum((sp.Integer(x) ** r) for x in w))
    return out

def _eval_partition(P: Dict[int, sp.Rational], part: Partition) -> sp.Rational:
    v = sp.Integer(1)
    for r in part:
        v *= P[r]
    return sp.Rational(v)

@lru_cache(maxsize=8192)
def _get_subset_sum_counts(others: Tuple[int, ...]) -> Tuple[Tuple[int, ...], ...]:
    m1 = len(others)
    if m1 == 0:
        return ((1,),)
    w_max = sum(others)
    dp = [[0] * (w_max + 1) for _ in range(m1 + 1)]
    dp[0][0] = 1
    for w in others:
        for s in range(m1, 0, -1):
            for L in range(w_max, w - 1, -1):
                dp[s][L] += dp[s - 1][L - w]
    return tuple(tuple(row) for row in dp)

def expected_sv_payment_D_exact(wD: List[int], wI: List[int], p: sp.Rational, d: int) -> sp.Rational:
    if not isinstance(p, sp.Rational):
        p = sp.Rational(str(p))
    w_all = list(map(int, wD)) + list(map(int, wI))
    nD = len(wD)
    n = len(w_all)
    isD_all = [True] * nD + [False] * (n - nD)

    one_minus_p = sp.Integer(1) - p
    total = sp.Integer(0)

    fact = [math.factorial(i) for i in range(n + 1)]

    for bits in itertools.product([0, 1], repeat=n):
        k = sum(bits)
        if k == 0:
            continue
        prob = (p ** k) * (one_minus_p ** (n - k))

        idx = [i for i, b in enumerate(bits) if b]
        wA = [w_all[i] for i in idx]
        isDA = [isD_all[i] for i in idx]
        m = len(wA)

        fact_m = fact[m]
        weight_by_s = [fact[s] * fact[m - 1 - s] for s in range(m)]

        state_num = 0

        for i in range(m):
            if not isDA[i]:
                continue
            wi = wA[i]
            
            others = tuple(sorted(wA[:i] + wA[i + 1:]))
            dp_res = _get_subset_sum_counts(others)

            player_sum = 0
            for s in range(m):
                coeff = weight_by_s[s]
                if coeff == 0:
                    continue
                
                sub = 0
                for L, cnt in enumerate(dp_res[s]):
                    if cnt:
                        sub += cnt * (((L + wi) ** d) - (L ** d))
                player_sum += coeff * sub
                
            state_num += player_sum

        if state_num:
            total += prob * sp.Rational(state_num, fact_m)

    return sp.simplify(total)

def _nonincreasing_weight_tuples(k: int, values: List[int]) -> List[Tuple[int, ...]]:
    out: List[Tuple[int, ...]] = []
    for comb in itertools.combinations_with_replacement(values, k):
        out.append(tuple(sorted(comb, reverse=True)))
    return out

@lru_cache(maxsize=256)
def get_sv_coeffs(d: int, p: float, nD: int, nI: int) -> Dict[BasisTerm, float]:
    if d < 2 or not (0 < p <= 1) or nD < 1 or nI < 0:
        raise ValueError("Invalid parameters")

    # Force universal incumbent size for coefficient fitting to ensure full rank
    effective_nI = d if nI > 0 else 0
    return _get_sv_coeffs_internal(d, p, nD, effective_nI, nI == 0)

@lru_cache(maxsize=256)
def _get_sv_coeffs_internal(d: int, p: float, nD: int, eval_nI: int, is_nI_zero: bool) -> Dict[BasisTerm, float]:
    pR = sp.Rational(str(float(p)))
    
    # Generate the correct basis. If nD==1, use the compact single-deviator basis.
    basis = generate_basis_terms(d, nD)
    
    # Filter basis if nI == 0 (no incumbents, so partI must be empty)
    if is_nI_zero:
        basis = [t for t in basis if len(t[1]) == 0]

    m = len(basis)

    vals_D = [1, 2, 3, 5]
    vals_I = [1, 2, 3, 5]

    if nD == 1:
        eval_nD = 1
    else:
        eval_nD = max(nD, d)

    cand_D = _nonincreasing_weight_tuples(eval_nD, vals_D)
    cand_I = _nonincreasing_weight_tuples(eval_nI, vals_I) if eval_nI > 0 else [tuple()]

    cand_pairs = list(itertools.product(cand_D, cand_I))
    random.seed(42)
    random.shuffle(cand_pairs)

    rows: List[List[sp.Rational]] = []
    rhs: List[sp.Rational] = []

    for wD, wI in cand_pairs:
        PD = _power_sums(list(wD), d)
        PI = _power_sums(list(wI), d) if eval_nI > 0 else {r: sp.Integer(0) for r in range(1, d + 1)}

        row = []
        for pD_part, pI_part in basis:
            row.append(_eval_partition(PD, pD_part) * _eval_partition(PI, pI_part))
        val = expected_sv_payment_D_exact(list(wD), list(wI), pR, d)

        rows.append(row)
        rhs.append(val)

        if len(rows) >= m and len(rows) % max(1, m // 2) == 0:
            M = sp.Matrix(rows)
            rnk = M.rank()
            if rnk == m:
                pivots = sp.Matrix(rows).T.rref()[1]
                pivots = list(pivots)[:m]
                Msel = sp.Matrix([rows[i] for i in pivots])
                bsel = sp.Matrix([rhs[i] for i in pivots])
                coeff_vec = list(Msel.LUsolve(bsel))
                coeffs: Dict[BasisTerm, float] = {}
                for j, term in enumerate(basis):
                    c = sp.simplify(coeff_vec[j])
                    if c != 0:
                        coeffs[term] = float(c.evalf(40))
                return coeffs
                
        if len(rows) > max(300, m * 5):
            break

    import numpy as np
    M_np = np.array([[float(x) for x in row] for row in rows], dtype=float)
    b_np = np.array([float(x) for x in rhs], dtype=float)
    coeff_vec, *_ = np.linalg.lstsq(M_np, b_np, rcond=None)

    coeffs_out: Dict[BasisTerm, float] = {}
    for j, term in enumerate(basis):
        val = float(coeff_vec[j])
        if abs(val) > 1e-10:
            coeffs_out[term] = val
    return coeffs_out
