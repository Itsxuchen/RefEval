"""Inclusive equal-tail hypergeometric intervals with exact boundary comparisons.

SciPy evaluates ordinary tails. A near-threshold tolerance only triggers an
integer/rational recomputation; it never directly accepts or widens an endpoint.
The resulting interval is pointwise for an SRSWOR fixed population and sample n.
"""
from fractions import Fraction
from functools import lru_cache
from math import comb, isclose
from numbers import Integral

from scipy.stats import hypergeom


def _exact_tail_accept(M, K, n, x, upper, threshold):
    """Compare a tail to threshold without any floating-point probabilities."""
    support_lo, support_hi = max(0, n - (M - K)), min(n, K)
    lo, hi = (max(x, support_lo), support_hi) if upper else (support_lo, min(x, support_hi))
    numerator = sum(comb(K, j) * comb(M - K, n - j) for j in range(lo, hi + 1))
    return numerator * threshold.denominator >= comb(M, n) * threshold.numerator


def _tail_accept(M, K, n, x, upper, threshold):
    value = float(hypergeom.sf(x - 1, M, K, n) if upper else hypergeom.cdf(x, M, K, n))
    target = float(threshold)
    # This guard is deliberately much wider than the demonstrated ~1e-17 error.
    # All close decisions use exact arithmetic, including values just BELOW alpha/2.
    if isclose(value, target, rel_tol=1e-10, abs_tol=1e-14):
        return _exact_tail_accept(M, K, n, x, upper, threshold)
    return value >= target


@lru_cache(maxsize=100000)
def hypergeom_ci(M: int, n: int, x: int, alpha: float = .05) -> tuple[int, int]:
    """Unknown success-total bounds; include tails exactly equal to alpha/2."""
    if not (all(isinstance(v, Integral) for v in (M, n, x)) and 0 <= x <= n <= M and 0 < alpha < 1):
        raise ValueError("invalid hypergeometric observation")
    M, n, x = int(M), int(n), int(x)
    threshold = Fraction(str(alpha)) / 2
    if n == 0:
        return 0, M
    if n == M:
        return x, x
    lo, hi = x, M - n + x
    while lo < hi:
        mid = (lo + hi) // 2
        if _tail_accept(M, mid, n, x, True, threshold):
            hi = mid
        else:
            lo = mid + 1
    lower = lo
    lo, hi = x, M - n + x
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _tail_accept(M, mid, n, x, False, threshold):
            lo = mid
        else:
            hi = mid - 1
    return lower, lo
