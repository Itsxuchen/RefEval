from fractions import Fraction
from math import comb

import pytest

from src.finite_population_intervals import hypergeom_ci, _tail_accept


def pmf(M, K, n, x):
    if x < max(0, n - M + K) or x > min(n, K):
        return Fraction(0)
    return Fraction(comb(K, x) * comb(M-K, n-x), comb(M, n))


def test_inclusive_boundary_and_complement():
    assert pmf(16, 13, 2, 0) == Fraction(1, 40)
    assert hypergeom_ci(16, 2, 0) == (0, 13)
    assert hypergeom_ci(16, 2, 2) == (3, 16)
    # The tolerance triggers exact evaluation, not automatic acceptance.
    assert not _tail_accept(16, 13, 2, 0, False, Fraction(1, 40) + Fraction(1, 10**15))


def test_zero_and_census():
    assert hypergeom_ci(0, 0, 0) == (0, 0)
    assert hypergeom_ci(8, 0, 0) == (0, 8)
    for x in range(9):
        assert hypergeom_ci(8, 8, x) == (x, x)


@pytest.mark.parametrize('args', [(3,4,1),(3,2,3),(3,-1,0),(3,2,1,0),(3,2,1,1),(3.1,2,1)])
def test_invalid(args):
    with pytest.raises(ValueError):
        hypergeom_ci(*args)


def test_exhaustive_rational_acceptance_and_coverage_through_24():
    threshold = Fraction(1, 40)
    for M in range(1, 25):
        for n in range(M + 1):
            intervals = [hypergeom_ci(M, n, x) for x in range(n + 1)]
            for x, (lo, hi) in enumerate(intervals):
                assert hypergeom_ci(M, n, n-x) == (M-hi, M-lo)
                accepted = []
                for K in range(x, M-n+x+1):
                    lower_tail = sum(pmf(M,K,n,j) for j in range(x+1))
                    upper_tail = sum(pmf(M,K,n,j) for j in range(x,n+1))
                    if min(lower_tail,upper_tail) >= threshold:
                        accepted.append(K)
                assert (lo,hi) == (min(accepted),max(accepted))
            for K in range(M+1):
                coverage = sum(pmf(M,K,n,x) for x,(lo,hi) in enumerate(intervals) if lo <= K <= hi)
                assert coverage >= Fraction(19,20), (M,n,K,coverage)
