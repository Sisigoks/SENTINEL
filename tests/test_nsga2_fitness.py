"""NSGA-II + constrained multi-objective fitness tests."""

from __future__ import annotations

from sentinel.evolution.fitness import FitnessVector, SecurityFitness, dominates
from sentinel.evolution.nsga2 import (
    crowding_distance,
    fast_non_dominated_sort,
    select_survivors,
)


def _fv(asr, recall=0.9, precision=0.9, fpr=0.0, util_drop=0.0, lat=1.0, cost=100.0):
    return FitnessVector(asr, recall, precision, fpr, util_drop, lat, cost)


def test_pareto_dominance_feasible():
    a = _fv(0.1)  # lower ASR => higher (1-ASR) objective
    b = _fv(0.3)
    assert dominates(a, b)
    assert not dominates(b, a)


def test_feasible_beats_infeasible():
    feasible = _fv(0.4, util_drop=0.0)
    infeasible = _fv(0.1, util_drop=0.5)  # violates utility-drop constraint
    assert dominates(feasible, infeasible)
    assert not dominates(infeasible, feasible)


def test_non_dominated_sort_fronts():
    fits = [_fv(0.1, recall=0.5), _fv(0.5, recall=0.95), _fv(0.4, recall=0.4)]
    fronts = fast_non_dominated_sort(fits)
    assert fronts  # at least one front
    # the dominated-by-all point (0.4 asr, 0.4 recall) should not be in the first front
    assert 2 not in fronts[0]


def test_crowding_distance_endpoints_infinite():
    fits = [_fv(0.1), _fv(0.2), _fv(0.3)]
    cd = crowding_distance(fits, [0, 1, 2])
    assert cd[0] == float("inf") or cd[2] == float("inf")


def test_survivor_selection_size():
    fits = [_fv(a / 10) for a in range(10)]
    survivors = select_survivors(fits, 4)
    assert len(survivors) == 4
    # best (lowest asr) should survive
    assert 0 in survivors


def test_security_fitness_scalar_monotone():
    low_asr = SecurityFitness.scalar(_fv(0.1))
    high_asr = SecurityFitness.scalar(_fv(0.6))
    assert low_asr > high_asr
