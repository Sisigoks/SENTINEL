"""NSGA-II: fast non-dominated sorting + crowding distance (Deb et al., 2002).

The multi-objective spine of the evolution engine. Returns Pareto fronts over the
constrained objective vectors so the security-utility tradeoff is preserved as a
*front*, not collapsed to a scalar (design doc §3.5/§3.6).
"""

from __future__ import annotations

from .fitness import FitnessVector, dominates


def fast_non_dominated_sort(fits: list[FitnessVector]) -> list[list[int]]:
    n = len(fits)
    S: list[list[int]] = [[] for _ in range(n)]
    dom_count = [0] * n
    fronts: list[list[int]] = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(fits[p], fits[q]):
                S[p].append(q)
            elif dominates(fits[q], fits[p]):
                dom_count[p] += 1
        if dom_count[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt: list[int] = []
        for p in fronts[i]:
            for q in S[p]:
                dom_count[q] -= 1
                if dom_count[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return [f for f in fronts if f]


def crowding_distance(fits: list[FitnessVector], front: list[int]) -> dict[int, float]:
    dist = {i: 0.0 for i in front}
    if len(front) <= 2:
        return {i: float("inf") for i in front}
    objs = [f.objectives() for f in fits]
    m = len(objs[0])
    for k in range(m):
        order = sorted(front, key=lambda i: objs[i][k])
        dist[order[0]] = float("inf")
        dist[order[-1]] = float("inf")
        lo, hi = objs[order[0]][k], objs[order[-1]][k]
        span = (hi - lo) or 1.0
        for idx in range(1, len(order) - 1):
            dist[order[idx]] += (objs[order[idx + 1]][k] - objs[order[idx - 1]][k]) / span
    return dist


def select_survivors(fits: list[FitnessVector], k: int) -> list[int]:
    """NSGA-II survivor selection: fill by front, break ties by crowding distance."""
    fronts = fast_non_dominated_sort(fits)
    chosen: list[int] = []
    for front in fronts:
        if len(chosen) + len(front) <= k:
            chosen.extend(front)
        else:
            cd = crowding_distance(fits, front)
            remaining = sorted(front, key=lambda i: cd[i], reverse=True)
            chosen.extend(remaining[: k - len(chosen)])
            break
    return chosen
