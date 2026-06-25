"""Defensive architecture evolution (NSGA-II + MAP-Elites), human-gated."""

from .engine import EvolutionEngine
from .fitness import FitnessVector, SecurityFitness, dominates
from .genome import mutate, random_seed_genome
from .human_gate import HumanGate

__all__ = [
    "EvolutionEngine",
    "FitnessVector",
    "SecurityFitness",
    "dominates",
    "mutate",
    "random_seed_genome",
    "HumanGate",
]
