"""MAP-Elites quality-diversity archive (Mouret & Clune, 2015).

Maintains an archive of high-performing genomes across a discretized *behavior space*
so the search yields a *diversity* of defenses (the "which defenses emerge / reuse"
science) rather than collapsing to a single architecture.

Behavior descriptor (2-D, interpretable):
    bd_0 = number of modules (architectural complexity), binned
    bd_1 = fraction of modules targeting agentic (ASI) threats, binned

Each cell keeps the elite (best scalar SecurityFitness) genome for that niche.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.types import Genome
from .fitness import FitnessVector, SecurityFitness


@dataclass
class Elite:
    genome: Genome
    fitness: FitnessVector
    score: float


@dataclass
class MapElites:
    n_bins: int = 5
    max_modules: int = 8
    archive: dict[tuple[int, int], Elite] = field(default_factory=dict)

    def _descriptor(self, genome: Genome) -> tuple[int, int]:
        n = len(genome.modules)
        bd0 = min(int(n / self.max_modules * self.n_bins), self.n_bins - 1)
        agentic = sum(
            1 for m in genome.modules for t in m.targets if t.is_agentic
        )
        frac = agentic / max(sum(len(m.targets) for m in genome.modules), 1)
        bd1 = min(int(frac * self.n_bins), self.n_bins - 1)
        return (bd0, bd1)

    def add(self, genome: Genome, fitness: FitnessVector) -> bool:
        """Insert if it beats the current elite in its niche. Returns True if added."""
        cell = self._descriptor(genome)
        score = SecurityFitness.scalar(fitness)
        cur = self.archive.get(cell)
        if cur is None or score > cur.score:
            self.archive[cell] = Elite(genome=genome, fitness=fitness, score=score)
            return True
        return False

    def coverage(self) -> float:
        return len(self.archive) / (self.n_bins * self.n_bins)

    def best(self) -> Elite | None:
        if not self.archive:
            return None
        return max(self.archive.values(), key=lambda e: e.score)

    def elites(self) -> list[Elite]:
        return list(self.archive.values())
