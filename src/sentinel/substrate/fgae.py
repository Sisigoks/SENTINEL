"""FGAE (Failure-Guided Architecture Evolution) substrate interface.

Per the project context, FGAE is the validated learn-from-signal engine: a base LLM
solves tasks; a critic detects/classifies *failures*; an evolution layer restructures
the architecture from a failure graph. SENTINEL substitutes a *threat* signal for the
*reasoning-failure* signal in the critic slot, reusing the same hierarchy.

This module defines the substrate contract SENTINEL depends on and ships a reference
implementation (``ReferenceFGAE``) so the repo is self-contained. The real SAFE-R/FGAE
engine can be dropped in behind :class:`FGAESubstrate` without touching call sites.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..models.backend import GenerationConfig, ModelBackend


@dataclass(slots=True)
class TaskResult:
    answer: str
    prompt_tokens: int
    completion_tokens: int


class FGAESubstrate(ABC):
    """The base task-solving substrate. The Sentinel layer occupies the critic slot."""

    backend: ModelBackend

    @abstractmethod
    def solve(self, system_prompt: str, task: str, cfg: GenerationConfig) -> TaskResult: ...


class ReferenceFGAE(FGAESubstrate):
    """Reference substrate: a single deterministic LLM solve via the model backend.

    This is the 'Base Layer — Task Solving' of Table 3 ('unchanged from FGAE'). The
    reasoning-failure critic and evolution machinery of full FGAE are provided by the
    real SAFE-R engine when plugged in; for the security study the relevant critic slot
    is occupied by the Sentinel layer.
    """

    def __init__(self, backend: ModelBackend) -> None:
        self.backend = backend

    def solve(self, system_prompt: str, task: str, cfg: GenerationConfig) -> TaskResult:
        gen = self.backend.chat(system_prompt, task, cfg)
        return TaskResult(
            answer=gen.text,
            prompt_tokens=gen.prompt_tokens,
            completion_tokens=gen.completion_tokens,
        )
