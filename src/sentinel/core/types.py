"""Typed records shared across the whole system (Pydantic v2).

These types are the contract between layers. They also *encode the safety
invariants in the type system*: the evolution genome can only contain
``DefensiveModuleSpec`` instances drawn from a closed set of module kinds, and
mutation operators are a closed enum — so a dual-use "attack-generating" module
is unrepresentable, not merely disallowed (Invariants I1, I2).
"""

from __future__ import annotations

import hashlib
import time
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


# --------------------------------------------------------------------------- #
# Threat taxonomy (OWASP LLM Top 10 2025 + OWASP Agentic/ASI Top 10 2026)
# --------------------------------------------------------------------------- #
class ThreatClass(str, Enum):
    """The ten threat classes SENTINEL learns to defend against.

    Codes follow the OWASP identifiers used in the paper (Tables 1 and 2).
    """

    PROMPT_INJECTION = "LLM01"           # input reframes prior instructions
    SENSITIVE_DISCLOSURE = "LLM02"       # outputs trend toward secrets
    DATA_POISONING = "LLM04"             # retrieved content contradicts baseline
    IMPROPER_OUTPUT = "LLM05"            # executable payload passed downstream
    EXCESSIVE_AGENCY = "LLM06"           # action exceeds task-justified scope
    SYSTEM_PROMPT_LEAK = "LLM07"         # probes extract the system prompt
    VECTOR_WEAKNESS = "LLM08"            # adversarially similar malicious context
    GOAL_HIJACK = "ASI01"                # objective drifts mid-execution
    TOOL_MISUSE = "ASI02"                # tool calls with out-of-policy params
    PRIVILEGE_ABUSE = "ASI03"            # actions invoke credentials beyond scope
    MEMORY_POISONING = "ASI06"           # persistent store mutated by untrusted input

    @property
    def is_agentic(self) -> bool:
        return self.value.startswith("ASI")


BENIGN = "BENIGN"  # sentinel label for non-threat inputs


class InputChannel(str, Enum):
    """Where an input entered the agent — its trust level differs per channel."""

    USER_TASK = "user_task"           # the task itself
    RETRIEVED_DOC = "retrieved_doc"   # untrusted retrieved content
    TOOL_OUTPUT = "tool_output"       # output returned by a tool
    MEMORY_READ = "memory_read"       # content read back from persistent memory


# --------------------------------------------------------------------------- #
# Behavioral signature (hybrid: 6 interpretable axes + learned residual)
# --------------------------------------------------------------------------- #
class SignatureAxis(str, Enum):
    SEMANTIC_INTENT = "semantic_intent"
    OBJECTIVE_DRIFT = "objective_drift"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    INSTRUCTION_HIERARCHY_VIOLATION = "instruction_hierarchy_violation"
    TRUST_BOUNDARY_CROSSING = "trust_boundary_crossing"
    TOOL_ABUSE = "tool_abuse"


AXIS_ORDER: tuple[SignatureAxis, ...] = tuple(SignatureAxis)


class BehavioralSignature(BaseModel):
    """A threat's behavioral fingerprint.

    ``axes`` are six interpretable values in [0,1] (one per ``SignatureAxis``);
    ``residual`` is a learned embedding capturing discriminative structure the
    axes do not. The hybrid keeps migration/drift analysis interpretable while
    preserving classifier accuracy (design doc G3).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    axes: dict[SignatureAxis, float]
    residual: list[float] = Field(default_factory=list)

    @field_validator("axes")
    @classmethod
    def _check_axes(cls, v: dict[SignatureAxis, float]) -> dict[SignatureAxis, float]:
        missing = set(SignatureAxis) - set(v)
        if missing:
            raise ValueError(f"signature missing axes: {sorted(a.value for a in missing)}")
        for a, val in v.items():
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"axis {a} out of [0,1]: {val}")
        return v

    def vector(self) -> np.ndarray:
        """Full signature vector: [axis_0..axis_5, residual...]."""
        axis_vec = np.array([self.axes[a] for a in AXIS_ORDER], dtype=np.float64)
        if self.residual:
            return np.concatenate([axis_vec, np.asarray(self.residual, dtype=np.float64)])
        return axis_vec

    def axis_vector(self) -> np.ndarray:
        return np.array([self.axes[a] for a in AXIS_ORDER], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Detection / classification results
# --------------------------------------------------------------------------- #
class CascadeStage(str, Enum):
    RULE = "rule_screen"
    ANOMALY = "anomaly_screen"
    NEURAL = "neural_classifier"
    SIGNATURE = "signature_extractor"


class StageResult(BaseModel):
    """Output of one cascade stage — kept for auditability and ablations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    stage: CascadeStage
    flagged: bool
    score: float = 0.0
    detail: dict[str, Any] = Field(default_factory=dict)


class ThreatEvent(BaseModel):
    """The Sentinel layer's verdict on a single input."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    event_id: str
    channel: InputChannel
    raw_text: str
    is_threat: bool
    threat_class: ThreatClass | None = None
    confidence: float = 0.0
    signature: BehavioralSignature | None = None
    stage_results: list[StageResult] = Field(default_factory=list)
    timestamp: float = Field(default_factory=time.time)
    provenance_hash: str | None = None

    @staticmethod
    def hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Defenses
# --------------------------------------------------------------------------- #
class DefenseStrategy(str, Enum):
    """The closed library of defensive strategies the meta-defense selects among.

    There is deliberately no offensive option; the enum *is* the policy boundary.
    """

    INPUT_SANITIZATION = "input_sanitization"
    INSTRUCTION_DATA_SEPARATION = "instruction_data_separation"
    PRIVILEGE_NARROWING = "privilege_narrowing"
    RETRIEVAL_GROUNDING = "retrieval_grounding"
    MEMORY_QUARANTINE = "memory_quarantine"
    OUTPUT_VALIDATION = "output_validation"


class DefensePlan(BaseModel):
    """What the meta-defense decided to apply to a given threat event."""

    strategies: list[DefenseStrategy]
    rationale: str = ""
    selected_by: str = "meta_defense"  # or "static", "reflection", "vanilla"


class Outcome(BaseModel):
    """Result of resolving a task under an applied defense."""

    event_id: str
    attack_succeeded: bool          # ASR numerator
    blocked: bool                   # defense actively blocked
    leaked: bool = False            # canary / secret exfiltration detected
    drifted: bool = False           # objective drift detected
    clean_task_correct: bool | None = None  # utility on the underlying task
    latency_s: float = 0.0
    tokens: int = 0
    applied: DefensePlan | None = None


# --------------------------------------------------------------------------- #
# Defensive genome (the ONLY mutable evolutionary substrate — Invariant I2)
# --------------------------------------------------------------------------- #
class ModuleKind(str, Enum):
    """Closed set of defensive module kinds the evolution engine may add/tune.

    Adding an entry here is the *only* way to expand evolution's reach, and every
    entry is defensive by construction. Objectives, oversight, and the base agent
    are not module kinds and therefore cannot be mutation targets (Invariant I4).
    """

    INPUT_VALIDATOR = "input_validator"
    THREAT_DETECTOR = "threat_detector"
    OUTPUT_GUARD = "output_guard"
    PRIVILEGE_CHECK = "privilege_check"
    POLICY_CONSTRAINT = "policy_constraint"
    ROUTER = "router"


class DefensiveModuleSpec(BaseModel):
    """A configured defensive module within a genome."""

    kind: ModuleKind
    name: str
    params: dict[str, float] = Field(default_factory=dict)
    targets: list[ThreatClass] = Field(default_factory=list)
    enabled: bool = True


class MutationOp(str, Enum):
    """Closed set of allowed mutations (design doc §3.5, Invariant I2)."""

    ADD_VALIDATOR = "add_validator"
    TUNE_VALIDATOR = "tune_validator"
    ADD_DETECTOR = "add_detector"
    TUNE_DETECTOR = "tune_detector"
    ADD_POLICY_CONSTRAINT = "add_policy_constraint"
    REROUTE = "reroute"


class Genome(BaseModel):
    """The defensive architecture as an evolvable object.

    A genome holds *only* defensive modules. It does not and cannot reference the
    base agent's objectives or the oversight/human-gate machinery.
    """

    genome_id: str
    modules: list[DefensiveModuleSpec] = Field(default_factory=list)
    parent_id: str | None = None
    generation: int = 0

    def fingerprint(self) -> str:
        payload = ";".join(
            f"{m.kind.value}:{m.name}:{sorted(m.params.items())}:{m.enabled}"
            for m in sorted(self.modules, key=lambda m: (m.kind.value, m.name))
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class GateState(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEPLOYED = "deployed"


class Proposal(BaseModel):
    """An evolution candidate awaiting the human gate (Invariant I3).

    There is no method that sets ``state = DEPLOYED`` without first passing
    through ``state = APPROVED`` with an attached :class:`HumanGateDecision`.
    """

    proposal_id: str
    genome: Genome
    mutation: MutationOp
    objectives: dict[str, float] = Field(default_factory=dict)  # the fitness vector
    state: GateState = GateState.PENDING
    decision: HumanGateDecision | None = None


class HumanGateDecision(BaseModel):
    approved: bool
    reviewer: str
    rationale: str
    timestamp: float = Field(default_factory=time.time)


Proposal.model_rebuild()
