"""The Sentinel cascade — orchestrates Stages 1-4 with a fail-closed state machine.

State machine (design doc §1.3a):
    NEW -> RULE -> [clean & in-distribution] -> BENIGN
                -> ANOMALY -> [in-distribution & unflagged] -> BENIGN
                           -> CLASSIFY -> [p<tau_low] -> UNCERTAIN (conservative)
                                       -> [p>=tau_high] -> THREAT -> SIGNATURE -> EMIT
    any error -> FAIL_CLOSED (treat as threat)

The cascade is *ablatable*: each stage can be disabled via ``enabled_stages`` to
support the paper's ablation study (-classifier, rule-only, neural-only, etc.).
"""

from __future__ import annotations

import uuid

from ..core.logging import get_logger
from ..core.types import (
    BehavioralSignature,
    CascadeStage,
    InputChannel,
    ThreatClass,
    ThreatEvent,
)
from ..models.encoder import FrozenEncoder
from .anomaly_screen import AnomalyScreen
from .classifier import NeuralThreatClassifier
from .rule_screen import RuleScreen
from .signature import SignatureExtractor

log = get_logger(__name__)


class SentinelCascade:
    def __init__(
        self,
        encoder: FrozenEncoder,
        rule: RuleScreen,
        anomaly: AnomalyScreen,
        classifier: NeuralThreatClassifier,
        signature: SignatureExtractor,
        *,
        tau_low: float = 0.35,
        tau_high: float = 0.55,
        enabled_stages: frozenset[CascadeStage] | None = None,
    ) -> None:
        self.encoder = encoder
        self.rule = rule
        self.anomaly = anomaly
        self.classifier = classifier
        self.signature = signature
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.enabled = enabled_stages or frozenset(CascadeStage)

    def _on(self, stage: CascadeStage) -> bool:
        return stage in self.enabled

    def screen(
        self,
        text: str,
        channel: InputChannel = InputChannel.USER_TASK,
        *,
        granted_scope: set[str] | None = None,
        requested_scope: set[str] | None = None,
    ) -> ThreatEvent:
        event_id = uuid.uuid4().hex[:12]
        stage_results = []
        try:
            # Stage 1 — structural
            rule_res = self.rule.screen(text, channel)
            if self._on(CascadeStage.RULE):
                stage_results.append(rule_res)

            # embedding shared by stages 2-4
            embedding = self.encoder.encode_one(text)

            # Stage 2 — anomaly
            anom_res = self.anomaly.screen(embedding)
            if self._on(CascadeStage.ANOMALY):
                stage_results.append(anom_res)

            # Early benign exit: nothing structural and in-distribution
            rule_clear = (not rule_res.flagged) or (not self._on(CascadeStage.RULE))
            anom_clear = (not anom_res.flagged) or (not self._on(CascadeStage.ANOMALY))
            if rule_clear and anom_clear:
                return ThreatEvent(
                    event_id=event_id,
                    channel=channel,
                    raw_text=text,
                    is_threat=False,
                    confidence=0.0,
                    stage_results=stage_results,
                    provenance_hash=ThreatEvent.hash_text(text),
                )

            # Stage 3 — neural classification
            threat_class: ThreatClass | None = None
            confidence = max(rule_res.score, anom_res.score)
            if self._on(CascadeStage.NEURAL):
                clf_res = self.classifier.predict(embedding)
                stage_results.append(clf_res)
                label = clf_res.detail["label"]
                confidence = clf_res.score
                if label != "BENIGN":
                    threat_class = ThreatClass(label)
                # below tau_low and unflagged structurally -> treat benign
                if confidence < self.tau_low and not rule_res.flagged and label == "BENIGN":
                    return ThreatEvent(
                        event_id=event_id, channel=channel, raw_text=text, is_threat=False,
                        confidence=confidence, stage_results=stage_results,
                        provenance_hash=ThreatEvent.hash_text(text),
                    )

            # Stage 4 — signature
            sig: BehavioralSignature | None = None
            if self._on(CascadeStage.SIGNATURE):
                sig_res, sig = self.signature.extract(
                    text, channel, embedding, rule_res.detail, anom_res.detail,
                    granted_scope=granted_scope, requested_scope=requested_scope,
                )
                stage_results.append(sig_res)

            # fail-closed: anything that reached here is treated as a threat
            is_threat = True
            if threat_class is None:
                # rule/anomaly flagged but classifier off/uncertain -> default class by signal
                threat_class = self._fallback_class(rule_res.detail)

            return ThreatEvent(
                event_id=event_id, channel=channel, raw_text=text, is_threat=is_threat,
                threat_class=threat_class, confidence=float(confidence), signature=sig,
                stage_results=stage_results, provenance_hash=ThreatEvent.hash_text(text),
            )
        except Exception as exc:  # FAIL_CLOSED
            log.error("cascade error -> fail-closed", error=str(exc), event_id=event_id)
            return ThreatEvent(
                event_id=event_id, channel=channel, raw_text=text, is_threat=True,
                threat_class=ThreatClass.PROMPT_INJECTION, confidence=1.0,
                stage_results=stage_results, provenance_hash=ThreatEvent.hash_text(text),
            )

    @staticmethod
    def _fallback_class(rule_detail: dict) -> ThreatClass:
        if rule_detail.get("secret_probe"):
            return ThreatClass.SYSTEM_PROMPT_LEAK
        if rule_detail.get("tool_param_risk"):
            return ThreatClass.TOOL_MISUSE
        if rule_detail.get("role_redefinition"):
            return ThreatClass.GOAL_HIJACK
        return ThreatClass.PROMPT_INJECTION
