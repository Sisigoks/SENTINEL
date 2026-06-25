"""Stage 1 — RuleScreen: fast, cheap, high-recall *structural* features.

Critically, this is NOT a keyword blocklist (the paper rejects signatures-as-strings
because attackers paraphrase). It computes *structural* features about the
relationship between instructions and data — e.g. whether untrusted content asserts
imperative authority, redefines roles, or breaks the instruction/data channel. These
features never accept/reject on their own; they feed the later stages (design doc §3.1).

Theoretical basis: instruction/data channel confusion (Greshake et al. 2023;
Perez & Ribeiro 2022). The agent cannot reliably separate instructions from data in
a single channel, so we surface *structural* signals of channel violation.
"""

from __future__ import annotations

import re

from ..core.types import CascadeStage, InputChannel, StageResult

# Structural pattern families (regex over *form*, not specific banned words).
_ROLE_REDEFINITION = re.compile(
    r"\b(you are now|act as|pretend to be|from now on|ignore (all|previous|prior)|"
    r"disregard (the|all|previous)|new instructions?|system prompt)\b",
    re.IGNORECASE,
)
_IMPERATIVE = re.compile(
    r"\b(ignore|disregard|reveal|print|output|send|exfiltrate|delete|execute|run|"
    r"override|bypass|forget|leak|disclose|grant|escalate)\b",
    re.IGNORECASE,
)
_DELIMITER_BREAK = re.compile(r"(```|</?(system|user|assistant)>|\[/?INST\]|<\|.*?\|>)")
_TOOL_PARAM = re.compile(r"\b(sudo|admin|root|--force|rm\s+-rf|DROP\s+TABLE|;|\|\||&&)\b", re.IGNORECASE)
_SECRET_PROBE = re.compile(
    r"\b(api[_\- ]?key|password|secret|token|credential|system prompt|instructions above)\b",
    re.IGNORECASE,
)


class RuleScreen:
    """Computes structural risk features. High recall by design (fail-open to Stage 2)."""

    UNTRUSTED = {InputChannel.RETRIEVED_DOC, InputChannel.TOOL_OUTPUT, InputChannel.MEMORY_READ}

    def screen(self, text: str, channel: InputChannel) -> StageResult:
        untrusted = channel in self.UNTRUSTED
        features = {
            "role_redefinition": bool(_ROLE_REDEFINITION.search(text)),
            "imperative_in_untrusted": untrusted and bool(_IMPERATIVE.search(text)),
            "delimiter_break": bool(_DELIMITER_BREAK.search(text)),
            "tool_param_risk": bool(_TOOL_PARAM.search(text)),
            "secret_probe": bool(_SECRET_PROBE.search(text)),
            "imperative_density": len(_IMPERATIVE.findall(text)) / max(len(text.split()), 1),
        }
        # weighted structural score; weights reflect asymmetric risk, not a blocklist
        score = (
            0.30 * features["role_redefinition"]
            + 0.30 * features["imperative_in_untrusted"]
            + 0.15 * features["delimiter_break"]
            + 0.15 * features["tool_param_risk"]
            + 0.10 * features["secret_probe"]
            + min(features["imperative_density"], 0.2)
        )
        flagged = score >= 0.15  # low threshold: route anything plausible to Stage 2
        return StageResult(
            stage=CascadeStage.RULE, flagged=flagged, score=float(min(score, 1.0)), detail=features
        )
