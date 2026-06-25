"""Runtime compatibility shims for known vLLM <-> transformers <-> model skews.

These are *defensive* best-effort patches applied right before a vLLM engine is built.
Each is narrow, idempotent, and logged. The clean fix is always to pin a compatible
dependency set (see ``docs/02-troubleshooting.md``); these shims keep a Colab/A100 run
from hard-crashing on a known, well-understood version mismatch.

Currently handled:
  * ``<Tokenizer> has no attribute all_special_tokens_extended`` — newer transformers
    builds dropped this property from some *slow* tokenizer classes (e.g. Qwen2Tokenizer),
    while vLLM's tokenizer cache still reads it. We restore a safe fallback property.
"""

from __future__ import annotations

from ..core.logging import get_logger

log = get_logger(__name__)

_APPLIED = False


def _patch_all_special_tokens_extended() -> None:
    """Restore ``all_special_tokens_extended`` on tokenizer base classes if missing.

    vLLM's ``get_cached_tokenizer`` reads ``tokenizer.all_special_tokens_extended`` and
    caches it. On transformers builds where the slow tokenizer no longer defines it, that
    read raises ``AttributeError``. We add a property that returns the AddedToken objects
    when available (falling back to the plain special-token strings) — exactly what vLLM
    then caches and overrides.
    """
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    except Exception as exc:  # transformers not importable (non-GPU tooling) — nothing to do
        log.debug("compat: transformers not importable; skipping tokenizer patch", error=str(exc))
        return

    if hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        return  # property already present in this transformers build

    def _all_special_tokens_extended(self):  # noqa: ANN001
        # Prefer AddedToken objects (the "extended" form) from the decoder map.
        decoder = getattr(self, "added_tokens_decoder", None)
        if decoder:
            special = []
            seen: set[str] = set()
            for tok in decoder.values():
                content = str(tok)
                is_special = getattr(tok, "special", False)
                if is_special and content not in seen:
                    seen.add(content)
                    special.append(tok)
            if special:
                return special
        # Fallback: plain special-token strings.
        try:
            return list(self.all_special_tokens)
        except Exception:
            return []

    PreTrainedTokenizerBase.all_special_tokens_extended = property(_all_special_tokens_extended)
    log.warning(
        "compat: patched missing tokenizer.all_special_tokens_extended "
        "(transformers/vLLM version skew). Pin transformers per docs/02-troubleshooting.md "
        "for the clean fix."
    )


def apply_model_compat_patches() -> None:
    """Apply all known compatibility shims once. Safe to call repeatedly."""
    global _APPLIED
    if _APPLIED:
        return
    _patch_all_special_tokens_extended()
    _APPLIED = True
