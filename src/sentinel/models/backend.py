"""Model backends — real inference only (vLLM on the A100, or an OpenAI-compatible endpoint).

The :class:`ModelBackend` protocol is the single seam the rest of the system
depends on. Two concrete backends are provided:

* :class:`VLLMBackend` — in-process batched serving of local quantized weights
  (the paper's recommended A100 + vLLM 14B path; Table 7).
* :class:`OpenAICompatBackend` — talks to an OpenAI-compatible HTTP server
  (e.g. ``vllm serve`` or SGLang) for multi-GPU / remote serving.

No mock backend exists by design: SENTINEL runs against real models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..core.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class GenerationConfig:
    max_tokens: int = 1024
    temperature: float = 0.0          # deterministic by default for reproducibility
    top_p: float = 1.0
    seed: int | None = None
    stop: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int


class ModelBackend(ABC):
    """A served LLM. Implementations must be deterministic at temperature 0."""

    model_name: str

    @abstractmethod
    def generate(self, prompt: str, cfg: GenerationConfig) -> Generation: ...

    @abstractmethod
    def generate_batch(self, prompts: list[str], cfg: GenerationConfig) -> list[Generation]:
        """Batched generation — the throughput path that keeps the GPU saturated."""

    def chat(self, system: str, user: str, cfg: GenerationConfig) -> Generation:
        """Default chat formatting; backends may override with a tokenizer template."""
        return self.generate(f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n", cfg)


class VLLMBackend(ModelBackend):
    """In-process vLLM engine. Loads local (optionally quantized) weights and serves batched.

    Robustness choices (see docs/02-troubleshooting.md):
      * ``quantization=None`` (default) lets vLLM auto-detect AWQ/GPTQ from the model's
        ``config.json`` — passing ``quantization='awq'`` to a non-AWQ repo is a common crash.
      * ``tokenizer_mode`` / ``trust_remote_code`` are config-driven for model-specific needs
        (e.g. Mistral's native tokenizer).
      * Known vLLM<->transformers tokenizer skews are patched before engine init.
      * Engine-init failures raise an actionable error instead of a raw stack trace.
    """

    def __init__(
        self,
        model_name: str,
        *,
        quantization: str | None = None,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.90,
        dtype: str = "auto",
        seed: int = 0,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        enforce_eager: bool = False,
        show_progress: bool = False,
    ) -> None:
        from .compat import apply_model_compat_patches

        apply_model_compat_patches()  # patch known tokenizer/version skews first
        from vllm import LLM  # imported lazily so non-GPU tooling can import the module

        self.model_name = model_name
        self._show_progress = show_progress
        # Empty string / "auto"-ish values -> None so vLLM auto-detects from config.json.
        quant = quantization if quantization not in (None, "", "none", "auto") else None
        log.info("loading vllm", model=model_name, quantization=quant or "auto-detect",
                 max_model_len=max_model_len)
        try:
            self._llm = LLM(
                model=model_name,
                quantization=quant,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                dtype=dtype,
                seed=seed,
                tokenizer_mode=tokenizer_mode,
                trust_remote_code=trust_remote_code,
                enforce_eager=enforce_eager,
            )
        except Exception as exc:  # surface an actionable message
            raise RuntimeError(self._diagnose(exc, model_name)) from exc

    @staticmethod
    def _diagnose(exc: Exception, model_name: str) -> str:
        msg = str(exc)
        hints: list[str] = []
        low = msg.lower()
        if "all_special_tokens_extended" in msg:
            hints.append("transformers<->vLLM tokenizer skew: pin transformers to a vLLM-compatible "
                         "version (see docs/02-troubleshooting.md). A runtime shim is applied "
                         "automatically; if you still see this, the pin is required.")
        if "out of memory" in low or "kv cache" in low or "no available memory" in low:
            hints.append("GPU OOM: lower max_model_len (e.g. 4096) and/or gpu_memory_utilization "
                         "(e.g. 0.80), or use a smaller / AWQ-quantized model.")
        if "quantization" in low or "awq" in low or "gptq" in low:
            hints.append("Quantization mismatch: leave quantization unset (null) so vLLM auto-detects "
                         "from the model config, or point at an AWQ/GPTQ repo.")
        if "mistral" in model_name.lower():
            hints.append("Mistral models may need tokenizer_mode=mistral in the model config.")
        if "compute capability" in low or "not supported" in low:
            hints.append("This GPU may be pre-Ampere (CC<8.0); vLLM falls back to the V0 engine. "
                         "AWQ needs CC>=7.5. Prefer an A100/L4/H100 for the full grid.")
        joined = "\n  - ".join(hints) if hints else "(no specific hint matched)"
        return f"vLLM failed to load {model_name!r}: {msg}\nLikely fixes:\n  - {joined}"

    def _params(self, cfg: GenerationConfig):
        from vllm import SamplingParams

        return SamplingParams(
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            seed=cfg.seed,
            stop=cfg.stop or None,
        )

    def generate(self, prompt: str, cfg: GenerationConfig) -> Generation:
        return self.generate_batch([prompt], cfg)[0]

    def generate_batch(self, prompts: list[str], cfg: GenerationConfig) -> list[Generation]:
        # vLLM's own bar is noisy for the per-probe (size-1) calls SENTINEL makes; gate it.
        use_tqdm = self._show_progress and len(prompts) > 1
        outs = self._llm.generate(prompts, self._params(cfg), use_tqdm=use_tqdm)
        results: list[Generation] = []
        for o in outs:
            comp = o.outputs[0]
            results.append(
                Generation(
                    text=comp.text,
                    prompt_tokens=len(o.prompt_token_ids),
                    completion_tokens=len(comp.token_ids),
                )
            )
        return results

    def chat(self, system: str, user: str, cfg: GenerationConfig) -> Generation:
        tok = self._llm.get_tokenizer()
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.generate(prompt, cfg)


class OpenAICompatBackend(ModelBackend):
    """Talks to an OpenAI-compatible server (``vllm serve`` / SGLang / TGI)."""

    def __init__(self, model_name: str, base_url: str, api_key: str = "EMPTY") -> None:
        from openai import OpenAI

        self.model_name = model_name
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def generate(self, prompt: str, cfg: GenerationConfig) -> Generation:
        return self.chat("", prompt, cfg)

    def generate_batch(self, prompts: list[str], cfg: GenerationConfig) -> list[Generation]:
        return [self.generate(p, cfg) for p in prompts]

    def chat(self, system: str, user: str, cfg: GenerationConfig) -> Generation:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=msgs,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            seed=cfg.seed,
            stop=cfg.stop or None,
        )
        u = resp.usage
        return Generation(
            text=resp.choices[0].message.content or "",
            prompt_tokens=u.prompt_tokens if u else 0,
            completion_tokens=u.completion_tokens if u else 0,
        )


def build_backend(cfg: dict) -> ModelBackend:
    """Factory used by the config system. ``cfg['kind']`` selects the backend."""
    kind = cfg.get("kind", "vllm")
    if kind == "vllm":
        return VLLMBackend(
            model_name=cfg["model_name"],
            quantization=cfg.get("quantization", None),
            max_model_len=cfg.get("max_model_len", 8192),
            gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.90),
            dtype=cfg.get("dtype", "auto"),
            seed=cfg.get("seed", 0),
            tokenizer_mode=cfg.get("tokenizer_mode", "auto"),
            trust_remote_code=cfg.get("trust_remote_code", False),
            enforce_eager=cfg.get("enforce_eager", False),
            show_progress=cfg.get("show_progress", False),
        )
    if kind == "openai_compat":
        return OpenAICompatBackend(
            model_name=cfg["model_name"],
            base_url=cfg["base_url"],
            api_key=cfg.get("api_key", "EMPTY"),
        )
    raise ValueError(f"unknown backend kind: {kind!r}")
