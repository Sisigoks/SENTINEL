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
    """In-process vLLM engine. Loads local quantized weights and serves batched."""

    def __init__(
        self,
        model_name: str,
        *,
        quantization: str | None = "awq",
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.90,
        dtype: str = "auto",
        seed: int = 0,
    ) -> None:
        from vllm import LLM  # imported lazily so non-GPU tooling can import the module

        self.model_name = model_name
        self._tokenizer_template = True
        log.info("loading vllm", model=model_name, quantization=quantization)
        self._llm = LLM(
            model=model_name,
            quantization=quantization,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            seed=seed,
            enforce_eager=False,
        )

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
        outs = self._llm.generate(prompts, self._params(cfg))
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
            quantization=cfg.get("quantization", "awq"),
            max_model_len=cfg.get("max_model_len", 8192),
            gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.90),
            dtype=cfg.get("dtype", "auto"),
            seed=cfg.get("seed", 0),
        )
    if kind == "openai_compat":
        return OpenAICompatBackend(
            model_name=cfg["model_name"],
            base_url=cfg["base_url"],
            api_key=cfg.get("api_key", "EMPTY"),
        )
    raise ValueError(f"unknown backend kind: {kind!r}")
