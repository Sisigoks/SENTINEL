"""Model abstraction: vLLM and OpenAI-compatible backends, plus a frozen encoder.

Swapping Qwen3-14B <-> DeepSeek-R1-Distill-14B <-> Mistral-Small <-> Qwen3-32B is
a config change only; no call site changes (paper §model abstraction requirement).
"""

from .backend import GenerationConfig, ModelBackend, OpenAICompatBackend, VLLMBackend, build_backend
from .encoder import FrozenEncoder

__all__ = [
    "ModelBackend",
    "VLLMBackend",
    "OpenAICompatBackend",
    "GenerationConfig",
    "build_backend",
    "FrozenEncoder",
]
