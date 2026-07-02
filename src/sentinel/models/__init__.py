"""Model abstraction: vLLM and OpenAI-compatible backends, plus a frozen encoder.

Swapping any model in the six-model roster (Llama-3.1-8B <-> Phi-4 <->
DeepSeek-R1-Distill-14B <-> Mistral-Small-24B <-> Qwen3-32B <-> Llama-3.3-70B) is
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
