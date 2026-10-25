import mimetypes
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# Receives the model's thinking text as it streams in.
ThoughtCallback = Callable[[str], None]


@dataclass
class Image:
    data: bytes
    media_type: str  # "image/png" or "image/jpeg"

    @classmethod
    def from_path(cls, path: str | Path) -> "Image":
        media_type = mimetypes.guess_type(str(path))[0] or "image/png"
        return cls(Path(path).read_bytes(), media_type)


@dataclass
class CallResult:
    text: str
    input_tokens: int
    output_tokens: int  # includes thinking/reasoning tokens
    stop_reason: str | None
    thinking: str = ""  # thinking summary, when the provider returns one
    thinking_tokens: int | None = None  # part of output_tokens spent thinking; None if the provider does not say


@dataclass
class ModelResponse:
    model: str  # alias from configs/models.yaml
    model_id: str  # exact provider model ID, recorded for reproducibility
    text: str
    thinking: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    stop_reason: str | None
    thinking_tokens: int | None = None


class VisionModel(ABC):
    """Common interface: images + prompt in, text + thinking + usage + cost out."""

    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.model_id = cfg["model_id"]
        self.max_output_tokens = cfg.get("max_output_tokens", 64000)
        self.price_in = cfg["pricing"]["input"]
        self.price_out = cfg["pricing"]["output"]
        self.cfg = cfg

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.price_in + output_tokens * self.price_out) / 1e6

    def generate(
        self,
        prompt: str,
        images: list[Image],
        system: str | None = None,
        on_thought: ThoughtCallback | None = None,
    ) -> ModelResponse:
        start = time.perf_counter()
        r = self._call(prompt, images, system, on_thought or (lambda _: None))
        return ModelResponse(
            model=self.name,
            model_id=self.model_id,
            text=r.text,
            thinking=r.thinking,
            input_tokens=r.input_tokens,
            output_tokens=r.output_tokens,
            cost_usd=self.cost(r.input_tokens, r.output_tokens),
            latency_s=time.perf_counter() - start,
            stop_reason=r.stop_reason,
            thinking_tokens=r.thinking_tokens,
        )

    @abstractmethod
    def _call(
        self, prompt: str, images: list[Image], system: str | None, on_thought: ThoughtCallback
    ) -> CallResult: ...
