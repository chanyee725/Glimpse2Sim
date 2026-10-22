import mimetypes
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Image:
    data: bytes
    media_type: str  # "image/png" or "image/jpeg"

    @classmethod
    def from_path(cls, path: str | Path) -> "Image":
        media_type = mimetypes.guess_type(str(path))[0] or "image/png"
        return cls(Path(path).read_bytes(), media_type)


@dataclass
class ModelResponse:
    model: str  # alias from configs/models.yaml
    model_id: str  # exact provider model ID, recorded for reproducibility
    text: str
    input_tokens: int
    output_tokens: int  # includes thinking/reasoning tokens
    cost_usd: float
    latency_s: float
    stop_reason: str | None


class VisionModel(ABC):
    """Common interface: images + prompt in, text + usage + cost out."""

    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.model_id = cfg["model_id"]
        self.max_output_tokens = cfg.get("max_output_tokens", 64000)
        self.price_in = cfg["pricing"]["input"]
        self.price_out = cfg["pricing"]["output"]
        self.cfg = cfg

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.price_in + output_tokens * self.price_out) / 1e6

    def generate(self, prompt: str, images: list[Image], system: str | None = None) -> ModelResponse:
        start = time.perf_counter()
        text, n_in, n_out, stop = self._call(prompt, images, system)
        return ModelResponse(
            model=self.name,
            model_id=self.model_id,
            text=text,
            input_tokens=n_in,
            output_tokens=n_out,
            cost_usd=self.cost(n_in, n_out),
            latency_s=time.perf_counter() - start,
            stop_reason=stop,
        )

    @abstractmethod
    def _call(
        self, prompt: str, images: list[Image], system: str | None
    ) -> tuple[str, int, int, str | None]:
        """Return (text, input_tokens, output_tokens, stop_reason)."""
