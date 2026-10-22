from ..config import load_config
from .base import Image, ModelResponse, VisionModel
from .claude import ClaudeModel
from .gemini import GeminiModel
from .gpt import GPTModel

PROVIDERS = {"anthropic": ClaudeModel, "openai": GPTModel, "google": GeminiModel}


def build_model(name: str) -> VisionModel:
    """Build a model by its alias in configs/models.yaml (claude, gpt, gemini-argon, gemini-er)."""
    cfg = load_config("models")["models"][name]
    return PROVIDERS[cfg["provider"]](name, cfg)


def model_names() -> list[str]:
    return list(load_config("models")["models"])


__all__ = ["Image", "ModelResponse", "VisionModel", "build_model", "model_names"]
