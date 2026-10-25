from ..config import load_config
from .base import Image, ModelResponse, VisionModel
from .chat import ChatCompletionsModel
from .claude import ClaudeModel
from .gemini import GeminiModel
from .gpt import GPTModel

PROVIDERS = {"anthropic": ClaudeModel, "openai": GPTModel, "google": GeminiModel, "zai": ChatCompletionsModel}


# The config key each provider uses for how deeply the model thinks.
THINKING_KEYS = {"google": "thinking_level", "anthropic": "effort", "openai": "reasoning_effort"}
THINKING_LEVELS = ["low", "medium", "high"]


def build_model(name: str, thinking: str | None = None) -> VisionModel:
    """Build a model by its alias in configs/models.yaml (claude, gpt, gemini-argon, gemini-er).

    `thinking` (low, medium, high) overrides the model's configured thinking depth for this call.
    """
    cfg = dict(load_config("models")["models"][name])
    if thinking:
        key = THINKING_KEYS.get(cfg["provider"])
        if key is None:
            raise ValueError(f"{name}: no thinking depth setting for provider {cfg['provider']}")
        cfg[key] = thinking
    return PROVIDERS[cfg["provider"]](name, cfg)


def thinking_setting(name: str) -> str:
    """The thinking depth configs/models.yaml gives a model, or "default" when it leaves it to the provider."""
    cfg = load_config("models")["models"][name]
    return str(cfg.get(THINKING_KEYS.get(cfg["provider"], ""), "default"))


def model_names(include_debug: bool = True) -> list[str]:
    models = load_config("models")["models"]
    return [n for n, cfg in models.items() if include_debug or not cfg.get("debug")]


__all__ = ["THINKING_LEVELS", "Image", "ModelResponse", "VisionModel", "build_model", "model_names", "thinking_setting"]
