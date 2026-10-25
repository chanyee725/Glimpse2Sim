import base64
import os

import openai

from .base import CallResult, Image, VisionModel


class ChatCompletionsModel(VisionModel):
    """Any provider with an OpenAI-style /chat/completions endpoint (Z.ai GLM, ...).

    configs/models.yaml sets `base_url` and `api_key_env`; `thinking: true` sends Z.ai's thinking switch.
    """

    def __init__(self, name: str, cfg: dict):
        super().__init__(name, cfg)
        self.client = openai.OpenAI(base_url=cfg["base_url"], api_key=os.environ[cfg["api_key_env"]])

    def _call(self, prompt, images: list[Image], system, on_thought):
        content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{img.media_type};base64,{base64.b64encode(img.data).decode()}"},
            }
            for img in images
        ]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": content})

        extra = {"thinking": {"type": "enabled" if self.cfg.get("thinking") else "disabled"}}
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=self.max_output_tokens,
            extra_body=extra,
        )
        msg = resp.choices[0].message
        thinking = getattr(msg, "reasoning_content", None) or ""
        if thinking:
            on_thought(thinking)
        return CallResult(
            msg.content or "", resp.usage.prompt_tokens, resp.usage.completion_tokens,
            resp.choices[0].finish_reason, thinking,
            getattr(getattr(resp.usage, "completion_tokens_details", None), "reasoning_tokens", None),
        )
