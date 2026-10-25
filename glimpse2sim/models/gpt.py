import base64

import openai

from .base import CallResult, Image, VisionModel


class GPTModel(VisionModel):
    # Reasoning summaries are not wired up yet, so `thinking` stays empty.

    def __init__(self, name: str, cfg: dict):
        super().__init__(name, cfg)
        self.client = openai.OpenAI()

    def _call(self, prompt, images: list[Image], system, on_thought):
        content = [
            {
                "type": "input_image",
                "image_url": f"data:{img.media_type};base64,{base64.b64encode(img.data).decode()}",
            }
            for img in images
        ]
        content.append({"type": "input_text", "text": prompt})

        resp = self.client.responses.create(
            model=self.model_id,
            instructions=system,
            input=[{"role": "user", "content": content}],
            max_output_tokens=self.max_output_tokens,
            reasoning={"effort": self.cfg.get("reasoning_effort", "high")},
            store=False,  # do not keep captured images on the provider side
        )
        stop = resp.incomplete_details.reason if resp.incomplete_details else resp.status
        details = getattr(resp.usage, "output_tokens_details", None)
        return CallResult(resp.output_text, resp.usage.input_tokens, resp.usage.output_tokens, stop,
                          thinking_tokens=getattr(details, "reasoning_tokens", None))
