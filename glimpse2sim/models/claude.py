import base64

import anthropic

from .base import Image, VisionModel


class ClaudeModel(VisionModel):
    # Fable 5.1 always thinks; adaptive is accepted and effort sets the depth.
    # No refusal fallback on purpose: a fallback would silently swap the model under test.
    # A refusal is recorded through stop_reason instead.

    def __init__(self, name: str, cfg: dict):
        super().__init__(name, cfg)
        self.client = anthropic.Anthropic()

    def _call(self, prompt, images: list[Image], system):
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img.media_type,
                    "data": base64.standard_b64encode(img.data).decode(),
                },
            }
            for img in images
        ]
        content.append({"type": "text", "text": prompt})

        kwargs = {}
        if system:
            kwargs["system"] = system
        # Streaming avoids HTTP timeouts on long scene-spec outputs.
        with self.client.messages.stream(
            model=self.model_id,
            max_tokens=self.max_output_tokens,
            thinking={"type": "adaptive"},
            output_config={"effort": self.cfg.get("effort", "high")},
            messages=[{"role": "user", "content": content}],
            **kwargs,
        ) as stream:
            msg = stream.get_final_message()

        text = "".join(b.text for b in msg.content if b.type == "text")
        return text, msg.usage.input_tokens, msg.usage.output_tokens, msg.stop_reason
