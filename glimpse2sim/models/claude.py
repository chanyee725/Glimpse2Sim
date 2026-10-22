import base64

import anthropic

from .base import CallResult, Image, VisionModel


class ClaudeModel(VisionModel):
    # Fable 5.1 always thinks; adaptive is accepted and effort sets the depth.
    # No refusal fallback on purpose: a fallback would silently swap the model under test.
    # A refusal is recorded through stop_reason instead.

    def __init__(self, name: str, cfg: dict):
        super().__init__(name, cfg)
        self.client = anthropic.Anthropic()

    def _call(self, prompt, images: list[Image], system, on_thought):
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
            # Thinking text is omitted by default; "summarized" returns a readable summary.
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": self.cfg.get("effort", "high")},
            messages=[{"role": "user", "content": content}],
            **kwargs,
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta" and event.delta.type == "thinking_delta":
                    on_thought(event.delta.thinking)
            msg = stream.get_final_message()

        return CallResult(
            text="".join(b.text for b in msg.content if b.type == "text"),
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
            stop_reason=msg.stop_reason,
            thinking="".join(b.thinking for b in msg.content if b.type == "thinking"),
        )
