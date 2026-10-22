from google import genai
from google.genai import types

from .base import CallResult, Image, VisionModel


class GeminiModel(VisionModel):
    """Shared client for all Gemini models (only model_id differs)."""

    def __init__(self, name: str, cfg: dict):
        super().__init__(name, cfg)
        self.client = genai.Client()  # reads GEMINI_API_KEY

    def _call(self, prompt, images: list[Image], system, on_thought):
        contents = [types.Part.from_bytes(data=img.data, mime_type=img.media_type) for img in images]
        contents.append(prompt)

        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json",
            # include_thoughts returns thought summaries as parts flagged with `thought`.
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_level=self.cfg.get("thinking_level"),
            ),
        )

        text, thinking = [], []
        usage, finish = None, None
        for chunk in self.client.models.generate_content_stream(
            model=self.model_id, contents=contents, config=config
        ):
            usage = chunk.usage_metadata or usage
            if not chunk.candidates:
                continue
            cand = chunk.candidates[0]
            finish = cand.finish_reason or finish
            for part in (cand.content.parts if cand.content else None) or []:
                if not part.text:
                    continue
                if part.thought:
                    thinking.append(part.text)
                    on_thought(part.text)
                else:
                    text.append(part.text)

        n_in = (usage.prompt_token_count or 0) if usage else 0
        # Thinking tokens are billed as output but reported separately.
        n_out = ((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)) if usage else 0
        return CallResult(
            text="".join(text),
            input_tokens=n_in,
            output_tokens=n_out,
            stop_reason=finish.name if finish else None,
            thinking="".join(thinking),
        )
