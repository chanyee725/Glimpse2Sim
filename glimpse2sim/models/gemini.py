from google import genai
from google.genai import types

from .base import Image, VisionModel


class GeminiModel(VisionModel):
    """Shared client for Gemini 4 Argon and Gemini Robotics-ER (only model_id differs)."""

    def __init__(self, name: str, cfg: dict):
        super().__init__(name, cfg)
        self.client = genai.Client()  # reads GEMINI_API_KEY

    def _call(self, prompt, images: list[Image], system):
        contents = [types.Part.from_bytes(data=img.data, mime_type=img.media_type) for img in images]
        contents.append(prompt)

        resp = self.client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=self.max_output_tokens,
            ),
        )
        usage = resp.usage_metadata
        n_in = usage.prompt_token_count or 0
        # Thinking tokens are billed as output but reported separately.
        n_out = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
        stop = resp.candidates[0].finish_reason.name if resp.candidates else None
        return resp.text or "", n_in, n_out, stop
