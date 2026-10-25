"""Translate a run's thinking.log into Korean for reading. The original log stays the record."""

import logging
from pathlib import Path

from google import genai

from ..config import load_config

log = logging.getLogger(__name__)

PROMPT = """Translate the following reasoning log of a vision model into natural Korean.
Keep the markdown structure (bold headings, paragraphs). Keep object ids, JSON field names, numbers, and units as they are.
Output only the translation.

"""


def translate_thinking(run_dir: Path) -> Path | None:
    """Write run_dir/thinking_ko.md from thinking.log with the free debug model. Return None if skipped or failed."""
    src = run_dir / "thinking.log"
    text = src.read_text() if src.exists() else ""
    if not text.strip():
        return None
    model_id = load_config("models")["models"]["gemini-flash"]["model_id"]
    try:
        client = genai.Client()  # keep a reference: a temporary client is closed before the request is sent
        resp = client.models.generate_content(model=model_id, contents=PROMPT + text)
    except Exception as e:  # translation is a reading aid; never fail the run over it
        log.warning("thinking translation failed: %s", e)
        return None
    out = run_dir / "thinking_ko.md"
    out.write_text(resp.text or "")
    log.info("thinking translated -> %s", out.name)
    return out
