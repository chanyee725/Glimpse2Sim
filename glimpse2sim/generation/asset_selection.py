"""Asset selection: the scene model itself picks each object's library model from retrieved candidates.

The embedding search (glimpse2sim/assets.py) only lists the TOP_K entries nearest to each `asset_query`; a second
call to the same model that wrote the scene decides which candidate, if any, is that object. So the asset is the
model's answer like the rest of the spec, and two models with different queries can end up with different assets.

Writes to the run folder:
    asset_prompt.md     the exact prompt sent (text only; no images)
    asset_thinking.log  the model's thinking during selection
    asset_response.txt  raw model output
"""

import json
import logging
from pathlib import Path

import numpy as np

from ..assets import AssetLibrary
from ..config import CONFIG_DIR
from ..models import VisionModel

log = logging.getLogger(__name__)

PROMPT = CONFIG_DIR / "prompts" / "asset_selection.md"


def _size(o: dict) -> np.ndarray | None:
    s = [(o.get("size_mm") or {}).get(k) for k in ("length", "width", "height")]
    return np.array(s, float) if all(isinstance(v, (int, float)) and v > 0 for v in s) else None


def _listing(objects: list[dict], candidates: list[list[dict]]) -> str:
    blocks = []
    for o, cands in zip(objects, candidates):
        lines = [f"- id `{o['id']}`: {o.get('identified_as') or ''}; asset_query \"{o['asset_query']}\""]
        lines += [f"  {i}. [{c['source']}] {c['text']}" for i, c in enumerate(cands, 1)]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def select_assets(scene: dict, model: VisionModel, system: str, out: Path, parse_json) -> dict | None:
    """Fill objects[].asset_match in place from the model's own choice among the retrieved candidates.

    Every object with an `asset_query` gets an asset_match: `candidates` shown, `choice` (1-based or None),
    `reason`, and when a model was chosen and found, its `source`, `text`, `path`, ... (`path` None otherwise).
    Return the call's usage (tokens, cost, latency), or None if no object had a query.
    """
    objects = [o for o in scene.get("objects", []) if isinstance(o.get("asset_query"), str) and o["asset_query"].strip()]
    if not objects:
        return None
    library = AssetLibrary()
    query_vecs, candidates = library.candidates([o["asset_query"] for o in objects])

    prompt = PROMPT.read_text().replace("{objects}", _listing(objects, candidates))
    (out / "asset_prompt.md").write_text(f"# System\n\n{system}\n\n# User\n\n{prompt}\n")
    with open(out / "asset_thinking.log", "w") as thinking_log:
        resp = model.generate(prompt, [], system=system, on_thought=thinking_log.write)
    (out / "asset_response.txt").write_text(resp.text)

    parsed = parse_json(resp.text) or {}
    choices = {c.get("id"): c for c in parsed.get("choices", []) if isinstance(c, dict)}
    if not choices:
        log.warning("asset selection output has no choices; see asset_response.txt")

    for o, qv, cands in zip(objects, query_vecs, candidates):
        c = choices.get(o["id"], {})
        n = c.get("candidate")
        n = n if isinstance(n, int) and 1 <= n <= len(cands) else None
        match = {"query": o["asset_query"], "choice": n, "reason": c.get("reason"), "source": None, "text": None,
                 "path": None, "candidates": [{k: x[k] for k in ("source", "text", "score")} for x in cands]}
        if n is not None:
            found = library.fetch(cands[n - 1], o["asset_query"], qv, _size(o))
            if found:
                match |= found
            else:
                log.warning("%s: chosen category %s has no usable Objaverse model", o["id"], cands[n - 1]["text"])
        o["asset_match"] = match

    found = sum(o["asset_match"]["path"] is not None for o in objects)
    log.info("asset selection: %d of %d objects got a model", found, len(objects))
    return {"input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens,
            "thinking_tokens": resp.thinking_tokens, "cost_usd": resp.cost_usd, "latency_s": resp.latency_s,
            "stop_reason": resp.stop_reason, "assets_found": found}
