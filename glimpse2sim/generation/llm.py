"""Scene generation: captured images + prompts -> model -> scene spec JSON, with the model's thinking logged.

Every run is saved under results/<scene>/<experiment>/<model folder>/<condition>/<time>/ and listed in
results/<scene>/<experiment>/index.csv; the experiment round is `current` in configs/experiment.yaml.
"""

import csv
import hashlib
import json
import logging
import re
import shutil
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path


from ..config import CONFIG_DIR, ROOT, load_config
from ..models import Image, build_model, thinking_setting
from .asset_selection import select_assets
from .boxes import draw_boxes
from .scene import current_experiment, experiment_dir, run_dir
from .translate import translate_thinking

INDEX_FIELDS = [
    "time", "camera", "scene", "condition", "model", "model_id", "thinking", "n_images", "input_tokens",
    "output_tokens", "thinking_tokens", "cost_usd", "latency_s", "stop_reason", "valid_json", "num_objects",
    "prompt_sha", "asset_input_tokens", "asset_output_tokens", "asset_cost_usd", "assets_found", "run_dir",
]

PROMPT_DIR = CONFIG_DIR / "prompts"

log = logging.getLogger(__name__)


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def scene_images(scene_dir: Path, with_depth: bool = True, frames: list[str] | str | None = None) -> list[Path]:
    """Color images, then depth images, of the given frames in that order.

    `None` or "all" means every frame, front first.
    """
    if frames is None or frames == "all":
        names = sorted(p.name.removesuffix("_color.png") for p in scene_dir.glob("*_color.png"))
        frames = sorted(names, key=lambda n: n != "front")  # stable: front first, the rest by name
    paths = [scene_dir / f"{f}_color.png" for f in frames]
    if with_depth:
        paths += [scene_dir / f"{f}_depth_vis.png" for f in frames]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing in {scene_dir}: {', '.join(missing)}")
    return paths


def load_prompts(camera: str) -> tuple[str, str]:
    """Return (system, user) prompts with camera-specific values filled in."""
    near, far = load_config("cameras")[camera]["depth_vis_range_m"]
    # Plain replace instead of str.format: the prompts contain literal JSON braces.
    system = (
        (PROMPT_DIR / "system.md").read_text()
        .replace("{depth_near_m}", str(near))
        .replace("{depth_far_m}", str(far))
    )
    listing = lambda catalog: "\n".join(f"  - `{name}`: {t['looks_like']}" for name, t in load_config(catalog).items())  # noqa: E731
    return system, (
        (PROMPT_DIR / "scene_generation.md").read_text()
        .replace("{texture_list}", listing("textures"))
    )


def condition_inputs(scene_dir: Path, condition: str) -> tuple[list[Path], str]:
    """The images to send and the text to add to the prompt for one input condition."""
    cond = load_config("conditions")[condition]
    images = scene_images(scene_dir, cond.get("depth", False), cond["frames"])
    # Models never see file names, so name each image's view in the order the images are sent.
    return images, "\n\nImages, in the order given:\n" + "\n".join(
        f"{i}. {p.name.removesuffix('.png')}" for i, p in enumerate(images, 1)
    )


def _append_index(scene: str, row: dict) -> None:
    index = experiment_dir(scene) / "index.csv"
    rows = []
    if index.exists():
        with open(index, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != INDEX_FIELDS:  # columns were added since: rewrite the old rows under the new header
                rows = list(reader)
    if rows or not index.exists():
        with open(index, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
            w.writeheader()
            w.writerows({k: r.get(k) for k in INDEX_FIELDS} for r in rows)
    with open(index, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=INDEX_FIELDS).writerow({k: row.get(k) for k in INDEX_FIELDS})


def parse_json(text: str) -> dict | None:
    """Parse the model output, tolerating a ```json fence around it."""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    try:
        return json.loads(m.group(1) if m else text)
    except json.JSONDecodeError:
        return None


def generate_scene(model_name: str, scene_dir: Path, camera: str, condition: str, thinking: str | None = None) -> Path:
    """Run one model on one captured scene under one input condition (configs/conditions.yaml).

    `thinking` (low, medium, high) overrides the model's thinking depth; such runs go to a separate model folder
    with the level appended (e.g. gemini/argon-high), so they never mix with runs at the configured depth.

    Writes to results/<scene>/<experiment>/<model folder>/<condition>/<time>/ (the folder is `results` in
    configs/models.yaml, e.g. gemini/flash) and appends a row to results/<scene>/<experiment>/index.csv:

    thinking.log  model's thinking, streamed as it arrives
    thinking_ko.md  Korean translation of thinking.log, for reading only
    scene.json    parsed scene spec (missing if the output was not valid JSON)
    *_boxes.png   sent color images with the model's 2D boxes and object ids drawn on
    asset_*       the same model choosing each object's library model (see asset_selection.py)
    response.txt  raw model output
    run.json      model ID, tokens, cost, latency, inputs, condition, prompt hash
    prompt.md     the exact system and user prompts sent
    """
    image_paths, extra = condition_inputs(scene_dir, condition)
    images = [Image.from_path(p) for p in image_paths]
    system, prompt = load_prompts(camera)
    prompt += extra
    prompt_sha = hashlib.sha256((system + "\0" + prompt).encode()).hexdigest()[:12]
    model = build_model(model_name, thinking)

    started = datetime.now()
    folder = load_config("models")["models"][model_name]["results"] + (f"-{thinking}" if thinking else "")
    out = run_dir(scene_dir.name, folder, condition, started)
    out.mkdir(parents=True)
    (out / "prompt.md").write_text(f"# System\n\n{system}\n\n# User\n\n{prompt}\n")
    log.info("%s (%s): %d images -> %s", model_name, model.model_id, len(images), _rel(out))

    with open(out / "thinking.log", "w") as thinking_log:
        def on_thought(text: str) -> None:
            sys.stdout.write(text)
            sys.stdout.flush()
            thinking_log.write(text)
            thinking_log.flush()

        print("----- thinking -----")
        try:
            resp = model.generate(prompt, images, system=system, on_thought=on_thought)
        except Exception:
            shutil.rmtree(out)  # a failed call (quota, 503, network) leaves no half-filled run folder behind
            raise
        print("\n----- end thinking -----")

    (out / "response.txt").write_text(resp.text)
    scene = parse_json(resp.text)
    assets = None
    if scene is not None:
        (out / "scene.json").write_text(json.dumps(scene, indent=2, ensure_ascii=False))
        draw_boxes(scene, image_paths, out)
        try:  # the same model picks a library model for each object; a failure only leaves the shapes in place
            assets = select_assets(scene, model, system, out, parse_json)
        except Exception as e:
            log.warning("asset selection failed: %s", e)
        (out / "scene.json").write_text(json.dumps(scene, indent=2, ensure_ascii=False))
    else:
        log.warning("output is not valid JSON; see response.txt")

    record = asdict(resp)
    record.pop("text")  # stored in response.txt
    record.pop("thinking")  # the thinking text, stored in thinking.log; "thinking" below is the depth setting
    record |= {
        "experiment": current_experiment(),
        "camera": camera,
        "scene": scene_dir.name,
        "condition": condition,
        "thinking": thinking or thinking_setting(model_name),
        "prompt_sha": prompt_sha,
        "images": [_rel(p) for p in image_paths],
        "valid_json": scene is not None,
        "num_objects": len(scene.get("objects", [])) if scene else None,
        "asset_selection": assets,  # the second call; its cost is not in cost_usd above
    }
    (out / "run.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
    _append_index(scene_dir.name, record | {
        "time": f"{started:%Y-%m-%d %H:%M:%S}", "n_images": len(image_paths),
        "cost_usd": round(resp.cost_usd, 5), "latency_s": round(resp.latency_s, 1), "run_dir": _rel(out),
    } | ({"asset_input_tokens": assets["input_tokens"], "asset_output_tokens": assets["output_tokens"],
          "asset_cost_usd": round(assets["cost_usd"], 5), "assets_found": assets["assets_found"]} if assets else {}))

    log.info(
        "%s: in=%d out=%d (thinking %s) $%.4f %.1fs stop=%s objects=%s",
        model_name, resp.input_tokens, resp.output_tokens, resp.thinking_tokens, resp.cost_usd,
        resp.latency_s, resp.stop_reason, record["num_objects"],
    )
    translate_thinking(out)
    return out
