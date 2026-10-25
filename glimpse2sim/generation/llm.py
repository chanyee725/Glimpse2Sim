"""Scene generation: captured images + prompts -> model -> scene spec JSON, with the model's thinking logged."""

import json
import logging
import re
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from ..config import CONFIG_DIR, ROOT, load_config
from ..models import Image, build_model
from .boxes import draw_boxes
from .translate import translate_thinking

PROMPT_DIR = CONFIG_DIR / "prompts"

log = logging.getLogger(__name__)


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def scene_images(scene_dir: Path, with_depth: bool = True, frames: list[str] | None = None) -> list[Path]:
    """Color images, then depth images, of the given frames in that order (default: all frames)."""
    if frames is None:
        frames = sorted(p.name.removesuffix("_color.png") for p in scene_dir.glob("*_color.png"))
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
    textures = "\n".join(f"  - `{name}`: {t['looks_like']}" for name, t in load_config("textures").items())
    return system, (PROMPT_DIR / "scene_generation.md").read_text().replace("{texture_list}", textures)


def parse_json(text: str) -> dict | None:
    """Parse the model output, tolerating a ```json fence around it."""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    try:
        return json.loads(m.group(1) if m else text)
    except json.JSONDecodeError:
        return None


def generate_scene(
    model_name: str, scene_dir: Path, camera: str, with_depth: bool = True, frames: list[str] | None = None
) -> Path:
    """Run one model on one captured scene and write everything to scene_dir/generations/<model>_<time>/.

    thinking.log  model's thinking, streamed as it arrives
    thinking_ko.md  Korean translation of thinking.log, for reading only
    scene.json    parsed scene spec (missing if the output was not valid JSON)
    *_boxes.png   sent color images with the model's 2D boxes and object ids drawn on
    response.txt  raw model output
    run.json      model ID, tokens, cost, latency, inputs
    """
    image_paths = scene_images(scene_dir, with_depth, frames)
    if not image_paths:
        raise FileNotFoundError(f"no *_color.png in {scene_dir}")
    images = [Image.from_path(p) for p in image_paths]
    system, prompt = load_prompts(camera)
    # Models never see file names, so name each image's view in the order the images are sent.
    prompt += "\n\nImages, in the order given:\n" + "\n".join(
        f"{i}. {p.name.removesuffix('.png')}" for i, p in enumerate(image_paths, 1)
    )
    model = build_model(model_name)

    run_dir = scene_dir / "generations" / f"{model_name}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True)
    log.info("%s (%s): %d images -> %s", model_name, model.model_id, len(images), _rel(run_dir))

    with open(run_dir / "thinking.log", "w") as thinking_log:
        def on_thought(text: str) -> None:
            sys.stdout.write(text)
            sys.stdout.flush()
            thinking_log.write(text)
            thinking_log.flush()

        print("----- thinking -----")
        resp = model.generate(prompt, images, system=system, on_thought=on_thought)
        print("\n----- end thinking -----")

    (run_dir / "response.txt").write_text(resp.text)
    scene = parse_json(resp.text)
    if scene is not None:
        (run_dir / "scene.json").write_text(json.dumps(scene, indent=2, ensure_ascii=False))
        draw_boxes(scene, image_paths, run_dir)
    else:
        log.warning("output is not valid JSON; see response.txt")

    record = asdict(resp) | {
        "camera": camera,
        "images": [_rel(p) for p in image_paths],
        "valid_json": scene is not None,
        "num_objects": len(scene.get("objects", [])) if scene else None,
    }
    record.pop("text")  # stored in response.txt
    record.pop("thinking")  # stored in thinking.log
    (run_dir / "run.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))

    log.info(
        "%s: in=%d out=%d $%.4f %.1fs stop=%s objects=%s",
        model_name, resp.input_tokens, resp.output_tokens, resp.cost_usd,
        resp.latency_s, resp.stop_reason, record["num_objects"],
    )
    translate_thinking(run_dir)
    return run_dir
