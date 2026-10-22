"""Send captured images + a prompt to one model and save the response with token usage and cost.

uv run scripts/query_model.py --model claude --scene scene01 --prompt configs/prompts/stage1_scene_understanding.md
uv run scripts/query_model.py --model all --scene scene01 --prompt configs/prompts/stage2_scene_spec.md --stage1 <stage1 output>
"""

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from glimpse2sim.config import ROOT, load_config
from glimpse2sim.models import Image, build_model, model_names


def collect_images(scene_dir: Path, with_depth: bool) -> list[Path]:
    paths = sorted(scene_dir.rglob("*_color.png"))
    if with_depth:
        paths += sorted(scene_dir.rglob("*_depth_vis.png"))
    return paths


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help=f"one of {model_names()} or 'all'")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--system", type=Path, default=ROOT / "configs" / "prompts" / "system.md")
    parser.add_argument("--stage1", type=Path, help="Stage 1 output JSON, filled into {stage1_json}")
    parser.add_argument("--no-depth", action="store_true", help="send RGB only")
    args = parser.parse_args()

    scene_dir = ROOT / "data" / "raw" / args.scene
    image_paths = collect_images(scene_dir, with_depth=not args.no_depth)
    images = [Image.from_path(p) for p in image_paths]

    # Plain replace instead of str.format: the prompts contain literal JSON braces.
    near, far = load_config("cameras")["d435"]["depth_vis_range_m"]
    system = (
        args.system.read_text()
        .replace("{depth_near_m}", str(near))
        .replace("{depth_far_m}", str(far))
    )
    prompt = args.prompt.read_text()
    if args.stage1:
        prompt = prompt.replace("{stage1_json}", args.stage1.read_text())
    names = model_names() if args.model == "all" else [args.model]

    run_dir = ROOT / "experiments" / datetime.now().strftime("%Y%m%d_%H%M%S") / args.scene
    run_dir.mkdir(parents=True, exist_ok=True)

    total = 0.0
    for name in names:
        resp = build_model(name).generate(prompt, images, system=system)
        record = asdict(resp) | {
            "images": [str(p.relative_to(ROOT)) for p in image_paths],
            "prompt_file": str(args.prompt),
            "system_file": str(args.system),
        }
        (run_dir / f"{name}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
        total += resp.cost_usd
        print(
            f"{name:<13} in={resp.input_tokens:>7} out={resp.output_tokens:>7} "
            f"${resp.cost_usd:.4f} {resp.latency_s:.1f}s stop={resp.stop_reason}"
        )
    print(f"total ${total:.4f} -> {run_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
