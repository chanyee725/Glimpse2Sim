"""Glimpse2Sim entry point.

Capture: live preview; each left click saves one RGB-D frame to outputs/<camera>/<name>.
The connected camera is detected automatically.
    uv run main.py capture scene01
    Left click: save the current frame (000, 001, ...). q / Esc / closing the window: quit.

Generate: send a captured scene to a model and build the scene spec, logging the model's thinking.
The camera is looked up from the scene folder.
    uv run main.py generate scene01
    uv run main.py generate scene01 --model all

View: build the latest generated scene spec in Isaac Sim (extra args go to the Isaac Lab launcher).
    uv run main.py view scene01
    uv run main.py view scene01 --spec outputs/d435i/scene01/generations/<run>/scene.json
    uv run main.py view scene01 --headless --max-steps 200
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

from glimpse2sim.capture import D435Camera, detect_camera, find_scene, output_dir
from glimpse2sim.config import ROOT, load_config
from glimpse2sim.generation import build_scene, generate_scene, latest_scene_spec
from glimpse2sim.models import model_names

PREVIEW_WIDTH = 640  # per panel; color and depth are shown side by side


def capture(name: str, camera: str | None) -> None:
    cameras = load_config("cameras")
    camera = camera or detect_camera(cameras)
    cfg = cameras[camera]
    out_dir = output_dir(name, camera)
    vis_range = tuple(cfg["depth_vis_range_m"])
    window = f"{camera} / {name}"

    # Continue numbering if the folder already has frames.
    count = len(list(out_dir.glob("*_color.png"))) if out_dir.exists() else 0
    clicked = False

    def on_mouse(event, *_):
        nonlocal clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked = True

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    with D435Camera(cfg) as cam:
        while True:
            frame = cam.capture()
            if clicked:
                frame_name = f"{count:03d}"
                frame.save(out_dir, frame_name, vis_range)
                count += 1
                clicked = False
                print(f"saved {frame_name}")

            depth = frame.depth_colormap(vis_range)
            h = round(frame.color.shape[0] * PREVIEW_WIDTH / frame.color.shape[1])
            panel = np.hstack([cv2.resize(img, (PREVIEW_WIDTH, h)) for img in (frame.color, depth)])
            cv2.putText(panel, f"saved: {count}  (left click: save, q: quit)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(window, panel)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break

    cv2.destroyAllWindows()
    print(f"{count} frame(s) in {out_dir.relative_to(ROOT)}")


def generate(name: str, camera: str | None, model: str, with_depth: bool) -> None:
    load_dotenv(ROOT / ".env")
    if camera:
        scene_dir = output_dir(name, camera)
    else:
        camera, scene_dir = find_scene(name)
    for m in model_names(include_debug=False) if model == "all" else [model]:
        generate_scene(m, scene_dir, camera, with_depth)


def view(name: str, camera: str | None, spec: Path | None, max_steps: int, launcher_argv: list[str]) -> None:
    # Isaac Lab is imported here so capture/generate never start Kit (or its EULA prompt).
    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab.physics import PhysicsCfg

    import isaaclab.sim as sim_utils

    if spec is None:
        scene_dir = output_dir(name, camera) if camera else find_scene(name)[1]
        spec = latest_scene_spec(scene_dir)

    launcher = argparse.ArgumentParser(prog="main.py view", conflict_handler="resolve")
    launcher.add_argument(
        "--physics", default="isaacsim_physx", choices=["isaacsim_physx", "physx", "newton_mjwarp"],
        help="physics backend (launch_simulation reads this to resolve PhysicsCfg)",
    )
    launcher.add_argument("--headless", action="store_true", help="run without a viewer window")
    add_launcher_args(launcher)
    launcher.set_defaults(visualizer=["kit"])
    launcher_args = launcher.parse_args(launcher_argv)
    if launcher_args.headless:
        launcher_args.visualizer = None  # Isaac Lab 3.0 has no --headless; no visualizer means headless

    with launch_simulation(cfg=PhysicsCfg(), launcher_args=launcher_args) as physics_cfg:
        sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(dt=0.01, device=launcher_args.device, physics=physics_cfg)
        )
        sim.set_camera_view(eye=[1.2, 1.0, 0.8], target=[0.25, 0.0, 0.0])
        spawned, total = build_scene(spec)
        logging.info("spec %s: %d/%d objects spawned", spec, spawned, total)
        sim.reset()

        steps = 0
        while sim.is_running() and (max_steps < 0 or steps < max_steps):
            sim.step()
            steps += 1


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cameras = list(load_config("cameras"))

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("capture", help="click to save RGB-D frames")
    p.add_argument("name", help="folder name under outputs/<camera> (any value, stored as a string)")
    p.add_argument("--camera", choices=cameras, help="default: the connected camera")

    p = sub.add_parser("generate", help="build a scene spec from captured frames")
    p.add_argument("name", help="captured scene folder under outputs/<camera>")
    p.add_argument("--camera", choices=cameras, help="only needed if the name exists for several cameras")
    p.add_argument("--model", choices=[*model_names(), "all"], default="gemini-flash")
    p.add_argument("--no-depth", action="store_true", help="send RGB only")

    p = sub.add_parser("view", help="show a generated scene spec in Isaac Sim")
    p.add_argument("name", help="captured scene folder under outputs/<camera>")
    p.add_argument("--camera", choices=cameras, help="only needed if the name exists for several cameras")
    p.add_argument("--spec", type=Path, help="scene.json to load (default: the latest generation)")
    p.add_argument("--max-steps", type=int, default=-1, help="stop after this many steps; -1 runs until closed")

    args, extra = parser.parse_known_args()
    if args.command == "view":
        view(str(args.name), args.camera, args.spec, args.max_steps, extra)
    elif extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    elif args.command == "capture":
        capture(str(args.name), args.camera)
    else:
        generate(str(args.name), args.camera, args.model, not args.no_depth)


if __name__ == "__main__":
    main()
