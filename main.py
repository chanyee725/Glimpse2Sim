"""Glimpse2Sim entry point.

Capture: live preview; each left click saves one RGB-D frame to outputs/<camera>/<name>.
The connected camera is detected automatically.
    uv run main.py capture scene01
    Left click: save the current frame (000, 001, ...). q / Esc / closing the window: quit.

Generate: send a captured scene to a model and build the scene spec, logging the model's thinking.
The camera is looked up from the scene folder.
    uv run main.py generate scene01
    uv run main.py generate scene01 --model all
    uv run main.py generate scene01 --frame front --no-depth

Fuse: place all captured frames of a scene with the ChArUco board on the table, correct each frame's depth
against the board, fuse them into one TSDF and ray cast it into the reference frame as one depth image
(<scene>/fused). --icp also refines the board poses with colored ICP and a pose graph; --view shows the result
in Rerun.
    uv run main.py fuse scene03 --icp --view

View: build the latest generated scene spec in Isaac Sim (extra args go to the Isaac Lab launcher).
    uv run main.py view scene01
    uv run main.py view scene01 --spec outputs/d435/scene01/generations/<run>/scene.json
    uv run main.py view scene01 --headless --max-steps 200
    uv run main.py view scene01 --capture       # save sim_capture.png next to the scene.json and exit
"""

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

from glimpse2sim.capture import D435Camera, detect_camera, find_scene, output_dir
from glimpse2sim.config import ROOT, load_config
from glimpse2sim.fusion import fuse_scene
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


def fuse(name: str, camera: str | None, reference: str, max_depth: float, voxel: float, icp: bool,
         frames: list[str] | None, view: bool) -> None:
    scene_dir = output_dir(name, camera) if camera else find_scene(name)[1]
    fuse_scene(scene_dir, reference, max_depth, voxel, icp, frames, view)


def generate(name: str, camera: str | None, model: str, with_depth: bool, frames: list[str] | None) -> None:
    load_dotenv(ROOT / ".env")
    if camera:
        scene_dir = output_dir(name, camera)
    else:
        camera, scene_dir = find_scene(name)
    for m in model_names(include_debug=False) if model == "all" else [model]:
        generate_scene(m, scene_dir, camera, with_depth, frames)


def view(
    name: str, camera: str | None, spec: Path | None, max_steps: int, capture: bool, launcher_argv: list[str]
) -> None:
    # Isaac Lab is imported here so capture/generate never start Kit (or its EULA prompt).
    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab.physics import PhysicsCfg

    import isaaclab.sim as sim_utils

    scene_dir = output_dir(name, camera) if camera else find_scene(name)[1]
    if spec is None:
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
    if launcher_args.headless and capture:
        raise SystemExit("--capture needs the viewer window; drop --headless")
    if launcher_args.headless:
        launcher_args.visualizer = None  # Isaac Lab 3.0 has no --headless; no visualizer means headless

    with launch_simulation(cfg=PhysicsCfg(), launcher_args=launcher_args) as physics_cfg:
        sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(dt=0.01, device=launcher_args.device, physics=physics_cfg)
        )
        _photo_camera(sim, spec, scene_dir)
        spawned, total = build_scene(spec)
        logging.info("spec %s: %d/%d objects spawned", spec, spawned, total)
        sim.reset()

        if capture:
            out = spec.parent / "sim_capture.png"
            _capture_view(sim, out)
            _side_by_side(scene_dir / "front_color.png", out, spec.parent / "compare.png")
            return

        steps = 0
        while sim.is_running() and (max_steps < 0 or steps < max_steps):
            sim.step()
            steps += 1


def _photo_camera(sim, spec: Path, scene_dir: Path, frame: str = "front") -> None:
    """Put the viewer where the model placed the front camera, with the real camera's field of view.

    Falls back to a fixed view from the front (-x) when the spec has no camera for that image.
    """
    cam = json.loads(spec.read_text()).get("environment", {}).get("cameras", {}).get(f"{frame}_color")
    try:
        eye, target = [float(v) for v in cam["position"]], [float(v) for v in cam["look_at"]]
    except (KeyError, TypeError, ValueError):
        eye, target = [-0.9, 0.0, 0.7], [0.0, 0.0, 0.0]
        logging.info("no %s_color camera in the spec; using the default front view", frame)
    sim.set_camera_view(eye=eye, target=target)

    meta = scene_dir / f"{frame}_meta.json"
    if not meta.exists():
        return
    import omni.usd

    k = json.loads(meta.read_text())["intrinsics"]
    persp = omni.usd.get_context().get_stage().GetPrimAtPath("/OmniverseKit_Persp")
    if persp:  # aperture / focal length = image size / focal length in pixels gives the same field of view
        focal = persp.GetAttribute("focalLength").Get()
        persp.GetAttribute("horizontalAperture").Set(focal * k["width"] / k["fx"])
        persp.GetAttribute("verticalAperture").Set(focal * k["height"] / k["fy"])


def _capture_view(sim, out: Path) -> None:
    """Save the viewer as the scene was spawned (physics not stepped), so objects are where the spec put them."""
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

    for _ in range(60):  # let the renderer warm up and load materials
        sim.render()
    out.unlink(missing_ok=True)
    capture_viewport_to_file(get_active_viewport(), str(out))
    for _ in range(300):  # the capture is written a few frames later
        sim.render()
        if out.exists():
            logging.info("captured %s", out)
            return
    logging.error("capture timed out: %s was not written", out)


def _side_by_side(photo: Path, render: Path, out: Path) -> None:
    """Photo on the left, render on the right, at the render's height."""
    if not (photo.exists() and render.exists()):
        return
    a, b = cv2.imread(str(photo)), cv2.imread(str(render))
    a = cv2.resize(a, (round(a.shape[1] * b.shape[0] / a.shape[0]), b.shape[0]))
    cv2.imwrite(str(out), np.hstack([a, b]))
    logging.info("photo | render -> %s", out)


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
    p.add_argument("--frame", nargs="+", help="only send these frames, in this order, e.g. --frame front top (default: all)")

    p = sub.add_parser("fuse", help="fuse all frames of a scene into one depth image in the reference view")
    p.add_argument("name", help="captured scene folder under outputs/<camera>")
    p.add_argument("--camera", choices=cameras, help="only needed if the name exists for several cameras")
    p.add_argument("--ref", default="front", help="reference frame the fused depth is rendered in")
    p.add_argument("--max-depth", type=float, default=1.5, help="m; farther depth is not fused")
    p.add_argument("--voxel", type=float, default=0.004, help="TSDF voxel size, m")
    p.add_argument("--icp", action="store_true", help="refine the board poses with colored ICP + pose graph")
    p.add_argument("--frame", nargs="+", help="only fuse these frames, e.g. --frame 000 001 002 (default: all)")
    p.add_argument("--view", action="store_true", help="show the fused mesh, cameras and depth in Rerun")

    p = sub.add_parser("view", help="show a generated scene spec in Isaac Sim")
    p.add_argument("name", help="captured scene folder under outputs/<camera>")
    p.add_argument("--camera", choices=cameras, help="only needed if the name exists for several cameras")
    p.add_argument("--spec", type=Path, help="scene.json to load (default: the latest generation)")
    p.add_argument("--max-steps", type=int, default=-1, help="stop after this many steps; -1 runs until closed")
    p.add_argument("--capture", action="store_true", help="save the spawned scene as sim_capture.png and exit")

    args, extra = parser.parse_known_args()
    if args.command == "view":
        view(str(args.name), args.camera, args.spec, args.max_steps, args.capture, extra)
    elif extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    elif args.command == "fuse":
        fuse(str(args.name), args.camera, args.ref, args.max_depth, args.voxel, args.icp, args.frame, args.view)
    elif args.command == "capture":
        capture(str(args.name), args.camera)
    else:
        generate(str(args.name), args.camera, args.model, not args.no_depth, args.frame)


if __name__ == "__main__":
    main()
