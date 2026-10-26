"""Load a generated scene spec and build it in Isaac Sim."""

import json
from datetime import datetime
from pathlib import Path

from ..config import ROOT, load_config


RESULTS_DIR = ROOT / "results"


def current_experiment() -> str:
    """The experiment round new runs go to (configs/experiment.yaml)."""
    return str(load_config("experiment")["current"])


def experiments(scene: str) -> list[str]:
    """The experiment rounds a scene has results in, oldest first."""
    return sorted(p.name for p in (RESULTS_DIR / scene).iterdir() if p.is_dir()) if (RESULTS_DIR / scene).exists() else []


def experiment_dir(scene: str, experiment: str | None = None) -> Path:
    """results/<scene>/<experiment>, the current round by default."""
    return RESULTS_DIR / scene / (experiment or current_experiment())


def run_dir(scene: str, model_folder: str, condition: str, started: datetime) -> Path:
    """results/<scene>/<experiment>/<model folder>/<condition>/<time> in the current round,
    e.g. results/scene01/02/gemini/flash/L1/20261025_134746."""
    return experiment_dir(scene) / model_folder / condition / f"{started:%Y%m%d_%H%M%S}"


def latest_scene_spec(scene: str, condition: str | None = None, model_folder: str | None = None,
                      experiment: str | None = None) -> Path:
    """Most recent scene.json of a scene in one experiment round, optionally only of one condition and one model."""
    root = experiment_dir(scene, experiment)
    pattern = f"{model_folder or '*/*'}/{condition or '*'}/*/scene.json"
    specs = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not specs:
        raise FileNotFoundError(f"no {root.relative_to(ROOT)}/{pattern}")
    return specs[-1]


def split_spec(spec: dict) -> tuple[dict | None, list[dict]]:
    """Return (environment, objects). Older specs kept the desk at the top level."""
    env = spec.get("environment")
    if env is None and "desk" in spec:
        env = {"desk": spec["desk"]}
    return env, spec.get("objects", [])


def load_scene_spec(path: Path) -> tuple[dict | None, list[dict]]:
    return split_spec(json.loads(path.read_text()))


def place_on_floor(furniture: list[dict], desk: dict | None) -> list[dict]:
    """Furniture comes with [x, y] only; set z so each item stands on the floor (desk top is z = 0)."""
    floor = -float(desk.get("height_from_floor") or 0.75) if desk else 0.0
    placed = []
    for f in furniture:
        d, shape = f.get("dimensions", {}), f.get("shape")
        half = {"box": d.get("z", 0), "cylinder": d.get("height", 0), "capsule": d.get("height", 0)}.get(shape, 0) / 2
        if shape == "sphere":
            half = d.get("radius", 0)
        if f.get("parts"):  # part offsets are measured from the floor
            half = 0.0
        x, y = f.get("position", [0.0, 0.0])[:2]
        placed.append(f | {"position": [x, y, floor + float(half)]})
    return placed


def build_scene(path: Path) -> tuple[int, int]:
    """Spawn the environment and objects of one scene spec. Return (spawned, total) objects."""
    # Isaac Lab modules are imported here so they load only after the app is launched.
    from .environment import spawn_environment
    from .objects import spawn_objects

    env, objects = load_scene_spec(path)
    spawn_environment(env)
    furniture = place_on_floor((env or {}).get("furniture") or [], (env or {}).get("desk"))
    spawn_objects(furniture, root="/World/Environment/Furniture", static=True)
    return spawn_objects(objects), len(objects)
