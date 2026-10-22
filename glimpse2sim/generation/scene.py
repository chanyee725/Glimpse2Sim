"""Load a generated scene spec and build it in Isaac Sim."""

import json
from pathlib import Path


def latest_scene_spec(scene_dir: Path) -> Path:
    """Most recent generations/*/scene.json under a captured scene folder."""
    specs = sorted(scene_dir.glob("generations/*/scene.json"), key=lambda p: p.stat().st_mtime)
    if not specs:
        raise FileNotFoundError(f"no generations/*/scene.json under {scene_dir}")
    return specs[-1]


def split_spec(spec: dict) -> tuple[dict | None, list[dict]]:
    """Return (environment, objects). Older specs kept the desk at the top level."""
    env = spec.get("environment")
    if env is None and "desk" in spec:
        env = {"desk": spec["desk"]}
    return env, spec.get("objects", [])


def load_scene_spec(path: Path) -> tuple[dict | None, list[dict]]:
    return split_spec(json.loads(path.read_text()))


def write_placements(run_dir: Path) -> Path | None:
    """Write run_dir/placements.json (offsets from the desk corner) from run_dir/scene.json."""
    from .placement import placements

    spec_path = run_dir / "scene.json"
    if not spec_path.exists():
        return None
    result = placements(*load_scene_spec(spec_path))
    if result is None:
        return None
    out = run_dir / "placements.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return out


def build_scene(path: Path) -> tuple[int, int]:
    """Spawn the environment and objects of one scene spec. Return (spawned, total) objects."""
    # Isaac Lab modules are imported here so they load only after the app is launched.
    from .environment import spawn_environment
    from .objects import spawn_objects

    env, objects = load_scene_spec(path)
    spawn_environment(env)
    return spawn_objects(objects), len(objects)
