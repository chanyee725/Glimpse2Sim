"""Spawn the objects of a scene spec as primitive rigid bodies.

Import only after the simulation app has been launched.
"""

import logging
import math
import re

import isaaclab.sim as sim_utils

from .environment import _material

MIN_SIZE = 0.002  # meters; guards against zero or negative sizes from the model

log = logging.getLogger(__name__)


def _yaw_quat(yaw_deg: float) -> tuple[float, float, float, float]:
    # Isaac Lab 3.0 orientations are (x, y, z, w).
    half = math.radians(yaw_deg) / 2
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _shape_cfg(obj: dict, **common):
    d = obj["dimensions"]
    size = lambda v: max(float(v), MIN_SIZE)  # noqa: E731
    match obj["shape"]:
        case "box":
            return sim_utils.CuboidCfg(size=(size(d["x"]), size(d["y"]), size(d["z"])), **common)
        case "cylinder":
            return sim_utils.CylinderCfg(radius=size(d["radius"]), height=size(d["height"]), **common)
        case "sphere":
            return sim_utils.SphereCfg(radius=size(d["radius"]), **common)
        case "capsule":
            # The spec gives the full end-to-end length; USD capsule height excludes the two caps.
            r = size(d["radius"])
            return sim_utils.CapsuleCfg(radius=r, height=size(float(d["height"]) - 2 * r), **common)
    raise ValueError(f"unknown shape {obj['shape']!r}")


def _prim_name(obj_id: str, used: set[str]) -> str:
    name = re.sub(r"\W", "_", obj_id) or "object"
    if name[0].isdigit():
        name = f"obj_{name}"
    base, i = name, 1
    while name in used:
        name, i = f"{base}_{i}", i + 1
    used.add(name)
    return name


def spawn_objects(objects: list[dict]) -> int:
    """Spawn every valid object; skip and log the ones that cannot be built. Return the spawned count."""
    used: set[str] = set()
    spawned = 0
    for obj in objects:
        obj_id = obj.get("id", "object")
        try:
            cfg = _shape_cfg(
                obj,
                rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=bool(obj.get("is_static", False))),
                collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
                mass_props=sim_utils.MassCfg(mass=max(float(obj.get("mass_kg", 0.1)), 1e-3)),
                physics_material=_material(obj),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(obj.get("color_rgb", (0.5, 0.5, 0.5)))),
            )
            cfg.func(
                f"/World/Objects/{_prim_name(obj_id, used)}",
                cfg,
                translation=tuple(obj["position"]),
                orientation=_yaw_quat(obj.get("yaw_deg", 0.0)),
            )
            spawned += 1
            log.info("object %-20s %-8s mass=%.3f kg pos=%s", obj_id, obj["shape"], obj.get("mass_kg", 0.1), obj["position"])
        except (KeyError, TypeError, ValueError) as e:
            log.warning("skipped object %s: %r", obj_id, e)
    return spawned
