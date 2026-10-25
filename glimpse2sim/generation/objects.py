"""Spawn the objects of a scene spec as primitive rigid bodies, or as one rigid body built from several parts.

Import only after the simulation app has been launched.
"""

import logging
import math
import re

import isaaclab.sim as sim_utils
from isaaclab.sim import schemas
from isaaclab.sim.spawners.utils import apply_schema_props

from .environment import _material, _visual

MIN_SIZE = 0.002  # meters; guards against zero or negative sizes from the model

log = logging.getLogger(__name__)

Quat = tuple[float, float, float, float]  # Isaac Lab 3.0 orientations are (x, y, z, w).


def _quat_mul(a: Quat, b: Quat) -> Quat:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _axis_quat(axis: int, deg: float) -> Quat:
    half = math.radians(deg) / 2
    q = [0.0, 0.0, 0.0, math.cos(half)]
    q[axis] = math.sin(half)
    return tuple(q)


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> Quat:
    """Rotation by roll about x, then pitch about y, then yaw about z (all in degrees, fixed axes)."""
    return _quat_mul(_axis_quat(2, yaw), _quat_mul(_axis_quat(1, pitch), _axis_quat(0, roll)))


def _orientation(obj: dict) -> Quat:
    """Yaw about +z; a lying single cylinder or capsule is first tipped 90 deg about +y so its axis lies along +x."""
    lying = obj.get("lying") and obj["shape"] in ("cylinder", "capsule")
    return quat_from_rpy(0.0, 90.0 if lying else 0.0, obj.get("yaw_deg", 0.0))


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


def spawn_parts(path: str, parts: list[dict], owner: dict, **physics) -> int:
    """Spawn each part as a child shape of `path` at its offset and rpy. Return the spawned count.

    A part without its own appearance takes the owner's. `physics` (collision, rigid, mass props) applies to
    every part; pass no rigid or mass props when the parent prim is the rigid body.
    """
    used: set[str] = set()
    spawned = 0
    for i, part in enumerate(parts):
        try:
            look = part if "color_rgb" in part or "texture" in part else owner
            cfg = _shape_cfg(part, physics_material=_material(owner), visual_material=_visual(look, (0.5, 0.5, 0.5)),
                             **physics)
            cfg.func(
                f"{path}/{_prim_name(str(part.get('name') or f'part_{i}'), used)}",
                cfg,
                translation=tuple(float(v) for v in part.get("offset", (0.0, 0.0, 0.0))),
                orientation=quat_from_rpy(*(float(v) for v in part.get("rpy_deg", (0.0, 0.0, 0.0)))),
            )
            spawned += 1
        except (KeyError, TypeError, ValueError) as e:
            log.warning("skipped part %d of %s: %r", i, path, e)
    return spawned


def spawn_objects(objects: list[dict], root: str = "/World/Objects", static: bool = False) -> int:
    """Spawn every valid object under `root`; skip and log the ones that cannot be built. Return the spawned count.

    `static` makes them all fixed (used for furniture standing on the floor). An object with `parts` becomes one
    rigid body whose collision shapes are the parts.
    """
    used: set[str] = set()
    spawned = 0
    for obj in objects:
        obj_id = obj.get("id", "object")
        path = f"{root}/{_prim_name(obj_id, used)}"
        rigid = sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=static or bool(obj.get("is_static", False)))
        mass = sim_utils.MassCfg(mass=max(float(obj.get("mass_kg", 0.1)), 1e-3))
        collision = sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)
        try:
            if obj.get("parts"):
                sim_utils.create_prim(path, "Xform", translation=tuple(obj["position"]), orientation=_orientation(obj))
                if not spawn_parts(path, obj["parts"], obj, collision_props=collision):
                    raise ValueError("no part could be built")
                apply_schema_props(mass, path, schemas.apply_mass_properties, schemas.define_mass_properties, None)
                apply_schema_props(rigid, path, schemas.apply_rigid_body_properties,
                                   schemas.define_rigid_body_properties, None)
                shape = f"{len(obj['parts'])} parts"
            else:
                cfg = _shape_cfg(
                    obj,
                    rigid_props=rigid,
                    collision_props=collision,
                    mass_props=mass,
                    physics_material=_material(obj),
                    visual_material=_visual(obj, (0.5, 0.5, 0.5)),
                )
                cfg.func(path, cfg, translation=tuple(obj["position"]), orientation=_orientation(obj))
                shape = obj["shape"]
            spawned += 1
            log.info("object %-20s %-8s mass=%.3f kg pos=%s", obj_id, shape, obj.get("mass_kg", 0.1), obj["position"])
        except (KeyError, TypeError, ValueError) as e:
            log.warning("skipped object %s: %r", obj_id, e)
    return spawned
