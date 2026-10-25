"""Spawn the objects of a scene spec as primitive rigid bodies, or as one rigid body built from several parts.

An object may also have a library model found for its `asset_query` (`asset_match.path`, see glimpse2sim/assets.py):
it is shown in place of the shapes, scaled to the object's dimensions and painted with the object's own color and
finish, while the shapes stay its invisible collision body. So the library lends only a form; size, place, color,
and physics all stay the model's answer.
Import only after the simulation app has been launched.
"""

import itertools
import logging
import math
import re

import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.sim import schemas
from isaaclab.sim.spawners.utils import apply_schema_props
from isaaclab.utils.assets import NUCLEUS_ASSET_ROOT_DIR
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from ..config import ROOT
from .environment import _material, _visual

CONVERTED_DIR = ROOT / "assets" / "cache" / "objaverse"  # GLB downloads converted to USD once, by uid

ASSET_TURN_PENALTY = 0.35  # log-stretch an axis swap must save before the asset is turned on its side

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


def _target_box(obj: dict) -> tuple[np.ndarray, np.ndarray]:
    """(extent, center) the object's shapes occupy in its own frame, which the asset is fitted to."""
    if obj.get("parts"):  # origin at the footprint center on the surface
        size = obj.get("size_mm") or {}
        ext = np.array([size.get("length", 0), size.get("width", 0), size.get("height", 0)], float) / 1000
        return np.maximum(ext, MIN_SIZE), np.array([0.0, 0.0, ext[2] / 2])
    d = obj["dimensions"]
    ext = {
        "box": lambda: [d["x"], d["y"], d["z"]],
        "cylinder": lambda: [2 * d["radius"], 2 * d["radius"], d["height"]],
        "capsule": lambda: [2 * d["radius"], 2 * d["radius"], d["height"]],
        "sphere": lambda: [2 * d["radius"]] * 3,
    }[obj["shape"]]()
    return np.maximum(np.array(ext, float), MIN_SIZE), np.zeros(3)


def _fit_matrix(src_min: np.ndarray, src_max: np.ndarray, ext: np.ndarray, center: np.ndarray) -> np.ndarray:
    """4x4 (column vectors) that rotates the asset's bounding box by right angles so its sides match the target's
    by length, then scales and moves it onto the target box. The asset keeps its own orientation unless turning it
    clearly reduces the stretch (a mug with a handle is nearly as wide as tall and must stay upright)."""
    src_ext = np.maximum(src_max - src_min, 1e-6)
    best = None
    for perm in itertools.permutations(range(3)):  # target axis i takes asset axis perm[i]
        moved = sum(i != j for i, j in enumerate(perm))
        distortion = np.ptp(np.log(ext / src_ext[list(perm)])) + ASSET_TURN_PENALTY * moved
        if best is None or distortion < best[0]:
            best = (distortion, perm)
    perm = best[1]
    rot = np.zeros((3, 3))
    for i, j in enumerate(perm):
        rot[i, j] = 1.0
    if np.linalg.det(rot) < 0:  # keep it a rotation, not a mirror image
        rot[2] *= -1
    scale = ext / src_ext[list(perm)]
    m = np.eye(4)
    m[:3, :3] = np.diag(scale) @ rot
    m[:3, 3] = center - m[:3, :3] @ ((src_min + src_max) / 2)
    return m


def _glb_to_usd(glb: str, uid: str) -> str:
    """Convert a downloaded GLB to USD with Kit's asset converter (once per uid). Return the USD path."""
    out = CONVERTED_DIR / uid / "model.usd"
    if out.exists():
        return str(out)
    import asyncio

    from isaaclab.sim.utils import enable_extension

    enable_extension("omni.kit.asset_converter")
    import omni.kit.asset_converter as converter

    context = converter.AssetConverterContext()
    context.ignore_animations = context.ignore_camera = context.ignore_light = True
    context.merge_all_meshes = True
    context.use_meter_as_world_unit = True
    context.baking_scales = True
    out.parent.mkdir(parents=True, exist_ok=True)
    task = converter.get_instance().create_converter_task(glb, str(out), None, context)
    if not asyncio.get_event_loop().run_until_complete(task.wait_until_finished()):
        raise RuntimeError(f"could not convert {glb}: {task.get_error_message()}")
    return str(out)


def _asset_usd(match: dict) -> str:
    if match.get("source") == "objaverse":
        return _glb_to_usd(match["path"], match["uid"])
    return f"{NUCLEUS_ASSET_ROOT_DIR}/{match['path']}"


def _up_to_z(usd_path: str) -> np.ndarray:
    """3x3 rotation that brings the file's up axis onto +z (converted GLBs are y-up)."""
    if UsdGeom.GetStageUpAxis(Usd.Stage.Open(usd_path)) == UsdGeom.Tokens.y:
        return np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])  # +90 deg about x: y -> z
    return np.eye(3)


def attach_asset(path: str, obj: dict) -> None:
    """Show the object's matched library model in place of its shapes, scaled to them; the shapes keep colliding."""
    stage = sim_utils.get_current_stage()
    for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
        if prim.IsA(UsdGeom.Gprim):
            UsdGeom.Imageable(prim).MakeInvisible()
    usd_path = _asset_usd(obj["asset_match"])
    asset = sim_utils.create_prim(f"{path}/asset", usd_path=usd_path)
    for prim in Usd.PrimRange(asset):  # the asset is only a look: its own physics would fight the object's body
        for api in (UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI, UsdPhysics.CollisionAPI, UsdPhysics.MeshCollisionAPI):
            if prim.HasAPI(api):
                prim.RemoveAPI(api)
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    bound = box.ComputeUntransformedBound(asset).ComputeAlignedRange()
    if bound.IsEmpty():
        raise ValueError(f"asset {obj['asset_match']['path']!r} has no geometry")
    up = _up_to_z(usd_path)
    lo, hi = np.array(bound.GetMin()), np.array(bound.GetMax())
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) @ up.T
    ext, center = _target_box(obj)
    m = _fit_matrix(corners.min(0), corners.max(0), ext, center)
    m[:3, :3] = m[:3, :3] @ up  # rotate the file z-up first, then fit
    xform = UsdGeom.Xformable(asset)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(Gf.Matrix4d(m.T.tolist()))  # USD multiplies row vectors
    # Paint it with the object's own look (color_rgb or texture, roughness, metallic) over the file's materials.
    look = _visual(obj, (0.5, 0.5, 0.5))
    look.func(f"{path}/asset_look", look)
    sim_utils.bind_visual_material(f"{path}/asset", f"{path}/asset_look", stronger_than_descendants=True)


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
            asset_path = (obj.get("asset_match") or {}).get("path")
            if asset_path:
                try:
                    attach_asset(path, obj)
                    shape += f" + asset {asset_path.rsplit('/', 1)[-1]}"
                except Exception as e:  # a missing or broken asset falls back to the visible shapes
                    log.warning("object %s: asset %s not used: %r", obj_id, asset_path, e)
                    for prim in Usd.PrimRange(sim_utils.get_current_stage().GetPrimAtPath(path)):
                        if prim.IsA(UsdGeom.Gprim):
                            UsdGeom.Imageable(prim).MakeVisible()
            spawned += 1
            log.info("object %-20s %-8s mass=%.3f kg pos=%s", obj_id, shape, obj.get("mass_kg", 0.1), obj["position"])
        except (KeyError, TypeError, ValueError) as e:
            log.warning("skipped object %s: %r", obj_id, e)
    return spawned
