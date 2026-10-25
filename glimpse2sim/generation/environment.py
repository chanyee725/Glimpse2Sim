"""Spawn the environment part of a scene spec: lights, ground, and optional desk (top outline + legs).

Furniture standing on the floor is spawned with the objects (see scene.build_scene).

World frame = scene-spec frame: origin at the center of the desk top (or on the ground), +x away from the
front camera, +z up.
Import only after the simulation app has been launched.
"""

import logging
import math
from collections.abc import Callable
from dataclasses import MISSING

import isaaclab.sim as sim_utils
import trimesh
from isaaclab.sim.spawners.meshes.meshes import _spawn_mesh_geom_from_mesh
from isaaclab.utils import configclass
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR

from ..config import load_config

DESK_DEFAULT_HEIGHT = 0.75  # used when the model omits height_from_floor
DESK_DEFAULT_TOP = 0.02  # top thickness when the model omits it
OUTLINE_SEGMENTS = 256  # points around a curved desk outline; enough that no facets show
FLOOR_SIZE = 20.0  # meters; plain floor slab (the default ground plane has a colored origin marker)
TEXTURES = load_config("textures")
TEXTURE_COLOR_TOLERANCE = 0.35  # sRGB distance; a texture this far from the model's own color would look wrong

log = logging.getLogger(__name__)


def _material(spec: dict) -> sim_utils.RigidBodyMaterialCfg:
    return sim_utils.RigidBodyMaterialCfg(
        static_friction=spec.get("static_friction", 0.5),
        dynamic_friction=spec.get("dynamic_friction", spec.get("static_friction", 0.5)),
        restitution=spec.get("restitution", 0.0),
    )


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _visual(spec: dict, default_color: tuple[float, float, float]) -> sim_utils.VisualMaterialCfg:
    """The named texture from configs/textures.yaml if the spec picked one, else a plain colored surface."""
    texture = spec.get("texture")
    color = spec.get("color_rgb")
    if texture in TEXTURES and color and math.dist(color, TEXTURES[texture]["color"]) > TEXTURE_COLOR_TOLERANCE:
        log.info("texture %r looks %s, far from color %s; using the plain color", texture, TEXTURES[texture]["color"], color)
        texture = None
    if texture in TEXTURES:
        return sim_utils.MdlFileCfg(
            mdl_path=f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/{TEXTURES[texture]['path']}", project_uvw=True
        )
    if texture:
        log.warning("unknown texture %r; using the plain color", texture)
    clamp = lambda v: min(max(float(v), 0.0), 1.0)  # noqa: E731
    # Models give colors as they look in the photo (sRGB); USD shading expects linear values.
    return sim_utils.PreviewSurfaceCfg(
        diffuse_color=tuple(_srgb_to_linear(clamp(c)) for c in spec.get("color_rgb", default_color)),
        roughness=clamp(spec.get("roughness", 0.5)),
        metallic=clamp(spec.get("metallic", 0.0)),
    )


def _arc(cx: float, cy: float, rx: float, ry: float, a0: float, a1: float, n: int) -> list[tuple[float, float]]:
    return [(cx + rx * math.cos(a), cy + ry * math.sin(a)) for a in (a0 + (a1 - a0) * i / n for i in range(n))]


def desk_outline(desk: dict, n: int = OUTLINE_SEGMENTS) -> list[tuple[float, float]]:
    """Counter-clockwise outline of the desk top seen from above, centered on the origin."""
    sx, sy = desk["size"]["x"], desk["size"]["y"]
    hx, hy = sx / 2, sy / 2
    shape = desk.get("top_shape", "rectangle")
    if shape in ("ellipse", "circle"):
        return _arc(0, 0, hx, hy, 0, 2 * math.pi, n)
    if shape == "stadium":
        r = min(hx, hy)
        if sx >= sy:  # half-circles at the +x and -x ends
            return _arc(hx - r, 0, r, r, -math.pi / 2, math.pi / 2, n // 2) + _arc(r - hx, 0, r, r, math.pi / 2, 3 * math.pi / 2, n // 2)
        return _arc(0, hy - r, r, r, 0, math.pi, n // 2) + _arc(0, r - hy, r, r, math.pi, 2 * math.pi, n // 2)
    r = min(float(desk.get("corner_radius") or 0.0), hx, hy) if shape == "rounded_rectangle" else 0.0
    if r <= 0:
        return [(hx, -hy), (hx, hy), (-hx, hy), (-hx, -hy)]
    q, cx, cy = n // 4, hx - r, hy - r
    return (_arc(cx, -cy, r, r, -math.pi / 2, 0, q) + _arc(cx, cy, r, r, 0, math.pi / 2, q)
            + _arc(-cx, cy, r, r, math.pi / 2, math.pi, q) + _arc(-cx, -cy, r, r, math.pi, 3 * math.pi / 2, q))


def _extrude_convex(outline: list[tuple[float, float]], thickness: float) -> trimesh.Trimesh:
    """Prism from a convex counter-clockwise outline, top face at z = 0 and bottom at z = -thickness."""
    n = len(outline)
    top = [(x, y, 0.0) for x, y in outline]
    bottom = [(x, y, -thickness) for x, y in outline]
    faces = [(0, i, i + 1) for i in range(1, n - 1)]  # top fan, facing up
    faces += [(n, n + i + 1, n + i) for i in range(1, n - 1)]  # bottom fan, facing down
    for i in range(n):
        j = (i + 1) % n
        faces += [(i, n + i, n + j), (i, n + j, j)]  # side quad, facing out
    return trimesh.Trimesh(vertices=top + bottom, faces=faces, process=False)


def spawn_slab(prim_path: str, cfg: "SlabCfg", translation=None, orientation=None, **kwargs):
    """A flat mesh: the convex `outline` extruded downward by `thickness`, so its top face is at the prim's z."""
    mesh = _extrude_convex(cfg.outline, cfg.thickness)
    return _spawn_mesh_geom_from_mesh(prim_path, cfg, mesh, translation, orientation)


@configclass
class SlabCfg(sim_utils.MeshCfg):
    func: Callable | str = spawn_slab
    outline: list = MISSING
    thickness: float = MISSING


def spawn_desk(desk: dict) -> float:
    """Spawn the desk as static pieces: a top from z = -thickness to 0 and legs down to the floor.

    Legs are parts in the desk frame (shape, dimensions, offset, rpy_deg); older specs gave posts as
    {position, radius}. Without legs the desk is a solid block from the floor up. Return the floor height.
    """
    from .objects import spawn_parts  # objects imports this module


    height = desk.get("height_from_floor") or DESK_DEFAULT_HEIGHT
    legs = desk.get("legs") or []
    t = float(desk.get("top_thickness") or desk.get("thickness") or DESK_DEFAULT_TOP) if legs else height
    common = dict(
        rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
        collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
        physics_material=_material(desk),
        visual_material=_visual(desk, (0.6, 0.45, 0.3)),
    )
    top = SlabCfg(outline=desk_outline(desk), thickness=t, **common)
    top.func("/World/Environment/Desk/top", top, translation=(0.0, 0.0, 0.0))
    leg_len = height - t
    posts = [  # older {position, radius} posts as vertical cylinder parts
        {"shape": "cylinder", "dimensions": {"radius": max(float(leg["radius"]), 0.003), "height": leg_len},
         "offset": [*leg["position"], -t - leg_len / 2]}
        for leg in legs if "shape" not in leg and "position" in leg
    ]
    parts = [leg for leg in legs if "shape" in leg] + posts
    if parts:
        sim_utils.create_prim("/World/Environment/Desk/legs", "Xform")
        spawn_parts("/World/Environment/Desk/legs", parts, desk | (desk.get("leg_appearance") or {}),
                    rigid_props=common["rigid_props"], collision_props=common["collision_props"])
    log.info("desk %s %.2f x %.2f m, top at z=0, height %.2f m, %d legs",
             desk.get("top_shape", "rectangle"), desk["size"]["x"], desk["size"]["y"], height, len(legs))
    return -height


def spawn_lights(lighting: dict | None) -> None:
    """A dome light for the ambient fill and a distant key light for shadows, as the spec describes them.

    Both stay neutral white: the photos are already white-balanced, so tinting the light would shift every color twice.
    """
    from .objects import quat_from_rpy  # objects imports this module

    lighting = lighting or {}
    ambient = min(max(float(lighting.get("ambient", 0.8)), 0.0), 1.0)
    dome = sim_utils.DomeLightCfg(intensity=300.0 + 1700.0 * ambient, color=(1.0, 1.0, 1.0))
    dome.func("/World/Light", dome)
    key = lighting.get("key_light") or {}
    strength = min(max(float(key.get("strength", 0.0)), 0.0), 1.0)
    if strength > 0:
        # A distant light shines along its own -z; tilt +z up toward the light's direction.
        orient = quat_from_rpy(0.0, 90.0 - float(key.get("elevation_deg", 60.0)), float(key.get("azimuth_deg", 0.0)))
        sun = sim_utils.DistantLightCfg(intensity=3000.0 * strength, angle=1.0 + 10.0 * (1.0 - strength))
        sun.func("/World/KeyLight", sun, orientation=orient)


def spawn_environment(env: dict | None) -> None:
    spawn_lights((env or {}).get("lighting"))

    desk = (env or {}).get("desk")
    floor_z = spawn_desk(desk) if desk else 0.0
    if not desk:
        log.info("no desk: objects rest on the ground at z=0")
    h = FLOOR_SIZE / 2
    floor = SlabCfg(
        outline=[(h, -h), (h, h), (-h, h), (-h, -h)], thickness=0.05,
        collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.5),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.55), roughness=0.8),
    )
    floor.func("/World/Environment/Ground", floor, translation=(0.0, 0.0, floor_z))
