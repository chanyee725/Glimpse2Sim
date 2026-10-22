"""Spawn the environment part of a scene spec: lights, ground, optional desk, and the SO-101.

World frame = scene-spec frame: origin at the robot base on its support surface, +z up.
Import only after the simulation app has been launched.
"""

import logging

import isaaclab.sim as sim_utils

from ..config import ROOT

ROBOT_USD = ROOT / "assets" / "so101" / "so101_new_calib.usd"
DESK_DEFAULT_HEIGHT = 0.75  # used when the model omits height_from_floor

log = logging.getLogger(__name__)


def _material(spec: dict) -> sim_utils.RigidBodyMaterialCfg:
    return sim_utils.RigidBodyMaterialCfg(
        static_friction=spec.get("static_friction", 0.5),
        dynamic_friction=spec.get("dynamic_friction", spec.get("static_friction", 0.5)),
        restitution=spec.get("restitution", 0.0),
    )


def spawn_desk(desk: dict) -> float:
    """Spawn the desk as a static block from the floor up to z = 0. Return the floor height."""
    sx, sy = desk["size"]["x"], desk["size"]["y"]
    cx, cy = desk.get("center", [0.0, 0.0])
    height = desk.get("height_from_floor") or DESK_DEFAULT_HEIGHT
    cfg = sim_utils.CuboidCfg(
        size=(sx, sy, height),
        rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
        collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
        physics_material=_material(desk),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(desk.get("color_rgb", (0.6, 0.45, 0.3)))),
    )
    cfg.func("/World/Environment/Desk", cfg, translation=(cx, cy, -height / 2))
    log.info("desk %.2f x %.2f m, top at z=0, height %.2f m", sx, sy, height)
    return -height


def spawn_environment(env: dict | None) -> None:
    light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9))
    light.func("/World/Light", light)

    desk = (env or {}).get("desk")
    floor_z = spawn_desk(desk) if desk else 0.0
    if not desk:
        log.info("no desk: robot stands on the ground at z=0")
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/Environment/Ground", ground, translation=(0.0, 0.0, floor_z))

    robot = sim_utils.UsdFileCfg(usd_path=str(ROBOT_USD))
    robot.func("/World/Robot", robot, translation=(0.0, 0.0, 0.0))
