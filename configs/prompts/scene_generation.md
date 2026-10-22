Rebuild the real scene in the images as an Isaac Sim scene specification.

Goal: the simulated scene should match the real one as closely as possible, split into the environment (the surface the robot is mounted on) and the objects on it, with their physical properties.

Environment:
- If the robot is mounted on a desk or table, describe it in `environment.desk`: size, height from the floor, surface color, and surface material.
- If there is no desk (for example the robot sits on the floor or on a board on the floor), set `environment.desk` to `null`. The surface the robot stands on is then the ground at z = 0.

Objects:
- Every object resting on the robot's support surface, including large ones such as a laptop or monitor.

Exclude:
- The SO-101 robot arm (it is already in the simulator).
- Cables and wires, including those attached to cameras or laptops.
- Walls and anything not resting on the support surface.

Physical properties:
- Give mass, friction, and restitution for the desk (if any) and every object.
- First identify what each object most likely is (for example "350 ml aluminum soda can" or "plastic storage box"), then derive its properties from typical real-world values for that kind of object.
- Record the reasoning behind every physical value in `physics_rationale`: what you identified the object as, the typical value you used, and how you adjusted it for the observed size or material.

Constraints:
- Every object must rest on the support surface or on another object, with no overlaps.
- Positions are the center of each shape in the coordinate frame defined in the system prompt. The support surface is at z = 0.

Return JSON with this structure:

```json
{
  "environment": {
    "desk": {
      "size": {"x": 0.0, "y": 0.0},
      "center": [0.0, 0.0],
      "height_from_floor": 0.0,
      "thickness": 0.0,
      "color_rgb": [0.0, 0.0, 0.0],
      "material": "",
      "static_friction": 0.0,
      "dynamic_friction": 0.0,
      "restitution": 0.0,
      "confidence": 0.0,
      "physics_rationale": {
        "identified_as": "",
        "friction": "",
        "restitution": ""
      }
    }
  },
  "objects": [
    {
      "id": "red_cube_1",
      "identified_as": "",
      "shape": "box",
      "dimensions": {"x": 0.0, "y": 0.0, "z": 0.0},
      "size_mm": {"length": 0, "width": 0, "height": 0},
      "position": [0.0, 0.0, 0.0],
      "yaw_deg": 0.0,
      "color_rgb": [0.0, 0.0, 0.0],
      "material": "",
      "mass_kg": 0.0,
      "static_friction": 0.0,
      "dynamic_friction": 0.0,
      "restitution": 0.0,
      "is_static": false,
      "confidence": 0.0,
      "physics_rationale": {
        "mass": "",
        "friction": "",
        "restitution": ""
      }
    }
  ]
}
```

Field definitions:
- `environment.desk`: `null` when there is no desk.
- `desk.center`: [x, y] of the desk's center. The desk can extend behind the robot (negative x).
- `identified_as`: what the object most likely is in the real world, as specific as you can tell.
- `shape`: one of `box`, `cylinder`, `sphere`, `capsule`. Approximate complex objects with the closest shape.
- `dimensions`: `box` uses x, y, z. `cylinder` uses radius and height. `capsule` uses radius and height, where height is the full end-to-end length including both rounded ends. `sphere` uses radius.
- `size_mm`: the object's bounding box in millimeters, as integers: `length` along the object's own x axis, `width` along its y axis, `height` along z. It must agree with `dimensions` (for a cylinder, length = width = 2 x radius). Only objects have this field, not the desk.
- `color_rgb`: 0 to 1 per channel.
- `is_static`: true for objects that should not move (heavy items, fixtures, a laptop).
- `confidence`: 0 to 1, for the object's identity, position, and size together.
- `physics_rationale`: one short sentence per property, e.g. "an empty 350 ml aluminum can typically weighs about 0.015 kg".
