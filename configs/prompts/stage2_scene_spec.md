Stage 2: simulation scene specification.

Build a scene specification that a physics simulator (Isaac Lab) will load directly. The robot, table, and cameras are already in the simulator; specify only the objects on the table.

Scene understanding from Stage 1:

```json
{stage1_json}
```

Use the images to refine positions and sizes, and add the physical properties a simulator needs. Every object must rest on the table or on another object without overlapping anything.

Return JSON with this structure:

```json
{
  "objects": [
    {
      "id": "red_cube_1",
      "shape": "box",
      "dimensions": {"x": 0.0, "y": 0.0, "z": 0.0},
      "position": [0.0, 0.0, 0.0],
      "yaw_deg": 0.0,
      "color_rgb": [0.0, 0.0, 0.0],
      "mass_kg": 0.0,
      "static_friction": 0.0,
      "dynamic_friction": 0.0,
      "restitution": 0.0,
      "is_static": false
    }
  ],
  "task": {
    "pick_object": "red_cube_1",
    "place_target": "blue_bowl_1"
  }
}
```

Field definitions:
- `shape`: one of `box`, `cylinder`, `sphere`, `capsule`.
- `dimensions`: `box` uses x, y, z. `cylinder` and `capsule` use radius and height. `sphere` uses radius.
- `position`: center of the shape in the robot base frame.
- `color_rgb`: 0 to 1 per channel.
- `mass_kg`, friction, `restitution`: estimate from the apparent material (wood, plastic, metal, foam, ...).
- `is_static`: true for objects that should not move (heavy containers, fixtures).
- `task`: the most natural pick-and-place pair in the scene. Use `null` for `place_target` if there is no container.
