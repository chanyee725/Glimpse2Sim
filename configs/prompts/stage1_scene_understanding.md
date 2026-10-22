Stage 1: scene understanding.

List every object on the table that the robot could interact with, and the spatial relations between them. Ignore the robot arm itself, the table, cables, and the background.

Return JSON with this structure:

```json
{
  "table": {"size_xy": [0.0, 0.0]},
  "objects": [
    {
      "id": "red_cube_1",
      "category": "cube",
      "color": "red",
      "position": [0.0, 0.0, 0.0],
      "size": [0.0, 0.0, 0.0],
      "yaw_deg": 0.0,
      "graspable": true,
      "confidence": 0.0
    }
  ],
  "relations": [
    {"subject": "red_cube_1", "predicate": "on", "object": "table"}
  ]
}
```

Field definitions:
- `id`: unique snake_case name.
- `position`: center of the object's bounding box in the robot base frame.
- `size`: axis-aligned bounding box extents [x, y, z] in the object's own frame.
- `yaw_deg`: rotation about +z.
- `predicate`: one of `on`, `in`, `left_of`, `right_of`, `in_front_of`, `behind`, `near`.
- `confidence`: 0 to 1.
