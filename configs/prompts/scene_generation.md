Rebuild the real scene in the images as an Isaac Sim scene specification.

Goal: a render of the rebuilt scene from the same camera should show the objects on the desk as they look in the photo. Put the effort into the objects: every one of them with the right shape, size, position, orientation, color, and surface finish. Above all, keep every object exactly where it is in the photos. The desk is rebuilt simply, as the surface they rest on; nothing else in the room is rebuilt. The same rules apply to any scene: count, measure, and place what is actually in these images.

How to work:
1. List every object on the desk. Look in every image, including small, thin, dark, transparent, and partly hidden items, and items at the image edges. Count repeated items one by one.
2. Identify each item as specifically as you can ("350 ml aluminum soda can", "folding tray table"). Use what such items typically measure only as a check on what you see, never instead of it.
3. Measure sizes and positions from the images: use the depth images when given, the desk and items of known size as scale references, and perspective. Cross-check between views.
4. Build each object from one simple shape, or a few parts when its main form needs them (see Parts).
5. Check the result against every image: positions, overlaps, what rests on what, and the 2D boxes.

Coordinate frame: as defined in the system prompt. The desk top is z = 0.

Positions:
- Place each object where it actually is, not where it would look neat. Do not center the group, spread objects apart, line them up, round coordinates to tidy values, or move an object to avoid a near-overlap that the photo shows.
- Measure each object's position against fixed references: its distance from the desk edges and from the desk center, and its distance and direction to its nearest neighbors.
- Keep the arrangement: the gaps between objects, which object is nearer the front camera, and which is further left must match the photos.
- Use the depth images, when given, for how far each object is from the camera, and the top view, when given, for the layout on the desk.
- Before answering, picture the rebuilt scene from the front camera and compare it with front_color: every object should land where its 2D box is.

Environment:
- `environment.desk`: the surface the objects rest on, with its outline (`top_shape`), size, top thickness, height from the floor, and simple vertical legs. `null` when the objects lie on the floor (the ground is then z = 0).
- `environment.lighting`: the main light as it appears in the photos.
- `environment.cameras`: where each color image was taken from.

Objects:
- Every item resting on the desk (or on the ground when there is no desk), or on another such item. Large items such as a laptop or monitor count.

Exclude:
- The floor, walls, and ceiling, and everything standing on the floor other than the desk (cabinets, shelves, chairs, fans, stands).
- Cables, wires, and other line-like details: seams, gaps, edges, grooves, printed lines, and patterns.
- The desk's frame details: braces, folding mechanisms, rims, and lips.

Parts:
- A single shape is enough when it matches the look (a box-shaped phone, a pen as one capsule).
- Use `parts` only when one shape would get the object's main form or main colors wrong: a mug is a cylinder plus a handle box, an open laptop is a base box plus a tilted screen box, a pen with a white cap is a colored capsule plus a white cylinder.
- Keep it to the few parts that make up the main form, at most 4. No thin decorative parts: no lines, seams, gaps, buttons, logos, or labels.
- Each part has its own shape, dimensions, offset from the item's origin, rotation (`rpy_deg`), and appearance.
- With parts, the item's `position` is the center of its footprint on the surface it rests on (z = that surface), and part offsets are measured from there in the item's own frame (x along `yaw_deg` 0, z up).

Ready-made models:
- For every object, write `asset_query`: a short English search phrase for a 3D model library that names the kind of object, ending with its noun, plus the details that set its look ("black wireless computer mouse", "white ceramic coffee mug", "red felt-tip pen with cap").
- The library is searched with your phrase, and you will then be shown the nearest candidates and choose the one that is the same kind of object, or none; with none, the object is shown with your shapes. So name the object as it is, not as something similar.
- Still give the object's `shape`, `dimensions`, and `size_mm` as usual: a found model is scaled to them, and the shapes remain the object's collision body.

Appearance:
- Give `color_rgb`, `roughness`, and `metallic` for every surface (item or part). The photos are white-balanced and the simulator light is neutral white, so give colors as they appear in the photos.
- When a surface has a visible texture (wood grain, stone, metal finish, cardboard), also set `texture` to the closest name from this list, else `null`:
{texture_list}

Physical properties:
- Give mass, friction, and restitution for the desk and every object, derived from what the object is and how big it looks.
- Record the reasoning behind every physical value in `physics_rationale`.

Constraints:
- Every object rests on the desk, the ground, or another object, with no overlaps between items.
- For a single-shape object, `position` is the center of the shape.

Focus object:
- The photos were taken around one object on the desk: the one the person was photographing. Decide which object it is and give its `id` in `focus_object.id`, with a short reason (where it sits in the frames, what every view keeps in sight).
- Still rebuild every object in the scene; the focus object is only named, not treated differently.

Spatial relations:
- Choose one object as the anchor and set `relations.anchor` to its `id`. Pick the object you can locate most reliably.
- In `relations.items`, describe every other object relative to the anchor, and give what the anchor itself rests on. Also give `on_top_of` for every object that rests on another object.
- Judge left, right, in front, and behind as seen in the front image (`front_color`; the image list at the end names each image).
- Use only these relations: `left_of`, `right_of`, `in_front_of` (closer to the front camera), `behind`, `on_top_of` (a rests on b), `inside`.
- An object may have more than one relation to the anchor (for example both `left_of` and `behind`); give each as its own entry.

Return JSON with this structure:

```json
{
  "environment": {
    "desk": {
      "identified_as": "",
      "top_shape": "rectangle",
      "size": {"x": 0.0, "y": 0.0},
      "corner_radius": 0.0,
      "top_thickness": 0.0,
      "height_from_floor": 0.0,
      "legs": [
        {"shape": "cylinder", "dimensions": {"radius": 0.0, "height": 0.0}, "offset": [0.0, 0.0, 0.0], "rpy_deg": [0.0, 0.0, 0.0]}
      ],
      "leg_appearance": {"color_rgb": [0.0, 0.0, 0.0], "roughness": 0.0, "metallic": 0.0, "texture": null},
      "color_rgb": [0.0, 0.0, 0.0],
      "material": "",
      "roughness": 0.0,
      "metallic": 0.0,
      "texture": null,
      "static_friction": 0.0,
      "dynamic_friction": 0.0,
      "restitution": 0.0,
      "confidence": 0.0,
      "physics_rationale": {"friction": "", "restitution": ""}
    },
    "lighting": {
      "ambient": 0.0,
      "key_light": {"azimuth_deg": 0.0, "elevation_deg": 0.0, "strength": 0.0}
    },
    "cameras": {
      "front_color": {"position": [0.0, 0.0, 0.0], "look_at": [0.0, 0.0, 0.0]}
    }
  },
  "objects": [
    {
      "id": "red_cube_1",
      "identified_as": "",
      "shape": "box",
      "dimensions": {"x": 0.0, "y": 0.0, "z": 0.0},
      "parts": null,
      "asset_query": "",
      "size_mm": {"length": 0, "width": 0, "height": 0},
      "position": [0.0, 0.0, 0.0],
      "yaw_deg": 0.0,
      "lying": false,
      "boxes_2d": {"front_color": [0, 0, 0, 0]},
      "color_rgb": [0.0, 0.0, 0.0],
      "material": "",
      "roughness": 0.0,
      "metallic": 0.0,
      "texture": null,
      "mass_kg": 0.0,
      "static_friction": 0.0,
      "dynamic_friction": 0.0,
      "restitution": 0.0,
      "is_static": false,
      "confidence": 0.0,
      "physics_rationale": {"mass": "", "friction": "", "restitution": ""}
    },
    {
      "id": "white_mug_1",
      "identified_as": "white ceramic coffee mug",
      "shape": "parts",
      "parts": [
        {"name": "body", "shape": "cylinder", "dimensions": {"radius": 0.04, "height": 0.095}, "offset": [0.0, 0.0, 0.0475], "rpy_deg": [0.0, 0.0, 0.0], "color_rgb": [0.95, 0.95, 0.93], "roughness": 0.2, "metallic": 0.0, "texture": null},
        {"name": "handle", "shape": "box", "dimensions": {"x": 0.03, "y": 0.012, "z": 0.06}, "offset": [0.052, 0.0, 0.05], "rpy_deg": [0.0, 0.0, 0.0], "color_rgb": [0.95, 0.95, 0.93], "roughness": 0.2, "metallic": 0.0, "texture": null}
      ],
      "size_mm": {"length": 110, "width": 80, "height": 95},
      "position": [0.0, 0.0, 0.0],
      "yaw_deg": 0.0,
      "...": "every other object field, as above"
    }
  ],
  "focus_object": {"id": "red_cube_1", "reason": ""},
  "relations": {
    "anchor": "red_cube_1",
    "items": [
      {"a": "red_cube_1", "rel": "on_top_of", "b": "desk"},
      {"a": "white_mug_1", "rel": "right_of", "b": "red_cube_1"}
    ]
  }
}
```

Field definitions:
- `identified_as`: what the item most likely is in the real world, as specific as you can tell.
- `shape`: `box`, `cylinder`, `sphere`, `capsule`, or `parts`. With `parts`, leave `dimensions` out and give the parts.
- `dimensions`: `box` uses x, y, z. `cylinder` uses radius and height. `capsule` uses radius and height, where height is the full end-to-end length including both rounded ends. `sphere` uses radius. For a cylinder or capsule, `height` is always the length along its own axis.
- `parts[]`: `name`, `shape` (box, cylinder, sphere, capsule), `dimensions`, `offset` [x, y, z] of the part's center from the item's origin, `rpy_deg` [roll, pitch, yaw] of the part in the item's frame (cylinders and capsules point along their own z before rotation, so pitch 90 lays one along x), and `color_rgb`, `roughness`, `metallic`, `texture`.
- `size_mm`: the item's whole bounding box as it rests, in millimeters, as integers: `length` along the item's own x axis, `width` along its y axis, `height` straight up from the surface. It must agree with the shape or parts (a lying cylinder has length = its height, width = height = 2 x radius).
- `position`: meters. A single shape: its center. Parts: the center of the footprint on the supporting surface.
- `yaw_deg`: rotation of the whole item about +z. For a lying cylinder or capsule, yaw 0 puts its axis along +x.
- `lying`: only for a single `cylinder` or `capsule`; true when its axis is horizontal, false when it stands upright. The `position` z is then its radius. Use parts with `rpy_deg` for any other tilt.
- `boxes_2d`: the item's 2D bounding box in each color image where it is visible, keyed by the image name from the image list (for example `front_color`). Each box is [ymin, xmin, ymax, xmax], normalized to 0-1000 over the image height and width. Leave out images where it is not visible; depth images get no boxes.
- `color_rgb`: 0 to 1 per channel, the dominant color of the surface as it appears in the photo (sRGB). For an item with parts, the item-level color is its overall dominant color.
- `roughness`, `metallic`: how the surface looks, 0 to 1. Roughness 0 is mirror-glossy (glass, polished phone back), 1 is fully matte (paper, rubber). Metallic is 1 for bare metal, 0 for everything else (plastic, painted, wood).
- `asset_query`: the search phrase described in Ready-made models. It changes only how the object looks.
- `texture`: a name from the texture list in Appearance, or `null` for a plain surface (most plastics and painted surfaces). A texture brings its own fixed colors and replaces `color_rgb`, so pick one only when its color also matches (white plastic stays `null`).
- `is_static`: true for objects that should not move (heavy items, fixtures).
- `confidence`: 0 to 1, for the item's identity, position, and size together.
- `desk.top_shape`: the outline of the desk top seen from above: `rectangle`, `rounded_rectangle` (corners rounded by `corner_radius`), `stadium` (two straight sides joined by two half-circles), `ellipse`, or `circle`. `size` is the full extent along x and y.
- `desk.legs`: one vertical post per place where the desk stands on the floor, as parts in the desk frame (origin at the center of the desk top; the floor is at z = -height_from_floor), each a cylinder or box from the floor to the underside of the top with `rpy_deg` [0, 0, 0]. Approximate a folding frame or panel legs by these posts. `leg_appearance` is their color and finish.
- `lighting.ambient`: 0 to 1, how bright the soft, shadowless light filling the room is.
- `lighting.key_light`: the main light that casts the visible shadows: `azimuth_deg` is the direction it comes from, measured from +x toward +y; `elevation_deg` is its height above the horizon (90 is straight above); `strength` 0 to 1 is how strong and sharp the shadows are (0 for no visible shadows).
- `environment.cameras`: for each color image, the camera's `position` and the point it looks at (`look_at`), in meters in the scene frame. The front camera defines the frame, so it is always on the -x side of the desk looking toward +x: its x is negative and its y is close to 0.
- `focus_object`: `id` is one object `id` from `objects`; `reason` is one short sentence.
- `relations.items`: `a` and `b` are object `id`s, or `desk` for the desk (or `ground` when there is no desk). Entries other than `on_top_of` have the anchor as `b`.
- `physics_rationale`: one short sentence per property, e.g. "an empty 350 ml aluminum can typically weighs about 0.015 kg".
