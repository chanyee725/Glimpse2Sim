"""Object placements measured from a desk corner, so they can be checked against tape-measured ground truth.

Reference corner: the desk's back-left corner in the robot frame (minimum x, maximum y),
i.e. behind the robot base on its left side. Distances are positive inward:
    from_back_mm  = (x - x_min) * 1000   along +x, away from the back edge
    from_left_mm  = (y_max - y) * 1000   along -y, away from the left edge
    height_mm     = z * 1000             object center above the desk surface
"""


def desk_corner(desk: dict) -> tuple[float, float]:
    """(x_min, y_max) of the desk in the robot frame."""
    cx, cy = desk.get("center", [0.0, 0.0])
    return cx - desk["size"]["x"] / 2, cy + desk["size"]["y"] / 2


def placements(env: dict | None, objects: list[dict]) -> dict | None:
    """Per-object offsets from the desk corner in mm, or None when there is no desk."""
    desk = (env or {}).get("desk")
    if not desk:
        return None
    x0, y0 = desk_corner(desk)
    rows = []
    for obj in objects:
        try:
            x, y, z = (float(v) for v in obj["position"])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append({
            "id": obj.get("id"),
            "identified_as": obj.get("identified_as"),
            "from_back_mm": round((x - x0) * 1000),
            "from_left_mm": round((y0 - y) * 1000),
            "height_mm": round(z * 1000),
            "size_mm": obj.get("size_mm"),
        })
    return {
        "reference": "desk back-left corner (min x, max y in the robot frame)",
        "desk_corner_m": [round(x0, 4), round(y0, 4)],
        "robot_base_from_corner_mm": {"from_back_mm": round(-x0 * 1000), "from_left_mm": round(y0 * 1000)},
        "objects": rows,
    }
