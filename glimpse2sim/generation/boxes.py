"""Draw the 2D boxes a model gave for each object onto the images it was sent."""

import logging
from pathlib import Path

import cv2

log = logging.getLogger(__name__)

# Distinct BGR colors, cycled over objects so the same object has the same color in every image.
COLORS = [
    (60, 76, 231), (113, 204, 46), (219, 152, 52), (15, 196, 241), (182, 89, 155),
    (34, 126, 230), (160, 160, 22), (133, 160, 255), (80, 62, 44), (0, 84, 211),
]


def draw_boxes(scene: dict, image_paths: list[Path], out_dir: Path) -> list[Path]:
    """Write <image>_boxes.png for every sent image that has at least one box. Return the written paths.

    Boxes are `objects[].boxes_2d[<image name>] = [ymin, xmin, ymax, xmax]`, normalized to 0-1000.
    """
    images = {p.name.removesuffix(".png"): p for p in image_paths}
    drawn: dict[str, list[tuple[str, list, tuple]]] = {}
    for i, obj in enumerate(scene.get("objects", [])):
        for name, box in (obj.get("boxes_2d") or {}).items():
            if name not in images:
                log.warning("%s: box for unknown image %r", obj.get("id"), name)
                continue
            if not (isinstance(box, list) and len(box) == 4):
                log.warning("%s: malformed box %r", obj.get("id"), box)
                continue
            drawn.setdefault(name, []).append((str(obj.get("id", i)), box, COLORS[i % len(COLORS)]))

    written = []
    for name, boxes in drawn.items():
        img = cv2.imread(str(images[name]))
        h, w = img.shape[:2]
        thick = max(2, round(w / 600))
        scale = w / 2400
        for label, (y0, x0, y1, x1), color in boxes:
            p0 = (round(x0 / 1000 * w), round(y0 / 1000 * h))
            p1 = (round(x1 / 1000 * w), round(y1 / 1000 * h))
            cv2.rectangle(img, p0, p1, color, thick)
        # Labels go on after every box, inside each box's top-left corner, so boxes never cover them.
        for label, (y0, x0, _, _), color in boxes:
            p0 = (round(x0 / 1000 * w), round(y0 / 1000 * h))
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            cv2.rectangle(img, p0, (p0[0] + tw + 4, p0[1] + th + base + 4), color, -1)
            cv2.putText(img, label, (p0[0] + 2, p0[1] + th + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1,
                        cv2.LINE_AA)
        out = out_dir / f"{name}_boxes.png"
        cv2.imwrite(str(out), img)
        written.append(out)
    return written
