import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class RGBDFrame:
    color: np.ndarray  # (H, W, 3) uint8, BGR
    depth: np.ndarray  # (H, W) uint16, raw depth units
    depth_scale: float  # meters per depth unit
    intrinsics: Intrinsics
    timestamp_ms: float

    @property
    def depth_m(self) -> np.ndarray:
        return self.depth.astype(np.float32) * self.depth_scale

    def depth_colormap(self, vis_range_m: tuple[float, float]) -> np.ndarray:
        """Colorize depth with a fixed range so captures are comparable. Invalid pixels are black."""
        near, far = vis_range_m
        d = self.depth_m
        norm = np.clip((d - near) / (far - near), 0.0, 1.0)
        vis = cv2.applyColorMap((255 * (1.0 - norm)).astype(np.uint8), cv2.COLORMAP_TURBO)
        vis[d <= 0] = 0
        return vis

    def save(self, out_dir: Path, name: str, vis_range_m: tuple[float, float]) -> dict[str, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "color": out_dir / f"{name}_color.png",
            "depth_raw": out_dir / f"{name}_depth.png",  # 16-bit PNG, lossless
            "depth_vis": out_dir / f"{name}_depth_vis.png",
            "meta": out_dir / f"{name}_meta.json",
        }
        cv2.imwrite(str(paths["color"]), self.color)
        cv2.imwrite(str(paths["depth_raw"]), self.depth)
        cv2.imwrite(str(paths["depth_vis"]), self.depth_colormap(vis_range_m))
        meta = {
            "depth_scale": self.depth_scale,
            "intrinsics": self.intrinsics.__dict__,
            "timestamp_ms": self.timestamp_ms,
            "depth_vis_range_m": list(vis_range_m),
        }
        paths["meta"].write_text(json.dumps(meta, indent=2))
        return paths


class D435Camera:
    """Aligned RGB-D capture from a RealSense D435.

    with D435Camera(cfg) as cam:
        frame = cam.capture()
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.pipeline = rs.pipeline()
        self.align = rs.align(rs.stream.color) if cfg.get("align_to_color", True) else None
        self.depth_scale = 0.0
        self.intrinsics: Intrinsics | None = None

    def start(self) -> None:
        c, d = self.cfg["color"], self.cfg["depth"]
        rs_cfg = rs.config()
        if self.cfg.get("serial"):
            rs_cfg.enable_device(self.cfg["serial"])
        rs_cfg.enable_stream(rs.stream.color, c["width"], c["height"], rs.format.bgr8, c["fps"])
        rs_cfg.enable_stream(rs.stream.depth, d["width"], d["height"], rs.format.z16, d["fps"])
        profile = self.pipeline.start(rs_cfg)

        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        # After alignment, depth shares the color intrinsics.
        k = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics = Intrinsics(k.width, k.height, k.fx, k.fy, k.ppx, k.ppy)

        for _ in range(self.cfg.get("warmup_frames", 30)):
            self.pipeline.wait_for_frames()

    def stop(self) -> None:
        self.pipeline.stop()

    def capture(self, timeout_ms: int = 5000) -> RGBDFrame:
        frames = self.pipeline.wait_for_frames(timeout_ms)
        if self.align is not None:
            frames = self.align.process(frames)
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not depth:
            raise RuntimeError("D435 returned an incomplete frameset")
        return RGBDFrame(
            color=np.asanyarray(color.get_data()).copy(),
            depth=np.asanyarray(depth.get_data()).copy(),
            depth_scale=self.depth_scale,
            intrinsics=self.intrinsics,
            timestamp_ms=frames.get_timestamp(),
        )

    def __enter__(self) -> "D435Camera":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
