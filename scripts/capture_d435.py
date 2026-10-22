"""Capture N aligned RGB-D frames from the D435.

uv run scripts/capture_d435.py --scene scene01 -n 5
"""

import argparse
import time
from pathlib import Path

from glimpse2sim.capture import D435Camera
from glimpse2sim.config import ROOT, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("-n", "--num-frames", type=int, default=5)
    parser.add_argument("--interval", type=float, default=0.5, help="seconds between frames")
    args = parser.parse_args()

    cfg = load_config("cameras")["d435"]
    out_dir = ROOT / "data" / "raw" / args.scene / "env"
    vis_range = tuple(cfg["depth_vis_range_m"])

    with D435Camera(cfg) as cam:
        for i in range(args.num_frames):
            paths = cam.capture().save(out_dir, f"{i:03d}", vis_range)
            print(f"[{i + 1}/{args.num_frames}] {paths['color'].name}")
            time.sleep(args.interval)
    print(f"saved to {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
