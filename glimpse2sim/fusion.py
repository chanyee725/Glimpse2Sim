"""Fuse several RGB-D captures of one scene into a single depth map (and mesh) in a reference view.

1. Camera poses from the ChArUco board on the table: PnP on the corners of every detected ArUco marker
   (Garrido-Jurado et al. 2014). The pose comes from the color image only.
2. Per-frame depth correction against the board. The D435 depth of each view is off from the board by a
   view-dependent scale and tilt (1-3 % and up to ~3 deg measured), so rigid alignment alone cannot make the views
   agree. A depth multiplier k(x, y) = a + b x + c y over normalized image coordinates (the per-pixel multiplier
   model of CLAMS, Teichman et al. RSS 2013, reduced to a plane) is fitted so that k * measured depth matches the
   depth of the board plane; frames that see only a small part of the board get a scale only (b = c = 0).
3. Optional refinement: pairwise multi-scale colored ICP (Park, Zhou, Koltun, ICCV 2017) from the board poses and
   a pose graph over all pairs with line processes (Choi, Zhou, Koltun, CVPR 2015).
4. TSDF fusion of all frames (Curless & Levoy 1996; KinectFusion, Newcombe et al. 2011).
5. Ray casting the fused surface into the reference camera gives one depth image on its color grid.

Writes <scene>/fused/: depth.png (16-bit, same depth_scale as the captures), depth_vis.png, mesh.ply, cloud.ply
and fusion.json (poses as reference-from-camera, per-frame corrections, diagnostics, parameters).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c
import rerun as rr
import rerun.blueprint as rrb

from .capture import RGBDFrame
from .capture.d435 import Intrinsics
from .config import load_config

ICP_VOXELS = (0.02, 0.01, 0.005)  # m, coarse to fine
ICP_ITERATIONS = (50, 30, 14)
MIN_EDGE_OVERLAP = 0.1  # pairs with fewer points within 1 cm of each other stay out of the pose graph
MIN_PLANE_FIT_PX = 50000  # board pixels needed to fit the tilt terms of the depth multiplier, not just its scale


@dataclass
class Frame:
    name: str
    color: np.ndarray  # (H, W, 3) uint8, RGB
    depth: np.ndarray  # (H, W) uint16
    depth_scale: float
    K: np.ndarray  # 3x3
    meta: dict

    @property
    def size(self) -> tuple[int, int]:
        return self.depth.shape[1], self.depth.shape[0]

    def intrinsic(self) -> o3d.camera.PinholeCameraIntrinsic:
        w, h = self.size
        return o3d.camera.PinholeCameraIntrinsic(w, h, self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2])

    def rgbd(self, max_depth: float) -> o3d.geometry.RGBDImage:
        return o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(self.color), o3d.geometry.Image(self.depth),
            depth_scale=1.0 / self.depth_scale, depth_trunc=max_depth, convert_rgb_to_intensity=False,
        )

    def normalized_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """(x, y) = ((u - cx) / fx, (v - cy) / fy) for every pixel."""
        w, h = self.size
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        return (u - self.K[0, 2]) / self.K[0, 0], (v - self.K[1, 2]) / self.K[1, 1]


def load_frame(scene_dir: Path, name: str) -> Frame:
    meta = json.loads((scene_dir / f"{name}_meta.json").read_text())
    k = meta["intrinsics"]
    K = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]])
    color = cv2.cvtColor(cv2.imread(str(scene_dir / f"{name}_color.png")), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(str(scene_dir / f"{name}_depth.png"), cv2.IMREAD_UNCHANGED)
    return Frame(name, color, depth, meta["depth_scale"], K, meta)


class Board:
    """The ChArUco board in configs/charuco.yaml; only its ArUco markers are used for the pose."""

    def __init__(self, cfg: dict):
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg["dictionary"]))
        self.board = cv2.aruco.CharucoBoard(tuple(cfg["squares"]), cfg["square_m"], cfg["marker_m"], dictionary)
        self.markers = {int(i): np.asarray(p, np.float32)
                        for i, p in zip(self.board.getIds().ravel(), self.board.getObjPoints())}
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(dictionary, params)
        sx, sy = self.board.getChessboardSize()
        side = self.board.getSquareLength()
        self.outline = np.array([[0, 0, 0], [sx * side, 0, 0], [sx * side, sy * side, 0], [0, sy * side, 0]],
                                np.float32)

    def detect(self, gray: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        """Marker corners and ids; far or oblique markers are searched again in upsampled images."""
        best = ([], np.empty(0, int))
        for scale in (1, 2, 3):
            img = gray if scale == 1 else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            corners, ids, _ = self.detector.detectMarkers(img)
            if ids is not None and len(ids) > len(best[1]):
                best = ([c[0] / scale for c in corners], ids.ravel())
        return best

    def pose(self, frame: Frame) -> tuple[np.ndarray, dict]:
        """Camera-from-board transform by PnP on the marker corners (the color stream has ~zero distortion)."""
        corners, ids = self.detect(cv2.cvtColor(frame.color, cv2.COLOR_RGB2GRAY))
        if len(ids) < 2:
            raise RuntimeError(f"frame {frame.name}: {len(ids)} board marker(s) found, need at least 2")
        obj = np.vstack([self.markers[int(i)] for i in ids])
        img = np.vstack(corners).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(obj, img, frame.K, None)
        if not ok:
            raise RuntimeError(f"frame {frame.name}: PnP on the board markers failed")
        proj, _ = cv2.projectPoints(obj, rvec, tvec, frame.K, None)
        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues(rvec)[0]
        T[:3, 3] = tvec.ravel()
        return T, {"markers": len(ids), "reprojection_px": float(np.sqrt(np.mean((proj.reshape(-1, 2) - img) ** 2)))}

    def mask(self, frame: Frame, cam_from_board: np.ndarray, margin_px: int = 7) -> np.ndarray:
        """Pixels inside the board outline (shrunk by margin_px), from the board pose."""
        rvec = cv2.Rodrigues(cam_from_board[:3, :3])[0]
        proj, _ = cv2.projectPoints(self.outline, rvec, cam_from_board[:3, 3], frame.K, None)
        mask = np.zeros(frame.depth.shape, np.uint8)
        cv2.fillConvexPoly(mask, proj.reshape(-1, 2).astype(np.int32), 1)
        return cv2.erode(mask, np.ones((2 * margin_px + 1, 2 * margin_px + 1), np.uint8)).astype(bool)


def board_depth(frame: Frame, cam_from_board: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Depth of the board plane along the rays through normalized coordinates (x, y)."""
    n, p0 = cam_from_board[:3, 2], cam_from_board[:3, 3]
    return (n @ p0) / (n[0] * x + n[1] * y + n[2])


def depth_correction(frame: Frame, board: Board, cam_from_board: np.ndarray) -> tuple[np.ndarray, dict]:
    """Fit the depth multiplier k = a + b x + c y on the board; returns (coefficients, diagnostics)."""
    x_all, y_all = frame.normalized_grid()
    on_board = board.mask(frame, cam_from_board) & (frame.depth > 0)
    x, y = x_all[on_board], y_all[on_board]
    measured = frame.depth[on_board] * frame.depth_scale
    target = board_depth(frame, cam_from_board, x, y)
    plane_fit = on_board.sum() >= MIN_PLANE_FIT_PX
    A = np.c_[np.ones_like(x), x, y] if plane_fit else np.ones((len(x), 1))
    coef = np.linalg.lstsq(A, target / measured, rcond=None)[0]
    coef = np.r_[coef, np.zeros(3 - len(coef))]
    corrected = measured * (coef[0] + coef[1] * x + coef[2] * y)
    # Tilt of the measured board surface against the board plane from the color pose.
    pts = np.c_[x * measured, y * measured, measured]
    normal = np.linalg.svd(pts - pts.mean(0), full_matrices=False)[2][2]
    tilt = np.degrees(np.arccos(min(1.0, abs(normal @ cam_from_board[:3, 2]))))
    return coef, {
        "board_px": int(on_board.sum()),
        "model": "a + b x + c y" if plane_fit else "a",
        "coefficients": coef.tolist(),
        "bias_mm": float(np.median(measured - target) * 1000),
        "tilt_deg": float(tilt),
        "rms_vs_board_mm": float((measured - target).std() * 1000),
        "rms_vs_board_corrected_mm": float((corrected - target).std() * 1000),
    }


def apply_correction(frame: Frame, coef: np.ndarray) -> None:
    x, y = frame.normalized_grid()
    k = coef[0] + coef[1] * x + coef[2] * y
    frame.depth = np.clip(np.round(frame.depth * k), 0, 65535).astype(np.uint16)


def icp_cloud(frame: Frame, max_depth: float, voxel: float) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(frame.rgbd(max_depth), frame.intrinsic())
    cloud = cloud.voxel_down_sample(voxel)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2 * voxel, max_nn=30))
    return cloud


def overlap(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, T: np.ndarray,
            dist: float = 0.01) -> float:
    """Fraction of source points within dist of the target after transforming the source by T."""
    moved = o3d.geometry.PointCloud(source).transform(T)
    return float((np.asarray(moved.compute_point_cloud_distance(target)) < dist).mean())


def colored_icp(clouds: dict, i: int, j: int, init: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Multi-scale colored ICP of frame i onto frame j; returns (T_j_i, information matrix)."""
    reg = o3d.pipelines.registration
    T = init
    for voxel, iterations in zip(ICP_VOXELS, ICP_ITERATIONS):
        T = reg.registration_colored_icp(
            clouds[voxel][i], clouds[voxel][j], voxel, T, reg.TransformationEstimationForColoredICP(),
            reg.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iterations),
        ).transformation
    fine = ICP_VOXELS[-1]
    return T, reg.get_information_matrix_from_point_clouds(clouds[fine][i], clouds[fine][j], fine, T)


def rotation_deg(T: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1))))


def refine_poses(frames: list[Frame], poses: list[np.ndarray], ref: int, max_depth: float) -> list[np.ndarray]:
    """Colored ICP over every overlapping pair, then pose graph optimization with the reference fixed."""
    clouds = {v: [icp_cloud(f, max_depth, v) for f in frames] for v in ICP_VOXELS}
    reg = o3d.pipelines.registration
    graph = reg.PoseGraph()
    for T in poses:
        graph.nodes.append(reg.PoseGraphNode(T))
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            init = np.linalg.inv(poses[j]) @ poses[i]
            if overlap(clouds[0.01][i], clouds[0.01][j], init) < MIN_EDGE_OVERLAP:
                continue
            T, info = colored_icp(clouds, i, j, init)
            delta = np.linalg.inv(init) @ T
            print(f"  icp {frames[i].name}->{frames[j].name}: moved {np.linalg.norm(delta[:3, 3]) * 1000:.1f} mm / "
                  f"{rotation_deg(delta):.2f} deg from the board poses")
            graph.edges.append(reg.PoseGraphEdge(i, j, T, info, uncertain=True))
    reg.global_optimization(
        graph, reg.GlobalOptimizationLevenbergMarquardt(), reg.GlobalOptimizationConvergenceCriteria(),
        reg.GlobalOptimizationOption(max_correspondence_distance=ICP_VOXELS[-1], edge_prune_threshold=0.25,
                                     reference_node=ref),
    )
    return [np.linalg.inv(graph.nodes[ref].pose) @ n.pose for n in graph.nodes]


def tsdf_fuse(frames: list[Frame], poses: list[np.ndarray], voxel: float,
              max_depth: float) -> o3d.t.geometry.VoxelBlockGrid:
    """Integrate every frame into one TSDF voxel grid in the reference camera frame."""
    grid = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"), attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)), voxel_size=voxel, block_resolution=16, block_count=100000)
    for f, T in zip(frames, poses):
        K = o3c.Tensor(f.K, o3c.float64)
        extrinsic = o3c.Tensor(np.linalg.inv(T), o3c.float64)  # camera-from-reference
        depth = o3d.t.geometry.Image(o3c.Tensor(f.depth))
        color = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(f.color)))
        blocks = grid.compute_unique_block_coordinates(depth, K, extrinsic, 1.0 / f.depth_scale, max_depth,
                                                       trunc_voxel_multiplier=4.0)
        grid.integrate(blocks, depth, color, K, K, extrinsic, 1.0 / f.depth_scale, max_depth,
                       trunc_voxel_multiplier=4.0)
    return grid


def raycast_depth(mesh: o3d.geometry.TriangleMesh, frame: Frame) -> np.ndarray:
    """Depth (m, along the optical axis) of the fused mesh seen from the frame's camera; 0 where no surface.

    VoxelBlockGrid.ray_cast is not used: in Open3D 0.20 it returns a few stray far hits (~0.1 % of pixels, up to
    0.3 m off) that change from run to run; ray casting the extracted mesh is exact and repeatable.
    """
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    w, h = frame.size
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        o3c.Tensor(frame.K, o3c.float64), o3c.Tensor(np.eye(4), o3c.float64), w, h)
    t_hit = scene.cast_rays(rays)["t_hit"].numpy()
    depth = t_hit * rays.numpy()[:, :, 5]  # hit = origin + t * direction, so z = t * direction_z
    depth[~np.isfinite(depth)] = 0.0
    return depth


def board_error(depth_m: np.ndarray, frame: Frame, board: Board, cam_from_board: np.ndarray) -> dict:
    """Depth against the board plane (from the color pose) over the board: a flat, independent reference."""
    x, y = frame.normalized_grid()
    m = board.mask(frame, cam_from_board) & (depth_m > 0)
    err = depth_m[m] - board_depth(frame, cam_from_board, x[m], y[m])
    return {"bias_mm": float(np.median(err) * 1000), "rms_mm": float(err.std() * 1000), "px": int(m.sum())}


def show_fusion(out_dir: Path, frames: list[Frame], poses: list[np.ndarray], reference: str,
                mesh: o3d.geometry.TriangleMesh, depth_raw: np.ndarray) -> None:
    """Rerun: fused mesh and every camera (pose, frustum, photo) in the reference camera frame, eye at the reference
    camera; the fused depth next to the reference photo."""
    rr.init(f"glimpse2sim/{out_dir.parent.name}/fused", spawn=True)
    rf = frames[[f.name for f in frames].index(reference)]
    look = float(np.median(depth_raw[depth_raw > 0])) * rf.depth_scale
    eye = rrb.EyeControls3D(kind=rrb.Eye3DKind.Orbital, position=[0, 0, 0], look_target=[0, 0, look],
                            eye_up=[0, -1, 0])
    # Images under a camera would be drawn on its frustum, right in front of the eye; keep them to the 2D views.
    hidden = [f"- /world/cameras/{f.name}/{image}" for f in frames for image in ("photo", "fused_depth")]
    rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(
        rrb.Spatial3DView(origin="world", contents=["+ /world/**", *hidden], eye_controls=eye),
        rrb.Vertical(rrb.Spatial2DView(name="fused depth", origin="world/cameras/" + reference,
                                       contents=["$origin/fused_depth"]),
                     rrb.Spatial2DView(name="reference photo", origin="world/cameras/" + reference,
                                       contents=["$origin/photo"])),
        column_shares=[3, 1],
    )))
    rr.log("world", rr.ViewCoordinates.RDF, static=True)
    rr.log("world/mesh", rr.Mesh3D(
        vertex_positions=np.asarray(mesh.vertices), triangle_indices=np.asarray(mesh.triangles),
        vertex_colors=np.asarray(mesh.vertex_colors), vertex_normals=np.asarray(mesh.vertex_normals)), static=True)
    for f, T in zip(frames, poses):
        cam = f"world/cameras/{f.name}"
        w, h = f.size
        rr.log(cam, rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]), rr.TransformAxes3D(0.05), static=True)
        rr.log(cam, rr.Pinhole(image_from_camera=f.K, width=w, height=h, image_plane_distance=0.1), static=True)
        rr.log(f"{cam}/photo", rr.Image(f.color).compress(), static=True)
    rr.log(f"world/cameras/{reference}/fused_depth", rr.DepthImage(depth_raw, meter=1.0 / rf.depth_scale), static=True)


def fuse_scene(scene_dir: Path, reference: str, max_depth: float, voxel: float, refine: bool,
               only: list[str] | None = None, view: bool = False) -> Path:
    names = sorted(p.name.removesuffix("_meta.json") for p in scene_dir.glob("*_meta.json"))
    if only:
        missing = sorted(set(only) - set(names))
        if missing:
            raise FileNotFoundError(f"no frame(s) {', '.join(missing)} in {scene_dir} (have {', '.join(names)})")
        names = [n for n in names if n in only]
    if reference not in names:
        raise FileNotFoundError(f"reference frame {reference} is not among the fused frames ({', '.join(names)})")
    frames = [load_frame(scene_dir, n) for n in names]
    ref = names.index(reference)
    rf = frames[ref]
    raw_ref_depth = rf.depth.astype(np.float64) * rf.depth_scale
    board = Board(load_config("charuco"))

    # 1-2. Board poses and depth corrections.
    cam_from_board, per_frame = [], {}
    for f in frames:
        T, pose_info = board.pose(f)
        coef, corr = depth_correction(f, board, T)
        apply_correction(f, coef)
        cam_from_board.append(T)
        per_frame[f.name] = {**pose_info, "depth_correction": corr}
        print(f"{f.name}: {pose_info['markers']} markers, reprojection {pose_info['reprojection_px']:.2f} px; depth vs "
              f"board: bias {corr['bias_mm']:+.1f} mm, tilt {corr['tilt_deg']:.2f} deg, RMS "
              f"{corr['rms_vs_board_mm']:.1f} -> {corr['rms_vs_board_corrected_mm']:.1f} mm ({corr['model']})")
    poses = [cam_from_board[ref] @ np.linalg.inv(T) for T in cam_from_board]  # reference-from-camera

    # 3. Optional ICP + pose graph refinement.
    if refine:
        poses = refine_poses(frames, poses, ref, max_depth)

    # 4. TSDF fusion in the reference camera frame. Surfaces seen by a single frame are kept: the extraction
    # threshold is strict (weight > threshold) and one frame adds a weight of 1.
    grid = tsdf_fuse(frames, poses, voxel, max_depth)
    mesh = grid.extract_triangle_mesh(weight_threshold=0.5).to_legacy()
    mesh.compute_vertex_normals()

    # 5. One depth image for the reference view.
    depth_raw = np.clip(np.round(raycast_depth(mesh, rf) / rf.depth_scale), 0, 65535).astype(np.uint16)

    clouds = [icp_cloud(f, max_depth, 0.01) for f in frames]
    pair_overlap = {f"{names[i]}-{names[j]}": overlap(clouds[i], clouds[j], np.linalg.inv(poses[j]) @ poses[i])
                    for i in range(len(frames)) for j in range(i + 1, len(frames))}
    evaluation = {
        "reference_raw": board_error(raw_ref_depth, rf, board, cam_from_board[ref]),
        "reference_corrected": board_error(rf.depth * rf.depth_scale, rf, board, cam_from_board[ref]),
        "fused": board_error(depth_raw * rf.depth_scale, rf, board, cam_from_board[ref]),
        "valid_px_reference": float((rf.depth > 0).mean()),
        "valid_px_fused": float((depth_raw > 0).mean()),
        "pair_overlap_1cm": pair_overlap,
    }

    out_dir = scene_dir / "fused"
    out_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(out_dir / "depth.png"), depth_raw)
    k = rf.meta["intrinsics"]
    fused = RGBDFrame(rf.color[:, :, ::-1], depth_raw, rf.depth_scale, Intrinsics(**k), rf.meta["timestamp_ms"])
    cv2.imwrite(str(out_dir / "depth_vis.png"), fused.depth_colormap(tuple(rf.meta["depth_vis_range_m"])))
    o3d.io.write_triangle_mesh(str(out_dir / "mesh.ply"), mesh)
    o3d.io.write_point_cloud(str(out_dir / "cloud.ply"), grid.extract_point_cloud(weight_threshold=0.5).to_legacy())
    (out_dir / "fusion.json").write_text(json.dumps({
        "reference": reference,
        "frames": names,
        "depth_scale": rf.depth_scale,
        "intrinsics": k,
        "params": {"max_depth_m": max_depth, "tsdf_voxel_m": voxel, "tsdf_trunc_m": 4 * voxel, "icp_refine": refine,
                   "icp_voxels_m": ICP_VOXELS, "icp_iterations": ICP_ITERATIONS,
                   "min_edge_overlap": MIN_EDGE_OVERLAP, "min_plane_fit_px": MIN_PLANE_FIT_PX,
                   "board": load_config("charuco")},
        "per_frame": per_frame,
        "evaluation": evaluation,
        "ref_from_camera": {n: T.tolist() for n, T in zip(names, poses)},
    }, indent=2))

    print("pair overlap (points within 1 cm): " + ", ".join(f"{p} {v:.2f}" for p, v in pair_overlap.items()))
    for key in ("reference_raw", "reference_corrected", "fused"):
        e = evaluation[key]
        print(f"{key:20s} vs board plane: bias {e['bias_mm']:+.1f} mm, RMS {e['rms_mm']:.2f} mm")
    print(f"valid depth: reference {evaluation['valid_px_reference']:.0%}, fused {evaluation['valid_px_fused']:.0%}")
    print(f"saved {out_dir}/{{depth.png,depth_vis.png,mesh.ply,cloud.ply,fusion.json}}")
    if view:
        show_fusion(out_dir, frames, poses, reference, mesh, depth_raw)
    return out_dir
