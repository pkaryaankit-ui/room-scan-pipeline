"""
pipeline/lidar_tier.py

Processes a LiDAR capture folder containing:
  - camera_matrix.csv   : 3x3 intrinsic matrix
  - odometry.csv        : per-frame camera poses (position + quaternion)
  - depth/XXXXXX.png    : per-frame depth maps
  - confidence/XXXXXX.png : per-frame depth confidence maps
  - rgb.mp4             : RGB walkthrough video

Returns: open3d PointCloud in world coordinates, with drift corrected.
"""

import os
import numpy as np
import cv2
import pandas as pd
import open3d as o3d
from scipy.spatial.transform import Rotation
from tqdm import tqdm


# ──────────────────────────────────────────────
# 1. Load camera intrinsics
# ──────────────────────────────────────────────

def load_camera_matrix(folder: str) -> np.ndarray:
    """Load the 3x3 camera intrinsic matrix from camera_matrix.csv."""
    path = os.path.join(folder, "camera_matrix.csv")
    K = np.loadtxt(path, delimiter=",")
    assert K.shape == (3, 3), f"Expected 3x3 camera matrix, got {K.shape}"
    return K


# ──────────────────────────────────────────────
# 2. Load odometry (camera poses per frame)
# ──────────────────────────────────────────────

def load_odometry(folder: str) -> pd.DataFrame:
    """
    Load odometry.csv.
    Columns: timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, ...
    Returns a DataFrame with one row per frame.
    """
    path = os.path.join(folder, "odometry.csv")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def pose_to_matrix(row: pd.Series) -> np.ndarray:
    """
    Convert a row from odometry (x,y,z, qx,qy,qz,qw) to a 4x4 pose matrix.
    """
    t = np.array([row["x"], row["y"], row["z"]])
    q = np.array([row["qx"], row["qy"], row["qz"], row["qw"]])
    R = Rotation.from_quat(q).as_matrix()   # scipy uses [x,y,z,w] order
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ──────────────────────────────────────────────
# 3. Back-project a depth frame to world points
# ──────────────────────────────────────────────

def depth_to_world_points(
    depth_img: np.ndarray,
    conf_img: np.ndarray,
    K: np.ndarray,
    pose: np.ndarray,
    conf_threshold: int = 1,
    depth_scale: float = 1000.0,
) -> np.ndarray:
    """
    Convert a depth image to a set of 3D world-coordinate points.

    Args:
        depth_img    : HxW uint16 depth image (millimetres if depth_scale=1000)
        conf_img     : HxW uint8 confidence map (0=low, 1=medium, 2=high)
        K            : 3x3 camera intrinsic matrix
        pose         : 4x4 camera-to-world transform
        conf_threshold: only use pixels with confidence >= this value
        depth_scale  : divide raw pixel value by this to get metres

    Returns:
        Nx3 array of (X,Y,Z) world-coordinate points
    """
    H, W = depth_img.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Build pixel coordinate grids
    u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))

    # Apply confidence mask
    mask = (conf_img >= conf_threshold) & (depth_img > 0)

    u_vals = u_grid[mask].astype(np.float32)
    v_vals = v_grid[mask].astype(np.float32)
    d_vals = depth_img[mask].astype(np.float32) / depth_scale  # metres

    # Back-project to camera space
    X_cam = (u_vals - cx) * d_vals / fx
    Y_cam = (v_vals - cy) * d_vals / fy
    Z_cam = d_vals

    # Stack into Nx4 homogeneous coords
    pts_cam = np.stack([X_cam, Y_cam, Z_cam, np.ones_like(Z_cam)], axis=1)  # Nx4

    # Transform to world space
    pts_world = (pose @ pts_cam.T).T  # Nx4
    return pts_world[:, :3]  # Nx3


# ──────────────────────────────────────────────
# 4. Drift correction (plane-anchored)
# ──────────────────────────────────────────────

def _fit_floor_plane(points: np.ndarray):
    """
    Fit a horizontal floor plane to a point cloud using Open3D RANSAC.
    Returns (plane_model, inlier_indices).
    plane_model = [a, b, c, d] where ax+by+cz+d=0
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.02,
        ransac_n=3,
        num_iterations=500,
    )
    return np.array(plane_model), inliers


def correct_drift(
    points_start: np.ndarray,
    points_end: np.ndarray,
    all_points: np.ndarray,
    split_fraction: float = 0.15,
) -> np.ndarray:
    """
    Plane-anchored drift correction.

    Detects the floor plane from the first and last `split_fraction` of points.
    If the planes differ, computes a vertical translation to align them and
    applies it to the second half of the full point cloud.

    Returns: drift-corrected Nx3 point array.
    """
    try:
        plane_start, _ = _fit_floor_plane(points_start)
        plane_end, _ = _fit_floor_plane(points_end)

        # Floor Z offset difference (c*z + d = 0  →  z_floor = -d/c)
        z_start = -plane_start[3] / (plane_start[2] + 1e-9)
        z_end = -plane_end[3] / (plane_end[2] + 1e-9)
        drift_z = z_end - z_start

        print(f"[Drift] floor Z at start: {z_start:.4f} m, at end: {z_end:.4f} m")
        print(f"[Drift] Vertical drift detected: {drift_z*100:.2f} cm")

        if abs(drift_z) < 0.001:  # < 1mm — no correction needed
            print("[Drift] Drift negligible. No correction applied.")
            return all_points

        # Apply linear correction: ramp drift correction over second half
        n = len(all_points)
        split = int(n * (1 - split_fraction))
        correction = np.zeros((n, 3))
        ramp = np.linspace(0, -drift_z, n - split)
        correction[split:, 2] = ramp

        corrected = all_points + correction
        print(f"[Drift] Correction applied to last {n - split} points.")
        return corrected

    except Exception as e:
        print(f"[Drift] Correction failed: {e}. Using raw points.")
        return all_points


# ──────────────────────────────────────────────
# 5. Main: build point cloud from a LiDAR folder
# ──────────────────────────────────────────────

def build_point_cloud(
    folder: str,
    frame_step: int = 30,
    voxel_size: float = 0.02,
    conf_threshold: int = 1,
) -> o3d.geometry.PointCloud:
    """
    Main entry for the LiDAR tier.

    Args:
        folder        : path to the capture folder
        frame_step    : process every N-th frame (10 = fast, 1 = dense)
        voxel_size    : voxel downsampling resolution in metres
        conf_threshold: minimum confidence level (0/1/2)

    Returns:
        Open3D PointCloud in world coordinates, drift-corrected
    """
    print(f"\n[LiDAR] Processing folder: {folder}")

    K = load_camera_matrix(folder)
    odom = load_odometry(folder)
    depth_dir = os.path.join(folder, "depth")
    conf_dir = os.path.join(folder, "confidence")

    # Get sorted list of depth frames
    depth_files = sorted(
        [f for f in os.listdir(depth_dir) if f.endswith(".png")]
    )
    print(f"[LiDAR] Found {len(depth_files)} depth frames. Using every {frame_step}th.")

    all_points = []
    odom_indexed = odom.set_index("frame") if "frame" in odom.columns else odom

    for i, fname in enumerate(tqdm(depth_files, desc="Building point cloud")):
        if i % frame_step != 0:
            continue

        frame_id = os.path.splitext(fname)[0]  # e.g. "000042"

        # Load depth
        depth_path = os.path.join(depth_dir, fname)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            continue

        # Load confidence
        conf_path = os.path.join(conf_dir, fname)
        if os.path.exists(conf_path):
            conf_img = cv2.imread(conf_path, cv2.IMREAD_UNCHANGED)
        else:
            conf_img = np.ones_like(depth_img, dtype=np.uint8) * 2  # assume high conf

        # Get pose for this frame
        frame_key = int(frame_id)
        if frame_key not in odom_indexed.index:
            # Find closest available frame
            closest = odom_indexed.index[
                (odom_indexed.index - frame_key).abs().argmin()
            ]
            row = odom_indexed.loc[closest]
        else:
            row = odom_indexed.loc[frame_key]

        # Handle duplicate frame index (take first)
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        pose = pose_to_matrix(row)

        # Infer depth scale from dtype
        depth_scale = 1000.0 if depth_img.dtype == np.uint16 else 1.0

        pts = depth_to_world_points(depth_img, conf_img, K, pose,
                                     conf_threshold, depth_scale)
        all_points.append(pts)

    if not all_points:
        raise RuntimeError("[LiDAR] No points collected. Check folder structure.")

    all_points_np = np.vstack(all_points)
    # Downsample directly in numpy to avoid Open3D memory layout segfaults on Mac
    all_points_np = all_points_np[::10]
    all_points_np = np.ascontiguousarray(all_points_np, dtype=np.float64)
    print(f"[LiDAR] Total raw points (sampled): {len(all_points_np):,}")

    # ── Voxel downsampling (early to save RAM) ────────────────────
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points_np)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    all_points_np = np.asarray(pcd.points)
    print(f"[LiDAR] Points after voxel downsampling ({voxel_size*100:.0f}cm): {len(all_points_np):,}")

    # ── Drift correction ──────────────────────
    n = len(all_points_np)
    split = int(n * 0.15)
    all_points_np = correct_drift(
        points_start=all_points_np[:split],
        points_end=all_points_np[-split:],
        all_points=all_points_np,
    )

    pcd.points = o3d.utility.Vector3dVector(all_points_np)

    # ── Remove statistical outliers ───────────
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"[LiDAR] Points after outlier removal: {len(pcd.points):,}")

    return pcd
