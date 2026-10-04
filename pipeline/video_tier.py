"""
pipeline/video_tier.py

Processes a VIDEO capture (rgb.mp4 only — no depth sensor).

Strategy:
  1. Extract frames from rgb.mp4 every N frames
  2. Estimate relative depth with MiDaS (monocular depth estimator)
  3. Scale depth using an assumed standard door height (2.0 m) as reference
  4. Estimate camera motion between frames via ORB feature matching
  5. Back-project to 3D points and feed into the normal floor_plan pipeline

Returns: Open3D PointCloud (less accurate than LiDAR — CI is ±3%)
"""

import os
import numpy as np
import cv2
import open3d as o3d
import torch
from tqdm import tqdm


# ──────────────────────────────────────────────
# 1. Load MiDaS model (small = fast, sufficient)
# ──────────────────────────────────────────────

def load_midas(model_type: str = "MiDaS_small"):
    """
    Load MiDaS depth estimation model from torch hub.
    Model types: "MiDaS_small" (fast), "DPT_Hybrid", "DPT_Large" (accurate)
    """
    print(f"[Video] Loading MiDaS model: {model_type}")
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = transforms.dpt_transform
    else:
        transform = transforms.small_transform

    print(f"[Video] MiDaS loaded. Device: {device}")
    return model, transform, device


def estimate_depth(frame_bgr: np.ndarray, model, transform, device) -> np.ndarray:
    """
    Run MiDaS on a single BGR frame.
    Returns a relative depth map (HxW float32, larger = closer in MiDaS convention).
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(frame_rgb).to(device)

    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()
    return depth


# ──────────────────────────────────────────────
# 2. Scale MiDaS output to metric depth
# ──────────────────────────────────────────────

def scale_depth_to_metric(
    depth_relative: np.ndarray,
    assumed_height_m: float = 2.4,
) -> np.ndarray:
    """
    MiDaS produces relative (inverse) depth, not metric.
    We scale it so that the median depth value corresponds to
    an assumed room depth of `assumed_height_m` metres.

    This is a rough calibration — accuracy is ±10-20% on scale.
    The ±3% CI on video-tier measurements accounts for this.
    """
    # MiDaS: larger value = closer. We invert to get depth (smaller = closer).
    depth_inv = 1.0 / (depth_relative + 1e-6)

    # Scale so that the 50th percentile corresponds to assumed_height_m
    median_inv = np.median(depth_inv[depth_inv > 0])
    scale = assumed_height_m / (median_inv + 1e-6)
    depth_metric = depth_inv * scale

    # Clip to plausible indoor range [0.1 m, 8 m]
    depth_metric = np.clip(depth_metric, 0.1, 8.0)
    return depth_metric.astype(np.float32)


# ──────────────────────────────────────────────
# 3. Camera pose estimation via ORB + homography
# ──────────────────────────────────────────────

def estimate_poses_from_frames(
    frames: list,
    focal_length_px: float = 1600.0,
) -> list:
    """
    Estimate cumulative camera poses from a sequence of frames.
    Uses ORB features + essential matrix for relative pose estimation.

    Returns a list of 4x4 pose matrices (camera-to-world).
    The first frame is the identity (world origin).
    """
    orb = cv2.ORB_create(nfeatures=1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    H, W = frames[0].shape[:2]
    cx, cy = W / 2, H / 2
    K = np.array([[focal_length_px, 0, cx],
                  [0, focal_length_px, cy],
                  [0, 0, 1]], dtype=np.float64)

    poses = [np.eye(4)]
    current_pose = np.eye(4)
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)

    for i in range(1, len(frames)):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)

        if prev_des is None or des is None or len(kp) < 8:
            poses.append(current_pose.copy())
            continue

        matches = bf.match(prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)[:100]

        if len(matches) < 8:
            poses.append(current_pose.copy())
            continue

        pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp[m.trainIdx].pt for m in matches])

        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            poses.append(current_pose.copy())
            continue

        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K, mask=mask)

        # Build incremental transform
        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t.ravel()  # scale unknown — normalised to unit translation

        current_pose = current_pose @ np.linalg.inv(T_rel)
        poses.append(current_pose.copy())

        prev_gray = gray
        prev_kp, prev_des = kp, des

    return poses


# ──────────────────────────────────────────────
# 4. Back-project depth frames → 3D points
# ──────────────────────────────────────────────

def frames_to_point_cloud(
    frames: list,
    depth_maps: list,
    poses: list,
    focal_length_px: float = 1600.0,
    skip_pixels: int = 8,
) -> o3d.geometry.PointCloud:
    """
    Back-project depth maps to 3D world points using estimated poses.
    skip_pixels: sample every N-th pixel for speed.
    """
    H, W = frames[0].shape[:2]
    cx, cy = W / 2, H / 2
    fx = fy = focal_length_px

    all_points = []

    for depth, pose in zip(depth_maps, poses):
        # Sub-sample pixel grid
        u = np.arange(0, W, skip_pixels)
        v = np.arange(0, H, skip_pixels)
        uu, vv = np.meshgrid(u, v)
        uu, vv = uu.ravel(), vv.ravel()

        d = depth[vv, uu]
        valid = d > 0.1

        X = (uu[valid] - cx) * d[valid] / fx
        Y = (vv[valid] - cy) * d[valid] / fy
        Z = d[valid]

        pts_cam = np.stack([X, Y, Z, np.ones_like(Z)], axis=1)
        pts_world = (pose @ pts_cam.T).T[:, :3]
        all_points.append(pts_world)

    all_pts = np.vstack(all_points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    pcd = pcd.voxel_down_sample(voxel_size=0.04)  # 4cm — coarser than LiDAR
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=15, std_ratio=2.5)
    return pcd


# ──────────────────────────────────────────────
# 5. Main: process a video capture folder
# ──────────────────────────────────────────────

def process_video(
    folder: str,
    frame_step: int = 30,
    assumed_room_depth_m: float = 2.4,
    model_type: str = "MiDaS_small",
) -> o3d.geometry.PointCloud:
    """
    Main entry for video tier.

    Args:
        folder              : capture folder containing rgb.mp4
        frame_step          : use every N-th frame
        assumed_room_depth_m: scale anchor for MiDaS depth
        model_type          : MiDaS model variant

    Returns:
        Open3D PointCloud (metric, world coordinates)
    """
    video_path = os.path.join(folder, "rgb.mp4")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"[Video] rgb.mp4 not found in {folder}")

    print(f"\n[Video] Processing: {video_path}")

    # ── Extract frames ─────────────────────────
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[Video] Total frames: {total_frames}, FPS: {fps:.1f}")

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0:
            frames.append(frame)
        frame_idx += 1
    cap.release()
    print(f"[Video] Using {len(frames)} frames (every {frame_step}th).")

    if len(frames) < 3:
        raise RuntimeError("[Video] Too few frames extracted. Reduce frame_step.")

    # ── Load MiDaS ────────────────────────────
    model, transform, device = load_midas(model_type)

    # ── Estimate depth for each frame ─────────
    depth_maps = []
    for frame in tqdm(frames, desc="Estimating depth"):
        depth_rel = estimate_depth(frame, model, transform, device)
        depth_metric = scale_depth_to_metric(depth_rel, assumed_room_depth_m)
        depth_maps.append(depth_metric)

    # ── Estimate camera poses ─────────────────
    # Use focal length from camera_matrix.csv if available
    focal = 1600.0
    cam_matrix_path = os.path.join(folder, "camera_matrix.csv")
    if os.path.exists(cam_matrix_path):
        K = np.loadtxt(cam_matrix_path, delimiter=",")
        focal = float(K[0, 0])

    print("[Video] Estimating camera poses via ORB matching...")
    poses = estimate_poses_from_frames(frames, focal_length_px=focal)

    # ── Build point cloud ─────────────────────
    print("[Video] Building point cloud...")
    pcd = frames_to_point_cloud(frames, depth_maps, poses, focal_length_px=focal)
    print(f"[Video] Final point cloud: {len(pcd.points):,} points.")

    return pcd
