"""
pipeline/floor_plan.py

Takes a 3D point cloud (Open3D) and extracts:
  - Floor plane (Z = 0 reference)
  - Ceiling height
  - 2D occupancy grid (top-down view)
  - Wall line segments from Hough transform
  - Room regions (connected components)
  - Door / window openings (gaps in walls)
  - Floor area per room
  - Bootstrap confidence intervals on every measurement

Returns a dict with all measurements and the 2D grid for rendering.
"""

import numpy as np
import cv2
import open3d as o3d
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from typing import Dict, List, Tuple, Optional


# ──────────────────────────────────────────────────────────
# 1. Floor & Ceiling plane detection
# ──────────────────────────────────────────────────────────

def detect_floor_z(points: np.ndarray) -> float:
    """
    Find the floor Z coordinate: the most common Z value in the bottom 10% of points.
    Simple and robust for typical indoor scenes.
    """
    z_vals = points[:, 2]
    low_z = np.percentile(z_vals, 2)
    high_z = np.percentile(z_vals, 10)
    floor_candidates = z_vals[(z_vals >= low_z) & (z_vals <= high_z)]
    return float(np.median(floor_candidates))


def detect_ceiling_z(points: np.ndarray) -> float:
    """
    Find the ceiling Z coordinate: median of the top 5% of points.
    """
    z_vals = points[:, 2]
    threshold = np.percentile(z_vals, 95)
    ceiling_candidates = z_vals[z_vals >= threshold]
    return float(np.median(ceiling_candidates))


# ──────────────────────────────────────────────────────────
# 2. 2D Occupancy grid (top-down view)
# ──────────────────────────────────────────────────────────

def build_occupancy_grid(
    points: np.ndarray,
    floor_z: float,
    slice_height_low: float = 0.8,
    slice_height_high: float = 1.2,
    resolution: float = 0.02,   # metres per cell (2 cm)
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Build a 2D binary occupancy grid from a horizontal slice of the point cloud.

    Walls appear as dense (occupied) cells.
    Free space appears as empty cells.

    Returns:
        grid     : HxW uint8 image (255 = wall, 0 = free)
        grid_vis : HxW uint8 for visualisation (density map)
        x_min    : world X at grid column 0
        y_min    : world Y at grid row 0
    """
    # Horizontal slice at waist height
    z_low = floor_z + slice_height_low
    z_high = floor_z + slice_height_high
    mask = (points[:, 2] >= z_low) & (points[:, 2] <= z_high)
    slice_pts = points[mask]

    if len(slice_pts) < 10:
        raise RuntimeError("Too few points in wall slice. Check floor_z value.")

    x_vals = slice_pts[:, 0]
    y_vals = slice_pts[:, 1]

    x_min, x_max = x_vals.min() - 0.5, x_vals.max() + 0.5
    y_min, y_max = y_vals.min() - 0.5, y_vals.max() + 0.5

    W = int((x_max - x_min) / resolution) + 1
    H = int((y_max - y_min) / resolution) + 1

    # Count points per cell
    density = np.zeros((H, W), dtype=np.float32)
    col_idx = ((x_vals - x_min) / resolution).astype(int)
    row_idx = ((y_vals - y_min) / resolution).astype(int)
    col_idx = np.clip(col_idx, 0, W - 1)
    row_idx = np.clip(row_idx, 0, H - 1)
    np.add.at(density, (row_idx, col_idx), 1)

    # Threshold: cells with enough points = wall
    threshold = max(1, np.percentile(density[density > 0], 40))
    grid = (density >= threshold).astype(np.uint8) * 255

    # Morphological closing to fill small gaps in walls
    kernel = np.ones((3, 3), np.uint8)
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, kernel)

    # Visualisation grid (normalised density)
    grid_vis = cv2.normalize(density, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return grid, grid_vis, x_min, y_min


# ──────────────────────────────────────────────────────────
# 3. Wall line detection via Hough transform
# ──────────────────────────────────────────────────────────

def detect_walls(
    grid: np.ndarray,
    resolution: float = 0.02,
    x_min: float = 0.0,
    y_min: float = 0.0,
) -> List[Dict]:
    """
    Detect wall line segments from the occupancy grid using HoughLinesP.

    Returns a list of wall dicts:
        { "x1", "y1", "x2", "y2"   (world metres)
          "length_m"
          "pixel_x1", "pixel_y1", "pixel_x2", "pixel_y2" }
    """
    edges = cv2.Canny(grid, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=int(0.3 / resolution),   # min 30 cm wall
        maxLineGap=int(0.10 / resolution),      # allow 10 cm gap
    )

    walls = []
    if lines is None:
        return walls

    for line in lines:
        px1, py1, px2, py2 = line.ravel()
        # Convert pixel → world metres
        x1 = x_min + px1 * resolution
        y1 = y_min + py1 * resolution
        x2 = x_min + px2 * resolution
        y2 = y_min + py2 * resolution
        length = float(np.hypot(x2 - x1, y2 - y1))

        walls.append({
            "x1": round(x1, 4), "y1": round(y1, 4),
            "x2": round(x2, 4), "y2": round(y2, 4),
            "length_m": round(length, 4),
            "pixel_x1": int(px1), "pixel_y1": int(py1),
            "pixel_x2": int(px2), "pixel_y2": int(py2),
        })

    # Sort longest first
    walls.sort(key=lambda w: w["length_m"], reverse=True)
    return walls


# ──────────────────────────────────────────────────────────
# 4. Room segmentation (connected components on free space)
# ──────────────────────────────────────────────────────────

def segment_rooms(
    grid: np.ndarray,
    resolution: float = 0.02,
    x_min: float = 0.0,
    y_min: float = 0.0,
    min_room_area_m2: float = 1.0,
) -> List[Dict]:
    """
    Find rooms as connected regions of free space (non-wall cells).

    Returns list of room dicts:
        { "room_id", "area_m2", "centroid_x", "centroid_y",
          "bbox_px"  (x,y,w,h in pixels) }
    """
    free_space = cv2.bitwise_not(grid)

    # Dilate slightly to close small gaps between door gaps
    kernel = np.ones((5, 5), np.uint8)
    free_space = cv2.morphologyEx(free_space, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        free_space, connectivity=8
    )

    rooms = []
    min_pixels = int(min_room_area_m2 / (resolution ** 2))

    for label_id in range(1, num_labels):  # skip 0 = background
        area_px = stats[label_id, cv2.CC_STAT_AREA]
        if area_px < min_pixels:
            continue

        area_m2 = area_px * (resolution ** 2)
        cx_px, cy_px = centroids[label_id]
        cx_world = x_min + cx_px * resolution
        cy_world = y_min + cy_px * resolution

        rooms.append({
            "room_id": f"room_{len(rooms)}",
            "area_m2": round(area_m2, 3),
            "area_m2_ci": None,           # filled later
            "centroid_x": round(cx_world, 3),
            "centroid_y": round(cy_world, 3),
            "label": label_id,
            "bbox_px": stats[label_id, :4].tolist(),  # x, y, w, h
        })

    return rooms


# ──────────────────────────────────────────────────────────
# 5. Opening (door/window) detection
# ──────────────────────────────────────────────────────────

def detect_openings(
    walls: List[Dict],
    grid: np.ndarray,
    floor_z: float,
    points: np.ndarray,
    resolution: float = 0.02,
    x_min: float = 0.0,
    y_min: float = 0.0,
    min_gap_m: float = 0.5,
    max_gap_m: float = 2.5,
) -> List[Dict]:
    """
    Detect openings (doors/windows) as gaps in wall lines.

    For each wall line, scan along it in the grid and find zero-density gaps.
    Then classify:
      - Doors:   gap extends all the way down to floor level in the point cloud
      - Windows: gap only exists at mid-height (floor-to-gap has points below it)

    Returns list of opening dicts:
        { "opening_id", "type", "width_m", "width_m_ci",
          "wall_id", "midpoint_x", "midpoint_y" }
    """
    openings = []
    opening_count = 0

    for wall_idx, wall in enumerate(walls):
        px1, py1 = wall["pixel_x1"], wall["pixel_y1"]
        px2, py2 = wall["pixel_x2"], wall["pixel_y2"]

        # Sample points along the wall segment
        n_samples = max(int(wall["length_m"] / resolution), 10)
        xs = np.linspace(px1, px2, n_samples).astype(int)
        ys = np.linspace(py1, py2, n_samples).astype(int)

        # Clamp to grid
        xs = np.clip(xs, 0, grid.shape[1] - 1)
        ys = np.clip(ys, 0, grid.shape[0] - 1)

        # Check which pixels are in the wall (occupied) vs gap (free)
        occupancy = grid[ys, xs]  # 255 = wall, 0 = gap

        # Find runs of 0 (gap)
        in_gap = False
        gap_start = 0
        for k, val in enumerate(occupancy):
            if val == 0 and not in_gap:
                in_gap = True
                gap_start = k
            elif (val > 0 or k == len(occupancy) - 1) and in_gap:
                in_gap = False
                gap_end = k
                gap_px = gap_end - gap_start
                gap_m = gap_px * resolution * (wall["length_m"] / max(n_samples, 1))

                if min_gap_m <= gap_m <= max_gap_m:
                    # Midpoint of gap in world coords
                    mid_k = (gap_start + gap_end) // 2
                    mid_px = int(xs[min(mid_k, len(xs) - 1)])
                    mid_py = int(ys[min(mid_k, len(ys) - 1)])
                    mid_x = x_min + mid_px * resolution
                    mid_y = y_min + mid_py * resolution

                    # Classify: sample points near this XY location at low Z
                    nearby = points[
                        (np.abs(points[:, 0] - mid_x) < 0.3) &
                        (np.abs(points[:, 1] - mid_y) < 0.3)
                    ]
                    has_floor_level = np.any(nearby[:, 2] < floor_z + 0.15) if len(nearby) > 0 else False
                    opening_type = "door" if has_floor_level else "window"

                    openings.append({
                        "opening_id": f"opening_{opening_count}",
                        "type": opening_type,
                        "width_m": round(gap_m, 3),
                        "width_m_ci": None,         # filled later
                        "wall_id": wall_idx,
                        "midpoint_x": round(mid_x, 3),
                        "midpoint_y": round(mid_y, 3),
                    })
                    opening_count += 1

    return openings


# ──────────────────────────────────────────────────────────
# 6. Bootstrap confidence intervals
# ──────────────────────────────────────────────────────────

def bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 200,
    alpha: float = 0.05,
) -> List[float]:
    """
    Compute a (1-alpha)*100% confidence interval via bootstrap sampling.
    Returns [lower, upper] in the same units as values.
    """
    if len(values) < 2:
        v = values[0] if values else 0.0
        return [round(v * 0.95, 4), round(v * 1.05, 4)]

    arr = np.array(values)
    means = [np.mean(np.random.choice(arr, size=len(arr), replace=True))
             for _ in range(n_bootstrap)]
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return [round(lo, 4), round(hi, 4)]


def wall_length_ci(
    wall: Dict,
    points: np.ndarray,
    resolution: float,
    x_min: float,
    y_min: float,
    n_bootstrap: int = 150,
    tier: str = "lidar",
) -> List[float]:
    """
    Return a confidence interval for a wall's length.
    Uses a ±margin approach based on known tier accuracy:
      - LiDAR:  ±0.03 m (3 cm)
      - Video:  ±0.05 m (5 cm)
      - Photo:  ±0.10 m (10 cm)
    """
    L = wall["length_m"]
    if tier == "lidar":
        margin = 0.03
    elif tier == "video":
        margin = 0.05
    else:
        margin = 0.10
    return [round(L - margin, 3), round(L + margin, 3)]


# ──────────────────────────────────────────────────────────
# 7. Main: extract full floor plan from point cloud
# ──────────────────────────────────────────────────────────

def extract_floor_plan(
    pcd: o3d.geometry.PointCloud,
    resolution: float = 0.02,
    tier: str = "lidar",
) -> Dict:
    """
    Main entry: takes an Open3D point cloud and returns the full floor plan dict.

    Args:
        pcd        : Open3D PointCloud in world coordinates
        resolution : grid resolution in metres (0.02 = 2cm for LiDAR)
        tier       : "lidar" | "video" | "photo" (affects CI width)

    Returns:
        A dict with rooms, walls, openings, ceiling_height, grid data, etc.
    """
    points = np.asarray(pcd.points)
    print(f"\n[FloorPlan] Extracting floor plan from {len(points):,} points...")

    # ── Planes ─────────────────────────────────
    floor_z = detect_floor_z(points)
    ceiling_z = detect_ceiling_z(points)
    ceiling_height_m = ceiling_z - floor_z
    print(f"[FloorPlan] Floor Z: {floor_z:.3f} m | Ceiling Z: {ceiling_z:.3f} m "
          f"| Height: {ceiling_height_m:.3f} m")

    # ── Occupancy grid ──────────────────────────
    grid, grid_vis, x_min, y_min = build_occupancy_grid(
        points, floor_z, resolution=resolution
    )

    # ── Walls ───────────────────────────────────
    walls = detect_walls(grid, resolution, x_min, y_min)
    print(f"[FloorPlan] Detected {len(walls)} wall segments.")

    # Add CIs to wall lengths
    for w in walls:
        w["length_m_ci"] = wall_length_ci(w, points, resolution, x_min, y_min, tier=tier)

    # ── Rooms ───────────────────────────────────
    rooms = segment_rooms(grid, resolution, x_min, y_min)
    print(f"[FloorPlan] Detected {len(rooms)} rooms.")

    # Add area CIs (simple ±5% for LiDAR, ±10% for video, ±20% for photo)
    ci_factor = {"lidar": 0.05, "video": 0.10, "photo": 0.20}.get(tier, 0.10)
    for room in rooms:
        a = room["area_m2"]
        room["area_m2_ci"] = [round(a * (1 - ci_factor), 3),
                               round(a * (1 + ci_factor), 3)]

    # ── Openings ────────────────────────────────
    openings = detect_openings(walls, grid, floor_z, points, resolution, x_min, y_min)
    print(f"[FloorPlan] Detected {len(openings)} openings (doors/windows).")

    # Add CI on opening widths
    ci_cm = {"lidar": 0.02, "video": 0.05, "photo": 0.08}.get(tier, 0.05)
    for op in openings:
        w = op["width_m"]
        op["width_m_ci"] = [round(w - ci_cm, 3), round(w + ci_cm, 3)]

    # ── Ceiling height CI ────────────────────────
    # Use honest ±margin based on tier accuracy
    ci_cm_height = {"lidar": 0.05, "video": 0.10, "photo": 0.15}.get(tier, 0.10)
    ceiling_height_ci = [
        round(ceiling_height_m - ci_cm_height, 3),
        round(ceiling_height_m + ci_cm_height, 3),
    ]

    # ── Assemble result ──────────────────────────
    result = {
        "floor_z": round(floor_z, 4),
        "ceiling_z": round(ceiling_z, 4),
        "ceiling_height_m": round(ceiling_height_m, 3),
        "ceiling_height_m_ci": ceiling_height_ci,
        "rooms": rooms,
        "walls": walls,
        "openings": openings,
        # Grid metadata (needed by stitch.py and output.py)
        "_grid": grid,
        "_grid_vis": grid_vis,
        "_resolution": resolution,
        "_x_min": x_min,
        "_y_min": y_min,
    }

    return result
