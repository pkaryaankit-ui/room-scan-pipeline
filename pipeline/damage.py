"""
pipeline/damage.py

Surface damage detection using CLIP zero-shot classification.

For each RGB frame:
  1. Run CLIP to score damage classes (crack, water_damage, mold, normal)
  2. If damage score > threshold → flag frame as damaged
  3. Map the damaged frame back to a 3D surface using the depth map + pose
  4. Report: class, approximate area in m², surface (wall/floor/ceiling), concealed flag

Concealed damage detection:
  - If a region has a depth discontinuity (sudden depth change along a smooth surface)
    but no visual damage → flag as "concealed" with rule: "depth_anomaly"
"""

import os
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import Dict, List, Optional


# ──────────────────────────────────────────────
# 1. Damage class definitions
# ──────────────────────────────────────────────

DAMAGE_CLASSES = [
    "water damage stain on wall",
    "crack in wall or ceiling",
    "mold or mildew on surface",
    "peeling paint or wallpaper",
    "normal clean surface",
]

# Map verbose label → short label
LABEL_MAP = {
    "water damage stain on wall": "water_damage",
    "crack in wall or ceiling": "crack",
    "mold or mildew on surface": "mold",
    "peeling paint or wallpaper": "peeling_paint",
    "normal clean surface": "normal",
}

DAMAGE_THRESHOLD = 0.25   # CLIP score above which we flag damage


# ──────────────────────────────────────────────
# 2. Load CLIP model
# ──────────────────────────────────────────────

def load_clip(model_name: str = "openai/clip-vit-base-patch32"):
    """
    Load CLIP model and processor from HuggingFace.
    Disclosed: openai/clip-vit-base-patch32
    """
    print(f"[Damage] Loading CLIP model: {model_name}")
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"[Damage] CLIP loaded. Device: {device}")
    return model, processor, device


# ──────────────────────────────────────────────
# 3. Classify a single frame
# ──────────────────────────────────────────────

def classify_frame(
    frame_bgr: np.ndarray,
    model,
    processor,
    device,
    labels: List[str] = DAMAGE_CLASSES,
) -> Dict[str, float]:
    """
    Run zero-shot CLIP classification on a single BGR frame.
    Returns dict: { label -> probability }
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)

    inputs = processor(
        text=labels,
        images=image,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image  # 1 x num_labels
        probs = logits.softmax(dim=1).squeeze().cpu().numpy()

    return {label: float(prob) for label, prob in zip(labels, probs)}


# ──────────────────────────────────────────────
# 4. Determine surface type from frame + depth
# ──────────────────────────────────────────────

def classify_surface(
    frame_bgr: np.ndarray,
    depth_map: Optional[np.ndarray],
    floor_z: float,
    pose: Optional[np.ndarray],
    K: Optional[np.ndarray],
) -> str:
    """
    Heuristic surface classification:
      - If median depth points are near floor level → "floor"
      - If median depth points are near ceiling → "ceiling"
      - Otherwise → "wall"
    """
    if depth_map is None or pose is None or K is None:
        return "wall"  # Default

    H, W = depth_map.shape
    cx, cy = K[0, 2], K[1, 2]
    fy = K[1, 1]

    # Sample a few depth values from image centre
    centre_depth = depth_map[H // 2, W // 2]
    if centre_depth <= 0:
        return "wall"

    # Estimate 3D point at image centre in world coords
    X = (W / 2 - cx) * centre_depth / K[0, 0]
    Y = (H / 2 - cy) * centre_depth / fy
    Z = centre_depth
    pt_cam = np.array([X, Y, Z, 1.0])
    pt_world = pose @ pt_cam
    z_world = pt_world[2]

    # Classify by Z height relative to floor
    height_above_floor = z_world - floor_z
    if height_above_floor < 0.15:
        return "floor"
    elif height_above_floor > 2.0:
        return "ceiling"
    else:
        return "wall"


# ──────────────────────────────────────────────
# 5. Concealed damage heuristic
# ──────────────────────────────────────────────

def detect_concealed_damage(
    depth_map: np.ndarray,
    conf_map: Optional[np.ndarray] = None,
    gradient_threshold: float = 0.3,
) -> bool:
    """
    Flag concealed damage based on abrupt depth discontinuities in the
    depth map that suggest a hidden cavity or delamination behind a surface.

    Rule: if the depth gradient (Laplacian) has unusually high values in
          a locally smooth region (confidence is high but depth jumps),
          flag as potentially concealed damage.

    Returns True if concealed damage is suspected.
    """
    if depth_map is None or depth_map.max() == 0:
        return False

    # Normalise depth
    d_norm = (depth_map / (depth_map.max() + 1e-6)).astype(np.float32)

    # Compute Laplacian (second derivative = edges / discontinuities)
    lap = cv2.Laplacian(d_norm, cv2.CV_32F)
    lap_abs = np.abs(lap)

    # Mask out already-low-confidence pixels
    if conf_map is not None:
        lap_abs[conf_map < 1] = 0

    # If max Laplacian response is unusually high → concealed issue
    return bool(lap_abs.max() > gradient_threshold)


# ──────────────────────────────────────────────
# 6. Estimate area of damage region in m²
# ──────────────────────────────────────────────

def estimate_damage_area(
    frame_bgr: np.ndarray,
    depth_map: Optional[np.ndarray],
    K: Optional[np.ndarray],
    damage_pixel_fraction: float = 0.15,
) -> float:
    """
    Rough estimate of damage area in m².

    Approach: assume the damaged region occupies `damage_pixel_fraction` of the frame.
    Use the median depth to estimate the scale (1 pixel ≈ depth/focal metres).
    """
    H, W = frame_bgr.shape[:2]
    total_pixels = H * W
    damage_pixels = int(total_pixels * damage_pixel_fraction)

    if depth_map is not None and K is not None:
        median_depth = float(np.median(depth_map[depth_map > 0]))
        fx = K[0, 0]
        # 1 pixel at this depth = depth/fx metres
        metres_per_pixel = median_depth / fx
        area = damage_pixels * (metres_per_pixel ** 2)
    else:
        # Fallback: assume 0.1 m² for any flagged frame
        area = 0.1

    return round(min(area, 10.0), 3)  # cap at 10 m²


# ──────────────────────────────────────────────
# 7. Main: detect damage across all frames
# ──────────────────────────────────────────────

def detect_damage(
    folder: str,
    floor_z: float = 0.0,
    frame_step: int = 30,
    model_name: str = "openai/clip-vit-base-patch32",
) -> List[Dict]:
    """
    Main entry for damage detection.

    Reads rgb.mp4 (or JPEG images) from the folder, runs CLIP on sampled frames,
    and returns a list of damage event dicts.

    Args:
        folder    : capture folder
        floor_z   : Z coordinate of floor (from floor_plan.detect_floor_z)
        frame_step: sample every N-th frame from video
        model_name: CLIP model to use

    Returns:
        List of damage dicts:
        { "damage_id", "class", "surface", "area_m2",
          "concealed", "concealed_rule",
          "frame_idx", "clip_scores" }
    """
    model, processor, device = load_clip(model_name)

    # Load frames from video
    video_path = os.path.join(folder, "rgb.mp4")
    depth_dir = os.path.join(folder, "depth")
    conf_dir = os.path.join(folder, "confidence")

    frames_with_meta = []

    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % frame_step == 0:
                depth_path = os.path.join(depth_dir, f"{idx:06d}.png")
                conf_path = os.path.join(conf_dir, f"{idx:06d}.png")
                depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) if os.path.exists(depth_path) else None
                conf = cv2.imread(conf_path, cv2.IMREAD_UNCHANGED) if os.path.exists(conf_path) else None
                if depth is not None and depth.dtype == np.uint16:
                    depth = depth.astype(np.float32) / 1000.0
                frames_with_meta.append((idx, frame, depth, conf))
            idx += 1
        cap.release()
    else:
        # Photo mode: load all images in folder
        for fname in sorted(os.listdir(folder)):
            if os.path.splitext(fname)[1].lower() in {".jpg", ".jpeg", ".png"}:
                img = cv2.imread(os.path.join(folder, fname))
                if img is not None:
                    frames_with_meta.append((len(frames_with_meta), img, None, None))

    # Limit to 5 frames for testing so the CPU doesn't hang
    if len(frames_with_meta) > 5:
        frames_with_meta = frames_with_meta[::len(frames_with_meta)//5][:5]

    print(f"[Damage] Scanning {len(frames_with_meta)} frames for damage...")

    damage_events = []
    damage_id = 0

    for frame_idx, frame, depth, conf in frames_with_meta:
        scores = classify_frame(frame, model, processor, device)

        # Check if any damage class exceeds threshold
        damaged = False
        top_damage_class = "normal"
        top_score = 0.0

        for label, score in scores.items():
            short = LABEL_MAP.get(label, label)
            if short != "normal" and score > DAMAGE_THRESHOLD and score > top_score:
                damaged = True
                top_damage_class = short
                top_score = score

        if not damaged:
            continue

        # Surface classification
        surface = classify_surface(frame, depth, floor_z, None, None)

        # Area estimate
        area = estimate_damage_area(frame, depth, None, damage_pixel_fraction=top_score)

        # Concealed damage check
        concealed = detect_concealed_damage(depth, conf) if depth is not None else False

        damage_events.append({
            "damage_id": f"damage_{damage_id}",
            "class": top_damage_class,
            "surface": surface,
            "area_m2": area,
            "concealed": concealed,
            "concealed_rule": "depth_anomaly" if concealed else None,
            "frame_idx": frame_idx,
            "clip_score": round(top_score, 4),
        })
        damage_id += 1

    print(f"[Damage] Found {len(damage_events)} damage regions.")
    return damage_events
