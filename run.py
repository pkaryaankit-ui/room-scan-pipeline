"""
run.py — Single-command entry point for the room scan pipeline.

Usage:
    python run.py <capture_folder>
    python run.py <capture_folder> --tier lidar
    python run.py <capture_folder> --tier video
    python run.py <capture_folder> --tier photo
    python run.py <capture_folder> --output_dir my_results/

The pipeline auto-detects the tier if not specified:
  - Has depth/ folder + odometry.csv → lidar
  - Has rgb.mp4 but no depth/        → video
  - Has only images (jpg/png)        → photo
"""

import argparse
import os
import sys
import time


def detect_tier(folder: str) -> str:
    """Auto-detect which tier the capture folder corresponds to."""
    has_depth = os.path.isdir(os.path.join(folder, "depth"))
    has_odom = os.path.isfile(os.path.join(folder, "odometry.csv"))
    has_video = os.path.isfile(os.path.join(folder, "rgb.mp4"))
    has_images = any(
        f.lower().endswith((".jpg", ".jpeg", ".png"))
        for f in os.listdir(folder)
    )

    if has_depth and has_odom:
        return "lidar"
    elif has_video:
        return "video"
    elif has_images:
        return "photo"
    else:
        # Check sub-folders for photo mode
        sub_folders = [
            d for d in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, d))
        ]
        if sub_folders:
            return "photo"
        raise ValueError(f"Cannot detect tier from folder: {folder}")


def run_pipeline(folder: str, tier: str, output_dir: str) -> dict:
    """
    Orchestrates the full pipeline for one capture.
    Returns the final JSON result dict.
    """
    capture_id = os.path.basename(os.path.normpath(folder))
    print(f"\n{'='*60}")
    print(f"  Room Scan Pipeline")
    print(f"  Capture : {capture_id}")
    print(f"  Tier    : {tier.upper()}")
    print(f"  Output  : {output_dir}")
    print(f"{'='*60}\n")

    t0 = time.time()

    # ── Step 1: Build point cloud ─────────────────────────────
    if tier == "lidar":
        from pipeline.lidar_tier import build_point_cloud
        pcd = build_point_cloud(folder)

    elif tier == "video":
        from pipeline.video_tier import process_video
        pcd = process_video(folder)

    elif tier == "photo":
        from pipeline.photo_tier import process_photos
        pcd = process_photos(folder)

    else:
        raise ValueError(f"Unknown tier: {tier}. Use lidar / video / photo")

    # ── Step 2: Extract floor plan ────────────────────────────
    from pipeline.floor_plan import extract_floor_plan
    # Use finer resolution for LiDAR, coarser for video/photo
    resolution = {"lidar": 0.02, "video": 0.04, "photo": 0.05}.get(tier, 0.03)
    floor_plan = extract_floor_plan(pcd, resolution=resolution, tier=tier)

    # ── Step 3: Stitch multi-room plan ────────────────────────
    from pipeline.stitch import stitch_floor_plan
    floor_plan = stitch_floor_plan(floor_plan)

    # ── Step 4: Detect damage ─────────────────────────────────
    from pipeline.damage import detect_damage
    floor_z = floor_plan.get("floor_z", 0.0)
    damage_events = detect_damage(folder, floor_z=floor_z)

    # ── Step 5: Write outputs ─────────────────────────────────
    from pipeline.output import write_outputs
    result = write_outputs(floor_plan, damage_events, capture_id, tier, output_dir)

    elapsed = time.time() - t0
    print(f"\n✅ Pipeline complete in {elapsed:.1f}s")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Room Scan Pipeline — turns iPhone sensor data into a floor plan."
    )
    parser.add_argument(
        "folder",
        help="Path to the capture folder (must contain depth/, rgb.mp4, or images)."
    )
    parser.add_argument(
        "--tier",
        choices=["lidar", "video", "photo"],
        default=None,
        help="Sensor tier. Auto-detected if not specified.",
    )
    parser.add_argument(
        "--output_dir",
        default="output",
        help="Directory to write JSON and PNG results (default: output/).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"ERROR: Folder not found: {args.folder}")
        sys.exit(1)

    # Auto-detect tier if not given
    tier = args.tier
    if tier is None:
        tier = detect_tier(args.folder)
        print(f"[run] Auto-detected tier: {tier}")

    result = run_pipeline(args.folder, tier, args.output_dir)

    # Print quick summary
    rooms = result.get("rooms", [])
    openings = result.get("openings", [])
    print(f"\n── Summary ──────────────────────────────")
    print(f"  Rooms      : {len(rooms)}")
    for r in rooms:
        print(f"    {r['room_id']}: {r['area_m2']} m²  "
              f"CI [{r['area_m2_ci'][0]}–{r['area_m2_ci'][1]}]")
    print(f"  Openings   : {len(openings)}")
    print(f"  Ceiling    : {result.get('ceiling_height_m')} m  "
          f"CI {result.get('ceiling_height_m_ci')}")
    total_damage = sum(len(r.get("damage", [])) for r in rooms)
    print(f"  Damage     : {total_damage} regions")
    print(f"  Plan PNG   : {result.get('stitched_plan_png')}")
    print(f"─────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
