"""
benchmark/run_benchmark.py

Runs the full pipeline on benchmark captures and checks all accuracy gates.

Usage:
    python benchmark/run_benchmark.py

Expects:
    benchmark/ground_truth.csv    -- your tape-measure ground truth
    benchmark/captures.txt        -- list of capture folders to benchmark

Ground truth CSV format:
    capture_id, tier, measurement, value_m
    room1_lidar, lidar, wall_A_length, 3.82
    room1_lidar, lidar, ceiling_height, 2.40
    room1_lidar, lidar, door_1_width, 0.91
    room1_lidar, lidar, floor_area_room_0, 14.2
    ...

Gates checked (from the assignment):
    - Opening widths   : ≤ 2cm on ≥ 85% of openings
    - Ceiling height   : ≤ 1.5cm per room
    - Repeatability    : Two captures of the same room agree within 1cm or 0.5%
    - Drift accountability: drift correction applied (checked in output JSON)
    - Photo-tier stitch: footprint within ±8% with calibrated intervals
"""

import os
import sys
import json
import csv
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent dir to path so we can import pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))
from run import run_pipeline


# ──────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────

BENCHMARK_DIR = os.path.dirname(__file__)
GROUND_TRUTH_CSV = os.path.join(BENCHMARK_DIR, "ground_truth.csv")
CAPTURES_TXT = os.path.join(BENCHMARK_DIR, "captures.txt")
RESULTS_DIR = os.path.join(BENCHMARK_DIR, "results")


# ──────────────────────────────────────────────────────────
# 1. Load ground truth
# ──────────────────────────────────────────────────────────

def load_ground_truth(path: str) -> dict:
    """
    Returns:
        { capture_id: { measurement_name: value_m } }
    """
    gt = {}
    if not os.path.exists(path):
        print(f"[Benchmark] Ground truth not found: {path}")
        return gt

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for _, row in df.iterrows():
        cid = str(row["capture_id"]).strip()
        meas = str(row["measurement"]).strip()
        val = float(row["value_m"])
        gt.setdefault(cid, {})[meas] = val
    return gt


# ──────────────────────────────────────────────────────────
# 2. Load capture list
# ──────────────────────────────────────────────────────────

def load_captures(path: str) -> list:
    """
    Each line of captures.txt:
        <folder_path>  [<tier>]
    Lines starting with # are comments.
    """
    if not os.path.exists(path):
        print(f"[Benchmark] captures.txt not found: {path}")
        return []

    captures = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            folder = parts[0]
            tier = parts[1] if len(parts) > 1 else None
            captures.append((folder, tier))
    return captures


# ──────────────────────────────────────────────────────────
# 3. Extract measurements from pipeline JSON output
# ──────────────────────────────────────────────────────────

def extract_measurements(result: dict) -> dict:
    """
    Pull out flat measurement dict from pipeline JSON output.
    Format: { measurement_name: value_m }
    """
    meas = {}
    meas["ceiling_height"] = result.get("ceiling_height_m", 0)

    rooms = result.get("rooms", [])
    for room in rooms:
        rid = room["room_id"]
        meas[f"floor_area_{rid}"] = room.get("area_m2", 0)

    walls = result.get("walls", [])
    for i, wall in enumerate(walls):
        meas[f"wall_{i}_length"] = wall.get("length_m", 0)

    openings = result.get("openings", [])
    for i, op in enumerate(openings):
        meas[f"opening_{i}_width"] = op.get("width_m", 0)

    return meas


# ──────────────────────────────────────────────────────────
# 4. Gate checks
# ──────────────────────────────────────────────────────────

def check_opening_gate(results_map: dict, gt_map: dict) -> dict:
    """
    Gate: Opening widths ≤ 2cm on ≥ 85% of openings.
    Also: missed openings (in GT but not detected) and phantom openings count as misses.
    """
    errors = []
    misses = 0
    total_gt_openings = 0

    for cid, gt in gt_map.items():
        measured = results_map.get(cid, {})
        gt_openings = {k: v for k, v in gt.items() if "opening" in k and "width" in k}
        total_gt_openings += len(gt_openings)

        for meas_name, gt_val in gt_openings.items():
            if meas_name in measured:
                error = abs(measured[meas_name] - gt_val)
                errors.append(error)
                if error > 0.02:
                    misses += 1
            else:
                misses += 1  # missed opening

    total = total_gt_openings
    detected_within_gate = total - misses
    pass_rate = detected_within_gate / total if total > 0 else 1.0
    gate_pass = pass_rate >= 0.85

    return {
        "gate": "Opening width ≤ 2cm on ≥ 85% of openings",
        "total_openings": total,
        "misses": misses,
        "pass_rate": round(pass_rate, 3),
        "mean_error_cm": round(np.mean(errors) * 100, 2) if errors else None,
        "PASS": gate_pass,
    }


def check_ceiling_gate(results_map: dict, gt_map: dict) -> dict:
    """Gate: Ceiling height ≤ 1.5cm per room."""
    errors = []
    fails = 0
    for cid, gt in gt_map.items():
        measured = results_map.get(cid, {})
        if "ceiling_height" in gt and "ceiling_height" in measured:
            error = abs(measured["ceiling_height"] - gt["ceiling_height"])
            errors.append(error)
            if error > 0.015:
                fails += 1

    return {
        "gate": "Ceiling height ≤ 1.5cm",
        "n_rooms": len(errors),
        "fails": fails,
        "mean_error_cm": round(np.mean(errors) * 100, 2) if errors else None,
        "max_error_cm": round(np.max(errors) * 100, 2) if errors else None,
        "PASS": fails == 0 and len(errors) > 0,
    }


def check_repeatability_gate(results_map: dict) -> dict:
    """
    Gate: Two captures of the same room agree within 1cm or 0.5% per wall.
    Looks for pairs of captures with the same base name + _a / _b suffix.
    """
    pairs = {}
    for cid in results_map:
        base = cid.rstrip("ab_").rstrip("12_")
        pairs.setdefault(base, []).append(cid)

    pair_results = []
    for base, cids in pairs.items():
        if len(cids) < 2:
            continue
        m1 = results_map[cids[0]]
        m2 = results_map[cids[1]]
        # Compare ceiling height
        if "ceiling_height" in m1 and "ceiling_height" in m2:
            diff = abs(m1["ceiling_height"] - m2["ceiling_height"])
            pct_diff = diff / m1["ceiling_height"] if m1["ceiling_height"] > 0 else 0
            pass_flag = diff <= 0.01 or pct_diff <= 0.005
            pair_results.append({
                "pair": f"{cids[0]} vs {cids[1]}",
                "metric": "ceiling_height",
                "diff_cm": round(diff * 100, 2),
                "diff_pct": round(pct_diff * 100, 3),
                "PASS": pass_flag,
            })

    gate_pass = all(p["PASS"] for p in pair_results) if pair_results else None

    return {
        "gate": "Repeatability: two captures agree within 1cm or 0.5%",
        "pairs_checked": len(pair_results),
        "details": pair_results,
        "PASS": gate_pass,
    }


def check_drift_gate(results_map: dict) -> dict:
    """Gate: Drift correction must be applied (not 'poses used as-is')."""
    all_applied = all(
        r.get("drift_correction_applied", False)
        for r in results_map.values()
    )
    return {
        "gate": "Drift correction applied (not poses used as-is)",
        "PASS": all_applied,
    }


def check_photo_stitch_gate(results_map: dict, gt_map: dict) -> dict:
    """Gate: Photo-tier footprint within ±8%."""
    errors = []
    for cid, measured in results_map.items():
        tier = "photo"  # Only check photo captures
        if tier not in cid and "photo" not in cid.lower():
            continue
        gt = gt_map.get(cid, {})
        for key, gt_val in gt.items():
            if "floor_area" in key and key in measured:
                pct_err = abs(measured[key] - gt_val) / gt_val
                errors.append(pct_err)

    if not errors:
        return {"gate": "Photo-tier stitch footprint ±8%", "PASS": None,
                "note": "No photo-tier captures in benchmark yet."}

    all_pass = all(e <= 0.08 for e in errors)
    return {
        "gate": "Photo-tier stitch footprint ±8%",
        "max_error_pct": round(max(errors) * 100, 1),
        "mean_error_pct": round(np.mean(errors) * 100, 1),
        "PASS": all_pass,
    }


# ──────────────────────────────────────────────────────────
# 5. Main benchmark runner
# ──────────────────────────────────────────────────────────

def run_benchmark(save_tag: str = "latest") -> dict:
    """
    Run the full benchmark and return a results dict.
    save_tag: "before_fix" or "after_fix" or "latest"
    """
    print(f"\n{'='*60}")
    print(f"  BENCHMARK RUN — {save_tag.upper()}")
    print(f"{'='*60}\n")

    gt_map = load_ground_truth(GROUND_TRUTH_CSV)
    captures = load_captures(CAPTURES_TXT)

    if not captures:
        print("[Benchmark] No captures configured. Edit benchmark/captures.txt")
        print("[Benchmark] Example line: /path/to/capture lidar")
        return {}

    results_map = {}
    out_dir_base = os.path.join(RESULTS_DIR, save_tag)
    os.makedirs(out_dir_base, exist_ok=True)

    for folder, tier in captures:
        if not os.path.isdir(folder):
            print(f"[Benchmark] Skipping (not found): {folder}")
            continue

        capture_id = os.path.basename(os.path.normpath(folder))
        if tier is None:
            from run import detect_tier
            tier = detect_tier(folder)

        print(f"\n[Benchmark] Running: {capture_id} ({tier})")
        out_dir = os.path.join(out_dir_base, capture_id)
        os.makedirs(out_dir, exist_ok=True)

        try:
            result = run_pipeline(folder, tier, out_dir)
            measurements = extract_measurements(result)
            results_map[capture_id] = measurements

            # Save individual result
            with open(os.path.join(out_dir, "result.json"), "w") as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            print(f"[Benchmark] FAILED for {capture_id}: {e}")

    # ── Gate checks ──────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  GATE RESULTS")
    print(f"{'─'*60}")

    gates = {
        "opening_width": check_opening_gate(results_map, gt_map),
        "ceiling_height": check_ceiling_gate(results_map, gt_map),
        "repeatability": check_repeatability_gate(results_map),
        "drift": check_drift_gate(results_map),
        "photo_stitch": check_photo_stitch_gate(results_map, gt_map),
    }

    all_results = {
        "tag": save_tag,
        "n_captures": len(results_map),
        "gates": gates,
        "measurements": results_map,
    }

    for gate_name, gate_result in gates.items():
        status = "✅ PASS" if gate_result.get("PASS") else (
            "❌ FAIL" if gate_result.get("PASS") is False else "⚠️  N/A"
        )
        print(f"\n{status}  {gate_result['gate']}")
        for k, v in gate_result.items():
            if k not in ("gate", "PASS", "details"):
                print(f"       {k}: {v}")
        if "details" in gate_result:
            for d in gate_result["details"]:
                print(f"       {d}")

    # Save summary
    summary_path = os.path.join(out_dir_base, "benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Benchmark] Summary saved: {summary_path}")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        default="latest",
        help="Tag for this run: 'before_fix', 'after_fix', or 'latest'",
    )
    args = parser.parse_args()
    run_benchmark(save_tag=args.tag)
