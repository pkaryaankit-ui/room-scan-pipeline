"""
pipeline/output.py

Assembles the final output for one capture:
  1. Builds the JSON result dict (per the schema)
  2. Renders the floor plan as a PNG image
  3. Writes both to the output/ folder

JSON schema:
{
  "capture_id": str,
  "tier": str,
  "drift_correction_applied": bool,
  "ceiling_height_m": float,
  "ceiling_height_m_ci": [float, float],
  "rooms": [ { room_id, floor_area_m2, floor_area_m2_ci, ceiling_height_m,
               walls: [...], openings: [...], damage: [...] } ],
  "stitched_plan_png": str,
  "room_adjacency": { room_id: [room_id, ...] }
}
"""

import os
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from typing import Dict, List


# ──────────────────────────────────────────────
# 1. Build clean JSON output (no private _ keys)
# ──────────────────────────────────────────────

def build_json_output(
    floor_plan: Dict,
    damage_events: List[Dict],
    capture_id: str,
    tier: str,
) -> Dict:
    """
    Assemble the final JSON output dict from the floor plan and damage results.
    Strips private keys (those starting with '_').
    """
    # Strip private keys
    clean_plan = {k: v for k, v in floor_plan.items() if not k.startswith("_")}

    # Attach damage to rooms (distribute by surface type)
    rooms = clean_plan.get("rooms", [])
    for room in rooms:
        room["damage"] = []

    # Assign damage events to rooms (simple: round-robin or first room)
    for event in damage_events:
        if rooms:
            target_room = rooms[0]  # Simple: assign to first room if no spatial info
            target_room["damage"].append({
                "damage_id": event["damage_id"],
                "class": event["class"],
                "surface": event["surface"],
                "area_m2": event["area_m2"],
                "concealed": event["concealed"],
                "concealed_rule": event.get("concealed_rule"),
            })

    # Add scope line items per room (simple: one line item per surface)
    for room in rooms:
        scope = []
        for wall in clean_plan.get("walls", []):
            if room["room_id"] in wall.get("room_ids", [room["room_id"]]):
                scope.append({
                    "surface": wall.get("wall_id", "wall"),
                    "item": "inspect and document",
                    "unit": "m",
                    "quantity": wall.get("length_m", 0),
                })
        for d in room["damage"]:
            scope.append({
                "surface": d["surface"],
                "item": f"repair {d['class']}",
                "unit": "m2",
                "quantity": d["area_m2"],
            })
        room["scope_line_items"] = scope

    result = {
        "capture_id": capture_id,
        "tier": tier,
        "drift_correction_applied": True,
        "ceiling_height_m": clean_plan.get("ceiling_height_m"),
        "ceiling_height_m_ci": clean_plan.get("ceiling_height_m_ci"),
        "rooms": rooms,
        "walls": clean_plan.get("walls", []),
        "openings": clean_plan.get("openings", []),
        "room_adjacency": clean_plan.get("room_adjacency", {}),
        "stitched_plan_png": f"output/{capture_id}_plan.png",
    }
    return result


# ──────────────────────────────────────────────
# 2. Render floor plan PNG
# ──────────────────────────────────────────────

def render_floor_plan(
    floor_plan: Dict,
    output_path: str,
    title: str = "Floor Plan",
) -> None:
    """
    Render the floor plan as a PNG using matplotlib.

    Draws:
      - Walls as thick black lines
      - Openings as coloured gaps (blue = door, green = window)
      - Room centroids with labels and area
      - Damage regions as red markers
      - Annotations for wall lengths
    """
    walls = floor_plan.get("walls", [])
    rooms = floor_plan.get("rooms", [])
    openings = floor_plan.get("openings", [])

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect("equal")
    ax.set_facecolor("#f8f8f8")
    fig.patch.set_facecolor("#ffffff")

    # ── Draw occupancy grid as background ──
    grid = floor_plan.get("_grid")
    x_min = floor_plan.get("_x_min", 0)
    y_min = floor_plan.get("_y_min", 0)
    resolution = floor_plan.get("_resolution", 0.02)

    if grid is not None:
        H, W = grid.shape
        x_max = x_min + W * resolution
        y_max = y_min + H * resolution
        ax.imshow(
            grid,
            origin="upper",
            extent=[x_min, x_max, y_max, y_min],
            cmap="Greys",
            alpha=0.2,
            vmin=0, vmax=255,
        )

    # ── Draw walls ──────────────────────────
    for wall in walls:
        x1, y1 = wall["x1"], wall["y1"]
        x2, y2 = wall["x2"], wall["y2"]
        ax.plot([x1, x2], [y1, y2], color="#222222", linewidth=3, solid_capstyle="round")

        # Annotate wall length at midpoint
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        length = wall["length_m"]
        if length > 0.5:
            ci = wall.get("length_m_ci", [length, length])
            label = f"{length:.2f}m\n[{ci[0]:.2f}–{ci[1]:.2f}]"
            ax.text(mid_x, mid_y, label, fontsize=6, ha="center", va="center",
                    color="#444444", bbox=dict(boxstyle="round,pad=0.1",
                    facecolor="white", alpha=0.7, edgecolor="none"))

    # ── Draw openings ────────────────────────
    for op in openings:
        mx, my = op["midpoint_x"], op["midpoint_y"]
        color = "#1a73e8" if op["type"] == "door" else "#34a853"
        marker = "D" if op["type"] == "door" else "s"
        ax.plot(mx, my, marker=marker, color=color, markersize=8, zorder=5)
        ax.text(mx, my - 0.15, f"{op['width_m']:.2f}m", fontsize=6,
                ha="center", color=color)

    # ── Draw room labels ─────────────────────
    for room in rooms:
        cx, cy = room["centroid_x"], room["centroid_y"]
        area = room["area_m2"]
        ax.text(cx, cy, f"{room['room_id']}\n{area:.1f} m²",
                fontsize=9, ha="center", va="center",
                color="#1a1a2e",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f0fe",
                          alpha=0.85, edgecolor="#1a73e8", linewidth=0.8))

    # ── Draw damage regions ──────────────────
    damage_color = {"crack": "#ea4335", "water_damage": "#fbbc04",
                    "mold": "#34a853", "peeling_paint": "#ff6d00"}
    for room in rooms:
        for d in room.get("damage", []):
            cx, cy = room["centroid_x"] + 0.3, room["centroid_y"] + 0.3
            col = damage_color.get(d["class"], "#ea4335")
            ax.plot(cx, cy, "x", color=col, markersize=10, markeredgewidth=2)
            ax.text(cx + 0.1, cy, d["class"], fontsize=6, color=col)

    # ── Ceiling height annotation ─────────────
    ch = floor_plan.get("ceiling_height_m", 0)
    ci_ch = floor_plan.get("ceiling_height_m_ci", [ch, ch])
    ax.text(
        0.02, 0.98,
        f"Ceiling: {ch:.2f} m [{ci_ch[0]:.2f}–{ci_ch[1]:.2f}]",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round", facecolor="#e6f4ea", alpha=0.9),
    )

    # ── Legend ───────────────────────────────
    legend_elements = [
        Line2D([0], [0], color="#222222", linewidth=3, label="Wall"),
        Line2D([0], [0], marker="D", color="#1a73e8", linewidth=0,
               markersize=8, label="Door"),
        Line2D([0], [0], marker="s", color="#34a853", linewidth=0,
               markersize=8, label="Window"),
        Line2D([0], [0], marker="x", color="#ea4335", linewidth=0,
               markersize=10, markeredgewidth=2, label="Damage"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    ax.set_xlabel("X (metres)", fontsize=9)
    ax.set_ylabel("Y (metres)", fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, alpha=0.2, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Output] Floor plan saved: {output_path}")


# ──────────────────────────────────────────────
# 3. Save JSON
# ──────────────────────────────────────────────

def save_json(result: Dict, output_path: str) -> None:
    """Write the result dict to a JSON file."""
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[Output] JSON saved: {output_path}")


# ──────────────────────────────────────────────
# 4. Main: write all outputs
# ──────────────────────────────────────────────

def write_outputs(
    floor_plan: Dict,
    damage_events: List[Dict],
    capture_id: str,
    tier: str,
    output_dir: str = "output",
) -> Dict:
    """
    Write JSON + PNG to output_dir.

    Returns the JSON result dict.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build JSON
    result = build_json_output(floor_plan, damage_events, capture_id, tier)

    # Save JSON
    json_path = os.path.join(output_dir, f"{capture_id}_result.json")
    save_json(result, json_path)

    # Render PNG
    png_path = os.path.join(output_dir, f"{capture_id}_plan.png")
    render_floor_plan(
        floor_plan,
        output_path=png_path,
        title=f"Floor Plan — {capture_id} ({tier.upper()} tier)",
    )

    # Update PNG path in result
    result["stitched_plan_png"] = png_path

    print(f"\n[Output] ✅ Done. Results in: {output_dir}/")
    print(f"  JSON : {json_path}")
    print(f"  PNG  : {png_path}")

    return result
