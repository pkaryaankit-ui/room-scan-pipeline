"""
pipeline/stitch.py

Multi-room stitching for the LiDAR and Video tiers.

Since all LiDAR frames share a single world coordinate system (from odometry),
the point cloud is already globally consistent.
Room stitching = segmenting the combined grid into connected free-space regions.

This module:
  1. Takes the combined floor plan dict (from floor_plan.extract_floor_plan)
  2. Labels each room in the 2D grid
  3. Assigns walls and openings to their rooms
  4. Computes a stitched adjacency graph (which rooms are connected via which openings)
  5. Returns the stitched floor plan ready for output
"""

import numpy as np
import cv2
from typing import Dict, List


# ──────────────────────────────────────────────
# 1. Assign walls to rooms
# ──────────────────────────────────────────────

def assign_walls_to_rooms(
    walls: List[Dict],
    rooms: List[Dict],
    grid: np.ndarray,
    resolution: float,
    x_min: float,
    y_min: float,
) -> List[Dict]:
    """
    For each wall, determine which room(s) it belongs to.
    A wall belongs to a room if its midpoint is adjacent to that room's free-space region.

    Modifies walls in-place, adds "room_ids" list.
    Returns the updated walls list.
    """
    # Build label map from connected components (same as segment_rooms used)
    free_space = cv2.bitwise_not(grid)
    kernel = np.ones((5, 5), np.uint8)
    free_space = cv2.morphologyEx(free_space, cv2.MORPH_OPEN, kernel)
    _, labels = cv2.connectedComponents(free_space, connectivity=8)

    # Build label → room_id mapping
    label_to_room = {}
    for room in rooms:
        label_to_room[room["label"]] = room["room_id"]

    for wall in walls:
        mid_x = (wall["x1"] + wall["x2"]) / 2
        mid_y = (wall["y1"] + wall["y2"]) / 2

        # Convert midpoint to pixel
        col = int((mid_x - x_min) / resolution)
        row = int((mid_y - y_min) / resolution)

        # Check 3x3 neighbourhood
        room_ids = set()
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                r, c = row + dr, col + dc
                if 0 <= r < labels.shape[0] and 0 <= c < labels.shape[1]:
                    lbl = labels[r, c]
                    if lbl in label_to_room:
                        room_ids.add(label_to_room[lbl])

        wall["room_ids"] = list(room_ids)

    return walls


# ──────────────────────────────────────────────
# 2. Build room adjacency graph from openings
# ──────────────────────────────────────────────

def build_adjacency_graph(
    rooms: List[Dict],
    openings: List[Dict],
    walls: List[Dict],
) -> Dict[str, List[str]]:
    """
    Build a room adjacency graph:
        { room_id: [adjacent_room_id, ...] }

    Two rooms are adjacent if they share an opening (door/window)
    through a wall that belongs to both rooms.
    """
    # Map wall_id → room_ids
    wall_rooms = {}
    for i, wall in enumerate(walls):
        wall_rooms[i] = wall.get("room_ids", [])

    adjacency = {room["room_id"]: [] for room in rooms}

    for opening in openings:
        wall_idx = opening.get("wall_id")
        if wall_idx is None:
            continue
        connected_rooms = wall_rooms.get(wall_idx, [])
        if len(connected_rooms) == 2:
            r1, r2 = connected_rooms[0], connected_rooms[1]
            if r2 not in adjacency.get(r1, []):
                adjacency.setdefault(r1, []).append(r2)
            if r1 not in adjacency.get(r2, []):
                adjacency.setdefault(r2, []).append(r1)

    return adjacency


# ──────────────────────────────────────────────
# 3. Main: stitch floor plan
# ──────────────────────────────────────────────

def stitch_floor_plan(floor_plan: Dict) -> Dict:
    """
    Takes a floor plan dict (from floor_plan.extract_floor_plan) and adds:
      - room_id assigned to each wall
      - room adjacency graph
      - stitched = True flag

    Returns the enriched floor plan dict.
    """
    grid = floor_plan["_grid"]
    resolution = floor_plan["_resolution"]
    x_min = floor_plan["_x_min"]
    y_min = floor_plan["_y_min"]
    rooms = floor_plan["rooms"]
    walls = floor_plan["walls"]
    openings = floor_plan["openings"]

    print(f"\n[Stitch] Assigning walls to rooms...")
    walls = assign_walls_to_rooms(walls, rooms, grid, resolution, x_min, y_min)

    print(f"[Stitch] Building room adjacency graph...")
    adjacency = build_adjacency_graph(rooms, openings, walls)

    floor_plan["walls"] = walls
    floor_plan["room_adjacency"] = adjacency
    floor_plan["stitched"] = True

    print(f"[Stitch] Room adjacency: {adjacency}")
    return floor_plan
