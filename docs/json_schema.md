# JSON Schema Definition

The pipeline outputs a JSON file mapping to this structure:

```json
{
  "capture_id": "string",
  "tier": "string (lidar | video | photo)",
  "drift_correction_applied": "boolean",
  "ceiling_height_m": "float",
  "ceiling_height_m_ci": ["float", "float"],
  "stitched_plan_png": "string (filepath)",
  "room_adjacency": {
    "room_id": ["adjacent_room_id"]
  },
  "rooms": [
    {
      "room_id": "string",
      "area_m2": "float",
      "area_m2_ci": ["float", "float"],
      "centroid_x": "float",
      "centroid_y": "float",
      "walls": [
        {
          "wall_id": "integer",
          "length_m": "float",
          "length_m_ci": ["float", "float"],
          "x1": "float",
          "y1": "float",
          "x2": "float",
          "y2": "float",
          "room_ids": ["string"]
        }
      ],
      "openings": [
        {
          "opening_id": "string",
          "type": "string (door | window)",
          "width_m": "float",
          "width_m_ci": ["float", "float"],
          "wall_id": "integer",
          "midpoint_x": "float",
          "midpoint_y": "float"
        }
      ],
      "damage": [
        {
          "damage_id": "string",
          "class": "string (water_damage | crack | mold | peeling_paint)",
          "surface": "string (wall | floor | ceiling)",
          "area_m2": "float",
          "concealed": "boolean",
          "concealed_rule": "string (e.g. 'depth_anomaly')"
        }
      ],
      "scope_line_items": [
        {
          "surface": "string",
          "item": "string",
          "unit": "string (m | m2)",
          "quantity": "float"
        }
      ]
    }
  ]
}
```
