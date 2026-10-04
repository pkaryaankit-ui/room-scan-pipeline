# Supported Device Matrix

| Device Type | Tier Supported | Limitations |
|-------------|----------------|-------------|
| iPhone 12 Pro / 13 Pro / 14 Pro / 15 Pro | **LiDAR** | Full accuracy (± 2cm). Drift correction applied automatically. |
| iPhone 11 / 12 / 13 / 14 / 15 (Standard) | **Video** | Depth inferred via MiDaS. Accuracy ± 5cm. Scale anchor assumption required (e.g., standard door height). |
| Older iPhones (iPhone X, 8, SE) | **Photo** | Rooms scanned via panoramic photos. High reliance on overlap. Accuracy ± 10cm. |
| Android Devices with ToF | **LiDAR** | Subject to ARCore compatibility. Data format translation required to standard pipeline inputs. |
| Standard Android Devices | **Video / Photo** | Supported via MiDaS inference tier. Requires steady camera movement to prevent motion blur. |
