# Recommended Capture Protocol

Based on the requirement for a robust, foolproof capture protocol for real estate agents, we have chosen **Route 2: Use an off-the-shelf 3D scanner app (Polycam/SiteScape)** rather than building an iOS app from scratch.

## Why Route 2?
Building a custom iOS ARKit app involves immense complexity regarding state management, memory leaks, and UI guidance. Off-the-shelf apps like Polycam have already solved these problems, providing real-time meshing feedback that ensures the user captures all corners of a room without missing data.

## Step-by-Step Instructions for the User

1. **Download:** Install Polycam (or SiteScape) from the iOS App Store.
2. **Setup:** Open the app, select the "LiDAR" or "Room" capture mode.
3. **Capture Sequence:**
   - Stand in the center of the room.
   - Point the camera at the floor.
   - Slowly pan the camera upwards towards the walls, walking the perimeter of the room in a smooth, continuous loop.
   - Ensure you point the camera specifically at door frames and windows.
4. **Export:** 
   - Tap "Done" and wait for the scan to process.
   - Tap "Export" and select "Raw Data" (which provides the `.csv` odometry, camera matrix, and depth maps).
   - Share the resulting `.zip` file via email or cloud drive to our processing backend.

## Edge Cases Handled
- **Too Fast:** Polycam will warn the user in real-time if they move too quickly.
- **Missed Spots:** The real-time AR mesh shows exactly what surfaces have been scanned, preventing holes in the data.
