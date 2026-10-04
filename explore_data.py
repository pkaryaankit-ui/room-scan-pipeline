"""
explore_data.py — Quick script to understand the provided sample data.

Run this FIRST to print key stats about each capture folder.
Usage:
    python explore_data.py <capture_folder>
    python explore_data.py ../1a8384c3f6
"""

import sys
import os
import numpy as np
import pandas as pd
import cv2

def explore(folder: str):
    print(f"\n{'='*55}")
    print(f"  Exploring: {folder}")
    print(f"{'='*55}\n")

    # camera_matrix.csv
    cam_path = os.path.join(folder, "camera_matrix.csv")
    if os.path.exists(cam_path):
        K = np.loadtxt(cam_path, delimiter=",")
        print(f"Camera Matrix (K):\n{K}\n")
        print(f"  fx={K[0,0]:.2f}  fy={K[1,1]:.2f}  cx={K[0,2]:.2f}  cy={K[1,2]:.2f}\n")

    # odometry.csv
    odom_path = os.path.join(folder, "odometry.csv")
    if os.path.exists(odom_path):
        odom = pd.read_csv(odom_path)
        odom.columns = odom.columns.str.strip()
        print(f"Odometry: {len(odom)} rows")
        print(f"  Columns: {list(odom.columns)}")
        print(f"  X range: [{odom['x'].min():.3f}, {odom['x'].max():.3f}] m")
        print(f"  Y range: [{odom['y'].min():.3f}, {odom['y'].max():.3f}] m")
        print(f"  Z range: [{odom['z'].min():.3f}, {odom['z'].max():.3f}] m\n")

    # imu.csv
    imu_path = os.path.join(folder, "imu.csv")
    if os.path.exists(imu_path):
        imu = pd.read_csv(imu_path)
        imu.columns = imu.columns.str.strip()
        print(f"IMU: {len(imu)} rows")
        print(f"  Columns: {list(imu.columns)}\n")

    # depth frames
    depth_dir = os.path.join(folder, "depth")
    if os.path.isdir(depth_dir):
        depth_files = sorted(os.listdir(depth_dir))
        print(f"Depth frames: {len(depth_files)}")
        if depth_files:
            sample = cv2.imread(os.path.join(depth_dir, depth_files[0]), cv2.IMREAD_UNCHANGED)
            print(f"  Shape: {sample.shape}  dtype: {sample.dtype}  "
                  f"min: {sample.min()}  max: {sample.max()}")
            print(f"  Max depth in metres (if /1000): {sample.max()/1000:.2f} m\n")

    # confidence frames
    conf_dir = os.path.join(folder, "confidence")
    if os.path.isdir(conf_dir):
        conf_files = sorted(os.listdir(conf_dir))
        print(f"Confidence frames: {len(conf_files)}")
        if conf_files:
            sample = cv2.imread(os.path.join(conf_dir, conf_files[0]), cv2.IMREAD_UNCHANGED)
            unique_vals = np.unique(sample)
            print(f"  Shape: {sample.shape}  dtype: {sample.dtype}  "
                  f"unique values: {unique_vals}\n")

    # video
    video_path = os.path.join(folder, "rgb.mp4")
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        duration = total / fps if fps > 0 else 0
        print(f"Video: {w}x{h}  {fps:.1f} fps  {total} frames  {duration:.1f}s\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python explore_data.py <capture_folder>")
        sys.exit(1)
    explore(sys.argv[1])
