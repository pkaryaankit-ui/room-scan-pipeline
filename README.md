# Room Scan Pipeline

A simplified, robust pipeline to turn iPhone sensor data (LiDAR, Video, Photo) into accurate 2D floor plans with damage detection. Built as a submission for the Applied AI Engineer Case Study.

## 🚀 Quickstart

1. **Install Dependencies**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run the Pipeline**
   Point the pipeline at any capture folder. It auto-detects if it's LiDAR, Video, or Photo.
   ```bash
   python run.py /path/to/capture_folder
   ```

3. **Explore Data**
   Use the exploration script to print summary statistics of any raw capture.
   ```bash
   python explore_data.py /path/to/capture_folder
   ```

4. **Run the Benchmark**
   Tests the pipeline against all company-defined accuracy gates.
   ```bash
   python benchmark/run_benchmark.py
   ```

## 📁 Repository Structure
- `pipeline/`: The core modules for point cloud generation, floor plan extraction, and damage detection.
- `benchmark/`: Ground truth definitions and the automated gate-checking script.
- `docs/`: All required deliverables for the case study (capture protocols, JSON schema, etc.).
- `run.py`: The single-command entry point.

## 🧠 Design Philosophy
- **Simple over complex:** Uses traditional computer vision (Open3D, OpenCV) for robust geometry extraction rather than training fragile end-to-end neural networks.
- **Zero-shot AI:** Uses CLIP for damage detection to avoid the need for custom training data.
- **Honest metrics:** Uses Bootstrap sampling to report rigorous confidence intervals on every measurement, as requested.
