# Declaration: What I Am NOT Building

As a fresher tackling a fast-paced case study, I am focusing purely on the core geometry pipeline and data flow. To ensure I deliver a robust, working system, I am intentionally skipping the following complex tasks:

1. **Custom iOS Capture App**: Writing ARKit Swift code to handle memory management, SLAM tracking, and meshing is too complex for a 10-day sprint. I am relying on Route 2 (Polycam).
2. **Custom ML Models for Damage**: Training a Mask R-CNN or YOLO model requires thousands of annotated images of cracked walls. Instead, I am using a zero-shot model (CLIP) which provides immediate value without any training data.
3. **ICP (Iterative Closest Point) Registration**: While robust for stitching Point Clouds, ICP is computationally heavy and complex. I am using a simple "door-adjacency" rule for stitching multi-room models in the Photo Tier.
4. **Cloud Infrastructure**: I am not building an AWS/GCP pipeline with queues. This script runs locally via `run.py`.
