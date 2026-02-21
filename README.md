# SHEILD

AI-powered bird deterrence and nest-awareness system built with YOLOv8, Streamlit, and ESP32 devices.

## Overview

SHEILD is a real-time monitoring and actuation project designed to reduce bird-related disturbance in protected or operational spaces.  
It combines computer vision with edge hardware to:

- detect birds in camera frames,
- trigger deterrence actions automatically via ESP32 control endpoints,
- detect nest/twig activity and suppress deterrence for safer handling.

This creates a practical "detect -> decide -> act" safety loop for field deployment.

## What This Project Includes

- `app.py`: Main Streamlit dashboard and runtime logic
- `obj.py`: Standalone OpenCV detection script for ESP32-CAM feed testing
- `yolov8n.pt`: Base YOLOv8 model used for bird/object detection
- `best.pt`: Custom model used for nest/twigs/nesting-material detection
- `Udgham.html`: Supporting static page asset

## Core Features

- Dual-model inference pipeline (general bird model + custom nest-activity model)
- Snapshot-based camera ingestion (optimized for unstable/slow Wi-Fi)
- ESP32 control integration using `.../bird/on` and `.../bird/off` endpoints
- Nest safety mode that suppresses automated deterrence during nesting activity
- Operator dashboard with Start/Stop controls, live detections, count chart, and FPS/confidence tuning

## High-Level Use Cases

- Campus/building rooftop bird deterrence
- Solar panel and utility asset protection
- Agriculture/perimeter monitoring
- Facilities where non-lethal, AI-assisted bird management is needed
- Safety-first scenarios where nesting activity must be identified before actuation

## System Flow

1. ESP32-CAM provides frame snapshots.
2. SHEILD runs bird and nest detection models on each frame.
3. If bird is detected, deterrence endpoint is triggered.
4. If nest/twigs are detected, nest alert is raised and automated deterrence is limited/suppressed.
5. Operator monitors everything in Streamlit UI.

## Tech Stack

- Python
- Streamlit
- Ultralytics YOLOv8
- OpenCV
- NumPy / Pandas
- Requests
- ESP32-CAM + ESP32 controller endpoints

## Quick Start

1. Install dependencies:

```bash
pip install streamlit ultralytics opencv-python numpy pandas requests
```

2. Ensure model files are present in project root:
- `yolov8n.pt`
- `best.pt`

3. Update network endpoints in `app.py` as needed:
- `ESP32_CAM_URL`
- `ESP32_CTRL_IP`

4. Run dashboard:

```bash
streamlit run app.py
```

5. Open the Streamlit URL and use **Start** to begin live detection.

## Notes

- Current endpoints are configured for a local network setup.
- The project focuses on live inference and control logic; production hardening (logging, auth, retries, deployment packaging) can be added as next-stage work.
"# SHIELD" 
"# SHIELD" 
