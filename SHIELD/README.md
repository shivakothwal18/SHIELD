# 🛡️ SHIELD – AI Bird Deterrence System

**Real-time AI-powered bird detection and nest-aware deterrence system** using YOLOv8, Streamlit, and ESP32 devices.

---

## 📋 Overview

SHIELD is an intelligent monitoring and actuation system designed to manage bird-related disturbances in protected or operational spaces. It combines advanced computer vision with edge hardware to create a practical **"detect → decide → act"** safety loop:

- **Detects** birds in real-time using YOLOv8 models
- **Identifies** nesting activity to prevent harm during critical periods
- **Triggers** automated deterrence actions via ESP32 controllers
- **Monitors** everything through an intuitive Streamlit dashboard

SHIELD is particularly valuable where non-lethal, AI-assisted bird management is critical and nest activity must be monitored.

---

## ✨ Core Features

- **Dual-Model Detection Pipeline**
  - YOLOv8 base model for general bird & object detection
  - Custom trained model for nest, twigs, and nesting material identification

- **Snapshot-Based Camera Integration**
  - Optimized for ESP32-CAM feeds over unstable/slow Wi-Fi
  - Adaptive frame capture methods (snapshot + OpenCV fallback)
  - Real-time inference with configurable processing delays

- **Smart Deterrence Logic**
  - Automatic bird detection → ESP32 deterrence trigger
  - Nest safety mode: suppresses deterrence when nesting activity detected
  - Operator override controls in dashboard

- **Live Monitoring Dashboard**
  - Real-time detection feed with bounding boxes
  - Bird & nest detection counts and trends
  - FPS monitoring and confidence threshold tuning
  - Quick start/stop controls

---

## 🎯 Use Cases

- Campus/building rooftop bird management
- Solar panel and utility infrastructure protection
- Agricultural perimeter monitoring
- Facilities requiring non-lethal bird deterrence
- Safety-first deployments where nesting detection is mandatory

---

## 📁 Project Structure

```
SHIELD/
├── app.py                  # Main Streamlit dashboard application
├── obj.py                  # Standalone detection script for testing
├── yolov8n.pt             # YOLOv8 nano model (bird detection)
├── best.pt                # Custom trained model (nest detection)
├── Udgham.html            # Supporting asset
└── README.md              # This file
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ESP32-CAM Stream                         │
│                 (10.12.10.115 /stream)                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │   SHIELD Detection Pipeline        │
        │  ┌──────────────────────────────┐  │
        │  │ YOLOv8n (Bird Detection)     │  │
        │  │ yolov8n.pt                   │  │
        │  └──────────────────────────────┘  │
        │  ┌──────────────────────────────┐  │
        │  │ Custom Model (Nest Detection)│  │
        │  │ best.pt                      │  │
        │  └──────────────────────────────┘  │
        └────────────────────┬───────────────┘
                             │
         ┌───────────────────┴──────────────────┐
         │                                      │
         ▼                                      ▼
    ┌─────────────┐                    ┌──────────────┐
    │ Bird Alert? │                    │ Nest Alert?  │
    │             │                    │              │
    │ Trigger OR  │                    │ Skip OR       │
    │ Suppress    │                    │ Suppress act  │
    └─────────────┘                    └──────────────┘
         │
         ▼
    ┌────────────────────────────┐
    │ ESP32 Control Endpoints    │
    │ /bird/on  ←→  /bird/off    │
    │ (10.12.17.216)             │
    └────────────────────────────┘
         │
         ▼
    ┌────────────────┐
    │ Deterrence     │
    │ Action         │
    └────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- ESP32-CAM connected to your network
- ESP32 controller for deterrence actuation (optional for testing)

### Installation

1. **Clone/navigate to the project folder:**
   ```bash
   cd SHIELD
   ```

2. **Install dependencies:**
   ```bash
   pip install streamlit opencv-python numpy requests pandas ultralytics
   ```

3. **Configure ESP32 endpoints** (edit `app.py`):
   ```python
   ESP32_CAM_URL = "http://<YOUR_ESP32_CAM_IP>/stream"
   ESP32_CTRL_IP = "http://<YOUR_ESP32_CONTROLLER_IP>"
   ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

   The dashboard will open in your browser (typically `http://localhost:8501`)

### Testing Without Hardware

To test bird detection with live playback from the ESP32-CAM:

```bash
python obj.py
```

This script will:
- Attempt to connect to the ESP32-CAM
- Auto-detect the best frame capture method
- Display live detections with bounding boxes
- Press `q` to exit

---

## ⚙️ Configuration

### In `app.py` (Streamlit Dashboard)

**Hardware Endpoints:**
```python
ESP32_CAM_URL = "http://10.12.10.115/stream"       # Camera feed IP
ESP32_CTRL_IP = "http://10.12.17.216"              # Deterrence controller IP
```

**Nest Detection Classes:**
```python
NEST_CLASSES = {"nest", "twigs", "nesting_material"}
```

**Runtime Settings (via Sidebar UI):**
- **Confidence Threshold:** 0.1 – 1.0 (default: 0.4)
- **Loop Delay:** 250ms – 2000ms (default: 400ms)

### In `obj.py` (Standalone Detection)

```python
ESP_CAM_URL = "http://10.12.10.115/stream"  # Try snapshot or stream URL
```

**Supported URL formats:**
- `http://<IP>/stream` ← Primary
- `http://<IP>/capture` ← Fallback
- Direct stream URLs for custom ESP32 firmware

---

## 📊 Dashboard Features

| Feature | Description |
|---------|-------------|
| **Live Feed** | Real-time detection visualization with bounding boxes |
| **Detections** | Current birds and nests detected on screen |
| **Count Metrics** | Running totals of birds & nests detected |
| **Trend Chart** | Historical detection counts over time |
| **Runtime Controls** | Start/Stop detection loop, confidence/delay sliders |
| **FPS Monitor** | Real-time frame processing speed |
| **Status Indicators** | Bird present, nest alert, ESP32 connection status |

---

## 🔌 ESP32 Integration

### Expected Endpoints

1. **Get Frame (Camera):**
   - `GET http://<CAM_IP>/stream` → MJPEG stream
   - `GET http://<CAM_IP>/capture` → Single JPEG snapshot

2. **Deterrence Actuation (Controller):**
   - `GET http://<CTRL_IP>/bird/on` → Activate deterrence
   - `GET http://<CTRL_IP>/bird/off` → Deactivate deterrence

### Example Workflow

```
Bird Detected (conf > 0.4)
    ↓
GET /bird/on  ← Trigger deterrence
    ↓
[Run for ~5 seconds]
    ↓
GET /bird/off  ← Stop deterrence
```

---

## 🛠️ Troubleshooting

### ESP32-CAM Connection Issues

**Problem:** "Failed to get frame, retrying..."

**Solutions:**
1. Verify ESP32-CAM is powered and connected to the same network
2. Test connection: `ping <ESP32_CAM_IP>`
3. Try alternative URL format in `obj.py` (stream vs. capture)
4. Increase `timeout` value in `get_snapshot()` function
5. Check firewall rules – ensure port 80 is open

### Model Loading Slow

**Problem:** Streamlit hangs on "Loading YOLOv8n..."

**Solutions:**
1. Models are cached in session state – reload Streamlit only on first run
2. Download models manually and place `.pt` files in the working directory
3. Use smaller model: `yolov8s.pt` instead of `yolov8n.pt`

### Low FPS / Lag

**Problem:** Detection feels sluggish

**Solutions:**
1. Increase "Loop Delay" slider in dashboard (allows longer processing)
2. Use snapshot method instead of stream in `obj.py`
3. Reduce image resolution if ESP32-CAM supports it
4. Lower confidence threshold to reduce false positives

### Nest Detection Not Triggering

**Problem:** Nesting material not being detected

**Solutions:**
1. Ensure `best.pt` is in the working directory
2. Verify model was trained on your nest/twig dataset
3. Test with `obj.py` standalone to isolate the model
4. Check confidence threshold – may need to lower it for nest detection

---

## 📦 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Computer Vision** | YOLOv8 (Ultralytics) |
| **Frontend** | Streamlit |
| **Image Processing** | OpenCV (cv2) |
| **Hardware Communication** | HTTP requests, ESP32 |
| **Data Processing** | NumPy, Pandas |
| **Language** | Python 3.8+ |

---

## 📋 Dependencies

```
streamlit>=1.0
opencv-python>=4.5
numpy>=1.20
requests>=2.25
pandas>=1.3
ultralytics>=8.0
```

Install with:
```bash
pip install -r requirements.txt
```

*(Create `requirements.txt` with the above lines)*

---

## 🔮 Future Enhancements

- [ ] Multi-camera support with load balancing
- [ ] Historical detection database (SQLite)
- [ ] Configurable deterrence strategies (sound, light, spray)
- [ ] Mobile app for remote monitoring
- [ ] Model fine-tuning dashboard for custom datasets
- [ ] Email/SMS alerts
- [ ] Docker containerization for easy deployment

---

## 📝 License

[Add your license here – e.g., MIT, Apache 2.0, etc.]

---

## 🤝 Contributing

Contributions welcome! Please:
1. Test changes with your ESP32-CAM setup
2. Document any new features or config options
3. Update this README as needed

---

## 📧 Support

For issues, questions, or feedback:
- Check the **Troubleshooting** section above
- Verify ESP32 endpoints are reachable
- Test `obj.py` in isolation to diagnose detection vs. hardware issues

---

**SHIELD** – Protecting spaces. Preserving wildlife. 🕊️
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
