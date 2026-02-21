import time
import streamlit as st
import cv2
import numpy as np
import requests
import pandas as pd
from ultralytics import YOLO

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="SHIELD | Bird Deterrence System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= STYLE =================
st.markdown("""
<style>
body { background-color:#0E1117; color:#FAFAFA; }
.block-container { padding:2rem 3rem; }
.card {
    background:#111827;
    padding:18px;
    border-radius:14px;
    border:1px solid #1F2937;
}
.metric {
    background:#111827;
    padding:14px;
    border-radius:12px;
    border:1px solid #1F2937;
    text-align:center;
}
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ================= CONFIG =================
ESP32_CAM_URL = "http://10.12.10.115/stream"
ESP32_CTRL_IP = "http://10.12.17.216"

BIRD_ON_URL  = f"{ESP32_CTRL_IP}/bird/on"
BIRD_OFF_URL = f"{ESP32_CTRL_IP}/bird/off"

NEST_CLASSES = {"nest", "twigs", "nesting_material"}

# ================= HELPERS =================
def notify_esp32(url):
    try:
        requests.get(url, timeout=0.3)
    except:
        pass

def get_snapshot(url, timeout=2.5):
    try:
        snap_url = url.replace("/stream", "/capture")
        r = requests.get(snap_url, timeout=timeout)
        if r.status_code == 200:
            img = np.frombuffer(r.content, np.uint8)
            return cv2.imdecode(img, cv2.IMREAD_COLOR)
    except:
        return None

# ================= SESSION STATE =================
st.session_state.setdefault("running", False)
st.session_state.setdefault("bird_present", False)
st.session_state.setdefault("bird_last_seen", 0.0)
st.session_state.setdefault("nest_alert", False)
st.session_state.setdefault("bird_count", 0)
st.session_state.setdefault("nest_count", 0)

# ================= LOAD MODELS =================
if "bird_model" not in st.session_state:
    with st.spinner("Loading YOLOv8n (Bird Detection)..."):
        st.session_state.bird_model = YOLO("yolov8n.pt")

if "nest_model" not in st.session_state:
    with st.spinner("Loading Custom Nest Model..."):
        st.session_state.nest_model = YOLO("best.pt")

bird_model = st.session_state.bird_model
nest_model = st.session_state.nest_model

# ================= HEADER =================
st.markdown("""
<h1>🛡️ SHIELD – AI Bird Deterrence System</h1>
<p style="color:#9CA3AF;">
Snapshot-based detection · Dual YOLO models · ESP32 actuation
</p>
<hr>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
with st.sidebar:
    st.header("⚙️ Runtime Settings")
    conf = st.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)
    loop_delay = st.slider("Loop Delay (ms)", 250, 2000, 400, 50)
    st.caption("Optimized for slow / unstable Wi-Fi")

# ================= CONTROLS =================
c1, c2, c3 = st.columns(3)

if c1.button("▶ Start", use_container_width=True):
    st.session_state.running = True

if c2.button("⏹ Stop", use_container_width=True):
    st.session_state.running = False
    notify_esp32(BIRD_OFF_URL)
    st.session_state.bird_present = False

status = "🟢 Running" if st.session_state.running else "🔴 Stopped"
c3.markdown(
    f"<div class='metric'><h3>Status</h3><h1>{status}</h1></div>",
    unsafe_allow_html=True
)

# ================= UI LAYOUT =================
left, right = st.columns([3, 1])
frame_box = left.empty()
info_box = right.empty()

# ================= MAIN LOOP =================
if st.session_state.running:
    last_time = time.time()

    while st.session_state.running:
        frame = get_snapshot(ESP32_CAM_URL)

        if frame is None:
            time.sleep(0.5)
            continue

        # ---------- RUN MODELS ----------
        bird_results = bird_model(frame, conf=conf, verbose=False)
        nest_results = nest_model(frame, conf=conf, verbose=False)

        bird_detected = False
        nest_detected = False
        detections = []

        # ---- Bird detection ----
        for r in bird_results:
            for box in r.boxes:
                cls = int(box.cls[0])
                name = bird_model.names[cls].lower()
                conf_v = float(box.conf[0])
                detections.append(f"{name} ({conf_v:.2f})")
                if name == "bird":
                    bird_detected = True

        # ---- Nest detection ----
        for r in nest_results:
            for box in r.boxes:
                cls = int(box.cls[0])
                name = nest_model.names[cls].lower()
                conf_v = float(box.conf[0])
                detections.append(f"{name} ({conf_v:.2f})")
                if name in NEST_CLASSES:
                    nest_detected = True

        # ---------- ESP32 LOGIC (BIRD ONLY) ----------
        now_ts = time.time()

        if bird_detected:
            st.session_state.bird_last_seen = now_ts
            if not st.session_state.bird_present:
                notify_esp32(BIRD_ON_URL)
                st.session_state.bird_present = True
                st.session_state.bird_count += 1
        else:
            if st.session_state.bird_present and (now_ts - st.session_state.bird_last_seen) > 3:
                notify_esp32(BIRD_OFF_URL)
                st.session_state.bird_present = False

        # ---------- NEST ALERT ----------
        st.session_state.nest_alert = nest_detected
        if nest_detected:
            st.session_state.nest_count += 1

        # ---------- DRAW FRAME ----------
        annotated = frame.copy()
        annotated = bird_results[0].plot(img=annotated)
        annotated = nest_results[0].plot(img=annotated)
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_box.image(annotated, use_container_width=True)

        # ---------- RIGHT PANEL ----------
        alert_html = ""
        if st.session_state.nest_alert:
            alert_html = """
            <div class='card' style='border:2px solid #DC2626;'>
                <h3 style='color:#F87171;'>🚨 Nest / Twigs Detected</h3>
                <p>
                Nesting activity detected.<br>
                <b>Automated deterrence suppressed.</b><br>
                Manual action recommended.
                </p>
            </div>
            """

        info_box.markdown(
            alert_html +
            "<div class='card'><h3>Detections</h3>" +
            ("<br>".join(detections) if detections else "No objects detected") +
            "</div>",
            unsafe_allow_html=True
        )

        # ---------- CHART ----------
        chart_data = pd.DataFrame({
            "Type": ["Birds", "Nest / Twigs"],
            "Count": [
                st.session_state.bird_count,
                st.session_state.nest_count
            ]
        })

        info_box.bar_chart(
            chart_data.set_index("Type"),
            use_container_width=True
        )

        # ---------- FPS ----------
        now = time.time()
        fps = 1 / max(now - last_time, 1e-6)
        last_time = now

        info_box.markdown(
            f"<div class='metric'><h3>System FPS</h3><h1>{fps:.1f}</h1></div>",
            unsafe_allow_html=True
        )

        time.sleep(loop_delay / 1000)

else:
    st.info("Click **Start** to begin detection.")

# ================= FOOTER =================
st.markdown("""
<hr>
<center><small>© 2026 SHIELD Systems · Final UI Build</small></center>
""", unsafe_allow_html=True)



