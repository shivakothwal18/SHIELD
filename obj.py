import cv2
import numpy as np
from ultralytics import YOLO
import requests

# Your ESP32-CAM URL - try these formats:
# Option 1: Stream endpoint
ESP_CAM_URL = "http://10.12.10.115/stream"
# Option 2: Snapshot endpoint (uncomment if stream doesn't work)
# ESP_CAM_URL = "http://10.250.6.216/capture"
# Option 3: Direct stream
# ESP_CAM_URL = "http://10.250.6.216/stream"

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

def get_snapshot(url):
    """Get single frame using snapshot method"""
    try:
        # Try capture endpoint first
        capture_url = url.replace('/stream', '/capture')
        response = requests.get(capture_url, timeout=3)
        if response.status_code == 200:
            img_array = np.frombuffer(response.content, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
    except Exception as e:
        print(f"Snapshot error: {e}")
    return None

def get_stream_opencv(url):
    """Use OpenCV's VideoCapture for stream"""
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret:
            return frame
    return None

def main():
    print("Starting ESP32-CAM Object Detection...")
    print(f"Connecting to: {ESP_CAM_URL}")
    print("Press 'q' to quit\n")
    
    # Try to determine which method works
    method = None
    
    while True:
        frame = None
        
        # Try snapshot method
        if method is None or method == 'snapshot':
            frame = get_snapshot(ESP_CAM_URL)
            if frame is not None and method is None:
                method = 'snapshot'
                print("✓ Using snapshot method")
        
        # Try OpenCV stream
        if frame is None and (method is None or method == 'opencv'):
            frame = get_stream_opencv(ESP_CAM_URL)
            if frame is not None and method is None:
                method = 'opencv'
                print("✓ Using OpenCV stream method")
        
        if frame is None:
            print("Failed to get frame, retrying...")
            cv2.waitKey(1000)  # Wait 1 second before retry
            continue
        
        # Run YOLOv8 inference
        results = model(frame, conf=0.4, verbose=False)
        
        # Visualize results
        annotated_frame = results[0].plot()
        
        # Print detections
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls]
                detections.append(f"{name} ({conf:.2f})")
        
        if detections:
            print(f"Detected: {', '.join(detections)}")
        
        # Show frame
        cv2.imshow('ESP32-CAM Object Detection', annotated_frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print("\nDetection stopped.")

if __name__ == "__main__":
    main()