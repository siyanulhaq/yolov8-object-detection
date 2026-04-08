import cv2
import csv
import os
import time
import threading
from datetime import datetime
from collections import defaultdict
from flask import Flask, Response, jsonify, render_template_string
from ultralytics import YOLO

app = Flask(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH = "best.pt" if os.path.exists("best.pt") else "yolov8n.pt"
DEFAULT_CONFIDENCE_THRESHOLD = 0.50
SCREENSHOT_DIR = "screenshots"
LOG_FILE = "detection_log.csv"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# ─── State ────────────────────────────────────────────────────────────────────
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)
lock = threading.Lock()

state = {
    "fps": 0,
    "total_detections": 0,
    "current_objects": {},
    "top_objects": defaultdict(int),
    "screenshots": 0,
    "alert_events": 0,
    "last_detection_epoch": 0.0,
    "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
    "auto_screenshot_enabled": True,
    "auto_screenshot_cooldown_sec": 5,
    "last_auto_screenshot_epoch": 0.0,
    "uptime_start": time.time(),
    "last_frame_time": time.time(),
}

# Init CSV log
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["timestamp", "object", "confidence", "screenshot"])

# ─── Frame Generator ──────────────────────────────────────────────────────────
def generate_frames():
    prev_time = time.time()
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        with lock:
            threshold = state["confidence_threshold"]
        results = model(frame, conf=threshold, verbose=False)
        annotated = results[0].plot(
            line_width=2,
            font_size=0.6,
        )

        # FPS
        frame_count += 1
        now = time.time()
        elapsed = now - prev_time
        if elapsed >= 1.0:
            with lock:
                state["fps"] = round(frame_count / elapsed, 1)
            frame_count = 0
            prev_time = now

        # Parse detections
        current = {}
        boxes = results[0].boxes
        screenshot_file = ""
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])
                current[label] = round(conf * 100, 1)

                with lock:
                    state["total_detections"] += 1
                    state["top_objects"][label] += 1
                    state["last_detection_epoch"] = now

        # Auto screenshot on detection (cooldown-protected)
        if current:
            with lock:
                auto_enabled = state["auto_screenshot_enabled"]
                cooldown = state["auto_screenshot_cooldown_sec"]
                last_auto = state["last_auto_screenshot_epoch"]

            if auto_enabled and (now - last_auto) >= cooldown:
                screenshot_file = f"screenshots/auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(screenshot_file, annotated)
                with lock:
                    state["screenshots"] += 1
                    state["last_auto_screenshot_epoch"] = now
                    state["alert_events"] += 1

        # Log to CSV
        if boxes is not None:
            with open(LOG_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                for box in boxes:
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    conf = float(box.conf[0])
                    writer.writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        label, round(conf, 3), screenshot_file
                    ])

        with lock:
            state["current_objects"] = current

        # Overlay: FPS + timestamp
        h, w = annotated.shape[:2]
        cv2.rectangle(annotated, (0, 0), (220, 50), (0, 0, 0), -1)
        cv2.putText(annotated, f"FPS: {state['fps']}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2)
        cv2.putText(annotated, datetime.now().strftime("%H:%M:%S"), (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")


# ─── Screenshot endpoint ──────────────────────────────────────────────────────
@app.route("/screenshot")
def screenshot():
    success, frame = cap.read()
    if success:
        with lock:
            threshold = state["confidence_threshold"]
        results = model(frame, conf=threshold, verbose=False)
        annotated = results[0].plot()
        fname = f"screenshots/snap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(fname, annotated)
        with lock:
            state["screenshots"] += 1
        return jsonify({"status": "saved", "file": fname})
    return jsonify({"status": "error"})


# ─── Stats endpoint ───────────────────────────────────────────────────────────
@app.route("/stats")
def stats():
    with lock:
        uptime = int(time.time() - state["uptime_start"])
        top = sorted(state["top_objects"].items(), key=lambda x: -x[1])[:5]
        return jsonify({
            "fps": state["fps"],
            "total_detections": state["total_detections"],
            "current_objects": state["current_objects"],
            "top_objects": top,
            "screenshots": state["screenshots"],
            "alert_events": state["alert_events"],
            "confidence_threshold": state["confidence_threshold"],
            "auto_screenshot_enabled": state["auto_screenshot_enabled"],
            "auto_screenshot_cooldown_sec": state["auto_screenshot_cooldown_sec"],
            "uptime": f"{uptime // 60}m {uptime % 60}s",
        })


@app.route("/config", methods=["POST"])
def config():
    payload = flask_request_json()
    with lock:
        if "confidence_threshold" in payload:
            raw = float(payload["confidence_threshold"])
            state["confidence_threshold"] = max(0.05, min(0.95, raw))
        if "auto_screenshot_enabled" in payload:
            state["auto_screenshot_enabled"] = bool(payload["auto_screenshot_enabled"])
        if "auto_screenshot_cooldown_sec" in payload:
            cooldown = float(payload["auto_screenshot_cooldown_sec"])
            state["auto_screenshot_cooldown_sec"] = max(1.0, min(60.0, cooldown))
    return jsonify({"status": "ok"})


def flask_request_json():
    from flask import request
    data = request.get_json(silent=True)
    if isinstance(data, dict):
        return data
    return {}


# ─── Video feed ───────────────────────────────────────────────────────────────
@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ─── Main UI ──────────────────────────────────────────────────────────────────
HTML = open("index.html", encoding="utf-8").read()

@app.route("/")
def index():
    return render_template_string(HTML)


if __name__ == "__main__":
    print("\n🚀 VisionAI running at http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
