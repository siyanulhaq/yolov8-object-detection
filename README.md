# VisionAI Object Detection Platform

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![Flask](https://img.shields.io/badge/Backend-Flask-black)
![OpenCV](https://img.shields.io/badge/CV-OpenCV-green)

Production-style real-time computer vision platform with live object detection, browser dashboard telemetry, runtime controls, and event logging.

## Demo
![VisionAI Demo](docs/demo.gif)


## Key Features
- Live MJPEG stream with YOLOv8 bounding boxes
- Runtime controls for confidence threshold and auto-capture
- Dashboard cards for FPS, uptime, detections, screenshots
- Structured CSV detection logs for analytics

## Quick Start
```bash
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000`.

## Resume Bullets
- Built a real-time computer vision platform using YOLOv8, Flask, and OpenCV with API-driven runtime controls and dashboard telemetry.
- Implemented event-driven detection logging and screenshot capture pipeline to support monitoring and analytics workflows.
- Designed a production-style architecture connecting inference, APIs, and frontend observability in a single deployable app.

## Repository
https://github.com/siyanulhaq/yolov8-object-detection