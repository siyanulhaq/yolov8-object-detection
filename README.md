# VisionAI - Real-Time Object Detection Platform

Production-style computer vision application that streams live detections to a browser dashboard, logs events for analytics, and exposes runtime controls for thresholding and auto-capture.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-API%20%2B%20UI-black?style=flat-square&logo=flask)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=flat-square&logo=opencv)

## Why This Project Matters

This project demonstrates the difference between a model demo and an industry-relevant system:
- Real-time inference pipeline with operational telemetry (FPS, uptime, event counts)
- Human-in-the-loop monitoring dashboard for live decisions
- Structured event logging (`detection_log.csv`) for post-analysis and BI
- Runtime controls for inference confidence and auto-capture behavior
- Alerting signal for detection-triggered events

## Core Features

- Live MJPEG video stream with YOLOv8 bounding boxes
- Real-time dashboard cards: FPS, total detections, screenshots, uptime
- Current detections and top-object leaderboard
- Manual snapshot capture from UI
- Auto-screenshot on detection with configurable cooldown (1-60s)
- Configurable confidence threshold (5%-95%) from UI
- Detection event alert (audio + toast)
- CSV logging with timestamp, object class, confidence, and screenshot path

## Architecture

```mermaid
flowchart LR
    A[Webcam] --> B[OpenCV Frame Capture]
    B --> C[YOLOv8 Inference]
    C --> D[Annotated Frame]
    D --> E[/video MJPEG Stream]
    C --> F[Detection Parser]
    F --> G[In-Memory State]
    F --> H[detection_log.csv]
    G --> I[/stats API]
    J[Dashboard UI] --> I
    J --> K[/config API]
    K --> G
    J --> L[/screenshot API]
```

## API Surface

- `GET /` - Dashboard UI
- `GET /video` - Live MJPEG stream
- `GET /stats` - Runtime telemetry and control-state snapshot
- `POST /config` - Update live config:
  - `confidence_threshold` (`0.05`-`0.95`)
  - `auto_screenshot_enabled` (`true`/`false`)
  - `auto_screenshot_cooldown_sec` (`1`-`60`)
- `GET /screenshot` - Manual snapshot

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Open [http://localhost:5000](http://localhost:5000).

> Model loading behavior: uses `best.pt` if present; otherwise falls back to `yolov8n.pt`.

## Demo Script (30-45 seconds)

1. Start app and show dashboard loads instantly.
2. Move objects in front of camera and highlight live detections.
3. Adjust confidence slider and show detections tighten/expand.
4. Enable/disable auto-screenshot and change cooldown live.
5. Trigger detection alert and show CSV logging output.

## Project Structure

```text
.
|-- app.py
|-- index.html
|-- requirements.txt
|-- README.md
|-- screenshots/              # generated at runtime
`-- detection_log.csv         # generated at runtime
```

## Industry Relevance

This architecture maps to practical use-cases:
- Retail traffic/object analytics
- Safety and compliance monitoring
- Smart surveillance event capture
- Edge AI proof-of-concept deployments

## Production Hardening Roadmap

- Replace in-memory state with Redis for multi-worker scaling
- Add authentication/authorization around control endpoints
- Containerize with Docker and health checks
- Add unit/integration tests for API and config validation
- Add CI pipeline (lint, tests, dependency/security scanning)

## Resume Bullet Ideas

- Built a real-time computer vision platform using YOLOv8, Flask, and OpenCV with live dashboard telemetry and configurable inference controls.
- Implemented event-driven auto-capture pipeline with cooldown management, alert signaling, and structured CSV logging for downstream analytics.
- Designed API-backed frontend controls enabling dynamic runtime reconfiguration without service restarts.
