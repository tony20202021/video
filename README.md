# Video Surveillance & Person Recognition System

Real-time video stream analysis from multiple IP cameras (iCSee) with person detection, group classification, and individual resident identification.

## Services

- **Backend** — FastAPI, RTSP stream ingestion, ML orchestration
- **ML Service** — ONNX Runtime inference (detect / classify / identify), training mode
- **Telegram Bot** — aiogram 3.0, event feed, person management

## Stack

Python 3.11+, FastAPI, MongoDB, ONNX Runtime, aiogram 3.0, Docker Compose

## Docs

See [docs/technical_spec.md](docs/technical_spec.md) for full technical specification.
