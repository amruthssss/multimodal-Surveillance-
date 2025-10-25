# Multi-Modal Surveillance Platform

A modular Flask-based surveillance and incident detection platform that fuses video (object + pose), emotion recognition, action recognition, and audio event classification. Generates alerts with optional email/SMS delivery and stores supporting media artifacts (clips, frames, audio snippets) for review in a web dashboard.

## Features
- User authentication (register/login) & per-user dashboards
- Real-time video streaming via Flask + Socket.IO
- Object detection & pose estimation (YOLOv8)
- Emotion recognition (Keras .h5 model placeholder)
- Action recognition (PyTorch EfficientNet placeholder)
- Audio event classification (PyTorch / librosa pipeline)
- Multi-modal fusion logic producing unified event labels + confidence
- Alerting (email/SMS via Twilio) and artifact recording
- SQLite + SQLAlchemy for events, users
- Pluggable async background worker (Celery or simple thread)
- Docker & docker-compose deployment ready

## Repository Structure
```
<root>
  app.py
  requirements.txt
  config/
  models/
  utils/
  templates/
  static/
  data/
  workers/
  deployment/
```
(See inline comments in tree for file purposes.)

## Quick Start
### 1. Create & Activate Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# (Optional GPU PyTorch) Choose appropriate CUDA wheel from https://pytorch.org/get-started/locally/
# Example (CUDA 12.1):
# pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

### 2. Environment Variables
Create a `.env` file at project root:
```
FLASK_SECRET=change_me
DATABASE_URL=sqlite:///data/logs/events.db
MAIL_SERVER=smtp.example.com
MAIL_PORT=587
MAIL_USERNAME=you@example.com
MAIL_PASSWORD=yourpassword
ALERT_FROM_EMAIL=alerts@example.com
ALERT_TO_EMAIL=destination@example.com
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_FROM=+15551234567
ALERT_TO_PHONE=+15557654321
```
Any unset values fall back to sane defaults in `config/config.py`.

### 3. Run Development Server
```bash
python app.py
```
Visit: http://127.0.0.1:5000

### 3b. Verify Installation
```bash
python scripts/verify_install.py
```

### 4. (Optional) Celery + Redis
If you enable Celery for async alert sending / ML tasks:
```bash
redis-server
celery -A workers.background_worker.celery_app worker --loglevel=INFO
```

## Multi-Modal Fusion (High-Level)
1. Frame -> YOLO: objects, persons, pose keypoints
2. Cropped face -> emotion model
3. Sequence of frames / features -> action model
4. Audio window -> audio classifier
5. Fusion: rules + weighted averaging to produce event label + confidence

## Minimal Fusion Pseudocode
```python
scores = {
  'violence': w1*action['fight'] + w2*pose['aggressive'] + w3*audio['shout'],
  'intrusion': w4*objects['person']*zone_factor + w5*time_factor,
}
label = max(scores, key=scores.get)
confidence = scores[label]
```

## ASCII Architecture
```
+--------------+     video frames      +------------------+
|   Camera     | --------------------> |   YOLO Detector  |
+--------------+                       +---------+--------+
          audio stream                           |
                |                                v
                |                        +---------------+
                +--------------------->  | Pose / Objects|
                                         +-------+-------+
                                                 |
                     +---------------------------+--------------------+
                     |                           |                    |
                     v                           v                    v
            +---------------+          +----------------+     +---------------+
            | Emotion Model |          | Action Model   |     | Audio Model   |
            +-------+-------+          +--------+-------+     +-------+-------+
                    \                        |                      /
                     \                       |                     /
                      \                      |                    /
                       +---------------------v-------------------+
                       |          Fusion & Alert Logic            |
                       +------------------+----------------------+
                                          |
                                          v
                                   +-------------+
                                   |  Dashboard  |
                                   +-------------+
```

## Deployment
### Docker (CPU)
```
docker build -t multimodal-surveillance ./deployment
# or if Dockerfile in root adjust path
```
### docker-compose
```
docker compose -f deployment/docker-compose.yml up --build
```

## Extending Models
- Replace placeholder model files in `models/` with trained weights.
- Ensure wrapper classes normalize inputs consistently.

## License
MIT (add LICENSE file as needed)

## Testing Fusion Logic
Run unit tests:
```bash
pytest -q
```
Run fusion demo simulation:
```bash
python scripts/fusion_demo.py
```

## Roadmap / TODO
- Add JWT API endpoints
- WebRTC ingest option
- GPU compose profile
- Advanced anomaly detection (temporal)
- Frontend charting of historical alerts
```
