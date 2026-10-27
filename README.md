<p align="center">
  <img src="docs/assets/sentinel-vision-hero.svg" alt="Animated Sentinel Vision operations display" width="100%" />
</p>

<p align="center">
  <a href="https://github.com/amruthssss/multimodal-Surveillance-"><img alt="Project" src="https://img.shields.io/badge/Sentinel-Vision-ff3b35?style=for-the-badge&logo=proton&logoColor=white"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12%2B-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="React" src="https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=101010">
  <img alt="Local AI" src="https://img.shields.io/badge/Inference-Local-222?style=for-the-badge&logo=ollama&logoColor=white">
</p>

<p align="center"><b>Local-first video and audio event monitoring with operator-reviewed alerts.</b></p>

---

## ◉ Mission

Sentinel Vision combines a React operations dashboard, a Node.js account/configuration API, and a Python computer-vision service. It accepts camera streams and recordings, analyzes visual and audio signals locally, and gives operators evidence to review before acting.

The project is a prototype. Its detections are advisory and need evaluation on representative clips before operational use.

## ✦ What it does

- **Connects video sources:** built-in camera, RTSP, IP camera, public YouTube stream, or local video file.
- **Analyzes visual events:** fire, smoke, traffic accidents, and explosion-like visual cues.
- **Analyzes audio events:** YAMNet classifies gunshot and explosion sounds when the operator enables microphone monitoring.
- **Displays an alert queue:** confirmed incidents include camera/source, event, severity, confidence, repeat count, and a locally saved image or audio sample.
- **Deduplicates and acknowledges incidents:** repeated detections update one incident for up to two minutes; higher severity escalates it. Operators can acknowledge alerts in the dashboard.
- **Learns from explicit review:** correct, false-alarm, and missed-event labels are saved and adjust per-event thresholds. This changes decision thresholds; it does not retrain neural-network weights.
- **Measures performance:** the Event Intelligence view reports reviewed-example precision/recall. An offline evaluator measures labeled video and audio clips.

## ⌁ Alert policy

| Signal | Confirmation | Escalation |
| --- | --- | --- |
| Fire, smoke, accident | Event-specific confidence threshold and 2 matching results in the last 3 inference samples | Critical requires 3 matching samples and at least 85% confidence |
| Explosion in video | Same 2-of-3 confirmation; visual evidence is checked for the event | A strong fast path can mark critical at 95% only when at least 3 explosion cues agree; otherwise critical requires 3 samples |
| Gunshot/explosion audio | YAMNet event score ≥0.65 in one window, or ≥0.30 in two consecutive windows | Strong matches respond in one ~1-second analysis window; lower scores need temporal confirmation. Score ≥0.75 is critical; lower confirmed scores are high. |
| Repeated same event | One camera/event incident is updated during a 120-second window | The incident keeps the highest severity and refreshed evidence on escalation |

Video evidence is saved as JPEG; audio evidence as WAV. Operators can preview snapshots or play audio evidence directly in the dashboard. Evidence stays under `data/alerts/` and is excluded from Git. External delivery remains behind a rollout gate: at least 20 human-reviewed examples per event class and ≥90% precision for that class. Those class metrics appear in Settings, but delivery providers must be configured separately; no external messages are sent yet. The intended first channel is email/push for confirmed high-severity alerts, with SMS reserved for critical alerts only.

## ▧ Architecture

```mermaid
flowchart LR
  CAM[Camera / YouTube / video] --> CAP[Python capture worker]
  CAP --> MJPEG[Latest-frame MJPEG feed]
  CAP --> INF[Python inference worker]
  INF --> VISION[YOLO + visual event heuristics]
  MIC[Operator microphone] --> AUDIO[YAMNet audio classifier]
  VISION --> POLICY[Temporal confirmation + thresholds]
  AUDIO --> POLICY
  POLICY --> ALERT[Deduplicated alert + local evidence]
  ALERT --> SQLITE[(SQLite incident and review records)]
  DASH[React dashboard] <--> API[Flask AI API]
  API <--> NODE[Node / Express / MongoDB]
  INF --> API
  ALERT --> DASH
```

## ◈ Models

| Capability | Current implementation | Notes |
| --- | --- | --- |
| General object context | YOLO26 Nano COCO checkpoint (`yolo26n.pt`) | Objects provide context; the COCO model is not an explosion classifier. |
| Fire/smoke | Optional YOLO event checkpoint plus visual heuristics | Configured through `FIRE_SMOKE_MODEL_REPO` / `FIRE_SMOKE_MODEL_FILE`; availability depends on model download or local weights. |
| Traffic accident | Optional YOLO accident checkpoint plus motion/vehicle context | Runs periodically to reduce inference load. |
| Video explosion | Temporal visual heuristics | No verified dedicated lightweight explosion checkpoint is bundled. |
| Gunshot/explosion audio | YAMNet ONNX and AudioSet class map | Microphone capture is opt-in. |

The model cards describe their authors’ datasets and results; those are not performance measurements for this app. No labeled evaluation set is bundled by default.

## ⌘ Run locally

### Requirements

- Windows 10/11 or a compatible desktop OS
- Python 3.12+
- Node.js 24+ and npm
- MongoDB for account/configuration services
- A webcam, supported network camera, or video/audio clips

### Configure

```powershell
git clone https://github.com/amruthssss/multimodal-Surveillance-.git
cd multimodal-Surveillance-
Copy-Item .env.example .env
```

Edit `.env`: set a unique `JWT_SECRET`, a `MONGODB_URI`, and any local model/source settings. URL-encode special characters in MongoDB passwords. Never commit `.env`, model weights, video, or alert evidence.

### Install

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Push-Location surveillance-backend
npm install
Pop-Location
Push-Location surveillance-frontend
npm install
Pop-Location
```

For NVIDIA GPU inference, install the PyTorch/CUDA wheel matching the machine from the [official PyTorch selector](https://pytorch.org/get-started/locally/) before installing the remaining Python requirements.

### Start

Run all services from the repository root:

```powershell
.\START.bat
```

Or start each in its own terminal:

```powershell
# Python camera and AI API
.\venv\Scripts\Activate.ps1
python app.py

# Node account/configuration API
Push-Location surveillance-backend
npm start

# React dashboard
Push-Location surveillance-frontend
npm run dev
```

Open **http://127.0.0.1:4300**. The Python API uses port **5000**; the Node API uses **5001**. Configure a source in the dashboard to start video inference. Microphone monitoring is separately opt-in.

## ◎ Review detections and measure accuracy

The dashboard’s Event Intelligence page lets an operator label a detected event as correct or a false alarm, or label a normal sample as a missed event. Reviewed labels are stored in SQLite and appear in the evaluation cards. Threshold updates are stored in the ignored local file `data/detection_feedback.json`.

For a separate clip-level evaluation set, copy `evaluation/manifest.example.csv` to `evaluation/manifest.csv`, add labeled local media paths, then run:

```powershell
python scripts/evaluate_labeled_videos.py --manifest evaluation/manifest.csv --output evaluation/report.json
```

The manifest supports `normal`, `explosion`, `fire`, `smoke`, `accident`, and `gunshot`. Use `media_type=video` for visual events and `media_type=audio` for gunshot/audio-explosion clips. Include normal clips and keep a final evaluation set separate from clips used to tune thresholds. See [evaluation/README.md](evaluation/README.md).

## ⇄ API reference

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Python service health |
| `GET /api/status` | Camera and model state |
| `GET /api/detections/recent` | Latest per-camera detections |
| `GET /api/alerts` | Recent persisted incidents |
| `POST /api/alerts/{id}/acknowledge` | Acknowledge an incident |
| `GET /api/alerts/{id}/evidence` | View its local JPEG/WAV evidence |
| `POST /api/detections/feedback` | Persist an operator ground-truth label |
| `GET /api/detections/reviews` | Retrieve reviewed examples |
| `GET /api/detections/evaluation` | Calculate per-event precision, recall, and F1 |
| `POST /api/camera/configure` | Configure a video source |
| `POST /api/audio-analysis` | Analyze an opt-in microphone window |

## ⚙ Configuration

| Variable | Purpose |
| --- | --- |
| `MONGODB_URI`, `JWT_SECRET` | Node API database and authentication |
| `FLASK_PORT`, `PORT` | Python and Node service ports |
| `VITE_API_URL`, `VITE_AI_URL` | Frontend API addresses |
| `VIDEO_SOURCE`, `FRAME_INTERVAL` | Camera device and inference sampling |
| `VIDEO_MAX_WIDTH`, `VIDEO_JPEG_QUALITY` | Stream bandwidth/quality balance |
| `EVENT_MODEL_INTERVAL` | Specialized visual model sampling interval |
| `FIRE_SMOKE_MODEL_REPO`, `TRAFFIC_ACCIDENT_MODEL_REPO`, `YAMNET_MODEL_REPO` | Optional Hugging Face model sources |
| `FIRE_SMOKE_CONFIDENCE`, `ACCIDENT_CONFIDENCE` | Event-model confidence cutoffs |
| `AUDIO_GUNSHOT_THRESHOLD`, `AUDIO_EXPLOSION_THRESHOLD` | Audio classifier cutoffs |
| `DATABASE_URL` | SQLite event/review database or compatible SQLAlchemy URL |

See [.env.example](.env.example) for the complete local configuration template.

## ⛨ Privacy and limitations

- Services bind to localhost by default. Do not expose them to the public internet without authentication, access controls, and a reviewed deployment design.
- Video and audio evidence remain on the machine running the Python service; the alert API serves evidence only on that local service.
- Account tokens are stored by the browser application. Review the authentication and deployment model before using real accounts.
- A few reviewed events are not enough to establish accuracy. Track false positives and misses per event on representative day/night, indoor/outdoor, and camera conditions.
- This prototype is not a substitute for human judgment or certified life-safety systems. Do not rely on it as the only means of detecting emergencies.

## License

Check the licenses of the project and each downloaded model before redistribution or commercial use. Model weights are not included in this repository.
