"""Main Flask application entry point.
Provides: auth (basic), dashboard, video feed route, Socket.IO events, and model init stubs.
This is a simplified scaffold – not production ready.
"""
from __future__ import annotations
import io
import json
import os
import base64
import shutil
import threading
import time
from collections import OrderedDict
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime
from typing import Generator
from uuid import uuid4
import numpy as np

import cv2
from flask import Flask, Response, request, redirect, session, flash, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

from config.config import Config
from utils import db as db_utils
from utils.feedback_learner import FeedbackThresholdLearner
from utils.detection_policy import TemporalDetectionPolicy
# The same visual agent handles live-stream and single-frame inference.
from enhanced_final_ultra_system import EnhancedUltraSystem


app = Flask(__name__, static_folder='new_frontend/build', static_url_path='/')
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024
app.config['SECRET_KEY'] = Config.SECRET_KEY
FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.getenv('FRONTEND_URL', 'http://localhost:3000').split(',')
    if origin.strip()
]

# Enable CORS for frontend integration with credentials support
CORS(app,
     origins=FRONTEND_ORIGINS,
     supports_credentials=True,
     allow_headers=['Content-Type', 'Authorization'],
     methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])

socketio = SocketIO(
    app,
    cors_allowed_origins=FRONTEND_ORIGINS,
    async_mode='threading',
)

db_utils.init_db()  # ensure tables

# In-memory user cache (replace with real User model / SQLAlchemy later)
_USERS = {}

# AI Detection System Instances
emotion_model = None
action_model = None
enhanced_system = None
ai_init_lock = threading.Lock()
inference_lock = threading.Lock()
detection_results_lock = threading.Lock()
audio_event_streaks = {'gunshot': 0, 'explosion': 0}
recent_detection_snapshots = OrderedDict()
alert_record_lock = threading.Lock()
feedback_learner = FeedbackThresholdLearner(os.path.join(Config.ROOT_DIR, 'data', 'detection_feedback.json'))
detection_policy = TemporalDetectionPolicy(feedback_learner)

def initialize_ai_systems():
    """Initialize all AI detection systems"""
    global emotion_model, action_model, enhanced_system
    
    try:
        print("\n" + "="*70)
        print("🧠 INITIALIZING ENHANCED ULTRA HYBRID SYSTEM")
        print("="*70 + "\n")
        
        # ==========================================
        # PRIORITY 1: Enhanced Ultra Hybrid System
        # ==========================================
        print("🚀 Loading Enhanced Ultra Hybrid System...")
        print("   Architecture: visual heuristics with optional YOLO models")
        print("   Custom event weights: optional (not included in the repository)")
        print("   Objects: available COCO model classes\n")
        
        enhanced_system = EnhancedUltraSystem(use_audio=False)
        
        print("\n✅ Enhanced Ultra Hybrid System Active")
        print("   Mode: heuristic event analysis plus available object detection")
        
        # ==========================================
        # PRIORITY 2: Smart Fusion Engine (Learning Mode) - DISABLED
        # ==========================================
        # print("\n🚀 Loading Smart Fusion Engine...")
        # print("   Strategy: YOLO primary (every frame)")
        # print("   Learning: Other models boost accuracy (periodic)")
        # print("   Result: Fast + Smart + Non-interfering\n")
        
        # smart_fusion = SmartFusionEngine(
        #     learning_interval=10,  # Run learning every 10 frames
        #     boost_threshold=0.6    # Boost when YOLO < 60% confident
        # )
        
        # # Get individual model references
        # emotion_model = smart_fusion.emotion
        # action_model = smart_fusion.action
        
        # print("\n✅ Smart Fusion Engine Active")
        # print(f"   Mode: Learning (models boost, don't interfere)")
        # print(f"   Speed: ~30 FPS (YOLO primary)")
        # print(f"   Accuracy: 85%+ (with model boosting)")
        
        print("\n" + "="*70)
        print("✅ AI SYSTEM READY")
        print("="*70)
        print("\n📊 SYSTEM ARCHITECTURE:")
        print("   🎯 Custom event weights are optional and must be supplied locally")
        print("   🚀 Object detection depends on the available YOLO model")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"❌ AI Systems initialization failed: {e}")
        import traceback
        traceback.print_exc()


def ensure_ai_system():
    """Initialize the existing visual agent once and return it."""
    with ai_init_lock:
        if enhanced_system is None:
            initialize_ai_systems()
        if enhanced_system is None:
            raise RuntimeError('The visual analysis agent could not be initialized')
        return enhanced_system


def decode_frame_payload(payload):
    """Decode a base64 image/data URL into an OpenCV BGR frame."""
    if not isinstance(payload, str) or not payload.strip():
        raise ValueError('An image payload is required')
    encoded = payload.split(',', 1)[1] if ',' in payload else payload
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise ValueError('Image payload must be valid base64') from error
    frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError('Image payload is not a decodable image')
    return frame


def analyze_frame_with_agent(frame, camera_id, frame_num=0):
    """Run the shared visual agent and store its normalized result."""
    system = ensure_ai_system()
    with inference_lock:
        result = system.process_frame_for_api(frame, frame_num)
    if result.get('event') == 'error':
        raise RuntimeError(result.get('error', 'Frame analysis failed'))

    result, final_result = detection_policy.apply(result, camera_id, frame_num)
    timestamp = time.time()
    with detection_results_lock:
        snapshot = {
            'timestamp': timestamp,
            'event_id': f'{camera_id}:{int(timestamp * 1000)}',
            'camera_id': str(camera_id),
            'results': {'fusion_result': final_result, 'agent_result': result},
        }
        detection_results[str(camera_id)] = snapshot
        recent_detection_snapshots[snapshot['event_id']] = snapshot
        recent_detection_snapshots.move_to_end(snapshot['event_id'])
        while len(recent_detection_snapshots) > 1000:
            recent_detection_snapshots.popitem(last=False)
    return result, final_result, timestamp

# Camera capture thread control
capture_lock = threading.Lock()
video_capture = None
last_frame = None
latest_camera_frame = None
camera_frame_sequence = 0
stream_condition = threading.Condition()
stream_stop_event = threading.Event()
capture_worker_thread = None
inference_worker_thread = None
stream_state = 'idle'
stream_error = None
stream_fps = 0.0
detection_results = {}  # Store latest detection results
app_started_monotonic = time.monotonic()
camera_configured = False  # Track if camera has been configured
camera_config = {
    'source': None,  # None until user configures (was 'builtin')
    'rtsp_url': None,
    'ip_url': None,
    'youtube_url': None,
    'video_file': None
}


def get_video_capture():
    global video_capture, camera_configured
    
    # Don't auto-open camera - wait for user configuration
    if not camera_configured:
        print("⏸️  Camera not configured yet. Waiting for user selection...")
        return None
    
    if video_capture is not None:
        return video_capture

    source = camera_config['source']
    if source == 'builtin':
        try:
            device_index = int(Config.VIDEO_SOURCE)
        except (TypeError, ValueError):
            device_index = Config.VIDEO_SOURCE
        print(f"📹 Opening built-in camera: {device_index}")
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if isinstance(device_index, int) else [cv2.CAP_ANY]
        for backend in backends:
            candidate = cv2.VideoCapture(device_index, backend)
            if candidate.isOpened():
                video_capture = candidate
                break
            candidate.release()
        if video_capture is not None:
            video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            video_capture.set(cv2.CAP_PROP_FPS, 30)
            video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    elif source == 'rtsp':
        stream_url = camera_config['rtsp_url']
        print("📹 Opening RTSP stream")
        video_capture = _open_network_capture(stream_url)
    elif source == 'ip':
        stream_url = camera_config['ip_url']
        print("📹 Opening IP camera stream")
        video_capture = _open_network_capture(stream_url)
    elif source == 'youtube':
        stream_url = _resolve_youtube_stream(camera_config['youtube_url'])
        print("📺 Opening extracted YouTube stream")
        video_capture = _open_network_capture(stream_url)
    elif source == 'video':
        video_capture = cv2.VideoCapture(camera_config['video_file'])

    if video_capture is not None and not video_capture.isOpened():
        video_capture.release()
        video_capture = None
    return video_capture


def _resolve_youtube_stream(page_url):
    """Resolve a public YouTube video/live URL to a progressive stream URL."""
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    try:
        from yt_dlp import YoutubeDL
    except ImportError as error:
        raise RuntimeError('YouTube support needs yt-dlp. Install requirements.txt and retry.') from error

    options = {
        'format': 'best[height<=720]/best',
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
        'socket_timeout': 15,
        'extractor_retries': 1,
    }
    node_path = shutil.which('node')
    if node_path:
        options['js_runtimes'] = {'node': {'path': node_path}}
    try:
        with YoutubeDL(options) as downloader:
            info = downloader.extract_info(page_url, download=False)
    except Exception as error:
        raise RuntimeError(f'Could not resolve that YouTube link: {error}') from error
    stream_url = info.get('url') if isinstance(info, dict) else None
    if not stream_url:
        raise RuntimeError('YouTube did not provide a compatible video stream for this link.')
    return stream_url


def _open_network_capture(stream_url):
    """Open an RTSP/HTTP stream with bounded open/read timeouts when supported."""
    params = [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10_000,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10_000,
    ]
    try:
        capture = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG, params)
    except (TypeError, cv2.error):
        capture = cv2.VideoCapture(stream_url)
    if not capture.isOpened():
        capture.release()
        return None
    return capture


def reset_video_capture():
    """Reset video capture to apply new configuration"""
    global video_capture, camera_configured, stream_stop_event
    global capture_worker_thread, inference_worker_thread, latest_camera_frame
    global last_frame, stream_state, stream_error, camera_frame_sequence, stream_fps
    detection_policy.reset()
    stream_stop_event.set()
    with stream_condition:
        stream_condition.notify_all()
    cap = video_capture
    video_capture = None
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    for worker in (capture_worker_thread, inference_worker_thread):
        if worker and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=1.5)
    capture_worker_thread = None
    inference_worker_thread = None
    with stream_condition:
        latest_camera_frame = None
        last_frame = None
        camera_frame_sequence = 0
        stream_state = 'idle'
        stream_error = None
        stream_fps = 0.0
        stream_condition.notify_all()
    camera_configured = False
    print("🔄 Video capture reset")


def start_stream_workers():
    """Capture/encode continuously; run expensive AI on a separate drop-old queue."""
    global stream_stop_event, capture_worker_thread, inference_worker_thread, stream_state
    if capture_worker_thread and capture_worker_thread.is_alive():
        return
    stream_stop_event = threading.Event()
    stop_event = stream_stop_event
    stream_state = 'connecting'
    capture_worker_thread = threading.Thread(
        target=_capture_latest_frames, args=(stop_event,), name='camera-capture', daemon=True
    )
    inference_worker_thread = threading.Thread(
        target=_infer_latest_frames, args=(stop_event,), name='camera-inference', daemon=True
    )
    capture_worker_thread.start()
    inference_worker_thread.start()


def _capture_latest_frames(stop_event):
    """Keep the newest JPEG fresh without waiting for inference or browser clients."""
    global latest_camera_frame, camera_frame_sequence, last_frame
    global stream_state, stream_error, stream_fps, video_capture
    failures = 0
    blank_frames = 0
    fps_started = time.monotonic()
    fps_frames = 0
    while not stop_event.is_set() and camera_configured:
        try:
            cap = get_video_capture()
        except Exception as error:
            failures += 1
            stream_state = 'reconnecting'
            stream_error = str(error)
            stop_event.wait(min(8.0, 0.5 * (2 ** min(failures, 4))))
            continue
        if cap is None:
            failures += 1
            stream_state = 'reconnecting'
            stream_error = stream_error or 'Unable to open the configured video source'
            if camera_config.get('source') == 'video':
                stream_state = 'ended'
                break
            stop_event.wait(min(8.0, 0.5 * (2 ** min(failures, 4))))
            continue

        ok, frame = cap.read()
        if not ok or frame is None:
            failures += 1
            stream_state = 'reconnecting'
            stream_error = 'No frames received; reconnecting'
            with capture_lock:
                if video_capture is cap:
                    video_capture = None
            try:
                cap.release()
            except Exception:
                pass
            if camera_config.get('source') == 'video':
                stream_state = 'ended'
                break
            stop_event.wait(min(8.0, 0.5 * (2 ** min(failures, 4))))
            continue

        failures = 0
        # A privacy shutter or disabled camera can still yield successful reads,
        # often as a stream of all-zero frames. Do not report that as a live feed.
        if float(frame.mean()) < 1.0 and float(frame.std()) < 1.0:
            blank_frames += 1
        else:
            blank_frames = 0
        if blank_frames >= 8:
            stream_state = 'blank'
            stream_error = 'Camera is returning a black image. Check its privacy shutter, Windows camera access, or selected device.'
        else:
            stream_state = 'online'
            stream_error = None
        fps_frames += 1
        elapsed = time.monotonic() - fps_started
        if elapsed >= 2.0:
            stream_fps = fps_frames / elapsed
            fps_frames = 0
            fps_started = time.monotonic()

        height, width = frame.shape[:2]
        max_width = max(320, int(os.getenv('VIDEO_MAX_WIDTH', '960')))
        if width > max_width:
            scaled_height = int(height * max_width / width)
            frame = cv2.resize(frame, (max_width, scaled_height), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(
            '.jpg', frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), max(45, min(90, int(os.getenv('VIDEO_JPEG_QUALITY', '78'))))],
        )
        if not ok:
            continue
        with stream_condition:
            latest_camera_frame = frame
            camera_frame_sequence += 1
            last_frame = encoded.tobytes()
            stream_condition.notify_all()


def _infer_latest_frames(stop_event):
    """Process only fresh frames and skip backlog when a model is slower than video."""
    global stream_state, stream_error
    last_sequence = 0
    interval = max(1, int(Config.FRAME_INTERVAL))
    while not stop_event.is_set() and camera_configured:
        with stream_condition:
            stream_condition.wait_for(
                lambda: stop_event.is_set() or camera_frame_sequence > last_sequence,
                timeout=1.0,
            )
            if stop_event.is_set():
                break
            if camera_frame_sequence <= last_sequence or latest_camera_frame is None:
                continue
            sequence = camera_frame_sequence
            frame = latest_camera_frame.copy()
            last_sequence = sequence
        if sequence % interval:
            continue
        try:
            result, final_result, timestamp = analyze_frame_with_agent(frame, 'camera_0', sequence)
            if final_result['should_alert']:
                record_alert_event(
                    final_result['label'], final_result['risk'], final_result['confidence'],
                    'camera_0', source='video', reasoning=final_result.get('reasoning'),
                    frame=frame, timestamp=timestamp,
                )
            detection_data = {
                **result,
                'timestamp': timestamp,
                'frame': sequence,
                'risk': final_result['risk'],
                'objects': result.get('objects', {}),
                'vehicles': result.get('vehicles', {}),
                'people': result.get('people', {}),
            }
            socketio.emit('detection_update', detection_data)
            socketio.emit('inference', {
                'event': final_result['label'],
                'confidence': final_result['confidence'],
                'risk': final_result['risk'],
                'objects': detection_data['objects'],
                'timestamp': timestamp,
            })
        except Exception as error:
            stream_error = f'Inference error: {error}'
            print(f"⚠️ Visual agent analysis failed: {error}")


def generate_frames() -> Generator[bytes, None, None]:
    sequence_sent = 0
    while camera_configured:
        with stream_condition:
            stream_condition.wait_for(
                lambda: not camera_configured or camera_frame_sequence > sequence_sent or stream_state == 'ended',
                timeout=5.0,
            )
            if not camera_configured:
                break
            if camera_frame_sequence <= sequence_sent:
                continue
            sequence_sent = camera_frame_sequence
            payload = last_frame
        if payload:
            yield b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(payload)).encode() + b'\r\n\r\n' + payload + b'\r\n'


@app.route('/')
def index():
    frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:3000').split(',', 1)[0].strip()
    return redirect(frontend_url.rstrip('/') + '/')


@app.route('/login')
def login_page():
    frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:3000').split(',', 1)[0].strip()
    return redirect(frontend_url.rstrip('/') + '/login')


@app.route('/api/auth/register', methods=['POST'])
def api_register():
    """Register new user with email/mobile verification"""
    data = request.get_json()
    username = data.get('username', '').strip()
    email = data.get('email', '').strip()
    mobile = data.get('mobile', '').strip()
    password = data.get('password', '')
    
    if not username or not password or not email or not mobile:
        return jsonify({'success': False, 'message': 'All fields required'}), 400
    
    if username in _USERS:
        return jsonify({'success': False, 'message': 'Username already exists'}), 400
    
    # Store pending user (not confirmed yet)
    import random
    otp = str(random.randint(100000, 999999))
    _USERS[username] = {
        'email': email,
        'mobile': mobile,
        'pw_hash': generate_password_hash(password),
        'otp': otp,
        'verified': False,
        'created_at': time.time()
    }
    
    # Email/SMS delivery is not configured; expose the OTP only for local development.
    print(f"\n📧 EMAIL OTP for {email}: {otp}")
    print(f"📱 SMS OTP for {mobile}: {otp}\n")
    
    return jsonify({
        'success': True, 
        'message': 'Registration successful. Enter the development OTP printed in the Python service terminal.',
        'requiresVerification': True,
        'email': email,
        'mobile': mobile
    })


@app.route('/api/auth/verify-otp', methods=['POST'])
def verify_otp():
    """Verify OTP code"""
    data = request.get_json()
    username = data.get('username', '').strip()
    otp = data.get('otp', '').strip()
    
    user = _USERS.get(username)
    if not user:
        return jsonify({'success': False, 'message': 'User not found'}), 404
    
    if user.get('otp') == otp:
        user['verified'] = True
        user.pop('otp', None)
        session['user'] = username
        return jsonify({
            'success': True, 
            'message': 'Email verified successfully!',
            'user': {'username': username, 'email': user['email']}
        })
    
    return jsonify({'success': False, 'message': 'Invalid OTP code'}), 400


@app.route('/api/auth/resend-otp', methods=['POST'])
def resend_otp():
    """Resend OTP code"""
    data = request.get_json()
    username = data.get('username', '').strip()
    
    user = _USERS.get(username)
    if not user:
        return jsonify({'success': False, 'message': 'User not found'}), 404
    
    # Generate new OTP
    import random
    otp = str(random.randint(100000, 999999))
    user['otp'] = otp
    
    print(f"\n📧 New OTP for {user['email']}: {otp}\n")
    
    return jsonify({'success': True, 'message': 'Development OTP regenerated and printed in the Python service terminal.'})


@app.route('/api/auth/login', methods=['POST'])
def api_login():
    """Login with email verification"""
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '')
    
    user = _USERS.get(username)
    if not user or not check_password_hash(user['pw_hash'], password):
        return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
    
    if not user.get('verified', False):
        return jsonify({
            'success': False, 
            'message': 'Please verify your email/mobile first',
            'requiresVerification': True,
            'username': username
        }), 403
    
    session['user'] = username
    return jsonify({
        'success': True, 
        'message': 'Login successful',
        'user': {'username': username, 'email': user['email'], 'mobile': user.get('mobile', '')}
    })


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(os.getenv('FRONTEND_URL', 'http://localhost:3000').rstrip('/') + '/login')


@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(os.getenv('FRONTEND_URL', 'http://localhost:3000').rstrip('/') + '/login')
    return redirect(os.getenv('FRONTEND_URL', 'http://localhost:3000').rstrip('/') + '/dashboard')


@app.route('/add_camera', methods=['POST'])
def add_camera():
    """Add a new camera to the system"""
    if 'user' not in session:
        return redirect(os.getenv('FRONTEND_URL', 'http://localhost:3000').rstrip('/') + '/login')
    
    rtsp_url = request.form.get('rtsp_url', '')
    if rtsp_url:
        # For now, just flash a message (you can implement actual camera storage later)
        flash(f'Camera added: {rtsp_url}', 'success')
    else:
        flash('Please provide a valid RTSP URL', 'error')
    
    return redirect(os.getenv('FRONTEND_URL', 'http://localhost:3000').rstrip('/') + '/dashboard')


@app.route('/video_feed')
def video_feed():
    if not camera_configured:
        return jsonify({'error': 'Configure a camera before requesting its video feed'}), 503
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'message': 'Python Flask Media Server is running',
        'services': {
            'camera': 'ready' if camera_configured else 'not_configured',
            'ai_processing': 'ready' if enhanced_system is not None else 'not_initialized',
            'websocket': 'active'
        }
    })

@app.route('/<path:path>')
def serve(path):
    return send_from_directory(app.static_folder, path)

@app.route('/dashboard_api')
def dashboard_api():
    """API endpoint for dashboard data"""
    return jsonify({
        'cameras': get_active_cameras(),
        'stats': get_system_stats(),
        'recent_detections': get_recent_detections_data(),
        'alerts': get_active_alerts_data()
    })

def get_active_cameras():
    """Report only the locally configured capture; never return stream credentials."""
    cap = video_capture
    if not camera_configured or cap is None or not cap.isOpened():
        return []
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    return [{
        'id': 'camera_0',
        'status': 'online',
        'location': {'name': 'Configured camera'},
        'type': camera_config['source'] or 'camera',
        'fps': fps,
        'resolution': f'{width}x{height}',
        'last_detection': None,
    }]

def get_system_stats():
    """Get system statistics"""
    import psutil
    cameras = get_active_cameras()
    
    return {
        'activeCameras': len(cameras),
        'totalDetections': len(detection_results),
        'systemStatus': 'operational' if cameras else 'waiting_for_camera',
        'uptimeSeconds': int(time.monotonic() - app_started_monotonic),
        'cpuUsage': f"{psutil.cpu_percent():.1f}%",
        'memoryUsage': f"{psutil.virtual_memory().percent:.1f}%",
        'diskUsage': f"{psutil.disk_usage(str(Config.ROOT_DIR)).percent:.1f}%"
    }

def get_recent_detections_data():
    """Return the latest real inference result for each configured camera."""
    with detection_results_lock:
        entries = list(detection_results.items())
    detections = []
    for camera_id, data in entries:
        final = data.get('results', {}).get('fusion_result', {})
        detections.append({
            'id': data.get('event_id', camera_id),
            'timestamp': datetime.fromtimestamp(data['timestamp']).isoformat(),
            'type': final.get('label', 'unknown'),
            'severity': final.get('risk', 'LOW'),
            'confidence': final.get('confidence', 0.0),
            'camera_id': camera_id,
            'candidate_event': final.get('candidate_event', final.get('label', 'normal')),
            'confirmation_count': final.get('confirmation_count', 0),
            'reasoning': final.get('reasoning', {}),
        })
    return sorted(detections, key=lambda item: item['timestamp'], reverse=True)

def get_active_alerts_data():
    """Return recent persisted incidents, including acknowledgement and evidence."""
    session = db_utils.get_session()
    try:
        rows = session.query(db_utils.AlertRecord).order_by(db_utils.AlertRecord.last_seen.desc()).limit(50).all()
        return [{
            'id': row.id,
            'timestamp': row.last_seen.isoformat(),
            'firstSeen': row.first_seen.isoformat(),
            'type': row.event,
            'risk': row.risk,
            'confidence': row.confidence,
            'camera': row.camera_id,
            'source': row.source,
            'occurrences': row.occurrences,
            'message': f"{row.occurrences} detection{'s' if row.occurrences != 1 else ''} · {row.camera_id}",
            'evidenceUrl': f"/api/alerts/{row.id}/evidence" if row.evidence_path else None,
            'acknowledged': row.acknowledged_at is not None,
            'acknowledgedAt': row.acknowledged_at.isoformat() if row.acknowledged_at else None,
            'reasoning': row.reasoning,
        } for row in rows]
    finally:
        session.close()


def _write_alert_evidence(frame=None, audio_samples=None, sample_rate=None):
    """Save one local JPEG or WAV evidence item; never upload media externally."""
    evidence_dir = Path(Config.DATA_DIR) / 'alerts'
    evidence_dir.mkdir(parents=True, exist_ok=True)
    stem = uuid4().hex
    if frame is not None:
        ok, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return None
        filename = f'{stem}.jpg'
        (evidence_dir / filename).write_bytes(encoded.tobytes())
        return filename
    if audio_samples is not None and sample_rate:
        import wave
        samples = np.asarray(audio_samples, dtype=np.float32).reshape(-1)
        pcm = (np.clip(np.nan_to_num(samples), -1.0, 1.0) * 32767).astype('<i2')
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(int(sample_rate))
            output.writeframes(pcm.tobytes())
        filename = f'{stem}.wav'
        (evidence_dir / filename).write_bytes(buffer.getvalue())
        return filename
    return None


def record_alert_event(event, risk, confidence, camera_id, source='video', reasoning=None,
                       frame=None, audio_samples=None, sample_rate=None, timestamp=None):
    """Persist a deduplicated alert; refresh incident evidence only on escalation."""
    event = str(event).lower()
    if event not in FeedbackThresholdLearner.BASE_THRESHOLDS:
        return None
    now = datetime.fromtimestamp(timestamp or time.time())
    key = f'{camera_id}:{event}'
    rank = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2, 'CRITICAL': 3}
    session = db_utils.get_session()
    try:
        with alert_record_lock:
            row = session.query(db_utils.AlertRecord).filter_by(alert_key=key).order_by(db_utils.AlertRecord.last_seen.desc()).first()
            if row and (now - row.last_seen).total_seconds() <= 120:
                changed = False
                escalated = rank.get(str(risk).upper(), 0) > rank.get(row.risk, 0)
                if escalated:
                    row.risk = str(risk).upper()
                    row.evidence_path = _write_alert_evidence(frame, audio_samples, sample_rate) or row.evidence_path
                    row.last_seen = now
                    changed = True
                elif (now - row.last_seen).total_seconds() >= 5:
                    row.occurrences += 1
                    row.last_seen = now
                    changed = True
                row.confidence = max(float(row.confidence or 0), float(confidence or 0))
                if reasoning:
                    row.reasoning = json.dumps(reasoning, ensure_ascii=False)[:12000]
                if changed:
                    session.commit()
                return row.id

            evidence = _write_alert_evidence(frame, audio_samples, sample_rate)
            row = db_utils.AlertRecord(
                alert_key=key,
                event=event,
                risk=str(risk).upper(),
                confidence=float(confidence or 0),
                camera_id=str(camera_id),
                source=str(source),
                occurrences=1,
                evidence_path=evidence,
                reasoning=json.dumps(reasoning or {}, ensure_ascii=False)[:12000],
                first_seen=now,
                last_seen=now,
            )
            session.add(row)
            session.commit()
            return row.id
    except Exception:
        session.rollback()
        app.logger.exception('Could not persist alert incident')
        return None
    finally:
        session.close()


# Socket.IO events
@socketio.on('connect')
def handle_connect():
    emit('status', {'message': 'Connected to Python Media Server'})
    print(f"🔌 Client connected: {request.sid}")


@socketio.on('disconnect')
def handle_disconnect():
    print(f"🔌 Client disconnected: {request.sid}")


@socketio.on('request_last_frame')
def handle_last_frame():
    global last_frame
    if last_frame:
        emit('frame', {'jpg': last_frame})


@socketio.on('analyze_frame')
def handle_analyze_frame(data):
    """Analyze an uploaded frame with the same agent used by the live feed."""
    try:
        data = data or {}
        camera_id = data.get('cameraId', 'camera_1')
        frame = decode_frame_payload(data.get('frameData', ''))
        result, final_result, timestamp = analyze_frame_with_agent(frame, camera_id)
        analysis_result = {
            'cameraId': str(camera_id),
            'timestamp': timestamp,
            'event': final_result['label'],
            'risk_level': final_result['risk'],
            'confidence': final_result['confidence'],
            'reasoning': result.get('reasoning', {}),
            'objects': result.get('objects', {}),
            'vehicles': result.get('vehicles', {}),
            'people': result.get('people', {}),
            'motion_spike': result.get('motion_spike', 0.0),
        }
        emit('analysis_result', analysis_result)
        socketio.emit('surveillance_update', {
            'type': 'agent_analysis',
            'data': analysis_result
        })
    except Exception as e:
        print(f"❌ Frame analysis error: {e}")
        emit('analysis_error', {'error': str(e)})


# ================================
# AI DETECTION API ENDPOINTS
# ================================

@app.route('/api/init-ai-systems', methods=['POST'])
def init_ai_systems():
    """Initialize AI detection systems"""
    try:
        ensure_ai_system()
        return jsonify({
            'success': True,
            'message': 'AI systems initialized successfully',
            'systems': {
                'enhanced_agent': enhanced_system is not None,
                'fire_smoke_model': enhanced_system is not None and enhanced_system.fire_smoke_model is not None,
                'accident_model': enhanced_system is not None and enhanced_system.accident_model is not None,
                'yamnet_model': enhanced_system is not None and enhanced_system.audio_event_model is not None,
                'emotion_model': emotion_model is not None,
                'action_model': action_model is not None
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/camera-analysis', methods=['POST'])
def analyze_camera_frame():
    """Analyze one uploaded image with the existing visual agent."""
    try:
        data = request.get_json(silent=True) or {}
        b64_img = data.get('image')
        camera_id = data.get('cameraId', 'camera_1')
        try:
            frame = decode_frame_payload(b64_img)
        except ValueError as error:
            return jsonify({'success': False, 'error': str(error)}), 400

        agent_result, final_result, timestamp = analyze_frame_with_agent(frame, camera_id)

        return jsonify({
            'success': True,
            'camera_id': str(camera_id),
            'timestamp': timestamp,
            'results': {
                'final_result': final_result,
                'agent_result': agent_result,
            }
        })

    except Exception as e:
        print(f"❌ Camera analysis error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/detection-status', methods=['GET'])
def get_detection_status():
    """Get current AI detection system status"""
    try:
        status = {
            'ai_systems_initialized': {
                'enhanced_agent': enhanced_system is not None,
                'emotion_model': emotion_model is not None,
                'action_model': action_model is not None
            },
            'detection_results_count': len(detection_results),
            'latest_detections': {}
        }
        
        # Get latest results for each camera
        for camera_id, data in detection_results.items():
            status['latest_detections'][camera_id] = {
                'timestamp': data['timestamp'],
                'age_seconds': time.time() - data['timestamp']
            }
            
            # Add summary of latest results
            results = data['results']
            if 'fusion_result' in results:
                fusion = results['fusion_result']
                status['latest_detections'][camera_id].update({
                    'fusion_label': fusion.get('label', 'none'),
                    'fusion_risk': fusion.get('risk', 'low'),
                    'fusion_confidence': fusion.get('confidence', 0)
                })
        
        if enhanced_system:
            status['enhanced_agent_status'] = 'ready'

        return jsonify(status)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def get_system_status():
    """Get comprehensive system status for dashboard"""
    try:
        cap = video_capture
        fps = round(stream_fps, 1) if stream_fps else (int(cap.get(cv2.CAP_PROP_FPS)) if cap else 0)
        
        status = {
            'hybridAI': {
                'active': enhanced_system is not None,
                'confidence': None
            },
            'emotionModel': {
                'active': emotion_model is not None,
                'confidence': None
            },
            'actionModel': {
                'active': action_model is not None,
                'confidence': None
            },
            'fireSmokeModel': {
                'active': enhanced_system is not None and enhanced_system.fire_smoke_model is not None,
                'confidence': None
            },
            'accidentModel': {
                'active': enhanced_system is not None and enhanced_system.accident_model is not None,
                'confidence': None
            },
            'audioModel': {
                'active': enhanced_system is not None and enhanced_system.audio_event_model is not None,
                'confidence': None
            },
            'cameraStatus': {
                'online': camera_configured and stream_state == 'online',
                'configured': camera_configured,
                'state': stream_state,
                'error': stream_error,
                'source': camera_config.get('source'),
                'fps': fps,
                'latency': None
            }
        }
        
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/detections/recent', methods=['GET'])
def get_recent_detections():
    """Get recent detection events"""
    try:
        recent = get_recent_detections_data()
        return jsonify([{
            'id': detection['id'],
            'timestamp': detection['timestamp'],
            'label': detection['type'],
            'risk': detection['severity'],
            'confidence': detection['confidence'],
            'camera': detection['camera_id'],
        } for detection in recent[:100]])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/alerts', methods=['GET'])
def get_active_alerts():
    """Get active alerts"""
    try:
        alerts = get_active_alerts_data()
        return jsonify([{
            'id': alert['id'],
            'timestamp': alert['timestamp'],
            'firstSeen': alert['firstSeen'],
            'type': alert['type'],
            'risk': alert['risk'],
            'confidence': alert['confidence'],
            'message': alert['message'],
            'camera': alert['camera'],
            'source': alert['source'],
            'occurrences': alert['occurrences'],
            'evidenceUrl': alert['evidenceUrl'],
            'acknowledged': alert['acknowledged'],
            'acknowledgedAt': alert['acknowledgedAt'],
        } for alert in alerts[:50]])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/alerts/<int:alert_id>/acknowledge', methods=['POST'])
def acknowledge_alert(alert_id):
    session = db_utils.get_session()
    try:
        alert = session.query(db_utils.AlertRecord).filter_by(id=alert_id).first()
        if alert is None:
            return jsonify({'success': False, 'message': 'Alert not found'}), 404
        alert.acknowledged_at = datetime.utcnow()
        session.commit()
        return jsonify({'success': True, 'message': 'Alert acknowledged'})
    finally:
        session.close()


@app.route('/api/alerts/<int:alert_id>/evidence', methods=['GET'])
def alert_evidence(alert_id):
    session = db_utils.get_session()
    try:
        alert = session.query(db_utils.AlertRecord).filter_by(id=alert_id).first()
        if alert is None or not alert.evidence_path:
            return jsonify({'error': 'Evidence not found'}), 404
        evidence_dir = Path(Config.DATA_DIR) / 'alerts'
        return send_from_directory(evidence_dir, alert.evidence_path, as_attachment=False)
    finally:
        session.close()


@app.route('/api/detections/feedback', methods=['POST'])
def submit_detection_feedback():
    """Persist a ground-truth label and use it as a reward for threshold learning."""
    data = request.get_json(silent=True) or {}
    event_id = data.get('eventId')
    actual = str(data.get('actualEvent', '')).lower()
    supported = {'normal', 'explosion', 'fire', 'accident', 'smoke', 'gunshot'}
    if not isinstance(event_id, str) or actual not in supported:
        return jsonify({'success': False, 'message': 'eventId and a supported actualEvent label are required'}), 400
    with detection_results_lock:
        match = recent_detection_snapshots.get(event_id)
    if match is None:
        return jsonify({'success': False, 'message': 'This event is no longer the latest result; refresh and try again.'}), 409
    final = match.get('results', {}).get('fusion_result', {})
    predicted = str(final.get('label', 'normal')).lower()
    outcome = 'correct' if actual == predicted else 'missed' if predicted == 'normal' else 'false_alarm'
    learned_event = predicted if predicted != 'normal' else actual if actual != 'normal' else None
    if outcome == 'missed' and actual not in feedback_learner.BASE_THRESHOLDS:
        return jsonify({'success': False, 'message': 'Could not associate the missed event with a supported class.'}), 400
    session = db_utils.get_session()
    try:
        existing = session.query(db_utils.DetectionReview).filter_by(event_id=event_id).first()
        if existing:
            return jsonify({'success': False, 'message': 'This detection has already been reviewed.'}), 409
        review = db_utils.DetectionReview(
            event_id=event_id,
            camera_id=str(match.get('camera_id', 'camera_0')),
            predicted_label=predicted,
            actual_label=actual,
            confidence=float(final.get('candidate_confidence', final.get('confidence', 0.0))),
            risk=str(final.get('risk', 'LOW')),
            reasoning=json.dumps(final.get('reasoning', {}), ensure_ascii=False)[:12000],
            created_at=datetime.utcfromtimestamp(match['timestamp']),
        )
        session.add(review)
        session.commit()
        learned = feedback_learner.record(learned_event, outcome) if learned_event else None
        missed_update = None
        if actual != 'normal' and actual != predicted:
            missed_update = feedback_learner.record(actual, 'missed')
    except ValueError as error:
        session.rollback()
        return jsonify({'success': False, 'message': str(error)}), 400
    except Exception:
        session.rollback()
        app.logger.exception('Could not persist detection review')
        return jsonify({'success': False, 'message': 'Could not save this review.'}), 500
    finally:
        session.close()
    return jsonify({'success': True, 'message': 'Review saved; event threshold updated.', 'learning': learned, 'missedEventLearning': missed_update})


@app.route('/api/detections/reviews', methods=['GET'])
def list_detection_reviews():
    limit = min(500, max(1, request.args.get('limit', default=100, type=int)))
    session = db_utils.get_session()
    try:
        rows = session.query(db_utils.DetectionReview).order_by(db_utils.DetectionReview.reviewed_at.desc()).limit(limit).all()
        return jsonify([{
            'eventId': row.event_id, 'camera': row.camera_id,
            'predicted': row.predicted_label, 'actual': row.actual_label,
            'confidence': row.confidence, 'risk': row.risk,
            'createdAt': row.created_at.isoformat(), 'reviewedAt': row.reviewed_at.isoformat(),
        } for row in rows])
    finally:
        session.close()


@app.route('/api/detections/evaluation', methods=['GET'])
def detection_evaluation():
    """Compute per-class metrics from human-reviewed ground truth only."""
    classes = ('explosion', 'fire', 'accident', 'smoke', 'gunshot')
    session = db_utils.get_session()
    try:
        rows = session.query(db_utils.DetectionReview).all()
        metrics = {}
        for label in classes:
            tp = sum(row.predicted_label == label and row.actual_label == label for row in rows)
            fp = sum(row.predicted_label == label and row.actual_label != label for row in rows)
            fn = sum(row.actual_label == label and row.predicted_label != label for row in rows)
            precision = tp / (tp + fp) if tp + fp else None
            recall = tp / (tp + fn) if tp + fn else None
            f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
            metrics[label] = {
                'truePositives': tp, 'falsePositives': fp, 'falseNegatives': fn,
                'precision': round(precision, 4) if precision is not None else None,
                'recall': round(recall, 4) if recall is not None else None,
                'f1': round(f1, 4) if f1 is not None else None,
                'labeledSamples': sum(row.predicted_label == label or row.actual_label == label for row in rows),
            }
        return jsonify({'labeledReviews': len(rows), 'metrics': metrics})
    finally:
        session.close()


@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get analytics data for charts"""
    try:
        return jsonify({
            'hourlyDetections': [],
            'threatDistribution': [],
            'modelAccuracy': [],
            'message': 'Analytics are unavailable until event history is persisted.'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/audio/toggle', methods=['POST'])
def toggle_audio():
    """Compatibility endpoint: browser microphone is controlled by the dashboard."""
    return jsonify({'success': True, 'message': 'Use the dashboard microphone control.'})


@app.route('/api/audio-analysis', methods=['POST', 'OPTIONS'])
def analyze_audio_window():
    """Analyze one mono float32 microphone window with local YAMNet."""
    if request.method == 'OPTIONS':
        return jsonify({'success': True}), 200
    try:
        payload = request.get_json(silent=True) or {}
        encoded = payload.get('audioData')
        sample_rate = int(payload.get('sampleRate', 0))
        if not isinstance(encoded, str) or not encoded:
            return jsonify({'error': 'audioData must be base64 float32 samples'}), 400
        if sample_rate < 8000 or sample_rate > 96000:
            return jsonify({'error': 'sampleRate must be between 8000 and 96000 Hz'}), 400
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > 512_000 or len(raw) % 4:
            return jsonify({'error': 'Audio window must be at most 128,000 float32 samples'}), 413
        samples = np.frombuffer(raw, dtype='<f4').copy()
        if not samples.size:
            return jsonify({'error': 'Audio window is empty'}), 400
        system = ensure_ai_system()
        with inference_lock:
            scores = system.classify_audio_events(samples, sample_rate)
        thresholds = {
            'gunshot': float(os.getenv('AUDIO_GUNSHOT_THRESHOLD', '0.20')),
            'explosion': float(os.getenv('AUDIO_EXPLOSION_THRESHOLD', '0.20')),
        }
        events = {}
        for label, score in scores.items():
            audio_event_streaks[label] = audio_event_streaks[label] + 1 if score >= thresholds[label] else 0
            events[label] = audio_event_streaks[label] >= 2
            if events[label]:
                record_alert_event(
                    label,
                    'CRITICAL' if score >= 0.75 else 'HIGH',
                    float(score),
                    'microphone',
                    source='audio',
                    reasoning={'model': 'YAMNet', 'score': float(score), 'threshold': thresholds[label]},
                    audio_samples=samples,
                    sample_rate=sample_rate,
                )
        return jsonify({'success': True, 'scores': scores, 'events': events})
    except (ValueError, TypeError, base64.binascii.Error) as error:
        return jsonify({'error': str(error)}), 400
    except Exception as error:
        return jsonify({'error': str(error)}), 503


@app.route('/api/camera/configure', methods=['POST', 'OPTIONS'])
def configure_camera():
    """Configure camera source (RTSP/IP/YouTube/Builtin)"""
    global camera_configured
    
    # Handle OPTIONS preflight request
    if request.method == 'OPTIONS':
        return jsonify({'success': True}), 200
    
    print("\n" + "=" * 60)
    print("🎬 CAMERA CONFIGURATION REQUEST RECEIVED")
    print("=" * 60)
    print(f"⏰ Timestamp: {datetime.now().strftime('%H:%M:%S')}")
    
    try:
        data = request.get_json()
        if not isinstance(data, dict):
            return jsonify({'success': False, 'message': 'A JSON camera configuration is required'}), 400
        
        source = data.get('source', 'builtin')
        if source == 'file':
            source = 'video'
        if source not in {'builtin', 'rtsp', 'ip', 'youtube', 'video'}:
            return jsonify({'success': False, 'message': 'Unsupported camera source'}), 400
        rtsp_url = (data.get('rtspUrl') or '').strip()
        ip_url = (data.get('ipUrl') or '').strip()
        youtube_url = (data.get('youtubeUrl') or '').strip()
        video_file = data.get('videoFilePath', '')
        
        print(f"📹 Camera Configuration Request:")
        print(f"   Source: {source}")
        
        # Validate URLs
        if source == 'rtsp' and not rtsp_url:
            return jsonify({'success': False, 'message': 'RTSP URL is required'}), 400
        if source == 'rtsp' and urlparse(rtsp_url).scheme.lower() not in {'rtsp', 'rtsps'}:
            return jsonify({'success': False, 'message': 'RTSP URLs must start with rtsp:// or rtsps://'}), 400
        
        if source == 'ip' and not ip_url:
            return jsonify({'success': False, 'message': 'IP Camera URL is required'}), 400
        if source == 'ip' and (urlparse(ip_url).scheme.lower() not in {'http', 'https', 'rtsp', 'rtsps'} or not urlparse(ip_url).netloc):
            return jsonify({'success': False, 'message': 'Enter a valid HTTP(S) or RTSP camera stream URL'}), 400
        
        if source == 'youtube' and not youtube_url:
            return jsonify({'success': False, 'message': 'YouTube URL is required'}), 400
        if source == 'youtube':
            host = (urlparse(youtube_url).hostname or '').lower()
            if urlparse(youtube_url).scheme.lower() != 'https' or host not in {'youtube.com', 'www.youtube.com', 'm.youtube.com', 'youtu.be', 'www.youtube-nocookie.com'}:
                return jsonify({'success': False, 'message': 'Enter a valid HTTPS YouTube video or live stream link'}), 400
        
        if source == 'video' and not video_file:
            return jsonify({'success': False, 'message': 'Video file path is required'}), 400
        
        # Validate video file exists if video source
        if source == 'video':
            import os
            if not os.path.exists(video_file):
                return jsonify({'success': False, 'message': f'Video file not found: {video_file}'}), 400
            if not video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v')):
                return jsonify({'success': False, 'message': 'Unsupported video format. Use: MP4, AVI, MOV, MKV, FLV, WMV, M4V'}), 400
        
        # Update global camera configuration
        camera_config['source'] = source
        camera_config['rtsp_url'] = rtsp_url if source == 'rtsp' else None
        camera_config['ip_url'] = ip_url if source == 'ip' else None
        camera_config['youtube_url'] = youtube_url if source == 'youtube' else None
        camera_config['video_file'] = video_file if source == 'video' else None
        
        # Reset video capture to apply new configuration
        reset_video_capture()
        
        # Mark as configured
        camera_configured = True
        
        # Test the new configuration
        cap = get_video_capture()
        if cap is None or not cap.isOpened():
            camera_configured = False
            reset_video_capture()
            if source == 'builtin':
                message = 'Could not open the computer camera. Check Windows camera permission, close apps using it, or choose another device index in VIDEO_SOURCE.'
            elif source == 'youtube':
                message = 'Could not open the YouTube stream. Check that it is public/live and that yt-dlp is installed and up to date.'
            else:
                message = 'Could not open that network stream. Check the address, credentials, firewall, and camera stream path.'
            return jsonify({
                'success': False,
                'message': message
            }), 400
        
        print(f"✅ Camera configured successfully: {source}")
        
        # Keep video flowing independently while large models initialize and infer.
        start_stream_workers()
        
        return jsonify({
            'success': True,
            'message': f'Camera configured to {source}',
            'config': {
                'source': camera_config['source']
            }
        })
    except RuntimeError as e:
        camera_configured = False
        reset_video_capture()
        return jsonify({'success': False, 'message': str(e)}), 400
    except Exception as e:
        print(f"❌ Camera configuration error: {e}")
        camera_configured = False
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    """Start camera feed processing"""
    global camera_configured
    try:
        if not camera_configured:
            return jsonify({'success': False, 'message': 'Camera not configured yet'}), 400
        
        print("▶️ Starting camera feed...")
        # Camera will automatically start processing in generate_frames()
        return jsonify({'success': True, 'message': 'Camera feed started'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    """Stop camera capture and inference workers and release the source."""
    global camera_configured
    try:
        print("⏹️ Stopping camera feed...")
        # Stop processing but keep configuration
        reset_video_capture()
        return jsonify({'success': True, 'message': 'Camera feed stopped'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/config/alerts', methods=['POST'])
def save_alert_config():
    """Save alert configuration"""
    try:
        config = request.get_json()
        # Save to database or config file
        print(f"💾 Alert configuration saved: {config}")
        return jsonify({'success': True, 'message': 'Configuration saved successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/events/<event_id>/pdf', methods=['GET'])
def download_event_pdf(event_id):
    """Generate and download event PDF report"""
    try:
        from io import BytesIO
        from flask import send_file
        
        # Create a simple text file as placeholder (replace with actual PDF generation)
        pdf_content = f"""
Event Report
============
Event ID: {event_id}
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
Type: Security Alert
Risk Level: HIGH
Confidence: 85%

Details:
--------
This is a placeholder PDF report.
Integrate with reportlab or similar library for actual PDF generation.
"""
        
        pdf_buffer = BytesIO()
        pdf_buffer.write(pdf_content.encode('utf-8'))
        pdf_buffer.seek(0)
        
        return send_file(
            pdf_buffer,
            mimetype='text/plain',
            as_attachment=True,
            download_name=f'event_{event_id}.txt'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import os
    port = int(os.environ.get('FLASK_PORT', 5000))
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    print(f"🐍 Starting Python Flask Media Server on port {port}")
    print(f"🎥 Video feed: http://localhost:{port}/video_feed") 
    print(f"🏥 Health check: http://localhost:{port}/health")
    print(f"📊 Dashboard API: http://localhost:{port}/dashboard")
    
    socketio.run(app, host=host, port=port, debug=False)
