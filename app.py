"""
Flask App — OpenVINO Face Counter
Versi 2 Kamera: /video_feed/0 dan /video_feed/1
Semua logika deteksi IDENTIK, hanya dijalankan 2 instance paralel.
"""
from flask import Flask, render_template, Response, jsonify, request, stream_with_context
from flask_cors import CORS
import cv2, time, os, sys, logging, threading, json, io, csv
import numpy as np
import socket
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
os.environ['OPENCV_LOG_LEVEL'] = 'FATAL'
import warnings; warnings.filterwarnings('ignore')

from core.config import Config
from core.face_counter import OpenVINOFaceCounter
from utils.session_manager import SessionManager

app  = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────────────────────────

# 2 counter — index 0 = kamera 1, index 1 = kamera 2
counters: list[OpenVINOFaceCounter | None] = [None, None]

session_manager        = None
detection_enabled      = True
stream_lock            = threading.Lock()
current_session_active = False
session_start_time     = None

# SSE subscribers
_sse_clients: list[list] = []
_sse_lock = threading.Lock()

# ─────────────────────────────────────────────────────────────────────────────
# SSE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _broadcast_sse(event: str, data: dict):
    payload = f"event: {event}\ndata: {json.dumps(data)}\n\n"
    with _sse_lock:
        for q in list(_sse_clients):
            q.append(payload)


def _on_new_face(face_id: str, meta: dict):
    _broadcast_sse('new_face', {
        'face_id':         face_id,
        'first_seen':      meta.get('first_seen'),
        'total_in_db':     meta.get('total_in_db'),
        'detection_count': meta.get('detection_count', 1),
    })


def _on_face_updated(face_id: str, meta: dict):
    _broadcast_sse('face_updated', {
        'face_id':         face_id,
        'last_seen':       meta.get('last_seen'),
        'detection_count': meta.get('detection_count'),
    })


# ─────────────────────────────────────────────────────────────────────────────
# INIT — inisialisasi 2 counter berdasarkan Config.CAMERAS
# ─────────────────────────────────────────────────────────────────────────────

def get_counter(cam_id: int = 0) -> OpenVINOFaceCounter:
    """Ambil atau buat counter untuk kamera cam_id (0 atau 1)."""
    global counters
    if counters[cam_id] is None:
        cam_cfg = Config.CAMERAS[cam_id]
        print(f"\n🎥 Inisialisasi {cam_cfg['label']} ({cam_cfg['ip']})...")
        c = OpenVINOFaceCounter(
            cam_cfg['urls'],
            cam_cfg['user'],
            cam_cfg['password'],
            Config
        )
        # SSE callbacks hanya daftarkan ke kamera 0 (shared database)
        if cam_id == 0:
            c.face_db.register_callback('new_face',     _on_new_face)
            c.face_db.register_callback('face_updated', _on_face_updated)
        c.start()
        counters[cam_id] = c
        print(f"✅ {cam_cfg['label']} started")
    return counters[cam_id]


def get_session_manager():
    global session_manager
    if session_manager is None:
        session_manager = SessionManager()
    return session_manager


def auto_start_session():
    global detection_enabled, current_session_active, session_start_time
    if not current_session_active and detection_enabled:
        sess = get_session_manager().start_session("CCTV Hall A (2 Kamera)")
        current_session_active = True
        session_start_time = time.time()
        print(f"🚀 Auto-started session: {sess['id']}")


def ensure_streams_alive():
    """Monitor kesehatan kedua stream, reconnect jika perlu."""
    while True:
        try:
            time.sleep(15)
            for cam_id, c in enumerate(counters):
                if c and c.is_running:
                    if c.cap is None or not c.cap.isOpened():
                        label = Config.CAMERAS[cam_id]['label']
                        print(f"⚠️ {label} stream tidak sehat, reconnecting...")
                        with stream_lock:
                            c._reconnect()
        except Exception as e:
            print(f"⚠️ Monitor error: {e}")
            time.sleep(5)

threading.Thread(target=ensure_streams_alive, daemon=True).start()

# ─────────────────────────────────────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/data-pengunjung')
def data_pengunjung():
    return render_template('data_pengunjung.html')

# ─────────────────────────────────────────────────────────────────────────────
# VIDEO STREAM — /video_feed/<cam_id>
# Backward-compat: /video_feed → kamera 0
# ─────────────────────────────────────────────────────────────────────────────

def _stream_generator(cam_id: int):
    """Generator MJPEG untuk kamera cam_id."""
    c = get_counter(cam_id)
    target_frame_time = 1.0 / Config.STREAM_FPS
    last_frame_time   = time.time()
    error_count       = 0
    label             = Config.CAMERAS[cam_id]['label']

    while True:
        try:
            elapsed = time.time() - last_frame_time
            if elapsed < target_frame_time:
                time.sleep(target_frame_time - elapsed)
                continue
            last_frame_time = time.time()

            if not detection_enabled:
                blank = np.zeros((Config.FRAME_HEIGHT, Config.FRAME_WIDTH, 3), dtype=np.uint8)
                cv2.putText(blank, f"{label} — DETECTION PAUSED",
                            (80, Config.FRAME_HEIGHT // 2),
                            cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 200, 0), 2)
                ret, buf = cv2.imencode('.jpg', blank,
                                        [cv2.IMWRITE_JPEG_QUALITY, Config.JPEG_QUALITY])
                if ret:
                    yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
                continue

            with stream_lock:
                frame = c.get_frame()

            if frame is None or frame.size == 0:
                error_count += 1
                if error_count > 10:
                    with stream_lock:
                        c._reconnect()
                    error_count = 0
                continue

            error_count = 0

            # Overlay label kamera di pojok kiri atas
            cv2.rectangle(frame, (0, 0), (180, 28), (0, 0, 0), -1)
            cv2.putText(frame, label, (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            ret, buf = cv2.imencode('.jpg', frame, [
                cv2.IMWRITE_JPEG_QUALITY, Config.JPEG_QUALITY,
                cv2.IMWRITE_JPEG_OPTIMIZE, 1])
            if ret:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'

        except GeneratorExit:
            break
        except Exception as e:
            print(f"❌ Stream {label} error: {e}")
            time.sleep(0.1)


@app.route('/video_feed')
def video_feed_default():
    """Backward-compat — redirect ke kamera 0."""
    return video_feed(0)


@app.route('/video_feed/<int:cam_id>')
def video_feed(cam_id: int):
    if cam_id not in (0, 1):
        return "Camera not found", 404
    return Response(
        _stream_generator(cam_id),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={'Cache-Control': 'no-cache', 'Connection': 'close'}
    )

# ─────────────────────────────────────────────────────────────────────────────
# SSE — real-time events
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/events')
def sse_events():
    queue = []
    with _sse_lock:
        _sse_clients.append(queue)

    def stream():
        try:
            stats = _merged_stats()
            yield f"event: stats\ndata: {json.dumps(stats)}\n\n"
        except Exception:
            pass

        try:
            while True:
                if queue:
                    yield queue.pop(0)
                else:
                    yield ": heartbeat\n\n"
                    time.sleep(5)
        except GeneratorExit:
            pass
        finally:
            with _sse_lock:
                if queue in _sse_clients:
                    _sse_clients.remove(queue)

    return Response(stream_with_context(stream()),
                    mimetype='text/event-stream',
                    headers={
                        'Cache-Control': 'no-cache',
                        'X-Accel-Buffering': 'no',
                        'Connection': 'keep-alive',
                    })

# ─────────────────────────────────────────────────────────────────────────────
# HELPER — gabungkan statistik kedua kamera
# ─────────────────────────────────────────────────────────────────────────────

def _merged_stats() -> dict:
    """
    Gabungkan stats dari kedua counter.
    current_count & active_trackers dijumlahkan.
    FPS diambil rata-rata.
    """
    active = [c for c in counters if c is not None]
    if not active:
        return {}

    merged = active[0].get_statistics().copy()

    if len(active) > 1:
        s1 = active[1].get_statistics()
        merged['current_count']  = (merged.get('current_count', 0)
                                    + s1.get('current_count', 0))
        merged['active_trackers'] = (merged.get('active_trackers', 0)
                                     + s1.get('active_trackers', 0))
        merged['fps']            = round(
            (merged.get('fps', 0) + s1.get('fps', 0)) / 2, 1)
        merged['processing_fps'] = round(
            (merged.get('processing_fps', 0) + s1.get('processing_fps', 0)) / 2, 1)
        # database_size tetap dari kamera 0 (shared)
        merged['camera_stats'] = [
            {
                'cam_id': i,
                'label':  Config.CAMERAS[i]['label'],
                'current_count':  active[0].get_statistics().get('current_count', 0)
                                  if i == 0 else s1.get('current_count', 0),
                'fps':    active[0].get_statistics().get('fps', 0)
                          if i == 0 else s1.get('fps', 0),
            }
            for i in range(len(active))
        ]

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API — STATS
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/stats')
def get_stats():
    stats = _merged_stats()
    return jsonify({
        'current_count':    stats.get('current_count', 0),
        'max_count':        stats.get('max_count', 0),
        'daily_total':      stats.get('daily_total', 0),
        'database_faces':   stats.get('database_size', 0),
        'fps':              stats.get('fps', 0),
        'processing_fps':   stats.get('processing_fps', 0),
        'active_ids':       stats.get('active_trackers', 0),
        'detection_enabled': detection_enabled,
        'camera_stats':     stats.get('camera_stats', []),
    })


@app.route('/api/stats/<int:cam_id>')
def get_stats_per_camera(cam_id: int):
    """Stats per kamera individual."""
    if cam_id not in (0, 1) or counters[cam_id] is None:
        return jsonify({'error': 'Camera not found'}), 404
    s = counters[cam_id].get_statistics()
    s['cam_id'] = cam_id
    s['label']  = Config.CAMERAS[cam_id]['label']
    return jsonify(s)


@app.route('/api/health')
def health_check():
    cam_health = []
    for i, c in enumerate(counters):
        cam_health.append({
            'cam_id':  i,
            'label':   Config.CAMERAS[i]['label'],
            'running': c.is_running if c else False,
            'fps':     round(c.fps, 1) if c else 0,
        })

    c0 = counters[0]
    return jsonify({
        'status':           'ok',
        'cameras':          cam_health,
        'detector':         c0.detector_type if c0 else 'N/A',
        'detection_enabled': detection_enabled,
        'sse_clients':      len(_sse_clients),
        'timestamp':        time.strftime('%Y-%m-%dT%H:%M:%S'),
    })


@app.route('/api/history')
def get_history():
    c = counters[0]
    if c is None:
        return jsonify({})
    return jsonify(c.get_historical_data())

# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API — FACE DATABASE
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/faces')
def get_faces():
    c     = get_counter(0)
    limit  = request.args.get('limit',  100, type=int)
    recent = request.args.get('recent', None, type=int)

    faces = (c.face_db.get_recent_faces(recent)
             if recent
             else c.face_db.get_all_faces_meta(limit))

    return jsonify({
        'success':  True,
        'total':    len(c.face_db.faces),
        'returned': len(faces),
        'faces':    faces,
    })


@app.route('/api/faces/<face_id>')
def get_face_detail(face_id):
    c    = get_counter(0)
    info = c.face_db.get_face_info(face_id)
    if not info:
        return jsonify({'success': False, 'message': 'Face not found'}), 404
    return jsonify({'success': True, 'face': info})


@app.route('/api/faces/stats')
def get_face_stats():
    c = get_counter(0)
    return jsonify({'success': True, **c.face_db.get_statistics()})


@app.route('/api/faces/reset', methods=['POST'])
def reset_face_db():
    get_counter(0).face_db.reset_database()
    _broadcast_sse('db_reset', {'message': 'Face database reset', 'timestamp': time.time()})
    return jsonify({'success': True, 'message': 'Face database reset'})

# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API — SESSIONS
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    mgr    = get_session_manager()
    date_q = request.args.get('date')
    loc_q  = request.args.get('location')
    limit  = request.args.get('limit', 200, type=int)

    if date_q:
        sessions = mgr.get_sessions_by_date(date_q)
    elif loc_q:
        sessions = mgr.get_sessions_by_location(loc_q)
    else:
        sessions = mgr.get_all_sessions(limit)

    c = get_counter(0)
    return jsonify({
        'success':         True,
        'sessions':        sessions,
        'statistics':      mgr.get_summary(),
        'database_faces':  len(c.face_db.faces),
        'current_session': mgr.get_current_session(),
    })


@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    ok = get_session_manager().delete_session(session_id)
    if ok:
        return jsonify({'success': True, 'message': f'Session {session_id} deleted'})
    return jsonify({'success': False, 'message': 'Session not found'}), 404


@app.route('/api/sessions/export')
def export_sessions_csv():
    mgr      = get_session_manager()
    csv_str  = mgr.export_csv()
    filename = f"data-pengunjung-{time.strftime('%Y-%m-%d')}.csv"
    return Response(
        csv_str,
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename="{filename}"'}
    )

# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API — DETECTION CONTROL
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/detection/toggle', methods=['POST'])
def toggle_detection():
    global detection_enabled, current_session_active, session_start_time
    data = request.get_json() or {}

    detection_enabled = data.get('enabled', not detection_enabled)
    mgr = get_session_manager()

    if detection_enabled and not current_session_active:
        loc     = data.get('camera_location', 'CCTV Hall A (2 Kamera)')
        session = mgr.start_session(loc)
        current_session_active = True
        session_start_time     = time.time()
    elif not detection_enabled and current_session_active:
        stats = _merged_stats()
        mgr.end_session(
            total_visitors = stats.get('daily_total', 0),
            max_concurrent = stats.get('max_count', 0),
            status         = 'Selesai',
            notes          = 'Manual stop',
        )
        current_session_active = False
        session_start_time     = None

    _broadcast_sse('detection_toggled', {'enabled': detection_enabled})
    return jsonify({'success': True, 'detection_enabled': detection_enabled,
                    'session_active': current_session_active})


@app.route('/api/reset', methods=['POST'])
def reset_stats():
    for c in counters:
        if c is not None:
            c.reset_daily_stats()
    return jsonify({'success': True, 'message': 'Statistics reset (semua kamera)'})

# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND: broadcast stats via SSE setiap 3 detik
# ─────────────────────────────────────────────────────────────────────────────

def _stats_broadcaster():
    time.sleep(5)
    while True:
        try:
            if _sse_clients and any(counters):
                stats = _merged_stats()
                _broadcast_sse('stats', {
                    'current_count':  stats.get('current_count', 0),
                    'daily_total':    stats.get('daily_total', 0),
                    'database_faces': stats.get('database_size', 0),
                    'fps':            stats.get('fps', 0),
                    'active_ids':     stats.get('active_trackers', 0),
                    'camera_stats':   stats.get('camera_stats', []),
                })
        except Exception as e:
            print(f"⚠️  SSE broadcast error: {e}")
        time.sleep(3)

threading.Thread(target=_stats_broadcaster, daemon=True).start()

# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    try:
        sys.modules['flask.cli'].show_server_banner = lambda *x: None
    except Exception:
        pass

    def get_local_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    local_ip = get_local_ip()

    Config.print_config()

    print("="*65)
    print(f"🌐 Dashboard  (lokal) : http://localhost:{Config.PORT}")
    print(f"🌐 Dashboard  (LAN)   : http://{local_ip}:{Config.PORT}")
    print(f"📡 Video Cam 0        : http://{local_ip}:{Config.PORT}/video_feed/0")
    print(f"📡 Video Cam 1        : http://{local_ip}:{Config.PORT}/video_feed/1")
    print(f"📋 API Stats          : http://{local_ip}:{Config.PORT}/api/stats")
    print(f"🔴 SSE Events         : http://{local_ip}:{Config.PORT}/api/events")
    print("="*65 + "\n")

    # Inisialisasi kedua kamera secara paralel
    print("🔄 Memulai kedua kamera secara paralel...")
    threads = []
    for cam_id in range(len(Config.CAMERAS)):
        t = threading.Thread(target=get_counter, args=(cam_id,), daemon=False)
        t.start()
        threads.append(t)

    for t in threads:
        t.join(timeout=30)  # tunggu maks 30 detik per kamera

    time.sleep(1)
    auto_start_session()

    try:
        app.run(host=Config.HOST, port=Config.PORT,
                debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n\n⏸️  Stopping...")
        for c in counters:
            if c is not None:
                c.stop()
        print("✅ Stopped\n")