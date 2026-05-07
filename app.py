"""
Flask App — OpenVINO Face Counter
Lengkap dengan Public API + SSE real-time untuk wajah baru.
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

counter              = None
session_manager      = None
detection_enabled    = True
stream_lock          = threading.Lock()
current_session_active = False
session_start_time   = None

# SSE subscribers — set of queue-like lists
_sse_clients: list[list] = []
_sse_lock = threading.Lock()

# ─────────────────────────────────────────────────────────────────────────────
# SSE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _broadcast_sse(event: str, data: dict):
    """Push event ke semua SSE client yang sedang connect."""
    payload = f"event: {event}\ndata: {json.dumps(data)}\n\n"
    with _sse_lock:
        for q in list(_sse_clients):
            q.append(payload)


def _on_new_face(face_id: str, meta: dict):
    """Dipanggil FaceDatabase saat wajah baru masuk."""
    _broadcast_sse('new_face', {
        'face_id':         face_id,
        'first_seen':      meta.get('first_seen'),
        'total_in_db':     meta.get('total_in_db'),
        'detection_count': meta.get('detection_count', 1),
    })


def _on_face_updated(face_id: str, meta: dict):
    """Dipanggil FaceDatabase saat wajah existing diupdate."""
    _broadcast_sse('face_updated', {
        'face_id':         face_id,
        'last_seen':       meta.get('last_seen'),
        'detection_count': meta.get('detection_count'),
    })


# ─────────────────────────────────────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────────────────────────────────────

def get_counter():
    global counter
    if counter is None:
        Config.print_config()
        counter = OpenVINOFaceCounter(Config.CCTV_URLS, Config.CCTV_USER, Config.CCTV_PASS, Config)
        # Daftarkan SSE callbacks ke face_db
        counter.face_db.register_callback('new_face',      _on_new_face)
        counter.face_db.register_callback('face_updated',  _on_face_updated)
        counter.start()
        print("✅ Face counter started")
    return counter


def get_session_manager():
    global session_manager
    if session_manager is None:
        session_manager = SessionManager()
    return session_manager


def auto_start_session():
    global detection_enabled, current_session_active, session_start_time
    if not current_session_active and detection_enabled:
        sess = get_session_manager().start_session("CCTV Hall A")
        current_session_active = True
        session_start_time = time.time()
        print(f"🚀 Auto-started session: {sess['id']}")


def ensure_stream_alive():
    global counter
    while True:
        try:
            time.sleep(15)
            if counter and counter.is_running:
                if counter.cap is None or not counter.cap.isOpened():
                    print("⚠️ Stream not healthy, reconnecting...")
                    with stream_lock:
                        counter._reconnect()
        except Exception as e:
            print(f"⚠️ Monitor error: {e}")
            time.sleep(5)

threading.Thread(target=ensure_stream_alive, daemon=True).start()

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
# VIDEO STREAM
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/video_feed')
def video_feed():
    def generate():
        c = get_counter()
        target_frame_time = 1.0 / Config.STREAM_FPS
        last_frame_time   = time.time()
        error_count       = 0

        while True:
            try:
                elapsed = time.time() - last_frame_time
                if elapsed < target_frame_time:
                    time.sleep(target_frame_time - elapsed)
                    continue
                last_frame_time = time.time()

                if not detection_enabled:
                    blank = np.zeros((Config.FRAME_HEIGHT, Config.FRAME_WIDTH, 3), dtype=np.uint8)
                    cv2.putText(blank, "DETECTION PAUSED",
                                (180, 180), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 200, 0), 2)
                    ret, buf = cv2.imencode('.jpg', blank, [cv2.IMWRITE_JPEG_QUALITY, Config.JPEG_QUALITY])
                    if ret:
                        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
                    continue

                with stream_lock:
                    frame = c.get_frame()

                if frame is None or frame.size == 0:
                    error_count += 1
                    if error_count > 10:
                        with stream_lock: c._reconnect()
                        error_count = 0
                    continue

                error_count = 0
                ret, buf = cv2.imencode('.jpg', frame, [
                    cv2.IMWRITE_JPEG_QUALITY, Config.JPEG_QUALITY,
                    cv2.IMWRITE_JPEG_OPTIMIZE, 1])
                if ret:
                    yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'

            except GeneratorExit:
                break
            except Exception as e:
                print(f"❌ Stream error: {e}")
                time.sleep(0.1)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame',
                    headers={'Cache-Control': 'no-cache', 'Connection': 'close'})

# ─────────────────────────────────────────────────────────────────────────────
# SSE — real-time events
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/events')
def sse_events():
    """
    Server-Sent Events endpoint.
    Client connect sekali, dapat push setiap ada wajah baru / update.

    Usage (JavaScript):
        const es = new EventSource('/api/events');
        es.addEventListener('new_face', e => console.log(JSON.parse(e.data)));
        es.addEventListener('face_updated', e => console.log(JSON.parse(e.data)));
        es.addEventListener('stats', e => console.log(JSON.parse(e.data)));
    """
    queue = []
    with _sse_lock:
        _sse_clients.append(queue)

    def stream():
        # Kirim stats awal
        try:
            c     = get_counter()
            stats = c.get_statistics()
            yield f"event: stats\ndata: {json.dumps(stats)}\n\n"
        except Exception:
            pass

        try:
            while True:
                if queue:
                    msg = queue.pop(0)
                    yield msg
                else:
                    # Heartbeat setiap 5 detik agar koneksi tidak timeout
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
                        'X-Accel-Buffering': 'no',  # disable nginx buffering
                        'Connection': 'keep-alive',
                    })

# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API — STATS
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/stats')
def get_stats():
    c     = get_counter()
    stats = c.get_statistics()
    return jsonify({
        'current_count':    stats.get('current_count', 0),
        'max_count':        stats.get('max_count', 0),
        'daily_total':      stats.get('daily_total', 0),
        'database_faces':   stats.get('database_size', 0),
        'fps':              stats.get('fps', 0),
        'processing_fps':   stats.get('processing_fps', 0),
        'active_ids':       stats.get('active_trackers', 0),
        'detection_enabled': detection_enabled,
    })

@app.route('/api/health')
def health_check():
    c = get_counter()
    return jsonify({
        'status':           'ok',
        'running':          c.is_running,
        'detector':         c.detector_type,
        'fps':              round(c.fps, 1),
        'detection_enabled': detection_enabled,
        'sse_clients':      len(_sse_clients),
        'timestamp':        time.strftime('%Y-%m-%dT%H:%M:%S'),
    })

@app.route('/api/history')
def get_history():
    return jsonify(get_counter().get_historical_data())

# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API — FACE DATABASE
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/faces')
def get_faces():
    """
    GET /api/faces
    Query params:
      limit  (int, default 100)  — maks wajah yang dikembalikan
      recent (int)               — jika diisi, ambil N wajah terbaru saja

    Response:
    {
      "success": true,
      "total": 42,
      "faces": [
        {
          "id": "0",
          "first_seen": "2026-02-14T11:29:12",
          "last_seen":  "2026-04-06T14:46:22",
          "detection_count": 139,
          "thumbnail_b64": null
        }, ...
      ]
    }
    """
    c     = get_counter()
    limit  = request.args.get('limit',  100, type=int)
    recent = request.args.get('recent', None, type=int)

    if recent:
        faces = c.face_db.get_recent_faces(recent)
    else:
        faces = c.face_db.get_all_faces_meta(limit)

    return jsonify({
        'success': True,
        'total':   len(c.face_db.faces),
        'returned': len(faces),
        'faces':   faces,
    })


@app.route('/api/faces/<face_id>')
def get_face_detail(face_id):
    """GET /api/faces/<face_id> — detail satu wajah."""
    c    = get_counter()
    info = c.face_db.get_face_info(face_id)
    if not info:
        return jsonify({'success': False, 'message': 'Face not found'}), 404
    return jsonify({'success': True, 'face': info})


@app.route('/api/faces/stats')
def get_face_stats():
    """GET /api/faces/stats — statistik database wajah."""
    c = get_counter()
    return jsonify({'success': True, **c.face_db.get_statistics()})


@app.route('/api/faces/reset', methods=['POST'])
def reset_face_db():
    """POST /api/faces/reset — hapus semua wajah dari database."""
    get_counter().face_db.reset_database()
    _broadcast_sse('db_reset', {'message': 'Face database reset', 'timestamp': time.time()})
    return jsonify({'success': True, 'message': 'Face database reset'})

# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API — SESSIONS
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """
    GET /api/sessions
    Query params:
      date     (YYYY-MM-DD)  — filter by tanggal
      location (string)      — filter by lokasi kamera
      limit    (int)         — maks hasil (default 200)

    Response:
    {
      "success": true,
      "sessions": [...],
      "statistics": { total_sessions, total_visitors_all_time, ... },
      "database_faces": 42,
      "current_session": { ... } | null
    }
    """
    mgr      = get_session_manager()
    date_q   = request.args.get('date')
    loc_q    = request.args.get('location')
    limit    = request.args.get('limit', 200, type=int)

    if date_q:
        sessions = mgr.get_sessions_by_date(date_q)
    elif loc_q:
        sessions = mgr.get_sessions_by_location(loc_q)
    else:
        sessions = mgr.get_all_sessions(limit)

    c = get_counter()
    return jsonify({
        'success':         True,
        'sessions':        sessions,
        'statistics':      mgr.get_summary(),
        'database_faces':  len(c.face_db.faces),
        'current_session': mgr.get_current_session(),
    })


@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """DELETE /api/sessions/<session_id>"""
    ok = get_session_manager().delete_session(session_id)
    if ok:
        return jsonify({'success': True, 'message': f'Session {session_id} deleted'})
    return jsonify({'success': False, 'message': 'Session not found'}), 404


@app.route('/api/sessions/export')
def export_sessions_csv():
    """GET /api/sessions/export — download CSV semua sessions."""
    mgr = get_session_manager()
    csv_str = mgr.export_csv()
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
    c   = get_counter()
    mgr = get_session_manager()

    if detection_enabled and not current_session_active:
        loc     = data.get('camera_location', 'CCTV Hall A')
        session = mgr.start_session(loc)
        current_session_active = True
        session_start_time     = time.time()
    elif not detection_enabled and current_session_active:
        stats = c.get_statistics()
        mgr.end_session(
            total_visitors  = stats.get('daily_total', 0),
            max_concurrent  = stats.get('max_count', 0),
            status          = 'Selesai',
            notes           = 'Manual stop',
        )
        current_session_active = False
        session_start_time     = None

    _broadcast_sse('detection_toggled', {'enabled': detection_enabled})
    return jsonify({'success': True, 'detection_enabled': detection_enabled,
                    'session_active': current_session_active})


@app.route('/api/reset', methods=['POST'])
def reset_stats():
    get_counter().reset_daily_stats()
    return jsonify({'success': True, 'message': 'Statistics reset'})

# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND: broadcast stats via SSE setiap 3 detik
# ─────────────────────────────────────────────────────────────────────────────

def _stats_broadcaster():
    time.sleep(5)   # tunggu counter init
    while True:
        try:
            if _sse_clients and counter:
                stats = counter.get_statistics()
                _broadcast_sse('stats', {
                    'current_count':  stats.get('current_count', 0),
                    'daily_total':    stats.get('daily_total', 0),
                    'database_faces': stats.get('database_size', 0),
                    'fps':            stats.get('fps', 0),
                    'active_ids':     stats.get('active_trackers', 0),
                })
        except Exception as e:
            print(f"⚠️  SSE broadcast error: {e}")
        time.sleep(3)

threading.Thread(target=_stats_broadcaster, daemon=True).start()

# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    try: sys.modules['flask.cli'].show_server_banner = lambda *x: None
    except: pass

    # Deteksi IP lokal (LAN)
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

    print("\n" + "="*65)
    print("🚀 OPENVINO FACE COUNTER")
    print("="*65)
    print(f"   CCTV IP   : {Config.CCTV_IP}")
    print(f"   Detector  : OpenVINO on {Config.OPENVINO_DEVICE}")
    print(f"   Resolution: {Config.FRAME_WIDTH}×{Config.FRAME_HEIGHT}")
    print(f"   Stream FPS: {Config.STREAM_FPS}")
    print(f"   Detect FPS: {Config.DETECTION_FPS}")
    print("="*65)
    print(f"\n🌐 Dashboard  (lokal) : http://localhost:{Config.PORT}")
    print(f"🌐 Dashboard  (LAN)   : http://{local_ip}:{Config.PORT}")
    print(f"📡 API Faces  (LAN)   : http://{local_ip}:{Config.PORT}/api/faces")
    print(f"📋 API Sess   (LAN)   : http://{local_ip}:{Config.PORT}/api/sessions")
    print(f"🔴 SSE Events (LAN)   : http://{local_ip}:{Config.PORT}/api/events")
    print("="*65)
    print(f"\n💡 Akses dari jaringan lain? Buka port {Config.PORT} di firewall:")
    print(f"   sudo ufw allow {Config.PORT}/tcp")
    print("="*65 + "\n")

    counter = get_counter()
    time.sleep(2)
    auto_start_session()

    try:
        app.run(host=Config.HOST, port=Config.PORT,
                debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n\n⏸️  Stopping...")
        if counter: counter.stop()
        print("✅ Stopped\n")