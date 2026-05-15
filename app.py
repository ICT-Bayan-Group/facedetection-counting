from flask import Flask, render_template, Response, jsonify, request, stream_with_context
from flask_cors import CORS
import cv2, time, os, sys, logging, threading, json
import numpy as np
import socket

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
os.environ['OPENCV_LOG_LEVEL'] = 'FATAL'
import warnings; warnings.filterwarnings('ignore')

from core.config import Config
from core.face_counter import OpenVINOFaceCounter
from utils.session_manager import SessionManager
from utils.daily_snapshot import DailySnapshotDB
from utils.midnight_reset import MidnightResetScheduler

app  = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────────────────────────

counters: list[OpenVINOFaceCounter | None] = [None, None]

session_manager        = None
detection_enabled      = True
stream_lock            = threading.Lock()
current_session_active = False
session_start_time     = None

# SSE subscribers
_sse_clients: list[list] = []
_sse_lock = threading.Lock()

# ── NEW: snapshot DB + scheduler ──────────────────────────────────────────
snapshot_db: DailySnapshotDB    = None
midnight_scheduler              = None


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
# INIT
# ─────────────────────────────────────────────────────────────────────────────

def get_counter(cam_id: int = 0) -> OpenVINOFaceCounter:
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


def init_snapshot_system():
    """Inisialisasi snapshot DB + midnight scheduler."""
    global snapshot_db, midnight_scheduler

    snapshot_db = DailySnapshotDB()

    def _on_reset_done(snap_date: str, total_visitors: int):
        """Callback setelah midnight reset: broadcast SSE."""
        _broadcast_sse('midnight_reset', {
            'date':           snap_date,
            'total_visitors': total_visitors,
            'message':        f'Data {snap_date} tersimpan, counter direset',
        })

    midnight_scheduler = MidnightResetScheduler(
        get_counters        = lambda: counters,
        snapshot_db         = snapshot_db,
        get_session_manager = get_session_manager,
        on_reset_done       = _on_reset_done,
    )
    midnight_scheduler.start()
    print("✅ Snapshot system initialized")


def auto_start_session():
    global detection_enabled, current_session_active, session_start_time
    if not current_session_active and detection_enabled:
        sess = get_session_manager().start_session("CCTV Hall A (2 Kamera)")
        current_session_active = True
        session_start_time = time.time()
        print(f"🚀 Auto-started session: {sess['id']}")


def ensure_streams_alive():
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
# VIDEO STREAM
# ─────────────────────────────────────────────────────────────────────────────

def _stream_generator(cam_id: int):
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
# SSE
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
                        'Cache-Control':    'no-cache',
                        'X-Accel-Buffering': 'no',
                        'Connection':       'keep-alive',
                    })

# ─────────────────────────────────────────────────────────────────────────────
# HELPER — merged stats
# ─────────────────────────────────────────────────────────────────────────────

def _merged_stats() -> dict:
    active = [c for c in counters if c is not None]
    if not active:
        return {}

    merged = active[0].get_statistics().copy()

    if len(active) > 1:
        s1 = active[1].get_statistics()
        merged['current_count']   = (merged.get('current_count', 0)
                                     + s1.get('current_count', 0))
        merged['active_trackers'] = (merged.get('active_trackers', 0)
                                     + s1.get('active_trackers', 0))
        merged['fps']             = round(
            (merged.get('fps', 0) + s1.get('fps', 0)) / 2, 1)
        merged['processing_fps']  = round(
            (merged.get('processing_fps', 0) + s1.get('processing_fps', 0)) / 2, 1)
        merged['camera_stats'] = [
            {
                'cam_id': i,
                'label':  Config.CAMERAS[i]['label'],
                'current_count': active[0].get_statistics().get('current_count', 0)
                                 if i == 0 else s1.get('current_count', 0),
                'fps': active[0].get_statistics().get('fps', 0)
                       if i == 0 else s1.get('fps', 0),
            }
            for i in range(len(active))
        ]

    return merged

# ─────────────────────────────────────────────────────────────────────────────
# API — STATS
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/stats')
def get_stats():
    stats = _merged_stats()
    return jsonify({
        'current_count':     stats.get('current_count', 0),
        'max_count':         stats.get('max_count', 0),
        'daily_total':       stats.get('daily_total', 0),
        'database_faces':    stats.get('database_size', 0),
        'fps':               stats.get('fps', 0),
        'processing_fps':    stats.get('processing_fps', 0),
        'active_ids':        stats.get('active_trackers', 0),
        'detection_enabled': detection_enabled,
        'camera_stats':      stats.get('camera_stats', []),
    })


@app.route('/api/stats/<int:cam_id>')
def get_stats_per_camera(cam_id: int):
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
        'status':            'ok',
        'cameras':           cam_health,
        'detector':          c0.detector_type if c0 else 'N/A',
        'detection_enabled': detection_enabled,
        'sse_clients':       len(_sse_clients),
        'timestamp':         time.strftime('%Y-%m-%dT%H:%M:%S'),
    })


@app.route('/api/history')
def get_history():
    c = counters[0]
    if c is None:
        return jsonify({})
    return jsonify(c.get_historical_data())

# ─────────────────────────────────────────────────────────────────────────────
# API — FACE DATABASE
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/faces')
def get_faces():
    c      = get_counter(0)
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
# API — SESSIONS
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
# API — DAILY SNAPSHOTS  ← NEW
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/daily')
def get_daily_list():
    """
    Daftar semua hari yang ada di snapshot DB.
    Query params:
      limit  (int, default 90)
      start  (YYYY-MM-DD)
      end    (YYYY-MM-DD)
    """
    if snapshot_db is None:
        return jsonify({'success': False, 'message': 'Snapshot DB not ready'}), 503

    start = request.args.get('start')
    end   = request.args.get('end')
    limit = request.args.get('limit', 90, type=int)

    if start and end:
        summaries = snapshot_db.get_date_range_summary(start, end)
    else:
        summaries = snapshot_db.get_all_daily_summaries(limit)

    overall = snapshot_db.get_overall_stats()

    return jsonify({
        'success':   True,
        'total':     len(summaries),
        'summaries': summaries,
        'overall':   overall,
    })


@app.route('/api/daily/<date_str>')
def get_daily_detail(date_str: str):
    """
    Ringkasan satu hari.
    date_str format: YYYY-MM-DD
    """
    if snapshot_db is None:
        return jsonify({'success': False, 'message': 'Snapshot DB not ready'}), 503

    summary = snapshot_db.get_summary_for_date(date_str)
    if not summary:
        return jsonify({'success': False, 'message': f'No data for {date_str}'}), 404

    return jsonify({'success': True, 'date': date_str, 'summary': summary})


@app.route('/api/daily/<date_str>/faces')
def get_daily_faces(date_str: str):
    """
    Semua wajah yang tertangkap pada tanggal tertentu.
    Query params:
      limit (int, default 1000)
    """
    if snapshot_db is None:
        return jsonify({'success': False, 'message': 'Snapshot DB not ready'}), 503

    limit = request.args.get('limit', 1000, type=int)
    faces = snapshot_db.get_faces_for_date(date_str, limit)

    return jsonify({
        'success': True,
        'date':    date_str,
        'total':   len(faces),
        'faces':   faces,
    })


@app.route('/api/daily/export/<date_str>')
def export_daily_csv(date_str: str):
    """Export data harian ke CSV."""
    if snapshot_db is None:
        return jsonify({'success': False}), 503

    faces   = snapshot_db.get_faces_for_date(date_str)
    summary = snapshot_db.get_summary_for_date(date_str)

    import csv, io
    output = io.StringIO()
    writer = csv.writer(output)

    # Header info
    writer.writerow(['Laporan Harian Face Counter'])
    writer.writerow(['Tanggal', date_str])
    if summary:
        writer.writerow(['Total Pengunjung', summary.get('total_visitors', 0)])
        writer.writerow(['Maks Bersamaan',   summary.get('max_concurrent', 0)])
    writer.writerow([])

    # Face table
    writer.writerow(['No', 'Face ID', 'Pertama Terlihat', 'Terakhir Terlihat',
                     'Jumlah Deteksi', 'Kamera'])
    for i, f in enumerate(faces, 1):
        writer.writerow([
            i,
            f.get('face_id', ''),
            f.get('first_seen', ''),
            f.get('last_seen', ''),
            f.get('detection_count', 1),
            f.get('camera_label', ''),
        ])

    filename = f"daily-{date_str}.csv"
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename="{filename}"'}
    )


@app.route('/api/daily/snapshot/manual', methods=['POST'])
def manual_snapshot():
    """
    Trigger simpan snapshot secara manual (tanpa reset).
    Berguna untuk test atau backup sewaktu-waktu.
    """
    if midnight_scheduler is None:
        return jsonify({'success': False, 'message': 'Scheduler not ready'}), 503

    data      = request.get_json() or {}
    today_str = time.strftime('%Y-%m-%d')
    counters_ = [c for c in counters if c is not None]

    if not counters_:
        return jsonify({'success': False, 'message': 'No active counters'}), 503

    total_visitors = 0
    max_concurrent = 0
    detection_method = ""

    for c in counters_:
        stats             = c.get_statistics()
        total_visitors   += stats.get('daily_total', 0)
        max_concurrent    = max(max_concurrent, stats.get('max_count', 0))
        if not detection_method:
            detection_method = stats.get('detection_method', '')

    ok = snapshot_db.save_daily_snapshot(
        snapshot_date    = today_str,
        total_visitors   = total_visitors,
        max_concurrent   = max_concurrent,
        detection_method = detection_method,
        notes            = data.get('notes', 'Manual snapshot'),
    )

    # Log faces
    try:
        face_db = counters_[0].face_db
        faces   = face_db.get_all_faces_meta(limit=10000)
        n = snapshot_db.log_faces_for_date(today_str, faces, "All Cameras")
    except Exception as e:
        n = 0
        print(f"⚠️  manual snapshot face log: {e}")

    return jsonify({
        'success':        ok,
        'date':           today_str,
        'total_visitors': total_visitors,
        'max_concurrent': max_concurrent,
        'faces_logged':   n,
    })

# ─────────────────────────────────────────────────────────────────────────────
# API — DETECTION CONTROL
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
        # Simpan snapshot saat deteksi dihentikan manual
        if snapshot_db:
            snapshot_db.save_daily_snapshot(
                snapshot_date  = time.strftime('%Y-%m-%d'),
                total_visitors = stats.get('daily_total', 0),
                max_concurrent = stats.get('max_count', 0),
                notes          = 'Detection manually stopped',
            )
            try:
                face_db = get_counter(0).face_db
                faces   = face_db.get_all_faces_meta(limit=10000)
                snapshot_db.log_faces_for_date(
                    time.strftime('%Y-%m-%d'), faces, "All Cameras"
                )
            except Exception as e:
                print(f"⚠️  toggle snapshot face log: {e}")

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
# BACKGROUND: broadcast stats SSE tiap 3 detik
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
    print(f"📅 API Daily          : http://{local_ip}:{Config.PORT}/api/daily")
    print(f"🔴 SSE Events         : http://{local_ip}:{Config.PORT}/api/events")
    print("="*65 + "\n")

    # ── Init snapshot system SEBELUM kamera ──────────────────────────────
    init_snapshot_system()

    # ── Inisialisasi kedua kamera secara paralel ──────────────────────────
    print("🔄 Memulai kedua kamera secara paralel...")
    threads = []
    for cam_id in range(len(Config.CAMERAS)):
        t = threading.Thread(target=get_counter, args=(cam_id,), daemon=False)
        t.start()
        threads.append(t)

    for t in threads:
        t.join(timeout=30)

    time.sleep(1)
    auto_start_session()

    try:
        app.run(host=Config.HOST, port=Config.PORT,
                debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        # ── Graceful shutdown: simpan snapshot sebelum mati ──────────────
        print("\n\n⏸️  Stopping — saving shutdown snapshot...")
        if midnight_scheduler:
            midnight_scheduler.save_shutdown_snapshot(notes="Server shutdown (KeyboardInterrupt)")
        print("✅ Stopped\n")