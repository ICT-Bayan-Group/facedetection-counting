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
# TIMEZONE WITA (UTC+8)
# TZ FIX: sekarang import dari utils/wita_time.py — single source of truth,
# bukan didefinisikan ulang di sini. Semua module (session_manager,
# face_counter, face_database, daily_snapshot, midnight_reset, app) pakai
# fungsi yang sama persis, jadi tidak akan ada drift antar file walau
# TZ setting Linux/Ubuntu server berubah.
# ─────────────────────────────────────────────────────────────────────────────
from utils.wita_time import now_wita, today_wita, seconds_until_midnight_wita

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

# Snapshot DB + scheduler
snapshot_db: DailySnapshotDB = None
midnight_scheduler           = None


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
            Config,
            cam_id=cam_id,
            label=cam_cfg['label'],
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
        # BUG #3 FIX: broadcast juga sinyal counter_reset generik, dan dorong
        # payload stats terbaru (yang sudah ter-reset) supaya semua SSE
        # listener (dashboard + data-pengunjung) langsung sinkron ke 0
        # tanpa nunggu polling/broadcaster 3 detik berikutnya.
        _broadcast_sse('counter_reset', {
            'date':    snap_date,
            'message': 'Counter direset otomatis (midnight)',
        })
        try:
            _broadcast_sse('stats', _build_stats_payload(_merged_stats()))
        except Exception:
            pass

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
        mgr = get_session_manager()

        # BUG FIX (carryover lintas-hari): kalau proses mati SEBELUM midnight
        # reset sempat jalan (misal mati jam 23:58, nyala lagi jam 00:05),
        # stats_manager tiap kamera akan mendeteksi ini saat load_statistics()
        # dan menaruh sisa data itu di pending_carryover (lihat stats_manager.py)
        # alih-alih ikut numpuk ke hitungan hari ini. Di sini kita ambil
        # carryover itu dan simpan ke daily_snapshot dengan TANGGAL YANG BENAR
        # (tanggal kemarin, bukan hari ini).
        if snapshot_db is not None:
            for c in [x for x in counters if x is not None]:
                try:
                    carry = c.stats_manager.pop_pending_carryover()
                except AttributeError:
                    carry = None
                if carry and carry.get('total_detected', 0) > 0:
                    snapshot_db.save_daily_snapshot(
                        snapshot_date  = carry['date'],
                        total_visitors = carry['total_detected'],
                        max_concurrent = carry['max_count'],
                        # BARU: kalau stats_manager punya angka raw detection carryover-nya
                        # ikut diselamatkan juga; kalau belum ada, default 0 (bukan hilang,
                        # cuma gak ke-carry — angka utama tetap total_visitors/max_concurrent).
                        raw_detections = carry.get('raw_detections', 0),
                        notes          = 'Auto-saved carryover (proses mati sebelum midnight reset)',
                    )
                    print(f"💾 Carryover {carry['date']} diselamatkan: "
                          f"{carry['total_detected']} visitors, {carry['max_count']} max concurrent")

        # BUG FIX (data hilang saat interrupted/restart):
        # Sebelumnya, sesi lama yang berstatus 'Active' saat server mati
        # (crash, restart, jaringan putus, redeploy) selalu ditutup dengan
        # total_visitors=0 (nilai default parameter) — bukan angka asli —
        # DAN datanya nggak pernah disimpan ke daily_snapshot sama sekali.
        # Akibatnya: kalau server restart di tengah hari, semua kunjungan
        # sebelum restart itu hilang dari laporan harian.
        #
        # Sekarang: ambil dulu angka aktual dari counter (stats_manager
        # sudah reload dari disk saat OpenVINOFaceCounter di-init ulang,
        # dan sudah bersih dari carryover hari lain berkat fix di atas),
        # lalu catat itu ke record sesi (bukan 0) DAN simpan sebagai
        # daily_snapshot untuk hari ini — sebelum sesi baru dibuka.
        stats          = _merged_stats()
        total_visitors = stats.get('daily_total', 0)
        max_concurrent = stats.get('max_count', 0)
        raw_detections = stats.get('raw_detections', 0)  # BARU

        n_closed = mgr.end_all_running_sessions(
            total_visitors = total_visitors,
            max_concurrent = max_concurrent,
            status          = 'Interrupted',
            notes           = 'Closed on server restart'
        )

        if n_closed > 0 and total_visitors > 0 and snapshot_db is not None:
            snapshot_db.save_daily_snapshot(
                snapshot_date  = today_wita(),
                total_visitors = total_visitors,
                max_concurrent = max_concurrent,
                raw_detections = raw_detections,  # BARU
                notes          = f'Auto-saved setelah restart/interrupt ({n_closed} sesi ditutup)',
            )
            print(f"💾 Data sesi yang ke-interrupt diselamatkan ke daily_snapshot: "
                  f"{total_visitors} visitors, {max_concurrent} max concurrent")

        sess = mgr.start_session("CCTV Hall A (2 Kamera)")
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
            yield f"event: stats\ndata: {json.dumps(_build_stats_payload(stats))}\n\n"
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
        # daily_total juga harus dijumlah dari kedua kamera
        merged['daily_total']     = (merged.get('daily_total', 0)
                                     + s1.get('daily_total', 0))
        merged['max_count']       = max(merged.get('max_count', 0),
                                        s1.get('max_count', 0))
        merged['raw_detections']  = (merged.get('raw_detections', 0)
                                 + s1.get('raw_detections', 0))
        merged['fps']             = round(
            (merged.get('fps', 0) + s1.get('fps', 0)) / 2, 1)
        merged['processing_fps']  = round(
            (merged.get('processing_fps', 0) + s1.get('processing_fps', 0)) / 2, 1)
        merged['camera_stats'] = [
            {
                'cam_id':        i,
                'label':         Config.CAMERAS[i]['label'],
                'current_count': active[0].get_statistics().get('current_count', 0)
                                 if i == 0 else s1.get('current_count', 0),
                'daily_total':   active[0].get_statistics().get('daily_total', 0)
                                 if i == 0 else s1.get('daily_total', 0),
                'fps':           active[0].get_statistics().get('fps', 0)
                                 if i == 0 else s1.get('fps', 0),
                'is_running':    active[i].is_running if i < len(active) else False,
            }
            for i in range(len(active))
        ]

    return merged

def _build_stats_payload(stats: dict) -> dict:
    nw = now_wita()
    today = nw.strftime('%Y-%m-%d')

    snapshot_saved_today = False
    if snapshot_db:
        snapshot_saved_today = snapshot_db.get_summary_for_date(today) is not None

    uptime = int(time.time() - session_start_time) if session_start_time else 0

    stream_health = []
    for i, c in enumerate(counters):
        stream_health.append({
            'cam_id':    i,
            'label':     Config.CAMERAS[i]['label'],
            'is_running': c.is_running if c else False,
            'fps':        round(c.fps, 1) if c else 0,
        })

    # ── Total keseluruhan (all-time, tidak ikut ke-reset harian) ────────
    total_raw_detections = 0
    if counters[0] is not None:
        try:
            total_raw_detections = counters[0].face_db.get_raw_detection_count()
        except Exception as e:
            print(f"⚠️  get_raw_detection_count error: {e}")

    return {
        'current_count':     stats.get('current_count', 0),
        'daily_total':       stats.get('daily_total', 0),
        'max_count':         stats.get('max_count', 0),
        'database_faces':    stats.get('database_size', 0),
        'raw_detections':    stats.get('raw_detections', 0),
        'total_raw_detections': total_raw_detections,   # ← BARU: all-time, semua kamera
        'fps':               stats.get('fps', 0),
        'processing_fps':    stats.get('processing_fps', 0),
        'active_ids':        stats.get('active_trackers', 0),
        'detection_enabled': detection_enabled,
        'session_active':    current_session_active,
        'session_uptime':    uptime,
        'today_date':        today,
        'wita_time':         nw.strftime('%H:%M:%S'),
        'next_reset_in':     seconds_until_midnight_wita(),
        'snapshot_saved_today': snapshot_saved_today,
        'camera_stats':      stats.get('camera_stats', []),
        'stream_health':     stream_health,
    }

# ─────────────────────────────────────────────────────────────────────────────
# API — STATS (FIXED & EXTENDED)
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/stats')
def get_stats():
    stats = _merged_stats()
    return jsonify(_build_stats_payload(stats))


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
    nw = now_wita()
    return jsonify({
        'status':            'ok',
        'cameras':           cam_health,
        'detector':          c0.detector_type if c0 else 'N/A',
        'detection_enabled': detection_enabled,
        'sse_clients':       len(_sse_clients),
        'scheduler_alive':   midnight_scheduler.is_alive() if midnight_scheduler else False,
        'wita_time':         nw.strftime('%H:%M:%S'),
        'next_reset_in':     seconds_until_midnight_wita(),
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
# API — DAILY SNAPSHOTS
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/daily')
def get_daily_list():
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
    if snapshot_db is None:
        return jsonify({'success': False, 'message': 'Snapshot DB not ready'}), 503

    summary = snapshot_db.get_summary_for_date(date_str)
    if not summary:
        return jsonify({'success': False, 'message': f'No data for {date_str}'}), 404

    return jsonify({'success': True, 'date': date_str, 'summary': summary})


@app.route('/api/daily/<date_str>/faces')
def get_daily_faces(date_str: str):
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
    if snapshot_db is None:
        return jsonify({'success': False}), 503

    faces   = snapshot_db.get_faces_for_date(date_str)
    summary = snapshot_db.get_summary_for_date(date_str)

    import csv, io
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(['Laporan Harian Face Counter'])
    writer.writerow(['Tanggal', date_str])
    if summary:
        writer.writerow(['Total Pengunjung', summary.get('total_visitors', 0)])
        writer.writerow(['Maks Bersamaan',   summary.get('max_concurrent', 0)])
        # BARU: total "terdeteksi manusia" (raw, sebelum dedup) untuk tanggal ini
        writer.writerow(['Total Terdeteksi Manusia', summary.get('raw_detections', 0)])
    writer.writerow([])

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


# ─────────────────────────────────────────────────────────────────────────────
# API — EXPORT REKAP HARIAN (total pengunjung per tanggal, semua hari)
#
# BEDA dengan /api/daily/export/<date_str> di atas:
#   - endpoint di atas = detail per WAJAH untuk SATU tanggal
#   - endpoint ini      = rekap TOTAL per tanggal untuk BANYAK tanggal sekaligus
#     (format: tanggal sekian, jumlah pengunjung sekian, jumlah terdeteksi manusia
#     sekian), sesuai tampilan kartu-kartu di grid "Data Per Hari" pada halaman
#     data-pengunjung.
# Query param opsional: ?start=YYYY-MM-DD&end=YYYY-MM-DD untuk export rentang
# tanggal tertentu (dipakai bareng filter di UI). Kalau kosong -> export semua.
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/daily/export-summary')
def export_daily_summary_csv():
    if snapshot_db is None:
        return jsonify({'success': False, 'message': 'Snapshot DB not ready'}), 503

    start = request.args.get('start')
    end   = request.args.get('end')

    if start and end:
        summaries = snapshot_db.get_date_range_summary(start, end)
    else:
        summaries = snapshot_db.get_all_daily_summaries(100000)

    import csv, io
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(['Rekap Pengunjung Harian — Face Counter'])
    writer.writerow(['Periode', f'{start} s/d {end}' if (start and end) else 'Semua Data'])
    writer.writerow(['Digenerate', now_wita().strftime('%Y-%m-%d %H:%M:%S WITA')])
    writer.writerow([])

    # BARU: kolom "Total Terdeteksi Manusia" — breakdown raw detection per tanggal
    writer.writerow(['No', 'Tanggal', 'Total Pengunjung', 'Maks Bersamaan',
                     'Total Terdeteksi Manusia', 'Metode Deteksi', 'Catatan'])

    sorted_summaries = sorted(summaries, key=lambda s: s.get('date', ''))
    grand_total       = 0
    grand_total_raw   = 0  # BARU
    for i, s in enumerate(sorted_summaries, 1):
        grand_total     += s.get('total_visitors', 0)
        grand_total_raw += s.get('raw_detections', 0)
        writer.writerow([
            i,
            s.get('date', ''),
            s.get('total_visitors', 0),
            s.get('max_concurrent', 0),
            s.get('raw_detections', 0),   # BARU
            s.get('detection_method', ''),
            s.get('notes', ''),
        ])

    writer.writerow([])
    writer.writerow(['', 'TOTAL KESELURUHAN (Pengunjung)',       grand_total,     '', '', '', ''])
    writer.writerow(['', 'TOTAL KESELURUHAN (Terdeteksi Manusia)', grand_total_raw, '', '', '', ''])
    writer.writerow(['', 'JUMLAH HARI TERCATAT', len(sorted_summaries), '', '', '', ''])

    if start and end:
        filename = f"rekap-pengunjung-{start}_sd_{end}.csv"
    else:
        filename = f"rekap-pengunjung-semua-{today_wita()}.csv"

    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename="{filename}"'}
    )


@app.route('/api/daily/snapshot/manual', methods=['POST'])
def manual_snapshot():
    """
    BUG #1 FIX (root cause 1): sebelumnya endpoint ini cuma memanggil
    save_daily_snapshot() tanpa mereset statistik. Sekarang lengkap:
    save snapshot -> save face log -> close session lama -> RESET semua
    counter -> buka session baru -> broadcast SSE (snapshot_saved +
    counter_reset) supaya dashboard & halaman data-pengunjung langsung
    balik ke 0 tanpa perlu refresh / restart server.
    """
    global current_session_active, session_start_time

    if midnight_scheduler is None or snapshot_db is None:
        return jsonify({'success': False, 'message': 'Scheduler/Snapshot DB not ready'}), 503

    data      = request.get_json() or {}
    today_str = today_wita()
    counters_ = [c for c in counters if c is not None]

    if not counters_:
        return jsonify({'success': False, 'message': 'No active counters'}), 503

    total_visitors   = 0
    max_concurrent   = 0
    total_raw        = 0  # BARU — total "terdeteksi manusia" hari ini, semua kamera
    detection_method = ""

    for c in counters_:
        stats            = c.get_statistics()
        total_visitors  += stats.get('daily_total', 0)
        max_concurrent   = max(max_concurrent, stats.get('max_count', 0))
        total_raw       += stats.get('raw_detections', 0)
        if not detection_method:
            detection_method = stats.get('detection_method', '')

    notes = data.get('notes', 'Manual snapshot')

    # 1. Simpan daily summary
    ok = snapshot_db.save_daily_snapshot(
        snapshot_date    = today_str,
        total_visitors   = total_visitors,
        max_concurrent   = max_concurrent,
        raw_detections   = total_raw,   # BARU
        detection_method = detection_method,
        notes            = notes,
    )

    # 2. Simpan daily face log
    n = 0
    try:
        face_db = counters_[0].face_db
        faces   = face_db.get_all_faces_meta(limit=10000)
        n = snapshot_db.log_faces_for_date(today_str, faces, "All Cameras")
    except Exception as e:
        print(f"⚠️  manual snapshot face log: {e}")

    # 3. Tutup sesi aktif (FR-04)
    mgr = get_session_manager()
    try:
        mgr.end_session(
            total_visitors = total_visitors,
            max_concurrent = max_concurrent,
            status         = 'Selesai',
            notes          = f'{notes} (manual reset)',
        )
    except Exception as e:
        print(f"⚠️  manual snapshot end_session: {e}")

    # 4. RESET semua counter harian — ini yang hilang sebelumnya (root cause bug #1)
    for c in counters_:
        try:
            c.reset_daily_stats()
        except Exception as e:
            print(f"⚠️  manual snapshot counter reset ({getattr(c,'label','?')}): {e}")

    # 5. Buat sesi baru (FR-05)
    try:
        mgr.start_session("CCTV Hall A (2 Kamera)")
        current_session_active = True
        session_start_time     = time.time()
    except Exception as e:
        print(f"⚠️  manual snapshot start_session: {e}")

    # 6. Broadcast SSE — snapshot_saved (info) + counter_reset (aksi UI reset ke 0)
    #    plus payload stats terbaru (yang sudah 0) supaya semua client sinkron instan.
    _broadcast_sse('snapshot_saved', {
        'date':           today_str,
        'total_visitors': total_visitors,
        'max_concurrent': max_concurrent,
        'raw_detections': total_raw,   # BARU
        'faces_logged':   n,
    })
    _broadcast_sse('counter_reset', {
        'date':    today_str,
        'message': 'Counter direset setelah manual snapshot',
    })
    try:
        _broadcast_sse('stats', _build_stats_payload(_merged_stats()))
    except Exception:
        pass

    return jsonify({
        'success':        ok,
        'date':           today_str,
        'total_visitors': total_visitors,
        'max_concurrent': max_concurrent,
        'raw_detections': total_raw,   # BARU
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
        if snapshot_db:
            snapshot_db.save_daily_snapshot(
                snapshot_date  = today_wita(),
                total_visitors = stats.get('daily_total', 0),
                max_concurrent = stats.get('max_count', 0),
                raw_detections = stats.get('raw_detections', 0),  # BARU
                notes          = 'Detection manually stopped',
            )
            try:
                face_db = get_counter(0).face_db
                faces   = face_db.get_all_faces_meta(limit=10000)
                snapshot_db.log_faces_for_date(today_wita(), faces, "All Cameras")
            except Exception as e:
                print(f"⚠️  toggle snapshot face log: {e}")

        current_session_active = False
        session_start_time = None

    _broadcast_sse('detection_toggled', {'enabled': detection_enabled})
    return jsonify({'success': True, 'detection_enabled': detection_enabled,
                    'session_active': current_session_active})


@app.route('/api/reset', methods=['POST'])
def reset_stats():
    for c in counters:
        if c is not None:
            c.reset_daily_stats()
    _broadcast_sse('counter_reset', {
        'date':    today_wita(),
        'message': 'Statistik direset manual dari panel kontrol',
    })
    return jsonify({'success': True, 'message': 'Statistics reset (semua kamera)'})

# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND: broadcast stats SSE tiap 3 detik (FIXED — semua field lengkap)
# ─────────────────────────────────────────────────────────────────────────────

def _stats_broadcaster():
    time.sleep(5)
    while True:
        try:
            if _sse_clients and any(counters):
                stats   = _merged_stats()
                payload = _build_stats_payload(stats)
                _broadcast_sse('stats', payload)
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

    init_snapshot_system()

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
        print("\n\n⏸️  Stopping — saving shutdown snapshot...")
        if midnight_scheduler:
            midnight_scheduler.save_shutdown_snapshot(notes="Server shutdown (KeyboardInterrupt)")
        print("✅ Stopped\n")