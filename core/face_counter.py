import cv2
import numpy as np
import os
import time
import uuid
import threading
from collections import defaultdict, deque
from datetime import datetime
from enum import Enum

from utils.wita_time import now_wita_iso  # TZ FIX — jangan pakai datetime.now() polos

# ─────────────────────────────────────────────
# OPENVINO
# ─────────────────────────────────────────────
try:
    from openvino.runtime import Core
    OPENVINO_AVAILABLE = True
    print("✅ OpenVINO available")
except ImportError:
    OPENVINO_AVAILABLE = False
    print("⚠️  OpenVINO not available, fallback to Haar Cascade")

# ─────────────────────────────────────────────
# FACENET
# ─────────────────────────────────────────────
try:
    import torch
    from PIL import Image
    from facenet_pytorch import InceptionResnetV1
    FACENET_AVAILABLE = True
    print("✅ FaceNet available")
except ImportError:
    FACENET_AVAILABLE = False
    print("⚠️  FaceNet not available, position-based tracking only")

# ─────────────────────────────────────────────
# SCIPY (Hungarian algorithm)
# ─────────────────────────────────────────────
try:
    from scipy.optimize import linear_sum_assignment
    HUNGARIAN_AVAILABLE = True
    print("✅ Hungarian matching (scipy) available")
except ImportError:
    HUNGARIAN_AVAILABLE = False
    print("⚠️  scipy tidak ada, fallback ke greedy matching (kurang akurat pas crowd)")

from utils.face_database import FaceDatabase
from utils.video_utils import VideoStreamHandler
from utils.stats_manager import StatisticsManager


# ─────────────────────────────────────────────────────────────────────────────
# FACE STATUS STATE MACHINE
# ─────────────────────────────────────────────────────────────────────────────

class FaceStatus(Enum):
    DETECTED         = "DETECTED"          # Frame 1–2   : baru muncul, belum diverifikasi
    VERIFYING        = "VERIFYING"         # Frame 3–4   : sedang dievaluasi
    WAJAH_BARU       = "WAJAH_BARU"        # Match DB: tidak ada  → disimpan ke database
    SUDAH_TERDETEKSI = "SUDAH_TERDETEKSI"  # Match DB: ada        → tidak disimpan lagi
    TRACKING         = "TRACKING"          # Sedang dilacak setelah diverifikasi
    LOST             = "LOST"              # Hilang dari frame


# Warna BGR per status
STATUS_COLOR: dict = {
    FaceStatus.DETECTED:         (255, 255, 255),  # putih
    FaceStatus.VERIFYING:        (0,   255, 255),  # kuning
    FaceStatus.WAJAH_BARU:       (0,   255,   0),  # hijau
    FaceStatus.SUDAH_TERDETEKSI: (255, 165,   0),  # oranye
    FaceStatus.TRACKING:         (150, 150, 150),  # abu-abu
    FaceStatus.LOST:             (0,     0, 255),  # merah
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def calculate_blur_score(face_roi: np.ndarray) -> float:
    """Laplacian variance — semakin tinggi semakin tajam."""
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity antara dua embedding vektor."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a / na, b / nb))


# ─────────────────────────────────────────────────────────────────────────────
# FACE TRACKER ENTRY
# ─────────────────────────────────────────────────────────────────────────────

class TrackerEntry:
    """Menyimpan state satu wajah yang sedang dilacak."""

    VERIFY_FRAME_THRESHOLD = 4  # frame minimum sebelum status berubah dari VERIFYING

    def __init__(self, face_id: int, cx: int, cy: int, ts: float):
        self.face_id       = face_id
        self.cx            = cx
        self.cy            = cy
        self.last_seen     = ts
        self.status        = FaceStatus.DETECTED
        self.frame_count   = 0
        self.quality_hist  = deque(maxlen=15)
        self.blur_hist     = deque(maxlen=15)
        self.embedding     = None
        self.last_embed_ts = 0.0
        self.db_id         = None
        self.last_box      = None  # [x,y,w,h] — disimpan untuk redraw antar detection frame

    def update_position(self, cx: int, cy: int, ts: float):
        self.cx = cx
        self.cy = cy
        self.last_seen = ts
        self.frame_count += 1

    def advance_status(self):
        """Transisi state berdasarkan frame_count."""
        if self.status == FaceStatus.DETECTED and self.frame_count >= 2:
            self.status = FaceStatus.VERIFYING

    def avg_quality(self) -> float:
        return float(np.mean(self.quality_hist)) if self.quality_hist else 0.0

    def avg_blur(self) -> float:
        return float(np.mean(self.blur_hist)) if self.blur_hist else 0.0

    def is_verified(self) -> bool:
        """Wajah dianggap layak diverifikasi ke database."""
        return (
            self.frame_count >= self.VERIFY_FRAME_THRESHOLD
            and self.avg_quality() > 0.60
            and self.avg_blur() > 40
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────

class OpenVINOFaceCounter:


    # ─── TUNABLE CONSTANTS ───────────────────────────────────────────────
    DETECTION_FPS     = 5           # OpenVINO inference rate
    TRACK_FPS         = 20          # render/tracking rate
    DETECTION_SIZE    = (640, 640)  # fallback size untuk Haar (OpenVINO pakai ov_w/ov_h asli)
    CONFIDENCE_THRESH = 0.55        # lebih rendah agar wajah jauh/miring tetap lolos
    QUALITY_THRESH    = 0.30        # longgar, kualitas dinilai dari blur saja di CCTV
    BLUR_THRESH       = 30          # CCTV jauh blur wajar, jangan terlalu ketat
    FRONTAL_THRESH    = 0.20        # deteksi wajah miring/menunduk tetap masuk
    EMBED_INTERVAL    = 2.0         # detik minimum antar embedding extraction
    EMBED_SIM_THRESH  = 0.72        # cosine similarity untuk pencocokan DB
    MAX_POSITION_DIST = 180         # lebih besar untuk kompensasi gerakan cepat
    ID_TIMEOUT        = 15.0        # box tetap tampil selama wajah ada di frame
    CROWD_FACE_COUNT  = 4           # >= segini di satu frame dianggap crowd -> relax frontal check
    STALL_TIMEOUT     = 8.0         # detik tanpa frame baru sebelum watchdog paksa reconnect
    # ─────────────────────────────────────────────────────────────────────

    def __init__(self, cctv_urls, user, password, config, cam_id: int = 0, label: str = None):
        print("🔄 Initializing Surveillance-Grade OpenVINO Face Counter...")

        self.config        = config
        self.cam_id         = cam_id
        self.label          = label or f"Kamera {cam_id + 1}"
        self.video_handler = VideoStreamHandler(
            cctv_urls, user, password,
            target_fps=getattr(self.config, 'STREAM_FPS', 25)
        )

        self.stats_manager = StatisticsManager(stats_file=self._build_cam_stats_path())
        self.face_db       = FaceDatabase()

        print(f"📊 Face Database: {len(self.face_db.faces)} known faces")

        # Device (FaceNet)
        self.device = (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if FACENET_AVAILABLE else None
        )

        # FaceNet
        self.resnet         = None
        self.use_embeddings = False
        if FACENET_AVAILABLE:
            try:
                self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
                self.use_embeddings = True
                print(f"✅ FaceNet loaded [{self.device}]")
            except Exception as e:
                print(f"⚠️  FaceNet init failed: {e}")

        # OpenVINO / fallback Haar
        self.detector_type  = "Unknown"
        self.use_openvino   = False
        self.ie             = None
        self.compiled_model = None
        self.input_layer    = None
        self.output_layer   = None
        self.ov_n = self.ov_c = self.ov_h = self.ov_w = None
        self.face_cascade    = None
        self.frontal_cascade = None
        self.eye_cascade     = None
        self._ov_input_buf   = None  # pre-allocated buffer untuk OpenVINO inference

        self._init_detector()
        self._init_frontal_validator()

        # ── Atomic frame slots (mengganti Queue untuk low-latency) ───────
        self._latest_frame  = None  # frame terbaru dari capture
        self._latest_result = None  # (frame, faces, ts) terbaru dari detection
        self._frame_lock    = threading.Lock()
        self._result_lock   = threading.Lock()
        self._cap_lock       = threading.Lock()  # PATCH: guard akses self.cap dari capture & watchdog

        # Tracker state
        self.trackers: dict    = {}
        self.next_id           = 0
        self.track_history     = defaultdict(lambda: deque(maxlen=40))

        # Runtime state
        self.frame          = None
        self.is_running     = False
        self.cap            = None
        self.fps            = 0.0
        self.processing_fps = 0.0
        self.frame_times    = deque(maxlen=30)
        self.last_frame_ts  = time.time()
        self.last_detect_ts = 0.0
        self.current_faces  = []

        # PATCH: dipantau watchdog — kapan terakhir kali frame BARU (bukan retry) berhasil diambil
        self._last_capture_success_ts = time.time()

        self.stats_manager.load_statistics()
        config.init_directories()

        print("✅ System ready")
        print(f"   Detector      : {self.detector_type}")
        print(f"   Detection Size: {self.DETECTION_SIZE[0]}x{self.DETECTION_SIZE[1]}")
        print(f"   Embeddings    : {'ON' if self.use_embeddings else 'OFF'}")
        print(f"   Blur Filter   : ON (threshold={self.BLUR_THRESH})")
        print(f"   Frontal Only  : ON (relaxed saat >= {self.CROWD_FACE_COUNT} wajah/frame)")
        print(f"   Matching      : {'Hungarian (optimal)' if HUNGARIAN_AVAILABLE else 'Greedy (fallback)'}")
        print(f"   Watchdog      : ON (stall timeout={self.STALL_TIMEOUT}s)")

    # ─────────────────────────────────────────────────────────────────────
    # INITIALIZERS
    # ─────────────────────────────────────────────────────────────────────

    def _init_detector(self):
        """
        Model priority:
          1. face-detection-adas-0001   (CCTV jarak jauh, small face)
          2. face-detection-retail-0044 (lebih ringan, modern)
          3. face-detection-retail-0004 (lama, fallback)
          4. Haar Cascade               (CPU only fallback)
        """
        if not (OPENVINO_AVAILABLE and getattr(self.config, 'USE_OPENVINO', True)):
            self._init_haar_fallback()
            return

        model_candidates = [
            (
                getattr(self.config, 'FACE_DETECTION_MODEL_XML',
                        'models/face-detection-adas-0001.xml'),
                getattr(self.config, 'FACE_DETECTION_MODEL_BIN',
                        'models/face-detection-adas-0001.bin'),
                "face-detection-adas-0001"
            ),
            (
                'models/face-detection-retail-0044.xml',
                'models/face-detection-retail-0044.bin',
                "face-detection-retail-0044"
            ),
            (
                'models/face-detection-retail-0004.xml',
                'models/face-detection-retail-0004.bin',
                "face-detection-retail-0004"
            ),
        ]

        device = getattr(self.config, 'OPENVINO_DEVICE', 'CPU')

        for xml_path, bin_path, model_name in model_candidates:
            if not os.path.exists(xml_path):
                continue
            try:
                print(f"🔄 Loading OpenVINO model: {model_name} on {device}...")
                self.ie = Core()
                model   = self.ie.read_model(model=xml_path, weights=bin_path)

                try:
                    from openvino.runtime import properties
                    config_hint = {
                        properties.hint.performance_mode(): properties.hint.PerformanceMode.THROUGHPUT
                    }
                    self.compiled_model = self.ie.compile_model(
                        model=model, device_name=device, config=config_hint)
                except Exception:
                    self.compiled_model = self.ie.compile_model(
                        model=model, device_name=device)

                self.input_layer  = self.compiled_model.input(0)
                self.output_layer = self.compiled_model.output(0)

                shape = self.input_layer.shape
                self.ov_n, self.ov_c, self.ov_h, self.ov_w = shape

                self.use_openvino  = True
                self.detector_type = f"OpenVINO {model_name} ({device})"
                print(f"✅ {self.detector_type} — input {shape}")
                return

            except Exception as e:
                print(f"⚠️  Failed to load {model_name}: {e}")

        self._init_haar_fallback()

    def _init_haar_fallback(self):
        print("🔄 Using Haar Cascade fallback...")
        path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(path)
        if self.face_cascade.empty():
            raise RuntimeError("Haar Cascade load failed — no detector available!")
        self.use_openvino  = False
        self.detector_type = "Haar Cascade (CPU)"
        print("✅ Haar Cascade loaded")

    def _init_frontal_validator(self):
        try:
            self.frontal_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml')
            print("✅ Frontal validator loaded")
        except Exception as e:
            print(f"⚠️  Frontal validator failed: {e}")
            self.frontal_cascade = self.eye_cascade = None

    # ─────────────────────────────────────────────────────────────────────
    # LIFECYCLE
    # ─────────────────────────────────────────────────────────────────────

    def start(self):
        if self.is_running:
            return
        self.cap        = self.video_handler.connect()
        self.is_running = True
        self._last_capture_success_ts = time.time()

        threading.Thread(target=self._capture_loop,   daemon=True, name="Capture").start()
        threading.Thread(target=self._detection_loop, daemon=True, name="Detection").start()
        threading.Thread(target=self._render_loop,    daemon=True, name="Render").start()
        threading.Thread(target=self._watchdog_loop,  daemon=True, name="Watchdog").start()

        if getattr(self.config, 'ENABLE_AUTO_CLEANUP', True):
            threading.Thread(target=self._cleanup_loop, daemon=True, name="Cleanup").start()

        print("▶️  Face detection started")

    def stop(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.stats_manager.save_statistics()
        self.face_db.save_database()
        print(f"💾 Database saved ({len(self.face_db.faces)} faces) | Detection stopped")

    # ─────────────────────────────────────────────────────────────────────
    # THREADS
    # ─────────────────────────────────────────────────────────────────────

    def _capture_loop(self):
        """
        Ambil frame dari CCTV secepat mungkin.
        Selalu overwrite slot dengan frame terbaru — tidak ada queue blocking.
        """
        interval = 1.0 / getattr(self.config, 'STREAM_FPS', 25)
        W = getattr(self.config, 'FRAME_WIDTH',  1080)
        H = getattr(self.config, 'FRAME_HEIGHT', 608)

        while self.is_running:
            t0 = time.time()
            try:
                with self._cap_lock:
                    cap = self.cap

                if cap is None:
                    time.sleep(0.05)
                    continue

                # Flush buffer — buang frame lama, ambil yang paling baru
                for _ in range(3):
                    cap.grab()

                ret, frame = cap.retrieve()
                if not ret or frame is None or frame.size == 0:
                    self._reconnect()
                    continue

                # INTER_LINEAR 3x lebih cepat dari INTER_AREA, cukup untuk streaming
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)

                with self._frame_lock:
                    self._latest_frame = frame

                # PATCH: tandai sukses — ini yang dipantau watchdog
                self._last_capture_success_ts = time.time()

            except Exception as e:
                print(f"❌ Capture: {e}")
                time.sleep(0.5)

            elapsed = time.time() - t0
            sleep_t = interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    def _watchdog_loop(self):
        """
        PATCH BARU — jaring pengaman kedua di luar stimeout FFmpeg.

        stimeout di FFmpeg options harusnya bikin cap.grab()/retrieve()
        gagal (return False) kalau socket macet > 5 detik, tapi di
        praktiknya beberapa build OpenCV/FFmpeg gak selalu menghormati
        opsi itu dengan konsisten (tergantung backend & versi). Watchdog
        ini independen: kalau gak ada frame BARU yang berhasil diambil
        selama STALL_TIMEOUT detik, paksa release() + reconnect() dari
        thread terpisah — release() dari thread lain biasanya cukup untuk
        "membangunkan" panggilan blocking di capture thread.
        """
        check_interval = 2.0
        while self.is_running:
            time.sleep(check_interval)
            stale_for = time.time() - self._last_capture_success_ts
            if stale_for > self.STALL_TIMEOUT:
                print(f"🐶 Watchdog: gak ada frame baru selama {stale_for:.1f}s — paksa reconnect")
                self._reconnect()
                self._last_capture_success_ts = time.time()

    def _detection_loop(self):
        """
        Jalankan inference pada frame terbaru.
        Throttle ke DETECTION_FPS agar CPU tidak penuh.

        PATCH: pakai letterbox resize (jaga rasio aspek + padding) langsung
        ke ukuran input model, bukan stretch paksa ke 640x640 lalu resize
        lagi. Distorsi rasio aspek numpuk sebelumnya bikin wajah di pinggir
        frame jadi gepeng — makin parah buat wajah kecil/miring pas crowd.
        """
        detect_interval = 1.0 / self.DETECTION_FPS
        FW = getattr(self.config, 'FRAME_WIDTH',  1080)
        FH = getattr(self.config, 'FRAME_HEIGHT', 608)

        # Target ukuran deteksi: kalau OpenVINO, langsung pakai ukuran input
        # model asli (skip resize dua kali). Kalau Haar, pakai DETECTION_SIZE.
        if self.use_openvino and self.ov_w and self.ov_h:
            target_w, target_h = self.ov_w, self.ov_h
        else:
            target_w, target_h = self.DETECTION_SIZE

        while self.is_running:
            now  = time.time()
            wait = detect_interval - (now - self.last_detect_ts)
            if wait > 0:
                time.sleep(wait)
                continue

            with self._frame_lock:
                frame = self._latest_frame
            if frame is None:
                time.sleep(0.01)
                continue

            t0 = time.time()

            det_frame, scale, pad_x, pad_y = self._letterbox_resize(frame, target_w, target_h)
            raw_faces = (
                self._detect_openvino(det_frame)
                if self.use_openvino
                else self._detect_haar(det_frame)
            )

            # PATCH: deteksi crowd — kalau kandidat wajah di frame ini udah
            # banyak, relax pengecekan frontal (eye-cascade) yang gak
            # reliable buat wajah kecil/miring, biar gak makin nyaring
            # wajah valid pas lagi rame.
            is_crowd = len(raw_faces) >= self.CROWD_FACE_COUNT

            validated = []
            for face in raw_faces:
                bx, by, bw, bh = face['box']

                # Map balik dari koordinat letterbox ke koordinat frame asli
                sx = int((bx - pad_x) / scale)
                sy = int((by - pad_y) / scale)
                sw = int(bw / scale)
                sh = int(bh / scale)

                # Clamp ke batas frame (bukan cuma reject) — letterbox bisa
                # bikin box nyenggol tepi karena pembulatan
                sx = max(0, sx)
                sy = max(0, sy)
                sw = min(sw, FW - sx)
                sh = min(sh, FH - sy)
                if sw <= 0 or sh <= 0:
                    continue

                roi = frame[sy:sy + sh, sx:sx + sw]
                if roi.size == 0:
                    continue

                blur = calculate_blur_score(roi)
                if blur < self.BLUR_THRESH:
                    continue

                # Frontal check di-skip total kalau lagi crowd
                if not is_crowd and face.get('confidence', 0) < 0.70:
                    if not self._is_frontal(frame, [sx, sy, sw, sh]):
                        continue

                face['box']  = [sx, sy, sw, sh]
                face['blur'] = blur
                validated.append(face)

            self.processing_fps = 1.0 / max(time.time() - t0, 1e-6)
            self.last_detect_ts = time.time()

            with self._result_lock:
                self._latest_result = (frame.copy(), validated, now)

    def _render_loop(self):
        """
        Render frame ke self.frame pada TARGET_FPS.
        Jika tidak ada result baru, tetap render frame dengan tracker lama.
        """
        render_interval = 1.0 / getattr(self.config, 'TARGET_FPS', 20)
        last_result_ts  = 0.0

        while self.is_running:
            t0 = time.time()

            with self._result_lock:
                result = self._latest_result

            if result is None:
                time.sleep(0.01)
                continue

            frame, faces, ts = result

            if ts == last_result_ts:
                with self._frame_lock:
                    display_frame = self._latest_frame
                if display_frame is not None:
                    draw_frame = display_frame.copy()
                    self._redraw_trackers(draw_frame)
                    self.frame = draw_frame
            else:
                last_result_ts = ts
                if faces is not None:
                    self._update_trackers(frame, faces, ts)
                self._redraw_trackers(frame)
                self.frame = frame

            now          = time.time()
            render_faces = [e for e in self.trackers.values()
                            if now - e.last_seen <= self.ID_TIMEOUT]
            self.current_faces = render_faces
            self.stats_manager.update(len(render_faces))

            elapsed            = time.time() - self.last_frame_ts
            self.last_frame_ts = time.time()
            self.frame_times.append(elapsed)
            if len(self.frame_times) > 5:
                avg      = sum(self.frame_times) / len(self.frame_times)
                self.fps = 1.0 / avg if avg > 0 else 0

            sleep_t = render_interval - (time.time() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    def _redraw_trackers(self, frame: np.ndarray):
        """Gambar semua tracker aktif ke frame."""
        now = time.time()
        for entry in list(self.trackers.values()):
            if now - entry.last_seen <= self.ID_TIMEOUT and entry.last_box is not None:
                self._draw_detection(frame, entry.last_box, entry)

    def _cleanup_loop(self):
        interval = getattr(self.config, 'CLEANUP_INTERVAL', 300)
        while self.is_running:
            time.sleep(interval)
            print("🧹 Periodic cleanup triggered")
            self.face_db.save_database()
            self.stats_manager.save_statistics()

    # ─────────────────────────────────────────────────────────────────────
    # RESIZE — PATCH BARU: letterbox (jaga rasio aspek)
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _letterbox_resize(frame: np.ndarray, target_w: int, target_h: int):
        """
        Resize frame ke (target_w, target_h) TANPA distorsi rasio aspek —
        pakai padding hitam (letterbox), bukan stretch.

        Return: (canvas, scale, pad_x, pad_y) — scale & pad dipakai buat
        mapping koordinat box hasil deteksi balik ke frame asli.
        """
        h, w = frame.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        return canvas, scale, pad_x, pad_y

    # ─────────────────────────────────────────────────────────────────────
    # TRACKER MATCHING — Hungarian (optimal assignment)
    # ─────────────────────────────────────────────────────────────────────

    def _match_detections_to_trackers(self, faces: list, tracker_ids: list):
        """
        Optimal assignment antara deteksi baru dan tracker yang sudah ada,
        pakai Hungarian algorithm — ganti greedy nearest-match lama yang
        gampang salah assign pas banyak orang berdekatan.
        """
        n_faces = len(faces)

        if n_faces == 0:
            return {}, []
        if not tracker_ids:
            return {}, list(range(n_faces))

        if not HUNGARIAN_AVAILABLE:
            return self._match_detections_greedy(faces, tracker_ids)

        n_trackers = len(tracker_ids)
        cost = np.full((n_faces, n_trackers), self.MAX_POSITION_DIST * 50.0, dtype=np.float64)

        for fi, face in enumerate(faces):
            x, y, w, h = face['box']
            cx = x + w // 2
            cy = y + h // 2
            for ti, tid in enumerate(tracker_ids):
                entry    = self.trackers[tid]
                pos_dist = float(np.hypot(cx - entry.cx, cy - entry.cy))
                emb_sim  = 0.0
                if face.get('embedding') is not None and entry.embedding is not None:
                    emb_sim = cosine_similarity(face['embedding'], entry.embedding)

                score = (pos_dist * (1.0 - emb_sim)
                         if self.use_embeddings and emb_sim > self.EMBED_SIM_THRESH
                         else pos_dist)
                cost[fi, ti] = score

        row_ind, col_ind = linear_sum_assignment(cost)

        matches: dict = {}
        matched_faces = set()
        for fi, ti in zip(row_ind, col_ind):
            if cost[fi, ti] < self.MAX_POSITION_DIST:
                matches[int(fi)] = tracker_ids[int(ti)]
                matched_faces.add(int(fi))

        unmatched_faces = [fi for fi in range(n_faces) if fi not in matched_faces]
        return matches, unmatched_faces

    def _match_detections_greedy(self, faces: list, tracker_ids: list):
        """Fallback kalau scipy gak ke-install."""
        matches: dict = {}
        used_tids = set()

        for fi, face in enumerate(faces):
            x, y, w, h = face['box']
            cx = x + w // 2
            cy = y + h // 2

            best_tid   = None
            best_score = float('inf')

            for tid in tracker_ids:
                if tid in used_tids:
                    continue
                entry    = self.trackers[tid]
                pos_dist = np.hypot(cx - entry.cx, cy - entry.cy)
                emb_sim  = 0.0
                if face.get('embedding') is not None and entry.embedding is not None:
                    emb_sim = cosine_similarity(face['embedding'], entry.embedding)
                score = (pos_dist * (1.0 - emb_sim)
                         if self.use_embeddings and emb_sim > self.EMBED_SIM_THRESH
                         else pos_dist)
                if score < best_score:
                    best_score = score
                    best_tid   = tid

            if best_tid is not None and best_score < self.MAX_POSITION_DIST:
                matches[fi] = best_tid
                used_tids.add(best_tid)

        unmatched_faces = [fi for fi in range(len(faces)) if fi not in matches]
        return matches, unmatched_faces

    # ─────────────────────────────────────────────────────────────────────
    # TRACKER UPDATE — core logic
    # ─────────────────────────────────────────────────────────────────────

    def _update_trackers(self, frame: np.ndarray, faces: list, now: float):
        # 1) Hapus tracker yang sudah timeout
        stale = [fid for fid, e in self.trackers.items()
                 if now - e.last_seen > self.ID_TIMEOUT]
        for fid in stale:
            self.trackers[fid].status = FaceStatus.LOST
            del self.trackers[fid]

        # 2) Optimal assignment (Hungarian)
        tracker_ids = list(self.trackers.keys())
        matches, unmatched_faces = self._match_detections_to_trackers(faces, tracker_ids)

        for face_idx, face in enumerate(faces):
            box        = face['box']
            x, y, w, h = box
            cx         = x + w // 2
            cy         = y + h // 2
            quality    = face.get('quality',    0.0)
            blur       = face.get('blur',       0.0)

            # 3) Assign ke tracker lama atau buat baru
            if face_idx in matches:
                entry = self.trackers[matches[face_idx]]
            else:
                entry = TrackerEntry(self.next_id, cx, cy, now)
                self.trackers[self.next_id] = entry
                self.next_id += 1

                # ── RAW DETECTION COUNTING ──────────────────────
                self.stats_manager.add_raw_detection()
                self.face_db.log_raw_detection(entry.face_id, self.cam_id)

            entry.update_position(cx, cy, now)
            entry.quality_hist.append(quality)
            entry.blur_hist.append(blur)
            entry.advance_status()
            entry.last_box = box
            self.track_history[entry.face_id].append((float(cx), float(cy)))

            # 4) Throttled embedding extraction
            if (self.use_embeddings
                    and entry.status in (FaceStatus.VERIFYING, FaceStatus.TRACKING,
                                         FaceStatus.WAJAH_BARU, FaceStatus.SUDAH_TERDETEKSI)
                    and now - entry.last_embed_ts >= self.EMBED_INTERVAL):
                emb = self._extract_embedding(frame, box)
                if emb is not None:
                    entry.embedding     = emb
                    entry.last_embed_ts = now

            # 5) Database logic — anti-duplicate
            if entry.status == FaceStatus.VERIFYING and entry.is_verified():
                if self.use_embeddings and entry.embedding is not None:
                    db_id, similarity = self.face_db.find_matching_face(entry.embedding)

                    if db_id is not None and similarity >= self.EMBED_SIM_THRESH:
                        entry.status = FaceStatus.SUDAH_TERDETEKSI
                        entry.db_id  = db_id
                        print(f"👤 Sudah terdeteksi: {db_id} (sim={similarity:.3f})")
                    else:
                        entry.status = FaceStatus.WAJAH_BARU
                        unique_id = f"{entry.face_id}_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
                        self.face_db.add_or_update_face(unique_id, entry.embedding)
                        self.stats_manager.add_unique_person()
                        print(f"✨ WAJAH BARU #{entry.face_id} "
                              f"| quality={entry.avg_quality():.2f} "
                              f"| blur={entry.avg_blur():.0f} "
                              f"| total={self.stats_manager.total_detected}")
                else:
                    entry.status = FaceStatus.WAJAH_BARU
                    self.stats_manager.add_unique_person()

            elif entry.status in (FaceStatus.WAJAH_BARU, FaceStatus.SUDAH_TERDETEKSI):
                entry.status = FaceStatus.TRACKING

    # ─────────────────────────────────────────────────────────────────────
    # DETECTORS
    # ─────────────────────────────────────────────────────────────────────

    def _detect_openvino(self, det_frame: np.ndarray) -> list:
        """OpenVINO inference dengan pre-allocated input buffer."""
        try:
            h, w = det_frame.shape[:2]

            # PATCH: det_frame sekarang sudah pas ukuran ov_w x ov_h dari
            # letterbox di _detection_loop, jadi resize ini praktis no-op
            # (dipertahankan sebagai safety net kalau ada mismatch).
            inp = (cv2.resize(det_frame, (self.ov_w, self.ov_h),
                              interpolation=cv2.INTER_LINEAR)
                   if h != self.ov_h or w != self.ov_w
                   else det_frame)

            if self._ov_input_buf is None:
                self._ov_input_buf = np.empty(
                    (1, self.ov_c, self.ov_h, self.ov_w), dtype=np.float32)

            np.copyto(self._ov_input_buf[0],
                      inp.transpose(2, 0, 1).astype(np.float32))

            results    = self.compiled_model([self._ov_input_buf])
            detections = results[self.output_layer]

            faces = []
            for det in detections[0][0]:
                conf = float(det[2])
                if conf < self.CONFIDENCE_THRESH:
                    continue

                xmin = max(0, int(det[3] * w))
                ymin = max(0, int(det[4] * h))
                xmax = min(w, int(det[5] * w))
                ymax = min(h, int(det[6] * h))
                bw   = xmax - xmin
                bh   = ymax - ymin

                if bw < 15 or bh < 15:
                    continue

                quality = self._validate_quality(det_frame, xmin, ymin, bw, bh, conf)
                if quality < self.QUALITY_THRESH:
                    continue

                faces.append({
                    'box':        [xmin, ymin, bw, bh],
                    'confidence': conf,
                    'quality':    quality,
                    'embedding':  None,
                })
            return faces

        except Exception as e:
            print(f"OpenVINO detection error: {e}")
            return []

    def _detect_haar(self, det_frame: np.ndarray) -> list:
        """Haar Cascade fallback — tuned untuk jarak jauh."""
        try:
            gray = cv2.cvtColor(det_frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            boxes = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=3,
                minSize=(20, 20), maxSize=(500, 500))

            faces = []
            for (x, y, w, h) in boxes:
                quality = self._validate_quality(det_frame, x, y, w, h, 0.80)
                if quality >= self.QUALITY_THRESH:
                    faces.append({
                        'box':        [int(x), int(y), int(w), int(h)],
                        'confidence': 0.80,
                        'quality':    quality,
                        'embedding':  None,
                    })
            return faces

        except Exception as e:
            print(f"Haar detection error: {e}")
            return []

    # ─────────────────────────────────────────────────────────────────────
    # VALIDATORS
    # ─────────────────────────────────────────────────────────────────────

    def _validate_quality(self, frame, x, y, w, h, conf) -> float:
        score = 0.0

        if   conf > 0.95: score += 0.30
        elif conf > 0.90: score += 0.25
        elif conf > 0.85: score += 0.20

        ar = w / float(h)
        if 0.70 < ar < 1.30:
            score += 0.20

        rel = (w * h) / (frame.shape[0] * frame.shape[1])
        if 0.001 < rel < 0.60:
            score += 0.20

        mg = 10
        if x > mg and y > mg and x + w < frame.shape[1] - mg and y + h < frame.shape[0] - mg:
            score += 0.15

        try:
            roi = frame[y:y + h, x:x + w]
            if roi.size > 0:
                bri = np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
                if 35 < bri < 225:
                    score += 0.15
        except Exception:
            pass

        return min(1.0, score)

    def _is_frontal(self, frame: np.ndarray, box: list) -> bool:
        """
        Multi-check frontal validation:
          1. Aspect ratio
          2. Eye detection + symmetry
          3. Frontal cascade confirmation

        Catatan: sekarang di-skip total kalau _detection_loop mendeteksi
        crowd (lihat CROWD_FACE_COUNT) — fungsi ini gak reliable buat
        wajah kecil/miring khas kerumunan, daripada nyaring wajah valid.
        """
        x, y, w, h = box
        if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
            return False
        try:
            roi = frame[y:y + h, x:x + w]
            if roi.size == 0:
                return False
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            score     = 0.0
            max_score = 3.5

            ar = w / float(h)
            if 0.75 < ar < 1.25:
                score += 1.0

            if self.eye_cascade is not None:
                eyes = self.eye_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=3,
                    minSize=(int(w * 0.08), int(h * 0.08)),
                    maxSize=(int(w * 0.45), int(h * 0.45)))

                if len(eyes) >= 2:
                    score += 1.0
                    ec = [(ex + ew // 2, ey + eh // 2) for (ex, ey, ew, eh) in eyes[:2]]

                    y_diff = abs(ec[0][1] - ec[1][1])
                    if y_diff < h * 0.15:
                        score += 0.5

                    x_sep = abs(ec[0][0] - ec[1][0])
                    if 0.25 * w < x_sep < 0.70 * w:
                        score += 0.5

                    eye_mid_x  = (ec[0][0] + ec[1][0]) / 2
                    face_mid_x = w / 2
                    if abs(eye_mid_x - face_mid_x) < w * 0.20:
                        score += 0.5

            if self.frontal_cascade is not None:
                ff = self.frontal_cascade.detectMultiScale(
                    gray, scaleFactor=1.05, minNeighbors=3)
                if len(ff) > 0:
                    score += 0.5

            return (score / max_score) >= self.FRONTAL_THRESH

        except Exception as e:
            print(f"Frontal check error: {e}")
            return False

    # ─────────────────────────────────────────────────────────────────────
    # EMBEDDING
    # ─────────────────────────────────────────────────────────────────────

    def _extract_embedding(self, frame: np.ndarray, box: list):
        """FaceNet embedding extraction dengan margin crop."""
        if not self.use_embeddings:
            return None
        try:
            x, y, w, h = box
            mg = int(min(w, h) * 0.20)
            x1 = max(0, x - mg)
            y1 = max(0, y - mg)
            x2 = min(frame.shape[1], x + w + mg)
            y2 = min(frame.shape[0], y + h + mg)

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return None

            img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            img = img.resize((160, 160), Image.LANCZOS)

            arr = (np.array(img, dtype=np.float32) - 127.5) / 128.0
            t   = torch.from_numpy(arr).permute(2, 0, 1).to(self.device)

            with torch.no_grad():
                emb = self.resnet(t.unsqueeze(0))

            return emb.cpu().numpy().flatten()

        except Exception as e:
            print(f"Embedding error: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────
    # DRAWING
    # ─────────────────────────────────────────────────────────────────────

    def _draw_detection(self, frame: np.ndarray, box: list, entry: TrackerEntry):
        x, y, w, h = box
        status     = entry.status
        color      = STATUS_COLOR.get(status, (200, 200, 200))
        corner_len = max(12, min(w // 4, 30))
        thickness  = 2

        cv2.line(frame, (x, y),         (x + corner_len, y),         color, thickness)
        cv2.line(frame, (x, y),         (x, y + corner_len),         color, thickness)
        cv2.line(frame, (x + w, y),     (x + w - corner_len, y),     color, thickness)
        cv2.line(frame, (x + w, y),     (x + w, y + corner_len),     color, thickness)
        cv2.line(frame, (x, y + h),     (x + corner_len, y + h),     color, thickness)
        cv2.line(frame, (x, y + h),     (x, y + h - corner_len),     color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), color, thickness)

        label       = f"#{entry.face_id} {status.value}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ly          = max(lh + 8, y)
        cv2.rectangle(frame, (x, ly - lh - 6), (x + lw + 8, ly + 2), color, -1)
        cv2.putText(frame, label, (x + 4, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        bar_y = y + h + 3
        if bar_y + 4 < frame.shape[0]:
            bar_fill = int(w * min(1.0, entry.avg_blur() / 250))
            cv2.rectangle(frame, (x, bar_y), (x + w, bar_y + 3), (40, 40, 40), -1)
            cv2.rectangle(frame, (x, bar_y), (x + bar_fill, bar_y + 3), color, -1)

    def _draw_trail(self, frame: np.ndarray, face_id: int):
        # Trail dinonaktifkan
        pass

    # ─────────────────────────────────────────────────────────────────────
    # UTILITY
    # ─────────────────────────────────────────────────────────────────────

    def _build_cam_stats_path(self) -> str:
        base_path = getattr(self.config, 'STATS_FILE', 'data/face_counter_stats.pkl')
        root, ext = os.path.splitext(base_path)
        return f"{root}_cam{self.cam_id}{ext}"

    def _reconnect(self):
        """
        PATCH: sekarang di-guard self._cap_lock supaya capture_loop dan
        watchdog_loop gak rebutan akses self.cap secara bersamaan (bisa
        dipanggil dari dua thread berbeda sekarang).
        """
        print("⚠️  Stream lost — reconnecting...")
        with self._cap_lock:
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
            time.sleep(2)
            self.cap = self.video_handler.connect()

    def get_frame(self) -> np.ndarray:
        if self.frame is None:
            W = getattr(self.config, 'FRAME_WIDTH',  1080)
            H = getattr(self.config, 'FRAME_HEIGHT', 608)
            return np.zeros((H, W, 3), dtype=np.uint8)
        return self.frame.copy()

    def get_frame_jpeg(self) -> bytes:
        q    = getattr(self.config, 'JPEG_QUALITY', 80)
        _, buf = cv2.imencode('.jpg', self.get_frame(),
                              [cv2.IMWRITE_JPEG_QUALITY, q,
                               cv2.IMWRITE_JPEG_OPTIMIZE, 1])
        return buf.tobytes()

    def get_statistics(self) -> dict:
        stats = self.stats_manager.get_stats()
        stats.update({
            'fps':                round(self.fps, 1),
            'processing_fps':     round(self.processing_fps, 1),
            'active_trackers':    len(self.trackers),
            'current_faces':      len(self.current_faces),
            'timestamp':          now_wita_iso(),
            'detection_method':   self.detector_type,
            'embedding_tracking': self.use_embeddings,
            'database_size':      len(self.face_db.faces),
        })
        return stats

    def get_historical_data(self):
        return self.stats_manager.get_historical_data()

    def get_database_stats(self):
        return self.face_db.get_statistics()

    def save_face_database(self):
        self.face_db.save_database()
        print("💾 Face database saved")

    def reset_face_database(self):
        self.face_db.reset_database()
        self.trackers.clear()
        print("🔄 Face database reset")

    def reset_daily_stats(self):
        self.stats_manager.reset_daily()
        self.trackers.clear()
        self.current_faces = []
        self.next_id       = 0
        self.face_db.save_database()
        print("🔄 Daily stats reset (database preserved)")


# Alias for backward compatibility
FaceCounter = OpenVINOFaceCounter