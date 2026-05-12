"""
Surveillance-Grade Face Detection Configuration
Sinkron dengan optimized_face_counter.py
"""
import os

class Config:

    # ========================================
    # OPENVINO & MODEL
    # ========================================
    USE_OPENVINO     = True
    OPENVINO_DEVICE  = 'CPU'

    # Priority: adas-0001 paling bagus untuk CCTV jarak jauh
    FACE_DETECTION_MODEL_XML = 'models/face-detection-adas-0001.xml'
    FACE_DETECTION_MODEL_BIN = 'models/face-detection-adas-0001.bin'

    # ========================================
    # FPS — JANGAN naikkan DETECTION_FPS
    # Detection berat di CPU, tracking yang mengisi sisanya
    # ========================================
    TARGET_FPS    = 20
    STREAM_FPS    = 25   # naikkan sedikit agar capture tidak lagging
    DETECTION_FPS = 5    # naikkan dari 3 → 5 (CPU masih aman, detection lebih responsif)

    FRAME_SKIP = 0       # jangan skip frame
    ENABLE_ADAPTIVE_FPS = False
    MIN_FPS = 15
    MAX_FPS = 25

    # ========================================
    # RESOLUSI
    # FRAME = resolusi display ke browser
    # DETECTION = resolusi input ke OpenVINO
    # ========================================
    FRAME_WIDTH  = 960
    FRAME_HEIGHT = 540

    # ← 640x640 WAJIB untuk deteksi wajah kecil di 8 meter
    # Jangan turunkan ke 320 atau 416
    DETECTION_WIDTH  = 640
    DETECTION_HEIGHT = 640

    JPEG_QUALITY = 82

    # ========================================
    # CCTV
    # ========================================
    CCTV_IP   = "10.2.22.30"
    CCTV_USER = "admin"
    CCTV_PASS = "ictb4y4n"

    CCTV_URLS = [
        f"rtsp://{CCTV_USER}:{CCTV_PASS}@{CCTV_IP}/streaming/channels/101",
        f"rtsp://{CCTV_USER}:{CCTV_PASS}@{CCTV_IP}/streaming/channels/102",
        f"rtsp://{CCTV_USER}:{CCTV_PASS}@{CCTV_IP}:554/stream1",
    ]

    # ========================================
    # THREADING
    # Queue kecil = latency rendah (jangan besarkan)
    # ========================================
    FRAME_QUEUE_SIZE  = 1   # ← turunkan dari 3 ke 1 (hanya frame terbaru)
    RESULT_QUEUE_SIZE = 1   # ← sama

    # ========================================
    # THRESHOLD DETEKSI
    # Nilai di bawah ini DIABAIKAN oleh face_counter
    # (face_counter punya konstannya sendiri)
    # Tapi tetap disimpan untuk referensi/legacy
    # ========================================
    CONFIDENCE_THRESHOLD = 0.55   # sinkron: CONFIDENCE_THRESH di face_counter
    MIN_FACE_SIZE        = 15     # sinkron: min 15px di OpenVINO inference
    MAX_FACE_SIZE        = 500

    # ========================================
    # TRACKING
    # ========================================
    TRACK_HISTORY_LENGTH   = 40    # sinkron: deque(maxlen=40) di face_counter
    MAX_TRACKING_DISTANCE  = 180   # sinkron: MAX_POSITION_DIST
    FACE_TIMEOUT           = 2.0
    ID_TIMEOUT             = 6.0   # sinkron: ID_TIMEOUT di face_counter

    DETECTION_COOLDOWN = 2.0

    # ========================================
    # DISPLAY
    # Trail hijau sudah dimatikan di face_counter._draw_trail()
    # Config ini untuk referensi saja
    # ========================================
    SHOW_TRACKING_BOXES = True    # corner bracket box tetap tampil semua status
    SHOW_NEW_ONLY       = False   # semua status tampil (DETECTED, VERIFYING, dst)
    SHOW_FACE_ID        = True
    SHOW_CONFIDENCE     = True
    SHOW_TRAIL          = False   # ← trail hijau MATI

    # Warna dihandle STATUS_COLOR di face_counter, ini untuk referensi
    NEW_FACE_COLOR    = (0, 255, 0)
    TRACKING_COLOR    = (150, 150, 150)
    BOX_THICKNESS     = 2
    FONT_SCALE        = 0.45
    FONT_THICKNESS    = 1

    # ========================================
    # MEMORY
    # ========================================
    MAX_EMBEDDING_HISTORY = 5
    MAX_QUALITY_HISTORY   = 15   # sinkron: deque(maxlen=15) di TrackerEntry

    ENABLE_AUTO_CLEANUP = True
    CLEANUP_INTERVAL    = 300

    # ========================================
    # FLASK
    # ========================================
    HOST     = '0.0.0.0'
    PORT     = 5000
    DEBUG    = False
    THREADED = True

    # ========================================
    # STORAGE
    # ========================================
    STATS_FILE    = 'data/face_counter_stats.pkl'
    HISTORY_FILE  = 'data/face_history.pkl'
    DATABASE_FILE = 'data/face_database.json'
    SESSIONS_FILE = 'data/sessions.json'

    # ========================================
    # OPENCV / RTSP
    # ========================================
    MAX_BUFFER_SIZE        = 3
    RTSP_TRANSPORT         = 'tcp'
    RTSP_TIMEOUT           = 10000
    RECONNECT_DELAY        = 2
    MAX_RECONNECT_ATTEMPTS = 5

    @staticmethod
    def init_directories():
        for d in ('data', 'templates', 'static', 'models'):
            os.makedirs(d, exist_ok=True)

    @staticmethod
    def download_openvino_models():
        import urllib.request

        base = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/face-detection-adas-0001/FP32"
        files = {
            Config.FACE_DETECTION_MODEL_XML: f"{base}/face-detection-adas-0001.xml",
            Config.FACE_DETECTION_MODEL_BIN: f"{base}/face-detection-adas-0001.bin",
        }

        print("📥 Downloading OpenVINO face-detection-adas-0001...")
        os.makedirs('models', exist_ok=True)

        for local_path, url in files.items():
            if os.path.exists(local_path):
                print(f"   ✅ Already exists: {local_path}")
                continue
            try:
                print(f"   ⬇️  {local_path} ...")
                urllib.request.urlretrieve(url, local_path)
                print(f"   ✅ Done")
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                print(f"      Manual: {url}")
                return False

        print("✅ Models ready!")
        return True

    @staticmethod
    def print_config():
        print("\n" + "="*60)
        print("⚙️  SURVEILLANCE FACE DETECTION CONFIG")
        print("="*60)
        print(f"   Device          : {Config.OPENVINO_DEVICE}")
        print(f"   Stream FPS      : {Config.STREAM_FPS}")
        print(f"   Detection FPS   : {Config.DETECTION_FPS}  ← CPU-friendly")
        print(f"   Frame Size      : {Config.FRAME_WIDTH}×{Config.FRAME_HEIGHT}")
        print(f"   Detection Size  : {Config.DETECTION_WIDTH}×{Config.DETECTION_HEIGHT}  ← 640 wajib!")
        print(f"   Confidence Min  : {Config.CONFIDENCE_THRESHOLD}")
        print(f"   Trail           : {'ON' if Config.SHOW_TRAIL else 'OFF'}")
        print("="*60 + "\n")