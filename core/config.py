"""
Surveillance-Grade Face Detection Configuration
Sinkron dengan optimized_face_counter.py
Versi: 2 kamera (10.2.22.30 + 10.2.22.15)
"""
import os

class Config:

    # ========================================
    # OPENVINO & MODEL
    # ========================================
    USE_OPENVINO     = True
    OPENVINO_DEVICE  = 'CPU'

    FACE_DETECTION_MODEL_XML = 'models/face-detection-adas-0001.xml'
    FACE_DETECTION_MODEL_BIN = 'models/face-detection-adas-0001.bin'

    # ========================================
    # FPS
    # ========================================
    TARGET_FPS    = 20
    STREAM_FPS    = 25
    DETECTION_FPS = 5

    FRAME_SKIP = 0
    ENABLE_ADAPTIVE_FPS = False
    MIN_FPS = 15
    MAX_FPS = 25

    # ========================================
    # RESOLUSI
    # ========================================
    FRAME_WIDTH  = 1080
    FRAME_HEIGHT = 608

    DETECTION_WIDTH  = 640
    DETECTION_HEIGHT = 640

    JPEG_QUALITY = 82

    # ========================================
    # KAMERA 1 — IP utama lama
    # ========================================
    CCTV_IP   = "192.168.20.3"
    CCTV_USER = "admin"
    CCTV_PASS = "ictb4y4n"

    CCTV_URLS = [
        f"rtsp://{CCTV_USER}:{CCTV_PASS}@{CCTV_IP}/streaming/channels/101",
        f"rtsp://{CCTV_USER}:{CCTV_PASS}@{CCTV_IP}/streaming/channels/102",
        f"rtsp://{CCTV_USER}:{CCTV_PASS}@{CCTV_IP}:554/stream1",
    ]

    # ========================================
    # KAMERA 2 — IP tambahan baru
    # ========================================
    CCTV_IP_2   = "192.168.20.2"
    CCTV_USER_2 = "admin"
    CCTV_PASS_2 = "ictb4y4n"

    CCTV_URLS_2 = [
        f"rtsp://{CCTV_USER_2}:{CCTV_PASS_2}@{CCTV_IP_2}/streaming/channels/101",
        f"rtsp://{CCTV_USER_2}:{CCTV_PASS_2}@{CCTV_IP_2}/streaming/channels/102",
        f"rtsp://{CCTV_USER_2}:{CCTV_PASS_2}@{CCTV_IP_2}:554/stream1",
    ]

    # ========================================
    # MULTI-CAMERA CONFIG
    # Urutan: [kamera_0, kamera_1]
    # ========================================
    CAMERAS = [
        {
            "id":       0,
            "label":    "Kamera 1",
            "ip":       CCTV_IP,
            "user":     CCTV_USER,
            "password": CCTV_PASS,
            "urls": [
                f"rtsp://{CCTV_USER}:{CCTV_PASS}@{CCTV_IP}/streaming/channels/101",
                f"rtsp://{CCTV_USER}:{CCTV_PASS}@{CCTV_IP}/streaming/channels/102",
                f"rtsp://{CCTV_USER}:{CCTV_PASS}@{CCTV_IP}:554/stream1",
            ],
        },
        {
            "id":       1,
            "label":    "Kamera 2",
            "ip":       CCTV_IP_2,
            "user":     CCTV_USER_2,
            "password": CCTV_PASS_2,
            "urls": [
                f"rtsp://{CCTV_USER_2}:{CCTV_PASS_2}@{CCTV_IP_2}/streaming/channels/101",
                f"rtsp://{CCTV_USER_2}:{CCTV_PASS_2}@{CCTV_IP_2}/streaming/channels/102",
                f"rtsp://{CCTV_USER_2}:{CCTV_PASS_2}@{CCTV_IP_2}:554/stream1",
            ],
        },
    ]

    # ========================================
    # THREADING
    # ========================================
    FRAME_QUEUE_SIZE  = 1
    RESULT_QUEUE_SIZE = 1

    # ========================================
    # THRESHOLD DETEKSI
    # ========================================
    CONFIDENCE_THRESHOLD = 0.55
    MIN_FACE_SIZE        = 15
    MAX_FACE_SIZE        = 500

    # ========================================
    # TRACKING
    # ========================================
    TRACK_HISTORY_LENGTH   = 40
    MAX_TRACKING_DISTANCE  = 180
    FACE_TIMEOUT           = 2.0
    ID_TIMEOUT             = 6.0

    DETECTION_COOLDOWN = 2.0

    # ========================================
    # DISPLAY
    # ========================================
    SHOW_TRACKING_BOXES = True
    SHOW_NEW_ONLY       = False
    SHOW_FACE_ID        = True
    SHOW_CONFIDENCE     = True
    SHOW_TRAIL          = False

    NEW_FACE_COLOR    = (0, 255, 0)
    TRACKING_COLOR    = (150, 150, 150)
    BOX_THICKNESS     = 2
    FONT_SCALE        = 0.45
    FONT_THICKNESS    = 1

    # ========================================
    # MEMORY
    # ========================================
    MAX_EMBEDDING_HISTORY = 5
    MAX_QUALITY_HISTORY   = 15

    ENABLE_AUTO_CLEANUP = True
    CLEANUP_INTERVAL    = 300

    # ========================================
    # FLASK
    # ========================================
    HOST     = '0.0.0.0'
    PORT     = 8000
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
        print("⚙️  SURVEILLANCE FACE DETECTION CONFIG — 2 KAMERA")
        print("="*60)
        print(f"   Device          : {Config.OPENVINO_DEVICE}")
        print(f"   Stream FPS      : {Config.STREAM_FPS}")
        print(f"   Detection FPS   : {Config.DETECTION_FPS}  ← CPU-friendly")
        print(f"   Frame Size      : {Config.FRAME_WIDTH}×{Config.FRAME_HEIGHT}")
        print(f"   Detection Size  : {Config.DETECTION_WIDTH}×{Config.DETECTION_HEIGHT}  ← 640 wajib!")
        print(f"   Confidence Min  : {Config.CONFIDENCE_THRESHOLD}")
        print(f"   Trail           : {'ON' if Config.SHOW_TRAIL else 'OFF'}")
        print(f"   Jumlah Kamera   : {len(Config.CAMERAS)}")
        for cam in Config.CAMERAS:
            print(f"     [{cam['id']}] {cam['label']} → {cam['ip']}")
        print("="*60 + "\n")