"""
OpenVINO-Optimized Configuration - Untuk Intel GPU/NPU
Mengurangi Beban CPU dengan OpenVINO Inference
"""
import os
import cv2

class Config:
    """Konfigurasi optimasi OpenVINO untuk Intel GPU/NPU"""
    
    # ========================================
    # OPENVINO & INFERENCE SETTINGS
    # ========================================
    USE_OPENVINO = True
    OPENVINO_DEVICE = 'GPU'  # Options: 'CPU', 'GPU', 'MYRIAD', 'AUTO'
    # 'GPU' = Intel integrated/discrete GPU
    # 'MYRIAD' = Intel Neural Compute Stick
    # 'AUTO' = Automatic device selection
    
    # Model paths for OpenVINO
    FACE_DETECTION_MODEL_XML = 'models/face-detection-adas-0001.xml'
    FACE_DETECTION_MODEL_BIN = 'models/face-detection-adas-0001.bin'
    
    # ========================================
    # FRAME RATE OPTIMIZATION (PENTING!)
    # ========================================
    TARGET_FPS = 10  # TURUN DRASTIS dari 50 ke 10 FPS
    STREAM_FPS = 5  # Stream output FPS
    DETECTION_FPS = 5  # Detection hanya 5 FPS (skip banyak frame)
    
    FRAME_SKIP = 1  # Process setiap frame ke-2
    
    # Adaptive FPS
    ENABLE_ADAPTIVE_FPS = True
    MIN_FPS = 5
    MAX_FPS = 15
    
    # ========================================
    # RESOLUTION OPTIMIZATION (LEBIH KECIL!)
    # ========================================
    FRAME_WIDTH = 640   # Turun dari 960 ke 640
    FRAME_HEIGHT = 360  # Turun dari 540 ke 360
    
    # Detection resolution (SANGAT KECIL untuk kecepatan)
    DETECTION_WIDTH = 320
    DETECTION_HEIGHT = 180
    
    JPEG_QUALITY = 80  # Turunkan quality untuk encoding lebih cepat
    
    # ========================================
    # CCTV SETTINGS
    # ========================================
    CCTV_IP = "10.2.22.30"
    CCTV_USER = "admin"
    CCTV_PASS = "ictb4y4n"
    
    CCTV_URLS = [
        f"rtsp://{CCTV_USER}:{CCTV_PASS}@{CCTV_IP}/streaming/channels/102",  # Lower quality stream
        f"rtsp://{CCTV_USER}:{CCTV_PASS}@{CCTV_IP}/streaming/channels/101",
        f"rtsp://{CCTV_USER}:{CCTV_PASS}@{CCTV_IP}:554/stream2",
    ]
    
    # ========================================
    # THREADING OPTIMIZATION
    # ========================================
    FRAME_QUEUE_SIZE = 1  # Minimal queue
    RESULT_QUEUE_SIZE = 1
    
    # ========================================
    # DETECTION OPTIMIZATION
    # ========================================
    # OpenVINO face detection thresholds
    CONFIDENCE_THRESHOLD = 0.7  # Lebih tinggi = lebih cepat
    
    # Tracking
    TRACK_HISTORY_LENGTH = 10  # Kurangi tracking history
    MAX_TRACKING_DISTANCE = 100
    FACE_TIMEOUT = 3.0
    ID_TIMEOUT = 4.0
    
    # Cooldown
    DETECTION_COOLDOWN = 5.0  # Detik
    
    # ========================================
    # MEMORY OPTIMIZATION
    # ========================================
    MAX_EMBEDDING_HISTORY = 2  # Minimal embedding storage
    MAX_QUALITY_HISTORY = 5
    
    ENABLE_AUTO_CLEANUP = True
    CLEANUP_INTERVAL = 180  # 3 menit
    
    # ========================================
    # FLASK SETTINGS
    # ========================================
    HOST = '0.0.0.0'
    PORT = 5000
    DEBUG = False
    THREADED = True
    
    # ========================================
    # STORAGE SETTINGS
    # ========================================
    STATS_FILE = 'data/face_counter_stats.pkl'
    HISTORY_FILE = 'data/face_history.pkl'
    DATABASE_FILE = 'data/face_database.json'
    SESSIONS_FILE = 'data/sessions.json'
    
    # ========================================
    # OPENCV OPTIMIZATION
    # ========================================
    MAX_BUFFER_SIZE = 1
    RTSP_TRANSPORT = 'tcp'
    RTSP_TIMEOUT = 5000
    RECONNECT_DELAY = 2
    MAX_RECONNECT_ATTEMPTS = 5
    
    @staticmethod
    def init_directories():
        """Create necessary directories"""
        os.makedirs('data', exist_ok=True)
        os.makedirs('templates', exist_ok=True)
        os.makedirs('static', exist_ok=True)
        os.makedirs('models', exist_ok=True)
    
    @staticmethod
    def download_openvino_models():
        """Download Intel OpenVINO face detection models"""
        import urllib.request
        
        model_xml_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/face-detection-adas-0001/FP32/face-detection-adas-0001.xml"
        model_bin_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/face-detection-adas-0001/FP32/face-detection-adas-0001.bin"
        
        print("📥 Downloading OpenVINO face detection models...")
        
        try:
            os.makedirs('models', exist_ok=True)
            
            if not os.path.exists(Config.FACE_DETECTION_MODEL_XML):
                print(f"   Downloading XML model...")
                urllib.request.urlretrieve(model_xml_url, Config.FACE_DETECTION_MODEL_XML)
                print("   ✅ XML downloaded")
            
            if not os.path.exists(Config.FACE_DETECTION_MODEL_BIN):
                print(f"   Downloading BIN model...")
                urllib.request.urlretrieve(model_bin_url, Config.FACE_DETECTION_MODEL_BIN)
                print("   ✅ BIN downloaded")
            
            print("✅ OpenVINO models ready!")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading models: {e}")
            print("\n💡 Manual download:")
            print(f"   XML: {model_xml_url}")
            print(f"   BIN: {model_bin_url}")
            return False
    
    @staticmethod
    def print_config():
        """Print configuration summary"""
        print("\n" + "="*70)
        print("⚙️  OPENVINO-OPTIMIZED CONFIGURATION")
        print("="*70)
        print(f"🎮 OpenVINO Device: {Config.OPENVINO_DEVICE}")
        print(f"🎬 Stream FPS: {Config.STREAM_FPS}")
        print(f"🔍 Detection FPS: {Config.DETECTION_FPS} (every ~{1/Config.DETECTION_FPS*Config.STREAM_FPS:.0f} frames)")
        print(f"📐 Stream Resolution: {Config.FRAME_WIDTH}x{Config.FRAME_HEIGHT}")
        print(f"🔍 Detection Size: {Config.DETECTION_WIDTH}x{Config.DETECTION_HEIGHT}")
        print(f"⏭️  Frame Skip: {Config.FRAME_SKIP}")
        print("="*70 + "\n")