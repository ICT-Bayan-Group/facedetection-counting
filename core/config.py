"""
OpenVINO-Optimized Configuration - 20 FPS Smooth Streaming
"""
import os
import cv2

class Config:
    """Konfigurasi optimasi untuk 20 FPS smooth streaming"""
    
    # ========================================
    # OPENVINO & INFERENCE SETTINGS
    # ========================================
    USE_OPENVINO = True
    OPENVINO_DEVICE = 'CPU'  # Use CPU since GPU not available
    
    # Model paths for OpenVINO
    FACE_DETECTION_MODEL_XML = 'models/face-detection-adas-0001.xml'
    FACE_DETECTION_MODEL_BIN = 'models/face-detection-adas-0001.bin'
    
    # ========================================
    # FRAME RATE OPTIMIZATION - 20 FPS!
    # ========================================
    TARGET_FPS = 25         # Target processing FPS
    STREAM_FPS = 25         # Stream output 20 FPS (smooth!)
    DETECTION_FPS = 10      # Detection setiap 2 frame (lebih cepat)
    
    FRAME_SKIP = 1          # Process setiap frame
    
    # Adaptive FPS
    ENABLE_ADAPTIVE_FPS = False  # Disable adaptive, use fixed 20 FPS
    MIN_FPS = 20
    MAX_FPS = 20
    
    # ========================================
    # RESOLUTION OPTIMIZATION
    # ========================================
    FRAME_WIDTH = 1080       # Resolution cukup untuk 20 FPS
    FRAME_HEIGHT = 720      
    
    # Detection resolution (keep small for speed)
    DETECTION_WIDTH = 320
    DETECTION_HEIGHT = 180
    
    JPEG_QUALITY = 75       # Quality sedang untuk balance speed/quality
    
    # ========================================
    # CCTV SETTINGS
    # ========================================
    CCTV_IP = "10.2.22.30"
    CCTV_USER = "admin"
    CCTV_PASS = "ictb4y4n"
    
    CCTV_URLS = [
        f"rtsp://{CCTV_USER}:{CCTV_PASS}@{CCTV_IP}/streaming/channels/102",
        f"rtsp://{CCTV_USER}:{CCTV_PASS}@{CCTV_IP}/streaming/channels/101",
        f"rtsp://{CCTV_USER}:{CCTV_PASS}@{CCTV_IP}:554/stream2",
    ]
    
    # ========================================
    # THREADING OPTIMIZATION
    # ========================================
    FRAME_QUEUE_SIZE = 2    # Slightly bigger queue for smoother flow
    RESULT_QUEUE_SIZE = 2
    
    # ========================================
    # DETECTION OPTIMIZATION
    # ========================================
    CONFIDENCE_THRESHOLD = 0.6  # Slightly lower for better detection
    
    # Tracking
    TRACK_HISTORY_LENGTH = 15
    MAX_TRACKING_DISTANCE = 100
    FACE_TIMEOUT = 3.0
    ID_TIMEOUT = 5.0
    
    # Cooldown
    DETECTION_COOLDOWN = 3.0  # Reduced cooldown
    
    # ========================================
    # MEMORY OPTIMIZATION
    # ========================================
    MAX_EMBEDDING_HISTORY = 3
    MAX_QUALITY_HISTORY = 5
    
    ENABLE_AUTO_CLEANUP = True
    CLEANUP_INTERVAL = 300  # 5 menit
    
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
    MAX_BUFFER_SIZE = 2     # Slightly bigger buffer
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
        print("⚙️  20 FPS SMOOTH STREAMING CONFIGURATION")
        print("="*70)
        print(f"🎮 OpenVINO Device: {Config.OPENVINO_DEVICE}")
        print(f"🎬 Stream FPS: {Config.STREAM_FPS} (SMOOTH!)")
        print(f"🔍 Detection FPS: {Config.DETECTION_FPS} (every ~{Config.STREAM_FPS/Config.DETECTION_FPS:.0f} frames)")
        print(f"📐 Stream Resolution: {Config.FRAME_WIDTH}x{Config.FRAME_HEIGHT}")
        print(f"🔍 Detection Size: {Config.DETECTION_WIDTH}x{Config.DETECTION_HEIGHT}")
        print(f"📊 JPEG Quality: {Config.JPEG_QUALITY}%")
        print("="*70 + "\n")