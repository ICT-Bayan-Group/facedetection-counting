"""
HD Face Detection Configuration - High Performance & Accuracy
Full HD video with optimized AI detection
"""
import os
import cv2

class Config:
    """HD Configuration for high-quality face detection"""
    
    # ========================================
    # OPENVINO & INFERENCE SETTINGS
    # ========================================
    USE_OPENVINO = True
    OPENVINO_DEVICE = 'CPU'  # Use CPU (GPU requires OpenCL)
    
    # Model paths for OpenVINO
    FACE_DETECTION_MODEL_XML = 'models/face-detection-adas-0001.xml'
    FACE_DETECTION_MODEL_BIN = 'models/face-detection-adas-0001.bin'
    
    # ========================================
    # HD VIDEO STREAMING - 30 FPS!
    # ========================================
    TARGET_FPS = 10         # High performance target
    STREAM_FPS = 10         # Smooth 30 FPS streaming
    DETECTION_FPS = 15      # Detect every 2 frames for accuracy
    
    FRAME_SKIP = 1
    
    # Adaptive FPS disabled for consistent performance
    ENABLE_ADAPTIVE_FPS = False
    MIN_FPS = 25
    MAX_FPS = 30
    
    # ========================================
    # FULL HD RESOLUTION!
    # ========================================
    FRAME_WIDTH = 960      # Full HD width
    FRAME_HEIGHT = 540      # HD 720p
    
    # Detection resolution - balanced for accuracy
    DETECTION_WIDTH = 640   # Higher resolution for better accuracy
    DETECTION_HEIGHT = 360
    
    JPEG_QUALITY = 85       # High quality HD video
    
    # ========================================
    # CCTV SETTINGS
    # ========================================
    CCTV_IP = "10.2.22.30"
    CCTV_USER = "admin"
    CCTV_PASS = "ictb4y4n"
    
    # Use highest quality stream (channel 101)
    CCTV_URLS = [
        f"rtsp://{CCTV_USER}:{CCTV_PASS}@{CCTV_IP}/streaming/channels/101",  # Main stream
        f"rtsp://{CCTV_USER}:{CCTV_PASS}@{CCTV_IP}/streaming/channels/102",  # Sub stream
        f"rtsp://{CCTV_USER}:{CCTV_PASS}@{CCTV_IP}:554/stream1",
    ]
    
    # ========================================
    # THREADING OPTIMIZATION
    # ========================================
    FRAME_QUEUE_SIZE = 3    # Bigger queue for smooth HD
    RESULT_QUEUE_SIZE = 3
    
    # ========================================
    # AI DETECTION OPTIMIZATION
    # ========================================
    # Improved thresholds for accuracy
    CONFIDENCE_THRESHOLD = 0.5  # Lower for better recall
    MIN_FACE_SIZE = 40          # Minimum face size in pixels
    MAX_FACE_SIZE = 500         # Maximum face size
    
    # Tracking - optimized for accuracy
    TRACK_HISTORY_LENGTH = 20
    MAX_TRACKING_DISTANCE = 150
    FACE_TIMEOUT = 2.0
    ID_TIMEOUT = 5.0
    
    # Cooldown
    DETECTION_COOLDOWN = 2.0
    
    # ========================================
    # DISPLAY SETTINGS
    # ========================================
    # Show only NEW detections (no tracking boxes)
    SHOW_TRACKING_BOXES = False  # Hide white tracking boxes!
    SHOW_NEW_ONLY = True         # Only show green boxes for new faces
    SHOW_FACE_ID = True
    SHOW_CONFIDENCE = True
    
    # Box styling
    NEW_FACE_COLOR = (0, 255, 0)      # Green for new
    TRACKING_COLOR = (100, 100, 100)   # Dark gray (won't show)
    BOX_THICKNESS = 3
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2
    
    # ========================================
    # MEMORY OPTIMIZATION
    # ========================================
    MAX_EMBEDDING_HISTORY = 5
    MAX_QUALITY_HISTORY = 10
    
    ENABLE_AUTO_CLEANUP = True
    CLEANUP_INTERVAL = 300
    
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
    MAX_BUFFER_SIZE = 3
    RTSP_TRANSPORT = 'tcp'
    RTSP_TIMEOUT = 10000
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
        print("⚙️  HD FACE DETECTION - HIGH PERFORMANCE")
        print("="*70)
        print(f"🎮 Device: {Config.OPENVINO_DEVICE}")
        print(f"🎬 Stream: {Config.STREAM_FPS} FPS")
        print(f"🔍 Detection: {Config.DETECTION_FPS} FPS")
        print(f"📐 Resolution: {Config.FRAME_WIDTH}x{Config.FRAME_HEIGHT} (HD)")
        print(f"🔍 Detection Size: {Config.DETECTION_WIDTH}x{Config.DETECTION_HEIGHT}")
        print(f"📊 Quality: {Config.JPEG_QUALITY}%")
        print(f"👁️  Display: {'NEW ONLY' if Config.SHOW_NEW_ONLY else 'ALL'}")
        print("="*70 + "\n")