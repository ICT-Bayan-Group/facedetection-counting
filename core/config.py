"""
Enhanced Configuration for Face Counter - Optimized Performance
"""
import os
import cv2

class Config:
    """Application configuration for face detection with performance optimizations"""
    
    # CCTV Settings
    CCTV_IP = "10.2.22.39"
    CCTV_USER = "admin"
    CCTV_PASS = "ictb4y4n"
    
    # Optimized CCTV URL Options (ordered by performance)
    CCTV_URLS = [
        # Try TCP first for better reliability
        f"rtsp://{CCTV_USER}:{CCTV_PASS}@{CCTV_IP}/streaming/channels/101",
        f"rtsp://{CCTV_USER}:{CCTV_PASS}@{CCTV_IP}/streaming/channels/101",
        # Then UDP for lower latency
        f"rtsp://{CCTV_USER}:{CCTV_PASS}@{CCTV_IP}:554/stream2",
        f"rtsp://{CCTV_USER}:{CCTV_PASS}@{CCTV_IP}:554/stream1",
        f"rtsp://{CCTV_USER}:{CCTV_PASS}@{CCTV_IP}/live",
        # HTTP fallback
        f"http://{CCTV_USER}:{CCTV_PASS}@{CCTV_IP}/video",
        f"http://{CCTV_IP}/mjpeg.cgi"
    ]
    
    # Flask Settings
    HOST = '0.0.0.0'
    PORT = 5000
    DEBUG = False
    THREADED = True
    
    # Face Detection Settings
    # Haar Cascade settings
    HAAR_SCALE_FACTOR = 1.1  # Smaller = more accurate but slower
    HAAR_MIN_NEIGHBORS = 6  # Higher = fewer false positives (increased)
    HAAR_MIN_SIZE = (50, 50)  # Minimum face size (increased)
    HAAR_MAX_SIZE = (400, 400)  # Maximum face size (prevent false positives)
    
    # Validation Settings
    ENABLE_EYE_VALIDATION = True  # Require eyes to be detected
    ENABLE_SKIN_VALIDATION = True  # Check for skin-like colors
    ENABLE_TEXTURE_VALIDATION = True  # Check texture variance
    QUALITY_THRESHOLD = 0.5  # Minimum quality score (0.0 - 1.0)
    MIN_CHECKS_PASSED = 0.4  # Minimum 40% of validation checks must pass
    
    # DNN Face Detection settings (if using DNN model)
    CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold for DNN
    DNN_INPUT_SIZE = (300, 300)  # Input size for DNN
    
    # Tracking Settings
    MAX_TRACKING_DISTANCE = 100  # Maximum distance to consider same face
    FACE_TIMEOUT = 1.0  # Seconds before face ID expires
    
    # Frame Settings - Optimized for real-time
    FRAME_WIDTH = 1280  # HD resolution
    FRAME_HEIGHT = 720
    JPEG_QUALITY = 80  # Balance quality and speed
    TARGET_FPS = 30
    
    # Performance Optimization
    FRAME_SKIP = 0  # Process every frame (set to 1 to skip alternate frames)
    MAX_BUFFER_SIZE = 2  # Small buffer for low latency
    
    # Tracking Settings
    TRACK_HISTORY_LENGTH = 40  # Longer trail
    FPS_WINDOW_SIZE = 20  # Responsive FPS calculation
    
    # Storage Settings
    STATS_FILE = 'data/face_counter_stats.pkl'
    HISTORY_FILE = 'data/face_history.pkl'
    
    # Network Settings for RTSP
    RTSP_TRANSPORT = 'tcp'  # tcp or udp (tcp more reliable, udp lower latency)
    RTSP_TIMEOUT = 5000  # milliseconds
    RECONNECT_DELAY = 2  # seconds
    MAX_RECONNECT_ATTEMPTS = 3
    
    # DNN Model Paths (optional - for better accuracy)
    DNN_PROTO = "deploy.prototxt"
    DNN_MODEL = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    
    @staticmethod
    def init_directories():
        """Create necessary directories"""
        os.makedirs('data', exist_ok=True)
        os.makedirs('templates', exist_ok=True)
        os.makedirs('static', exist_ok=True)
    
    @staticmethod
    def get_optimized_cv2_settings():
        """Get optimized OpenCV settings"""
        return {
            cv2.CAP_PROP_BUFFERSIZE: 1,  # Minimal buffer
            cv2.CAP_PROP_FPS: Config.TARGET_FPS,
            cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*'H264'),
            cv2.CAP_PROP_FRAME_WIDTH: Config.FRAME_WIDTH,
            cv2.CAP_PROP_FRAME_HEIGHT: Config.FRAME_HEIGHT
        }
    
    @staticmethod
    def download_dnn_models():
        """
        Download DNN models for better face detection accuracy
        Run this once to download the models
        """
        import urllib.request
        
        proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        
        print("Downloading DNN face detection models...")
        
        try:
            if not os.path.exists(Config.DNN_PROTO):
                print(f"Downloading {Config.DNN_PROTO}...")
                urllib.request.urlretrieve(proto_url, Config.DNN_PROTO)
                print("✅ Proto file downloaded")
            
            if not os.path.exists(Config.DNN_MODEL):
                print(f"Downloading {Config.DNN_MODEL} (this may take a while)...")
                urllib.request.urlretrieve(model_url, Config.DNN_MODEL)
                print("✅ Model file downloaded")
            
            print("✅ All DNN models ready!")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading models: {e}")
            print("💡 Tip: You can manually download from:")
            print(f"   Proto: {proto_url}")
            print(f"   Model: {model_url}")
            return False