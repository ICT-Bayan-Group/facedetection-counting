"""
Enhanced Configuration for Face Counter - Optimized for 25 FPS
"""
import os
import cv2

class Config:
    """Application configuration for face detection with 25 FPS optimization"""
    
    # CCTV Settings
    CCTV_IP = "10.2.22.30"
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
    HAAR_SCALE_FACTOR = 1.15  # Slightly larger for faster processing
    HAAR_MIN_NEIGHBORS = 5  # Balanced for speed vs accuracy
    HAAR_MIN_SIZE = (40, 40)  # Slightly smaller for better detection
    HAAR_MAX_SIZE = (350, 350)
    
    # Validation Settings
    ENABLE_EYE_VALIDATION = True
    ENABLE_SKIN_VALIDATION = True
    ENABLE_TEXTURE_VALIDATION = True
    QUALITY_THRESHOLD = 0.5
    MIN_CHECKS_PASSED = 0.4
    
    # DNN Face Detection settings
    CONFIDENCE_THRESHOLD = 0.5
    DNN_INPUT_SIZE = (300, 300)
    
    # Tracking Settings
    MAX_TRACKING_DISTANCE = 100
    FACE_TIMEOUT = 1.0

    # Frame Settings - OPTIMIZED FOR 50 FPS
    FRAME_WIDTH = 960  # Reduced from 1280 for better performance
    FRAME_HEIGHT = 540  # Reduced from 720 for better performance
    JPEG_QUALITY = 75  # Slightly lower for faster encoding
    TARGET_FPS = 50  # Target 50 FPS
    
    # Performance Optimization
    FRAME_SKIP = 0  # Process every frame
    MAX_BUFFER_SIZE = 1  # Minimal buffer for lowest latency
    
    # Tracking Settings
    TRACK_HISTORY_LENGTH = 30  # Reduced for faster processing
    FPS_WINDOW_SIZE = 15  # Faster FPS calculation
    
    # Storage Settings
    STATS_FILE = 'data/face_counter_stats.pkl'
    HISTORY_FILE = 'data/face_history.pkl'
    
    # Network Settings for RTSP - OPTIMIZED FOR 25 FPS
    RTSP_TRANSPORT = 'tcp'
    RTSP_TIMEOUT = 3000  # Reduced timeout
    RECONNECT_DELAY = 1  # Faster reconnect
    MAX_RECONNECT_ATTEMPTS = 3
    
    # DNN Model Paths
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
        """Get optimized OpenCV settings for 25 FPS"""
        return {
            cv2.CAP_PROP_BUFFERSIZE: 1,  # Minimal buffer
            cv2.CAP_PROP_FPS: 25,  # Target 25 FPS
            cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*'H264'),
            cv2.CAP_PROP_FRAME_WIDTH: Config.FRAME_WIDTH,
            cv2.CAP_PROP_FRAME_HEIGHT: Config.FRAME_HEIGHT
        }
    
    @staticmethod
    def download_dnn_models():
        """Download DNN models for better face detection accuracy"""
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