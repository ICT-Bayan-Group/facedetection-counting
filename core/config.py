"""
Enhanced Configuration for Optimized Performance
"""
import os

class Config:
    """Application configuration with performance optimizations"""
    
    # CCTV Settings
    CCTV_IP = "10.2.22.15"
    CCTV_USER = "user"
    CCTV_PASS = "Okedeh.12345"
    
    # Optimized CCTV URL Options (ordered by performance)
    CCTV_URLS = [
        # Try TCP first for better reliability
        f"rtsp://{CCTV_USER}:{CCTV_PASS}@{CCTV_IP}:554/stream2?tcp",
        f"rtsp://{CCTV_USER}:{CCTV_PASS}@{CCTV_IP}:554/stream1?tcp",
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
    
    # Enhanced Detection Settings
    YOLO_MODEL = 'yolov8n.pt'  # Nano model - fastest
    # Use 'yolov8s.pt' for better accuracy or 'yolov8m.pt' for best accuracy
    
    CONFIDENCE_THRESHOLD = 0.45  # Lower for better detection
    MIN_TRACKING_CONFIDENCE = 0.4  # Minimum to keep tracking
    NEW_ID_CONFIDENCE = 0.5  # Higher threshold for new person
    
    IOU_THRESHOLD = 0.4  # Lower for better separation
    TRACKER = "bytetrack.yaml"  # Fast and accurate tracker
    
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
    STATS_FILE = 'data/people_counter_stats.pkl'
    HISTORY_FILE = 'data/people_history.pkl'
    
    # Network Settings for RTSP
    RTSP_TRANSPORT = 'tcp'  # tcp or udp (tcp more reliable, udp lower latency)
    RTSP_TIMEOUT = 5000  # milliseconds
    RECONNECT_DELAY = 2  # seconds
    MAX_RECONNECT_ATTEMPTS = 3
    
    # GPU Settings
    USE_GPU = True  # Enable GPU acceleration if available
    USE_FP16 = True  # Use half precision on GPU for 2x speed
    
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

# Import cv2 for settings
import cv2