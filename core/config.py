"""
Configuration file for People Counter System
"""
import os

class Config:
    """Application configuration"""
    
    # CCTV Settings
    CCTV_IP = "10.2.22.50"
    CCTV_USER = "user"
    CCTV_PASS = "Okedeh.12345"
    
    # CCTV URL Options (will try in order)
    CCTV_URLS = [
        f"rtsp://{CCTV_USER}:{CCTV_PASS}@{CCTV_IP}:554/stream1",
        f"rtsp://{CCTV_USER}:{CCTV_PASS}@{CCTV_IP}:554/stream2",
        f"rtsp://{CCTV_USER}:{CCTV_PASS}@{CCTV_IP}/live",
        f"http://{CCTV_USER}:{CCTV_PASS}@{CCTV_IP}/video",
        f"http://{CCTV_IP}/mjpeg.cgi"
    ]
    
    # Flask Settings
    HOST = '0.0.0.0'
    PORT = 5000
    DEBUG = False
    
    # Detection Settings
    YOLO_MODEL = 'yolov8n.pt'  # Nano model for speed
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.45
    TRACKER = "bytetrack.yaml"
    
    # Frame Settings
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    JPEG_QUALITY = 85
    TARGET_FPS = 30
    
    # Tracking Settings
    TRACK_HISTORY_LENGTH = 30
    FPS_WINDOW_SIZE = 30
    
    # Storage Settings
    STATS_FILE = 'data/people_counter_stats.pkl'
    HISTORY_FILE = 'data/people_history.pkl'
    
    # Ensure data directory exists
    @staticmethod
    def init_directories():
        os.makedirs('data', exist_ok=True)
        os.makedirs('templates', exist_ok=True)
        os.makedirs('static', exist_ok=True)