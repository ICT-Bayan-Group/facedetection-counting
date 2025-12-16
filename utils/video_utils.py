"""
Video stream handling utilities
"""
import cv2
import os

class VideoStreamHandler:
    """Handles CCTV video stream connection"""
    
    def __init__(self, cctv_urls, user, password):
        self.cctv_urls = cctv_urls
        self.user = user
        self.password = password
        
        # Suppress FFmpeg/OpenCV error messages
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
        os.environ['OPENCV_LOG_LEVEL'] = 'FATAL'
        os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
    
    def connect(self):
        """Try multiple CCTV URL formats with optimized settings"""
        print("\n🎥 Connecting to CCTV...")
        
        for url in self.cctv_urls:
            try:
                print(f"   Trying: {url[:50]}...")
                
                # Create capture with FFmpeg backend
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                
                # Optimize settings for RTSP
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Small buffer to reduce latency
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                
                # Try to grab a few frames to clear initial corrupt frames
                for _ in range(5):
                    cap.grab()
                
                # Test if we can read a valid frame
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    print(f"✅ Connected successfully!")
                    print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
                    return cap
                else:
                    cap.release()
                    
            except Exception as e:
                print(f"   ❌ Failed: {str(e)}")
                continue
        
        print("\n⚠️  Could not connect to CCTV. Using demo mode.")
        return None
    
    def reconnect(self, old_cap):
        """Reconnect to CCTV stream"""
        if old_cap:
            old_cap.release()
        return self.connect()
    
    def is_valid_frame(self, frame):
        """Check if frame is valid"""
        if frame is None:
            return False
        if frame.size == 0:
            return False
        if frame.shape[0] == 0 or frame.shape[1] == 0:
            return False
        return True
    
    def optimize_capture(self, cap):
        """Apply optimal settings to capture"""
        if cap is None:
            return
        
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
        except Exception as e:
            print(f"⚠️  Could not optimize capture: {e}")
    
    def get_stream_info(self, cap):
        """Get stream information"""
        if cap is None:
            return None
        
        try:
            info = {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': int(cap.get(cv2.CAP_PROP_FPS)),
                'codec': int(cap.get(cv2.CAP_PROP_FOURCC)),
                'backend': cap.getBackendName()
            }
            return info
        except:
            return None