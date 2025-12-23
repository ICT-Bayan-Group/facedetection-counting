"""
Video stream handling utilities - PRODUCTION READY
"""
import cv2
import os
import time
import threading

class VideoStreamHandler:
    """Handles CCTV video stream connection with optimized settings"""
    
    def __init__(self, cctv_urls, user, password):
        self.cctv_urls = cctv_urls
        self.user = user
        self.password = password
        self.current_url = None
        self.lock = threading.Lock()
        
        # Suppress OpenCV/FFmpeg verbose output
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
        os.environ['OPENCV_LOG_LEVEL'] = 'FATAL'
        os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
        
        try:
            cv2.setLogLevel(0)
        except:
            pass
    
    def connect(self):
        """
        Try to connect to CCTV using multiple URL formats
        Returns: cv2.VideoCapture object or None
        """
        print("\n🎥 Connecting to CCTV...")
        
        for idx, url in enumerate(self.cctv_urls, 1):
            try:
                # Show safe URL (hide password)
                url_display = url.replace(self.password, "***")
                if len(url_display) > 70:
                    url_display = url_display[:67] + "..."
                
                print(f"   [{idx}/{len(self.cctv_urls)}] Trying: {url_display}")
                
                start_time = time.time()
                
                # Create VideoCapture with FFmpeg backend
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                
                if not cap.isOpened():
                    print(f"      ❌ Cannot open stream")
                    continue
                
                # Set basic properties
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                # Try to grab and read first frame
                max_attempts = 3
                frame_valid = False
                
                for attempt in range(max_attempts):
                    grabbed = cap.grab()
                    if not grabbed:
                        time.sleep(0.1)
                        continue
                    
                    ret, frame = cap.retrieve()
                    if ret and frame is not None and frame.size > 0:
                        frame_valid = True
                        break
                
                elapsed = time.time() - start_time
                
                if frame_valid:
                    height, width = frame.shape[:2]
                    print(f"      ✅ Connected! Resolution: {width}x{height} ({elapsed:.1f}s)")
                    self.current_url = url
                    return cap
                else:
                    cap.release()
                    print(f"      ❌ No valid frame received ({elapsed:.1f}s)")
                    
            except Exception as e:
                error_msg = str(e)
                if len(error_msg) > 50:
                    error_msg = error_msg[:47] + "..."
                print(f"      ❌ Error: {error_msg}")
                continue
        
        print("\n❌ Failed to connect to any CCTV URL")
        print("\n💡 Troubleshooting tips:")
        print(f"   1. Verify CCTV is accessible: ping {self.cctv_urls[0].split('@')[1].split('/')[0].split(':')[0]}")
        print(f"   2. Check credentials: user='{self.user}'")
        print("   3. Try with VLC or ffplay:")
        print(f"      ffplay \"{self.cctv_urls[0]}\"")
        print("   4. Check firewall/network settings")
        print("   5. Verify RTSP port 554 is open")
        
        return None
    
    def reconnect(self, old_cap):
        """
        Reconnect to CCTV stream
        Args:
            old_cap: Old VideoCapture object to release
        Returns:
            New VideoCapture object or None
        """
        with self.lock:
            if old_cap is not None:
                try:
                    old_cap.release()
                except:
                    pass
            
            print("\n🔄 Attempting to reconnect...")
            time.sleep(2)  # Brief delay before reconnect
            return self.connect()
    
    def is_valid_frame(self, frame):
        """
        Validate if frame is usable
        Args:
            frame: numpy array from VideoCapture.read()
        Returns:
            bool: True if frame is valid
        """
        if frame is None:
            return False
        
        if not hasattr(frame, 'shape'):
            return False
        
        if frame.size == 0:
            return False
        
        if len(frame.shape) < 2:
            return False
        
        if frame.shape[0] == 0 or frame.shape[1] == 0:
            return False
        
        return True
    
    def optimize_capture(self, cap):
        """
        Apply optimized settings to VideoCapture
        Args:
            cap: cv2.VideoCapture object
        """
        if cap is None or not cap.isOpened():
            return
        
        try:
            # Buffer settings for low latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            
            # Frame rate
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Codec
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
            
            # Resolution (optional - let camera decide)
            # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
        except Exception as e:
            print(f"⚠️  Warning: Could not apply all optimizations: {e}")
    
    def get_stream_info(self, cap):
        """
        Get stream information
        Args:
            cap: cv2.VideoCapture object
        Returns:
            dict: Stream properties or None
        """
        if cap is None or not cap.isOpened():
            return None
        
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            backend = cap.getBackendName()
            
            # Decode fourcc
            fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            info = {
                'width': width,
                'height': height,
                'fps': fps,
                'codec': fourcc_str,
                'backend': backend,
                'url': self.current_url.replace(self.password, "***") if self.current_url else None
            }
            
            return info
            
        except Exception as e:
            print(f"⚠️  Could not retrieve stream info: {e}")
            return None
    
    def test_url_quick(self, url, timeout=3):
        """
        Quick test if URL is accessible
        Args:
            url: RTSP URL to test
            timeout: Timeout in seconds
        Returns:
            tuple: (success: bool, time_taken: float)
        """
        start_time = time.time()
        
        try:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            
            if not cap.isOpened():
                return False, time.time() - start_time
            
            # Try to grab one frame
            success = cap.grab()
            elapsed = time.time() - start_time
            
            cap.release()
            
            return success, elapsed
            
        except Exception as e:
            return False, time.time() - start_time
    
    def get_best_url(self):
        """
        Test all URLs and return the fastest working one
        Returns:
            str: Best URL or None
        """
        print("\n🔍 Testing all CCTV URLs to find fastest...")
        
        best_url = None
        best_time = float('inf')
        
        for idx, url in enumerate(self.cctv_urls, 1):
            url_display = url.replace(self.password, "***")[:60]
            print(f"   [{idx}/{len(self.cctv_urls)}] {url_display}")
            
            success, elapsed = self.test_url_quick(url, timeout=3)
            
            if success:
                print(f"      ✅ Working ({elapsed:.2f}s)")
                if elapsed < best_time:
                    best_time = elapsed
                    best_url = url
            else:
                print(f"      ❌ Failed ({elapsed:.2f}s)")
        
        if best_url:
            print(f"\n✅ Best URL found (response time: {best_time:.2f}s)")
            return best_url
        else:
            print("\n❌ No working URLs found")
            return None
    
    def create_demo_stream(self, width=640, height=480, fps=30):
        """
        Create a demo video stream for testing
        Args:
            width: Frame width
            height: Frame height
            fps: Frames per second
        Returns:
            Generator yielding demo frames
        """
        import numpy as np
        
        print(f"\n🎬 Starting DEMO mode ({width}x{height} @ {fps}fps)")
        print("   Generating synthetic video stream...")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            # Create colored frame with moving rectangle
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = (40, 40, 40)  # Dark gray background
            
            # Moving rectangle
            x = int((frame_count % 100) * width / 100)
            cv2.rectangle(frame, (x, height//3), (x+50, height*2//3), (0, 255, 0), -1)
            
            # Add text
            cv2.putText(frame, "DEMO MODE - No CCTV Connected", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Frame: {frame_count}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            elapsed = time.time() - start_time
            actual_fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(frame, f"FPS: {actual_fps:.1f}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            frame_count += 1
            
            yield frame
            
            # Frame rate control
            time.sleep(1.0 / fps)