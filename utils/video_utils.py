"""
Video stream handling utilities - OPTIMIZED FOR 25 FPS
"""
import cv2
import os
import time
import threading

class VideoStreamHandler:
    """Handles CCTV video stream connection with 25 FPS optimization"""
    
    def __init__(self, cctv_urls, user, password):
        self.cctv_urls = cctv_urls
        self.user = user
        self.password = password
        self.current_url = None
        self.lock = threading.Lock()
        
        # Suppress OpenCV/FFmpeg verbose output
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|buffer_size;1024000|max_delay;500000'
        os.environ['OPENCV_LOG_LEVEL'] = 'FATAL'
        os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
        os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'
        
        try:
            cv2.setLogLevel(0)
        except:
            pass
    
    def connect(self):
        """
        Connect to CCTV with 25 FPS optimization
        Returns: cv2.VideoCapture object or None
        """
        print("\n🎥 Connecting to CCTV (25 FPS mode)...")
        
        for idx, url in enumerate(self.cctv_urls, 1):
            try:
                url_display = url.replace(self.password, "***")
                if len(url_display) > 70:
                    url_display = url_display[:67] + "..."
                
                print(f"   [{idx}/{len(self.cctv_urls)}] Trying: {url_display}")
                
                start_time = time.time()
                
                # Create VideoCapture with optimized settings
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                
                if not cap.isOpened():
                    print(f"      ❌ Cannot open stream")
                    continue
                
                # OPTIMIZED SETTINGS FOR 50 FPS
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
                cap.set(cv2.CAP_PROP_FPS, 50)  # Target 50 FPS
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                
                # Try to set resolution (optional, let camera decide if fails)
                try:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
                except:
                    pass
                
                # Try to grab and read first frame with faster timeout
                max_attempts = 2  # Reduced attempts
                frame_valid = False
                
                for attempt in range(max_attempts):
                    grabbed = cap.grab()
                    if not grabbed:
                        time.sleep(0.05)  # Shorter delay
                        continue
                    
                    ret, frame = cap.retrieve()
                    if ret and frame is not None and frame.size > 0:
                        frame_valid = True
                        break
                
                elapsed = time.time() - start_time
                
                if frame_valid:
                    height, width = frame.shape[:2]
                    actual_fps = cap.get(cv2.CAP_PROP_FPS)
                    print(f"      ✅ Connected! {width}x{height} @ {actual_fps:.0f}fps ({elapsed:.1f}s)")
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
        Fast reconnect to CCTV stream
        """
        with self.lock:
            if old_cap is not None:
                try:
                    old_cap.release()
                except:
                    pass
            
            print("\n🔄 Fast reconnecting...")
            time.sleep(1)  # Shorter delay for faster recovery
            return self.connect()
    
    def is_valid_frame(self, frame):
        """Validate if frame is usable"""
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
        """Apply 25 FPS optimized settings to VideoCapture"""
        if cap is None or not cap.isOpened():
            return
        
        try:
            # Minimal buffer for low latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Target 50 FPS
            cap.set(cv2.CAP_PROP_FPS, 50)
            
            # H264 codec
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
            
            # Optimized resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
            
        except Exception as e:
            print(f"⚠️  Warning: Could not apply all optimizations: {e}")
    
    def get_stream_info(self, cap):
        """Get stream information"""
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
    
    def test_url_quick(self, url, timeout=2):
        """Quick test with shorter timeout"""
        start_time = time.time()
        
        try:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            
            if not cap.isOpened():
                return False, time.time() - start_time
            
            success = cap.grab()
            elapsed = time.time() - start_time
            
            cap.release()
            
            return success, elapsed
            
        except Exception as e:
            return False, time.time() - start_time
    
    def get_best_url(self):
        """Test all URLs and return the fastest working one"""
        print("\n🔍 Testing all CCTV URLs (fast mode)...")
        
        best_url = None
        best_time = float('inf')
        
        for idx, url in enumerate(self.cctv_urls, 1):
            url_display = url.replace(self.password, "***")[:60]
            print(f"   [{idx}/{len(self.cctv_urls)}] {url_display}")
            
            success, elapsed = self.test_url_quick(url, timeout=2)
            
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
    
    def create_demo_stream(self, width=960, height=540, fps=50):
        """Create a demo video stream at 50 FPS"""
        import numpy as np
        
        print(f"\n🎬 Starting DEMO mode ({width}x{height} @ {fps}fps)")
        print("   Generating synthetic video stream...")
        
        frame_count = 0
        start_time = time.time()
        frame_time = 1.0 / fps
        
        while True:
            loop_start = time.time()
            
            # Create colored frame with moving rectangle
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = (40, 40, 40)
            
            # Moving rectangle
            x = int((frame_count % 100) * width / 100)
            cv2.rectangle(frame, (x, height//3), (x+50, height*2//3), (0, 255, 0), -1)
            
            # Add text
            cv2.putText(frame, "DEMO MODE - 25 FPS Target", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Frame: {frame_count}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            elapsed = time.time() - start_time
            actual_fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(frame, f"FPS: {actual_fps:.1f}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            frame_count += 1
            
            yield frame
            
            # Precise frame rate control
            elapsed = time.time() - loop_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)