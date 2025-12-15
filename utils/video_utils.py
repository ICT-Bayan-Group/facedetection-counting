"""
Video stream handling utilities
"""
import cv2

class VideoStreamHandler:
    """Handles CCTV video stream connection"""
    
    def __init__(self, cctv_urls, user, password):
        self.cctv_urls = cctv_urls
        self.user = user
        self.password = password
    
    def connect(self):
        """Try multiple CCTV URL formats"""
        print("\n🎥 Connecting to CCTV...")
        
        for url in self.cctv_urls:
            try:
                print(f"   Trying: {url[:50]}...")
                
                cap = cv2.VideoCapture(url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Test if we can read
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✅ Connected successfully!")
                    return cap
                else:
                    cap.release()
                    
            except Exception as e:
                print(f"   ❌ Failed: {str(e)}")
                continue
        
        print("\n⚠️  Could not connect to CCTV. Using demo mode.")
        return None