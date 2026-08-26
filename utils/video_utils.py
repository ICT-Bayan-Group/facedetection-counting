"""
Video stream handling utilities - OPTIMIZED FOR LOW LATENCY + STABILITAS JARINGAN
"""
import cv2
import os
import time
import threading


class VideoStreamHandler:
    """Handles CCTV video stream connection dengan optimasi latency & network resilience"""

    def __init__(self, cctv_urls, user, password, target_fps: int = 25):
        self.cctv_urls  = cctv_urls
        self.user       = user
        self.password   = password
        self.target_fps = target_fps  # PATCH: dulu hardcoded 50 di banyak tempat, sekarang satu sumber kebenaran
        self.current_url = None
        self.lock = threading.Lock()

        # ─────────────────────────────────────────────────────────────
        # PATCH — FFMPEG OPTIONS DIROMBAK:
        #
        # SEBELUM: 'rtsp_transport;tcp|buffer_size;1024000|max_delay;500000'
        #   - GAK ADA stimeout sama sekali. Kalau jaringan CCTV putus-nyambung
        #     (packet loss / RTSP server gak respon), cap.grab()/retrieve()
        #     bisa BLOCK TANPA BATAS WAKTU nunggu socket. Ini penyebab utama
        #     "lemot dikit-dikit" — bukan lemot, tapi macet sesaat nunggu
        #     socket sebelum akhirnya jalan lagi.
        #   - max_delay 500000 (500ms) = FFmpeg sengaja nunda setengah detik
        #     demi smoothness buffer, nambah lag yang gak kelihatan tapi nyata
        #     buat live counting.
        #
        # SESUDAH:
        #   - stimeout;5000000     -> socket timeout 5 detik. Kalau network
        #     hiccup lebih dari itu, FFmpeg nyerah dan return error, BUKAN
        #     block selamanya. Ini yang bikin _reconnect() bisa kepanggil.
        #   - max_delay;100000     -> 100ms, jauh lebih responsif dari 500ms
        #   - fflags;nobuffer      -> minimalkan internal buffering FFmpeg
        #   - flags;low_delay      -> prioritaskan latency rendah di decoder
        #   - reorder_queue_size;0 -> jangan nunggu re-order paket RTP,
        #     langsung proses begitu datang (trade-off: sedikit lebih rawan
        #     artefak kalau paket kacau urutannya, tapi CCTV lokal biasanya
        #     jaringannya stabil urutan paketnya)
        # ─────────────────────────────────────────────────────────────
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
            'rtsp_transport;tcp|'
            'stimeout;5000000|'
            'buffer_size;1024000|'
            'max_delay;100000|'
            'fflags;nobuffer|'
            'flags;low_delay|'
            'reorder_queue_size;0'
        )
        os.environ['OPENCV_LOG_LEVEL'] = 'FATAL'
        os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
        os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'

        try:
            cv2.setLogLevel(0)
        except:
            pass

    def connect(self):
        """
        Connect to CCTV dengan setting low-latency + timeout.
        Returns: cv2.VideoCapture object or None
        """
        print(f"\n🎥 Connecting to CCTV ({self.target_fps} FPS target, low-latency mode)...")

        for idx, url in enumerate(self.cctv_urls, 1):
            try:
                url_display = url.replace(self.password, "***")
                if len(url_display) > 70:
                    url_display = url_display[:67] + "..."

                print(f"   [{idx}/{len(self.cctv_urls)}] Trying: {url_display}")

                start_time = time.time()

                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

                if not cap.isOpened():
                    print(f"      ❌ Cannot open stream")
                    continue

                self.optimize_capture(cap)

                # Try to grab and read first frame with faster timeout
                max_attempts = 2
                frame_valid = False

                for attempt in range(max_attempts):
                    grabbed = cap.grab()
                    if not grabbed:
                        time.sleep(0.05)
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
        """Fast reconnect to CCTV stream"""
        with self.lock:
            if old_cap is not None:
                try:
                    old_cap.release()
                except:
                    pass

            print("\n🔄 Fast reconnecting...")
            time.sleep(1)
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
        """
        Apply low-latency settings to VideoCapture.

        PATCH: CAP_PROP_FPS sekarang pakai self.target_fps (default 25),
        BUKAN hardcoded 50. Sebelumnya kode minta 50fps ke kamera padahal
        STREAM_FPS di config biasanya 25 — mismatch ini bisa bikin
        RTSP server/encoder kamera "bingung" dan malah nge-drop frame lebih
        sering, apalagi kalau bandwidth jaringan pas-pasan.
        """
        if cap is None or not cap.isOpened():
            return

        try:
            # Minimal buffer for low latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # PATCH: selaras dengan FPS asli stream, bukan angka ambisius
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)

            # H264 codec
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))

            # Optimized resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 608)

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

    def create_demo_stream(self, width=1080, height=608, fps=25):
        """Create a demo video stream"""
        import numpy as np

        print(f"\n🎬 Starting DEMO mode ({width}x{height} @ {fps}fps)")
        print("   Generating synthetic video stream...")

        frame_count = 0
        start_time = time.time()
        frame_time = 1.0 / fps

        while True:
            loop_start = time.time()

            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = (40, 40, 40)

            x = int((frame_count % 100) * width / 100)
            cv2.rectangle(frame, (x, height//3), (x+50, height*2//3), (0, 255, 0), -1)

            cv2.putText(frame, "DEMO MODE", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Frame: {frame_count}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            elapsed = time.time() - start_time
            actual_fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(frame, f"FPS: {actual_fps:.1f}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            frame_count += 1

            yield frame

            elapsed = time.time() - loop_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)