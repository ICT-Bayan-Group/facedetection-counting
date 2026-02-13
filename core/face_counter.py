"""
OpenVINO-Optimized Face Counter
Menggunakan Intel OpenVINO untuk GPU/NPU inference
Mengurangi beban CPU secara drastis
"""
import cv2
import numpy as np
import os
from collections import defaultdict, deque
from datetime import datetime
import threading
import time
from queue import Queue, Empty

# OpenVINO imports
try:
    from openvino.runtime import Core
    OPENVINO_AVAILABLE = True
    print("✅ OpenVINO available")
except ImportError:
    OPENVINO_AVAILABLE = False
    print("⚠️  OpenVINO not available, using OpenCV DNN")

from utils.face_database import FaceDatabase
from utils.video_utils import VideoStreamHandler
from utils.stats_manager import StatisticsManager

class OpenVINOFaceCounter:
    """
    Face Counter dengan OpenVINO inference
    - Menggunakan Intel GPU/NPU untuk detection
    - Frame skipping agresif
    - Resolusi rendah untuk kecepatan
    """
    
    def __init__(self, cctv_urls, user, password, config):
        print("🔄 Initializing OpenVINO Face Counter...")
        
        self.config = config
        
        # Initialize components
        self.video_handler = VideoStreamHandler(cctv_urls, user, password)
        self.stats_manager = StatisticsManager()
        self.face_db = FaceDatabase()
        
        print(f"📊 Face Database: {len(self.face_db.faces)} known faces")
        
        # Initialize OpenVINO or fallback
        self._init_detector()
        
        # Queues dengan size minimal
        self.frame_queue = Queue(maxsize=config.FRAME_QUEUE_SIZE)
        self.result_queue = Queue(maxsize=config.RESULT_QUEUE_SIZE)
        
        # Tracking
        self.track_history = defaultdict(lambda: deque(maxlen=config.TRACK_HISTORY_LENGTH))
        self.detected_ids = set()
        self.current_faces = []
        self.next_id = 0
        self.face_trackers = {}
        
        # Quality tracking (minimal size)
        self.face_quality_history = defaultdict(lambda: deque(maxlen=config.MAX_QUALITY_HISTORY))
        
        # Detection cooldown
        self.detection_cooldown = {}
        
        # State
        self.frame = None
        self.is_running = False
        self.cap = None
        
        # FPS tracking
        self.fps = 0
        self.processing_fps = 0
        self.frame_times = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # Frame counter
        self.frame_count = 0
        self.last_detection_time = 0
        
        # Load saved data
        self.stats_manager.load_statistics()
        config.init_directories()
        
        print("✅ OpenVINO-Optimized system initialized")
        print(f"   ✓ Detector: {self.detector_type}")
        print(f"   ✓ Target FPS: {config.TARGET_FPS}")
        print(f"   ✓ Detection FPS: {config.DETECTION_FPS}")
    
    def _init_detector(self):
        """Initialize OpenVINO detector atau fallback ke OpenCV DNN"""
        
        if OPENVINO_AVAILABLE and self.config.USE_OPENVINO:
            try:
                print(f"🔄 Loading OpenVINO on {self.config.OPENVINO_DEVICE}...")
                
                # Check if models exist
                if not os.path.exists(self.config.FACE_DETECTION_MODEL_XML):
                    print("⚠️  Models not found, downloading...")
                    self.config.download_openvino_models()
                
                # Initialize OpenVINO
                self.ie = Core()
                
                # List available devices
                devices = self.ie.available_devices
                print(f"   Available devices: {devices}")
                
                # Load model
                model = self.ie.read_model(
                    model=self.config.FACE_DETECTION_MODEL_XML,
                    weights=self.config.FACE_DETECTION_MODEL_BIN
                )
                
                # Compile for specific device
                self.compiled_model = self.ie.compile_model(
                    model=model,
                    device_name=self.config.OPENVINO_DEVICE
                )
                
                # Get input/output layers
                self.input_layer = self.compiled_model.input(0)
                self.output_layer = self.compiled_model.output(0)
                
                # Get input shape
                self.n, self.c, self.h, self.w = self.input_layer.shape
                
                self.use_openvino = True
                self.detector_type = f"OpenVINO ({self.config.OPENVINO_DEVICE})"
                
                print(f"✅ OpenVINO loaded on {self.config.OPENVINO_DEVICE}")
                print(f"   Input shape: {self.input_layer.shape}")
                
            except Exception as e:
                print(f"❌ OpenVINO init failed: {e}")
                self._init_opencv_dnn_fallback()
        else:
            self._init_opencv_dnn_fallback()
    
    def _init_opencv_dnn_fallback(self):
        """Fallback to OpenCV DNN (CPU)"""
        print("🔄 Using OpenCV DNN as fallback...")
        
        try:
            # Use Haar Cascade for fastest CPU detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                raise Exception("Failed to load Haar Cascade")
            
            self.use_openvino = False
            self.detector_type = "Haar Cascade (CPU)"
            print("✅ Haar Cascade loaded (CPU fallback)")
            
        except Exception as e:
            print(f"❌ Fallback init failed: {e}")
            raise
    
    def start(self):
        """Start detection dengan multi-threading"""
        if self.is_running:
            return
        
        self.cap = self.video_handler.connect()
        self.is_running = True
        
        # Start threads
        threading.Thread(target=self._frame_capture_loop, daemon=True, name="Capture").start()
        threading.Thread(target=self._detection_loop, daemon=True, name="Detection").start()
        threading.Thread(target=self._render_loop, daemon=True, name="Render").start()
        
        # Cleanup thread
        if self.config.ENABLE_AUTO_CLEANUP:
            threading.Thread(target=self._cleanup_loop, daemon=True, name="Cleanup").start()
        
        print("✅ OpenVINO face detection started")
    
    def stop(self):
        """Stop detection"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        
        self.stats_manager.save_statistics()
        self.face_db.save_database()
        
        print("⏸️  Detection stopped")
    
    def _frame_capture_loop(self):
        """Frame capture thread - CONTROLLED RATE"""
        target_interval = 1.0 / self.config.STREAM_FPS
        
        while self.is_running:
            try:
                loop_start = time.time()
                
                if self.cap is None:
                    time.sleep(0.1)
                    continue
                
                ret = self.cap.grab()
                if not ret:
                    self._reconnect_stream()
                    continue
                
                # Only retrieve if queue not full
                if not self.frame_queue.full():
                    ret, frame = self.cap.retrieve()
                    if ret and frame is not None and frame.size > 0:
                        # Resize ke target resolution
                        frame = cv2.resize(
                            frame, 
                            (self.config.FRAME_WIDTH, self.config.FRAME_HEIGHT),
                            interpolation=cv2.INTER_AREA
                        )
                        self.frame_queue.put(frame)
                
                # Control capture rate
                elapsed = time.time() - loop_start
                if elapsed < target_interval:
                    time.sleep(target_interval - elapsed)
                
            except Exception as e:
                print(f"❌ Capture error: {e}")
                time.sleep(1)
    
    def _detection_loop(self):
        """Detection thread - AGGRESSIVE FRAME SKIPPING"""
        detection_interval = 1.0 / self.config.DETECTION_FPS
        
        while self.is_running:
            try:
                if self.frame_queue.empty():
                    time.sleep(0.01)
                    continue
                
                frame = self.frame_queue.get(timeout=0.1)
                self.frame_count += 1
                
                current_time = time.time()
                
                # AGGRESSIVE FRAME SKIPPING
                time_since_detection = current_time - self.last_detection_time
                should_detect = time_since_detection >= detection_interval
                
                if not should_detect:
                    # Skip detection
                    if not self.result_queue.full():
                        self.result_queue.put((frame, None, current_time))
                    continue
                
                # DETECTION
                start_time = time.time()
                
                # Resize untuk detection (lebih kecil = lebih cepat)
                detection_frame = cv2.resize(
                    frame,
                    (self.config.DETECTION_WIDTH, self.config.DETECTION_HEIGHT),
                    interpolation=cv2.INTER_AREA
                )
                
                # Detect faces
                if self.use_openvino:
                    faces = self._detect_faces_openvino(detection_frame)
                else:
                    faces = self._detect_faces_haar(detection_frame)
                
                # Scale boxes kembali ke resolusi asli
                scale_x = self.config.FRAME_WIDTH / self.config.DETECTION_WIDTH
                scale_y = self.config.FRAME_HEIGHT / self.config.DETECTION_HEIGHT
                
                for face in faces:
                    box = face['box']
                    face['box'] = [
                        int(box[0] * scale_x),
                        int(box[1] * scale_y),
                        int(box[2] * scale_x),
                        int(box[3] * scale_y)
                    ]
                
                # Calculate processing FPS
                process_time = time.time() - start_time
                self.processing_fps = 1.0 / process_time if process_time > 0 else 0
                
                self.last_detection_time = current_time
                
                # Put result
                if not self.result_queue.full():
                    self.result_queue.put((frame, faces, current_time))
                
            except Empty:
                continue
            except Exception as e:
                print(f"❌ Detection error: {e}")
                time.sleep(0.1)
    
    def _detect_faces_openvino(self, frame):
        """Detect faces menggunakan OpenVINO"""
        try:
            # Prepare input
            input_frame = cv2.resize(frame, (self.w, self.h))
            input_frame = input_frame.transpose((2, 0, 1))  # HWC -> CHW
            input_frame = np.expand_dims(input_frame, 0)  # Add batch dimension
            
            # Run inference
            results = self.compiled_model([input_frame])
            detections = results[self.output_layer]
            
            faces = []
            h, w = frame.shape[:2]
            
            # Process detections
            for detection in detections[0][0]:
                confidence = float(detection[2])
                
                if confidence < self.config.CONFIDENCE_THRESHOLD:
                    continue
                
                # Get box coordinates
                xmin = int(detection[3] * w)
                ymin = int(detection[4] * h)
                xmax = int(detection[5] * w)
                ymax = int(detection[6] * h)
                
                # Validate box
                box_w = xmax - xmin
                box_h = ymax - ymin
                
                if box_w < 30 or box_h < 30:
                    continue
                
                faces.append({
                    'box': [xmin, ymin, box_w, box_h],
                    'confidence': confidence,
                    'quality': confidence
                })
            
            return faces
            
        except Exception as e:
            print(f"OpenVINO detection error: {e}")
            return []
    
    def _detect_faces_haar(self, frame):
        """Detect faces menggunakan Haar Cascade (fallback)"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            
            boxes = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(30, 30),
                maxSize=(300, 300)
            )
            
            faces = []
            for (x, y, w, h) in boxes:
                faces.append({
                    'box': [int(x), int(y), int(w), int(h)],
                    'confidence': 0.8,
                    'quality': 0.7
                })
            
            return faces
            
        except Exception as e:
            print(f"Haar detection error: {e}")
            return []
    
    def _render_loop(self):
        """Rendering thread dengan tracking"""
        while self.is_running:
            try:
                if self.result_queue.empty():
                    time.sleep(0.01)
                    continue
                
                frame, faces, timestamp = self.result_queue.get(timeout=0.1)
                
                if faces is None:
                    self.frame = frame
                    continue
                
                current_time = time.time()
                
                # Track faces
                tracked_faces = self._track_faces(faces, current_time)
                self.current_faces = tracked_faces
                
                # Draw detections
                for box, face_id, quality, confidence, status in tracked_faces:
                    self._draw_detection(frame, box, face_id, quality, confidence, status)
                
                # Update stats
                self.stats_manager.update(len(tracked_faces))
                
                self.frame = frame
                
                # Calculate FPS
                elapsed = time.time() - self.last_frame_time
                self.last_frame_time = time.time()
                self.frame_times.append(elapsed)
                
                if len(self.frame_times) > 5:
                    avg_time = sum(self.frame_times) / len(self.frame_times)
                    self.fps = 1.0 / avg_time if avg_time > 0 else 0
                
            except Empty:
                continue
            except Exception as e:
                print(f"❌ Render error: {e}")
                time.sleep(0.1)
    
    def _track_faces(self, faces, current_time):
        """Simple face tracking"""
        MAX_DIST = self.config.MAX_TRACKING_DISTANCE
        COOLDOWN = self.config.DETECTION_COOLDOWN
        
        # Cleanup old trackers
        ids_to_remove = [
            fid for fid, (_, _, ts, _) in self.face_trackers.items()
            if current_time - ts > self.config.ID_TIMEOUT
        ]
        
        for fid in ids_to_remove:
            del self.face_trackers[fid]
            self.face_quality_history.pop(fid, None)
        
        # Clean cooldown
        expired = [k for k, v in self.detection_cooldown.items() 
                  if current_time - v > COOLDOWN]
        for k in expired:
            del self.detection_cooldown[k]
        
        tracked_faces = []
        used_ids = set()
        
        for face in faces:
            box = face['box']
            x, y, w, h = box
            confidence = face['confidence']
            quality = face['quality']
            
            cx = x + w // 2
            cy = y + h // 2
            
            # Find best tracker
            best_id = None
            best_dist = float('inf')
            
            for fid, (tx, ty, _, _) in self.face_trackers.items():
                if fid in used_ids:
                    continue
                
                dist = np.sqrt((cx - tx)**2 + (cy - ty)**2)
                
                if dist < MAX_DIST and dist < best_dist:
                    best_dist = dist
                    best_id = fid
            
            # Assign ID
            if best_id is not None:
                face_id = best_id
                status = 'tracking'
            else:
                face_id = self.next_id
                self.next_id += 1
                status = 'new'
                
                # Check cooldown
                if face_id not in self.detected_ids:
                    self.detected_ids.add(face_id)
                    self.stats_manager.add_unique_person()
                    self.detection_cooldown[face_id] = current_time
                    print(f"✨ NEW FACE #{face_id} | Total: {self.stats_manager.total_detected}")
            
            # Update tracker
            self.face_trackers[face_id] = (cx, cy, current_time, quality)
            used_ids.add(face_id)
            
            tracked_faces.append((box, face_id, quality, confidence, status))
        
        return tracked_faces
    
    def _draw_detection(self, frame, box, face_id, quality, confidence, status):
        """Draw detection box"""
        x, y, w, h = box
        
        # Color based on status
        if status == 'new':
            color = (0, 255, 0)  # Green
            label = "NEW"
        else:
            color = (200, 200, 200)  # Gray
            label = "TRACK"
        
        # Draw box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Label
        text = f"#{face_id} {label}"
        cv2.putText(frame, text, (x+5, y-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Center point
        cx = x + w // 2
        cy = y + h // 2
        cv2.circle(frame, (cx, cy), 3, color, -1)
    
    def _cleanup_loop(self):
        """Periodic cleanup thread"""
        while self.is_running:
            time.sleep(self.config.CLEANUP_INTERVAL)
            print("🧹 Periodic cleanup...")
    
    def _reconnect_stream(self):
        """Reconnect to stream"""
        print("⚠️  Stream disconnected, reconnecting...")
        if self.cap:
            self.cap.release()
        time.sleep(2)
        self.cap = self.video_handler.connect()
    
    def get_frame(self):
        """Get current frame"""
        if self.frame is None:
            return np.zeros((self.config.FRAME_HEIGHT, self.config.FRAME_WIDTH, 3), dtype=np.uint8)
        return self.frame.copy()
    
    def get_statistics(self):
        """Get statistics"""
        stats = self.stats_manager.get_stats()
        stats.update({
            'fps': round(self.fps, 1),
            'processing_fps': round(self.processing_fps, 1),
            'active_trackers': len(self.face_trackers),
            'current_faces': len(self.current_faces),
            'timestamp': datetime.now().isoformat(),
            'detection_method': self.detector_type,
            'database_size': len(self.face_db.faces)
        })
        return stats
    
    def get_historical_data(self):
        return self.stats_manager.get_historical_data()
    
    def get_database_stats(self):
        return self.face_db.get_statistics()
    
    def save_face_database(self):
        self.face_db.save_database()
    
    def reset_daily_stats(self):
        self.stats_manager.reset_daily()
        self.face_trackers.clear()
        self.current_faces = []
        self.face_quality_history.clear()
        self.next_id = 0
        self.face_db.save_database()
        print("🔄 Daily stats reset")

# Alias
FaceCounter = OpenVINOFaceCounter