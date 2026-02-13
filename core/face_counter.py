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

# Deep Learning untuk face recognition
try:
    import torch
    from PIL import Image
    from facenet_pytorch import InceptionResnetV1
    FACENET_AVAILABLE = True
    print("✅ FaceNet available for embeddings")
except ImportError:
    FACENET_AVAILABLE = False
    print("⚠️  FaceNet not available, using position-based tracking")

from utils.face_database import FaceDatabase
from utils.video_utils import VideoStreamHandler
from utils.stats_manager import StatisticsManager

class OpenVINOFaceCounter:
    """
    Enhanced Face Counter dengan:
    - OpenVINO untuk detection
    - FaceNet untuk embeddings
    - Frontal face detection only
    - Persistent face database
    """
    
    def __init__(self, cctv_urls, user, password, config):
        print("🔄 Initializing Improved OpenVINO Face Counter...")
        
        # Core configuration
        self.config = config
        
        # Component initialization
        self.video_handler = VideoStreamHandler(cctv_urls, user, password)
        self.stats_manager = StatisticsManager()
        self.face_db = FaceDatabase()
        
        print(f"📊 Face Database: {len(self.face_db.faces)} known faces")
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if FACENET_AVAILABLE else None
        if self.device:
            print(f"🖥️  Device: {self.device}")
        
        # FaceNet embedding system
        self.use_embeddings = False
        self.resnet = None
        if FACENET_AVAILABLE:
            try:
                self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
                self.use_embeddings = True
                print("✅ FaceNet loaded for embedding-based tracking")
            except Exception as e:
                print(f"⚠️  FaceNet init failed: {e}")
                self.use_embeddings = False
        
        # Detection system initialization
        self.detector_type = "Unknown"
        self.use_openvino = False
        self.ie = None
        self.compiled_model = None
        self.input_layer = None
        self.output_layer = None
        self.n = None
        self.c = None
        self.h = None
        self.w = None
        self.face_cascade = None
        self.frontal_cascade = None
        self.eye_cascade = None
        
        self._init_detector()
        self._init_frontal_detector()
        
        # Threading queues
        self.frame_queue = Queue(maxsize=config.FRAME_QUEUE_SIZE)
        self.result_queue = Queue(maxsize=config.RESULT_QUEUE_SIZE)
        
        # Tracking state
        self.track_history = defaultdict(lambda: deque(maxlen=config.TRACK_HISTORY_LENGTH))
        self.detected_ids = set()
        self.current_faces = []
        self.next_id = 0
        self.face_trackers = {}
        
        # Quality and embedding tracking
        self.face_quality_history = defaultdict(lambda: deque(maxlen=config.MAX_QUALITY_HISTORY))
        self.face_embeddings = {}
        self.detection_cooldown = {}
        
        # Video capture state
        self.frame = None
        self.is_running = False
        self.cap = None
        
        # Performance metrics
        self.fps = 0
        self.processing_fps = 0
        self.frame_times = deque(maxlen=30)
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.last_detection_time = 0
        
        # Detection thresholds
        self.confidence_threshold = 0.90
        self.embedding_threshold = 0.6
        self.frontal_threshold = 0.7
        
        # Load saved data
        self.stats_manager.load_statistics()
        config.init_directories()
        
        print("✅ Improved OpenVINO system initialized")
        print(f"   ✓ Detector: {self.detector_type}")
        print(f"   ✓ Embeddings: {'Enabled' if self.use_embeddings else 'Disabled'}")
        print(f"   ✓ Frontal Face Only: Enabled")
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
        """Fallback to OpenCV DNN"""
        print("🔄 Using OpenCV DNN as fallback...")
        
        try:
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
    
    def _init_frontal_detector(self):
        """Initialize frontal face detector untuk validasi"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
            self.frontal_cascade = cv2.CascadeClassifier(cascade_path)
            
            eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
            
            print("✅ Frontal face validator loaded")
            
        except Exception as e:
            print(f"⚠️  Frontal validator init failed: {e}")
            self.frontal_cascade = None
            self.eye_cascade = None
    
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
        
        print("✅ Improved face detection started")
    
    def stop(self):
        """Stop detection"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        
        self.stats_manager.save_statistics()
        self.face_db.save_database()
        
        print(f"💾 Database saved: {len(self.face_db.faces)} unique faces")
        print("⏸️  Detection stopped")
    
    def _frame_capture_loop(self):
        """Frame capture thread"""
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
        """Detection thread with aggressive frame skipping"""
        detection_interval = 1.0 / self.config.DETECTION_FPS
        
        while self.is_running:
            try:
                if self.frame_queue.empty():
                    time.sleep(0.01)
                    continue
                
                frame = self.frame_queue.get(timeout=0.1)
                self.frame_count += 1
                
                current_time = time.time()
                time_since_detection = current_time - self.last_detection_time
                should_detect = time_since_detection >= detection_interval
                
                if not should_detect:
                    if not self.result_queue.full():
                        self.result_queue.put((frame, None, current_time))
                    continue
                
                # DETECTION
                start_time = time.time()
                
                detection_frame = cv2.resize(
                    frame,
                    (self.config.DETECTION_WIDTH, self.config.DETECTION_HEIGHT),
                    interpolation=cv2.INTER_AREA
                )
                
                # Detect faces
                if self.use_openvino:
                    faces = self._detect_faces_openvino(detection_frame, frame)
                else:
                    faces = self._detect_faces_haar(detection_frame, frame)
                
                # Scale boxes
                scale_x = self.config.FRAME_WIDTH / self.config.DETECTION_WIDTH
                scale_y = self.config.FRAME_HEIGHT / self.config.DETECTION_HEIGHT
                
                validated_faces = []
                for face in faces:
                    box = face['box']
                    scaled_box = [
                        int(box[0] * scale_x),
                        int(box[1] * scale_y),
                        int(box[2] * scale_x),
                        int(box[3] * scale_y)
                    ]
                    
                    if self._is_frontal_face(frame, scaled_box):
                        face['box'] = scaled_box
                        
                        if self.use_embeddings:
                            embedding = self._extract_embedding(frame, scaled_box)
                            face['embedding'] = embedding
                        
                        validated_faces.append(face)
                
                process_time = time.time() - start_time
                self.processing_fps = 1.0 / process_time if process_time > 0 else 0
                
                self.last_detection_time = current_time
                
                if not self.result_queue.full():
                    self.result_queue.put((frame, validated_faces, current_time))
                
            except Empty:
                continue
            except Exception as e:
                print(f"❌ Detection error: {e}")
                time.sleep(0.1)
    
    def _detect_faces_openvino(self, detection_frame, original_frame):
        """Detect faces menggunakan OpenVINO"""
        try:
            h, w = detection_frame.shape[:2]
            
            # Prepare input
            input_frame = cv2.resize(detection_frame, (self.w, self.h))
            input_frame = input_frame.transpose((2, 0, 1))
            input_frame = np.expand_dims(input_frame, 0)
            
            # Run inference
            results = self.compiled_model([input_frame])
            detections = results[self.output_layer]
            
            faces = []
            
            for detection in detections[0][0]:
                confidence = float(detection[2])
                
                if confidence < self.confidence_threshold:
                    continue
                
                # Get box coordinates
                xmin = max(0, int(detection[3] * w))
                ymin = max(0, int(detection[4] * h))
                xmax = min(w, int(detection[5] * w))
                ymax = min(h, int(detection[6] * h))
                
                box_w = xmax - xmin
                box_h = ymax - ymin
                
                if box_w < 30 or box_h < 30:
                    continue
                
                quality_score = self._validate_face_quality(
                    detection_frame, xmin, ymin, box_w, box_h, confidence
                )
                
                if quality_score < 0.6:
                    continue
                
                faces.append({
                    'box': [xmin, ymin, box_w, box_h],
                    'confidence': confidence,
                    'quality': quality_score,
                    'embedding': None
                })
            
            return faces
            
        except Exception as e:
            print(f"OpenVINO detection error: {e}")
            return []
    
    def _detect_faces_haar(self, detection_frame, original_frame):
        """Detect faces menggunakan Haar Cascade"""
        try:
            gray = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            
            boxes = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40),
                maxSize=(300, 300)
            )
            
            faces = []
            for (x, y, w, h) in boxes:
                quality = self._validate_face_quality(
                    detection_frame, x, y, w, h, 0.85
                )
                
                if quality > 0.6:
                    faces.append({
                        'box': [int(x), int(y), int(w), int(h)],
                        'confidence': 0.85,
                        'quality': quality,
                        'embedding': None
                    })
            
            return faces
            
        except Exception as e:
            print(f"Haar detection error: {e}")
            return []
    
    def _is_frontal_face(self, frame, box):
        """Validate frontal face menggunakan multiple checks"""
        x, y, w, h = box
        
        if x < 0 or y < 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
            return False
        
        try:
            face_roi = frame[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                return False
            
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            frontal_score = 0
            max_score = 3
            
            # CHECK 1: Aspect ratio
            aspect_ratio = w / float(h)
            if 0.75 < aspect_ratio < 1.25:
                frontal_score += 1
            
            # CHECK 2: Eye detection
            if self.eye_cascade is not None:
                eyes = self.eye_cascade.detectMultiScale(
                    gray_face,
                    scaleFactor=1.1,
                    minNeighbors=3,
                    minSize=(int(w*0.1), int(h*0.1)),
                    maxSize=(int(w*0.4), int(h*0.4))
                )
                
                if len(eyes) >= 2:
                    frontal_score += 1
                    
                    if len(eyes) >= 2:
                        eye_centers = [(ex + ew//2, ey + eh//2) for (ex, ey, ew, eh) in eyes[:2]]
                        y_diff = abs(eye_centers[0][1] - eye_centers[1][1])
                        
                        if y_diff < h * 0.15:
                            frontal_score += 0.5
            
            # CHECK 3: Frontal cascade validation
            if self.frontal_cascade is not None:
                frontal_faces = self.frontal_cascade.detectMultiScale(
                    gray_face,
                    scaleFactor=1.05,
                    minNeighbors=3
                )
                
                if len(frontal_faces) > 0:
                    frontal_score += 0.5
            
            normalized_score = frontal_score / max_score
            
            return normalized_score >= self.frontal_threshold
            
        except Exception as e:
            print(f"Frontal validation error: {e}")
            return False
    
    def _validate_face_quality(self, frame, x, y, w, h, confidence):
        """Advanced face quality validation"""
        score = 0.0
        checks_passed = 0
        total_checks = 0
        
        # CHECK 1: Confidence
        total_checks += 1
        if confidence > 0.95:
            score += 0.3
            checks_passed += 1
        elif confidence > 0.90:
            score += 0.25
            checks_passed += 1
        elif confidence > 0.85:
            score += 0.2
            checks_passed += 1
        
        # CHECK 2: Aspect Ratio
        total_checks += 1
        aspect_ratio = w / float(h)
        if 0.7 < aspect_ratio < 1.3:
            score += 0.2
            checks_passed += 1
        
        # CHECK 3: Size validation
        total_checks += 1
        face_area = w * h
        frame_area = frame.shape[0] * frame.shape[1]
        relative_size = face_area / frame_area
        if 0.02 < relative_size < 0.5:
            score += 0.2
            checks_passed += 1
        
        # CHECK 4: Position validation
        total_checks += 1
        margin = 15
        if (x > margin and y > margin and 
            x+w < frame.shape[1]-margin and y+h < frame.shape[0]-margin):
            score += 0.15
            checks_passed += 1
        
        # CHECK 5: Brightness check
        total_checks += 1
        try:
            if x >= 0 and y >= 0 and x+w <= frame.shape[1] and y+h <= frame.shape[0]:
                face_roi = frame[y:y+h, x:x+w]
                gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray_roi)
                
                if 40 < brightness < 220:
                    score += 0.15
                    checks_passed += 1
        except:
            pass
        
        normalized_score = min(1.0, score)
        
        if checks_passed < total_checks * 0.7:
            return 0.0
        
        return normalized_score
    
    def _extract_embedding(self, frame, box):
        """Extract face embedding menggunakan FaceNet"""
        if not self.use_embeddings:
            return None
        
        try:
            x, y, w, h = box
            
            if x < 0 or y < 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
                return None
            
            margin = int(min(w, h) * 0.2)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)
            
            face_img = frame[y1:y2, x1:x2]
            
            if face_img.size == 0:
                return None
            
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            face_pil = face_pil.resize((160, 160), Image.LANCZOS)
            
            face_array = np.array(face_pil).astype(np.float32)
            face_array = (face_array - 127.5) / 128.0
            face_tensor = torch.from_numpy(face_array).permute(2, 0, 1).to(self.device)
            
            with torch.no_grad():
                embedding = self.resnet(face_tensor.unsqueeze(0))
            
            return embedding.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"Embedding extraction error: {e}")
            return None
    
    def _render_loop(self):
        """Rendering thread dengan enhanced tracking"""
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
                
                tracked_faces = self._track_faces(faces, current_time)
                self.current_faces = tracked_faces
                
                for box, face_id, quality, confidence, color_indicator in tracked_faces:
                    self._draw_detection(frame, box, face_id, quality, confidence, color_indicator)
                    self._draw_trail(frame, face_id)
                
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
        """Advanced face tracking dengan persistent database"""
        MAX_DISTANCE = 100
        EMBEDDING_THRESHOLD = self.embedding_threshold
        
        # Cleanup old trackers
        ids_to_remove = [
            fid for fid, tracker_data in self.face_trackers.items()
            if current_time - tracker_data[2] > self.config.ID_TIMEOUT
        ]
        
        for fid in ids_to_remove:
            del self.face_trackers[fid]
            self.face_quality_history.pop(fid, None)
            self.face_embeddings.pop(fid, None)
        
        tracked_faces = []
        used_ids = set()
        
        for face in faces:
            box = face['box']
            x, y, w, h = box
            confidence = face['confidence']
            quality = face['quality']
            embedding = face.get('embedding')
            
            cx = x + w // 2
            cy = y + h // 2
            
            # Check database untuk known faces
            is_known_face = False
            db_match_id = None
            db_similarity = 0
            
            if embedding is not None:
                db_match_id, db_similarity = self.face_db.find_matching_face(embedding)
                
                if db_match_id is not None:
                    is_known_face = True
                    print(f"🔍 Recognized known face: {db_match_id} (similarity: {db_similarity:.2f})")
            
            # Find best matching active tracker
            best_id = None
            best_score = float('inf')
            
            for fid, tracker_data in self.face_trackers.items():
                if fid in used_ids:
                    continue
                
                track_x, track_y, _, prev_quality, prev_embedding = tracker_data
                
                position_distance = np.sqrt((cx - track_x)**2 + (cy - track_y)**2)
                
                embedding_similarity = 0.0
                if embedding is not None and prev_embedding is not None:
                    try:
                        emb1 = embedding / np.linalg.norm(embedding)
                        emb2 = prev_embedding / np.linalg.norm(prev_embedding)
                        embedding_similarity = np.dot(emb1, emb2)
                    except:
                        embedding_similarity = 0.0
                
                if self.use_embeddings and embedding is not None and prev_embedding is not None:
                    if embedding_similarity > EMBEDDING_THRESHOLD:
                        combined_score = position_distance * (1 - embedding_similarity)
                    else:
                        combined_score = float('inf')
                else:
                    combined_score = position_distance
                
                if combined_score < best_score:
                    best_score = combined_score
                    best_id = fid
            
            # Assign ID
            if best_id is not None and best_score < MAX_DISTANCE:
                face_id = best_id
                color_indicator = 'tracking'
            else:
                face_id = self.next_id
                self.next_id += 1
                color_indicator = 'new'
            
            # Update quality history
            self.face_quality_history[face_id].append(quality)
            avg_quality = np.mean(list(self.face_quality_history[face_id]))
            
            if embedding is not None:
                self.face_embeddings[face_id] = embedding
            
            # Track only high quality faces
            if avg_quality > 0.6 and confidence > self.confidence_threshold:
                
                # Database check
                if embedding is not None:
                    is_new_face, matched_db_id, similarity = self.face_db.add_or_update_face(
                        str(face_id),
                        embedding
                    )
                    
                    if is_new_face:
                        if face_id not in self.detected_ids:
                            self.detected_ids.add(face_id)
                            self.stats_manager.add_unique_person()
                            color_indicator = 'new'
                            print(f"✨ NEW UNIQUE FACE! ID: {face_id} | Confidence: {confidence:.2f} | Quality: {avg_quality:.2f} | Total: {self.stats_manager.total_detected}")
                    else:
                        color_indicator = 'known'
                        if face_id not in self.detected_ids:
                            self.detected_ids.add(face_id)
                        print(f"👤 Known face detected: {matched_db_id} (similarity: {similarity:.2f})")
                else:
                    if face_id not in self.detected_ids:
                        self.detected_ids.add(face_id)
                        self.stats_manager.add_unique_person()
                        color_indicator = 'new'
                
                self.face_trackers[face_id] = (
                    cx, cy, current_time, quality,
                    self.face_embeddings.get(face_id)
                )
                used_ids.add(face_id)
                
                tracked_faces.append((box, face_id, quality, confidence, color_indicator))
                
                self.track_history[face_id].append((float(cx), float(cy)))
        
        return tracked_faces
    
    def _draw_detection(self, frame, box, face_id, quality, confidence, color_indicator):
        """Draw detection box dengan color indicator"""
        x, y, w, h = box
        
        # Color based on status
        if color_indicator == 'new':
            color = (0, 255, 0)
            label_text = "NEW FACE"
        elif color_indicator == 'known':
            color = (255, 165, 0)
            label_text = "KNOWN"
        else:
            color = (200, 200, 200)
            label_text = "TRACK"
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
        label = f"Face #{face_id} | {label_text} ({confidence:.0%})"
        
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_y = max(label_h + 10, y)
        cv2.rectangle(frame, (x, label_y-label_h-10), (x+label_w+10, label_y), color, -1)
        cv2.putText(frame, label, (x+5, label_y-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        cx = x + w // 2
        cy = y + h // 2
        cv2.circle(frame, (cx, cy), 5, color, -1)
        
        # Quality bar
        bar_width = w
        bar_height = 6
        bar_x = x
        bar_y = y + h + 5
        
        if bar_y + bar_height < frame.shape[0]:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                        (50, 50, 50), -1)
            fill_width = int(bar_width * ((quality + confidence) / 2))
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height),
                        color, -1)
    
    def _draw_trail(self, frame, face_id):
        """Draw tracking trail"""
        points = self.track_history[face_id]
        if len(points) > 1:
            color = (0, 255, 0)
            
            for i in range(1, len(points)):
                thickness = max(1, int(3 * (i / len(points))))
                cv2.line(frame,
                        (int(points[i-1][0]), int(points[i-1][1])),
                        (int(points[i][0]), int(points[i][1])),
                        color, thickness)
    
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
    
    def get_frame_jpeg(self):
        """Get JPEG encoded frame"""
        frame = self.get_frame()
        jpeg_quality = getattr(self.config, 'JPEG_QUALITY', 80)
        _, buffer = cv2.imencode('.jpg', frame, [
            cv2.IMWRITE_JPEG_QUALITY, jpeg_quality,
            cv2.IMWRITE_JPEG_OPTIMIZE, 1
        ])
        return buffer.tobytes()
    
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
            'embedding_tracking': self.use_embeddings,
            'database_size': len(self.face_db.faces)
        })
        return stats
    
    def get_historical_data(self):
        return self.stats_manager.get_historical_data()
    
    def get_database_stats(self):
        return self.face_db.get_statistics()
    
    def save_face_database(self):
        self.face_db.save_database()
        print("💾 Face database saved")
    
    def reset_face_database(self):
        self.face_db.reset_database()
        self.detected_ids.clear()
        print("🔄 Face database reset")
    
    def reset_daily_stats(self):
        """Reset daily stats (preserve face database)"""
        self.stats_manager.reset_daily()
        self.face_trackers.clear()
        self.current_faces = []
        self.face_quality_history.clear()
        self.face_embeddings.clear()
        self.next_id = 0
        
        self.face_db.save_database()
        
        print("🔄 Daily stats reset (face database preserved)")

# Alias
FaceCounter = OpenVINOFaceCounter