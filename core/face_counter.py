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
    - Multi-scale detection (jauh & dekat)
    - Occlusion handling (kacamata, topi, jilbab)
    - FaceNet untuk embeddings
    - Persistent face database
    """
    
    def __init__(self, cctv_urls, user, password, config):
        print("🔄 Initializing Enhanced OpenVINO Face Counter...")
        
        self.config = config
        
        # Initialize components
        self.video_handler = VideoStreamHandler(cctv_urls, user, password)
        self.stats_manager = StatisticsManager()
        self.face_db = FaceDatabase()
        
        print(f"📊 Face Database: {len(self.face_db.faces)} known faces")
        
        # Setup device (GPU if available)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if FACENET_AVAILABLE else None
        if self.device:
            print(f"🖥️  Device: {self.device}")
        
        # Initialize FaceNet for embeddings
        self.use_embeddings = False
        if FACENET_AVAILABLE:
            try:
                self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
                self.use_embeddings = True
                print("✅ FaceNet loaded for embedding-based tracking")
            except Exception as e:
                print(f"⚠️  FaceNet init failed: {e}")
                self.use_embeddings = False
        
        # Initialize OpenVINO detector
        self._init_detector()
        
        # Initialize multi-detector untuk berbagai kondisi
        self._init_multi_detectors()
        
        # Queues
        self.frame_queue = Queue(maxsize=config.FRAME_QUEUE_SIZE)
        self.result_queue = Queue(maxsize=config.RESULT_QUEUE_SIZE)
        
        # Enhanced tracking
        self.track_history = defaultdict(lambda: deque(maxlen=config.TRACK_HISTORY_LENGTH))
        self.detected_ids = set()
        self.current_faces = []
        self.next_id = 0
        self.face_trackers = {}  # {id: (cx, cy, timestamp, quality, embedding)}
        
        # Quality tracking
        self.face_quality_history = defaultdict(lambda: deque(maxlen=config.MAX_QUALITY_HISTORY))
        self.face_embeddings = {}
        
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
        
        # ENHANCED THRESHOLDS - Lebih permisif untuk deteksi dengan occlusion
        self.confidence_threshold = 0.70  # Turunkan dari 0.90 untuk deteksi jauh
        self.embedding_threshold = 0.50   # Turunkan dari 0.6 untuk similarity yang lebih loose
        self.frontal_threshold = 0.50     # Turunkan dari 0.7 untuk allow partial faces
        self.min_face_size = 25           # Turunkan dari 30 untuk detect wajah lebih kecil
        self.max_face_size = 400          # Naikkan untuk wajah dekat
        
        # Load saved data
        self.stats_manager.load_statistics()
        config.init_directories()
        
        print("✅ Enhanced OpenVINO system initialized")
        print(f"   ✓ Detector: {self.detector_type}")
        print(f"   ✓ Embeddings: {'Enabled' if self.use_embeddings else 'Disabled'}")
        print(f"   ✓ Multi-Scale Detection: Enabled")
        print(f"   ✓ Occlusion Handling: Enabled (glasses, hats, hijab)")
        print(f"   ✓ Min Face Size: {self.min_face_size}px")
        print(f"   ✓ Confidence Threshold: {self.confidence_threshold}")
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
            # Use Haar Cascade for CPU detection
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
    
    def _init_multi_detectors(self):
        """
        Initialize multiple detectors untuk berbagai kondisi:
        - Frontal face (standar)
        - Profile face (samping)
        - Upper body (untuk deteksi dengan jilbab/topi)
        """
        try:
            # Frontal face detector (primary)
            frontal_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
            self.frontal_cascade = cv2.CascadeClassifier(frontal_path)
            
            # Profile face detector (untuk wajah samping)
            profile_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'
            self.profile_cascade = cv2.CascadeClassifier(profile_path)
            
            # Eye detector (untuk validasi, tapi optional)
            eye_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            self.eye_cascade = cv2.CascadeClassifier(eye_path)
            
            # Upper body detector (backup untuk orang dengan hijab/topi)
            upperbody_path = cv2.data.haarcascades + 'haarcascade_upperbody.xml'
            self.upperbody_cascade = cv2.CascadeClassifier(upperbody_path)
            
            print("✅ Multi-detector system initialized:")
            print(f"   ✓ Frontal face detector")
            print(f"   ✓ Profile face detector")
            print(f"   ✓ Eye detector")
            print(f"   ✓ Upper body detector")
            
        except Exception as e:
            print(f"⚠️  Multi-detector init warning: {e}")
            self.frontal_cascade = None
            self.profile_cascade = None
            self.eye_cascade = None
            self.upperbody_cascade = None
    
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
        
        print("✅ Enhanced face detection started")
    
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
        """Detection thread with multi-scale detection"""
        detection_interval = 1.0 / self.config.DETECTION_FPS
        
        while self.is_running:
            try:
                if self.frame_queue.empty():
                    time.sleep(0.01)
                    continue
                
                frame = self.frame_queue.get(timeout=0.1)
                self.frame_count += 1
                
                current_time = time.time()
                
                # Frame skipping
                time_since_detection = current_time - self.last_detection_time
                should_detect = time_since_detection >= detection_interval
                
                if not should_detect:
                    if not self.result_queue.full():
                        self.result_queue.put((frame, None, current_time))
                    continue
                
                # DETECTION
                start_time = time.time()
                
                # Resize untuk detection
                detection_frame = cv2.resize(
                    frame,
                    (self.config.DETECTION_WIDTH, self.config.DETECTION_HEIGHT),
                    interpolation=cv2.INTER_AREA
                )
                
                # ENHANCED: Multi-scale detection
                if self.use_openvino:
                    faces = self._detect_faces_openvino_multiscale(detection_frame, frame)
                else:
                    faces = self._detect_faces_haar_multiscale(detection_frame, frame)
                
                # Scale boxes kembali ke resolusi asli
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
                    
                    # ENHANCED: Relaxed validation untuk occlusion
                    if self._is_valid_face_region(frame, scaled_box):
                        face['box'] = scaled_box
                        
                        # Extract embedding jika tersedia
                        if self.use_embeddings:
                            embedding = self._extract_embedding(frame, scaled_box)
                            face['embedding'] = embedding
                        
                        validated_faces.append(face)
                
                # Calculate processing FPS
                process_time = time.time() - start_time
                self.processing_fps = 1.0 / process_time if process_time > 0 else 0
                
                self.last_detection_time = current_time
                
                # Put result
                if not self.result_queue.full():
                    self.result_queue.put((frame, validated_faces, current_time))
                
            except Empty:
                continue
            except Exception as e:
                print(f"❌ Detection error: {e}")
                time.sleep(0.1)
    
    def _detect_faces_openvino_multiscale(self, detection_frame, original_frame):
        """
        ENHANCED: Multi-scale detection menggunakan OpenVINO
        Deteksi pada multiple scale untuk handle wajah jauh dan dekat
        """
        try:
            all_faces = []
            
            # Scale factors untuk multi-scale detection
            # Scale 1.0 = original, 0.5 = zoom in (untuk wajah kecil/jauh)
            scale_factors = [1.0, 0.7]  # Hanya 2 scale untuk efisiensi
            
            for scale in scale_factors:
                if scale != 1.0:
                    # Resize frame untuk scale berbeda
                    scaled_h = int(detection_frame.shape[0] * scale)
                    scaled_w = int(detection_frame.shape[1] * scale)
                    scaled_frame = cv2.resize(detection_frame, (scaled_w, scaled_h))
                else:
                    scaled_frame = detection_frame
                
                # Prepare input
                input_frame = cv2.resize(scaled_frame, (self.w, self.h))
                input_frame = input_frame.transpose((2, 0, 1))  # HWC -> CHW
                input_frame = np.expand_dims(input_frame, 0)
                
                # Run inference
                results = self.compiled_model([input_frame])
                detections = results[self.output_layer]
                
                h, w = scaled_frame.shape[:2]
                
                # Process detections
                for detection in detections[0][0]:
                    confidence = float(detection[2])
                    
                    # ENHANCED: Lower threshold
                    if confidence < self.confidence_threshold:
                        continue
                    
                    # Get box coordinates
                    xmin = int(detection[3] * w)
                    ymin = int(detection[4] * h)
                    xmax = int(detection[5] * w)
                    ymax = int(detection[6] * h)
                    
                    # Scale back ke ukuran original
                    if scale != 1.0:
                        xmin = int(xmin / scale)
                        ymin = int(ymin / scale)
                        xmax = int(xmax / scale)
                        ymax = int(ymax / scale)
                    
                    box_w = xmax - xmin
                    box_h = ymax - ymin
                    
                    # ENHANCED: Relaxed size validation
                    if box_w < self.min_face_size or box_h < self.min_face_size:
                        continue
                    if box_w > self.max_face_size or box_h > self.max_face_size:
                        continue
                    
                    # ENHANCED: Relaxed quality validation
                    quality_score = self._validate_face_quality_enhanced(
                        detection_frame, xmin, ymin, box_w, box_h, confidence
                    )
                    
                    if quality_score < 0.40:  # Turunkan dari 0.6
                        continue
                    
                    all_faces.append({
                        'box': [xmin, ymin, box_w, box_h],
                        'confidence': confidence,
                        'quality': quality_score,
                        'embedding': None,
                        'scale': scale
                    })
            
            # NMS untuk remove duplicates dari multi-scale
            faces = self._non_max_suppression(all_faces)
            
            return faces
            
        except Exception as e:
            print(f"OpenVINO multi-scale detection error: {e}")
            return []
    
    def _detect_faces_haar_multiscale(self, detection_frame, original_frame):
        """
        ENHANCED: Multi-detector Haar Cascade
        - Detect frontal faces
        - Detect profile faces
        - Detect dengan berbagai parameter untuk handle occlusion
        """
        try:
            all_faces = []
            gray = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2GRAY)
            
            # CLAHE untuk enhance contrast (bantu deteksi di kondisi cahaya buruk)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            # DETECTOR 1: Frontal face (primary)
            frontal_faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,  # Lebih fine-grained
                minNeighbors=3,     # Turunkan dari 5 untuk lebih sensitif
                minSize=(self.min_face_size, self.min_face_size),
                maxSize=(self.max_face_size, self.max_face_size),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            for (x, y, w, h) in frontal_faces:
                quality = self._validate_face_quality_enhanced(
                    detection_frame, x, y, w, h, 0.75
                )
                
                if quality > 0.35:  # Turunkan threshold
                    all_faces.append({
                        'box': [int(x), int(y), int(w), int(h)],
                        'confidence': 0.75,
                        'quality': quality,
                        'embedding': None,
                        'type': 'frontal'
                    })
            
            # DETECTOR 2: Profile face (samping)
            if self.profile_cascade is not None:
                # Left profile
                profile_faces = self.profile_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=3,
                    minSize=(self.min_face_size, self.min_face_size),
                    maxSize=(self.max_face_size, self.max_face_size)
                )
                
                for (x, y, w, h) in profile_faces:
                    quality = self._validate_face_quality_enhanced(
                        detection_frame, x, y, w, h, 0.70
                    )
                    
                    if quality > 0.35:
                        all_faces.append({
                            'box': [int(x), int(y), int(w), int(h)],
                            'confidence': 0.70,
                            'quality': quality,
                            'embedding': None,
                            'type': 'profile'
                        })
                
                # Right profile (flip image)
                flipped = cv2.flip(gray, 1)
                profile_faces_flipped = self.profile_cascade.detectMultiScale(
                    flipped,
                    scaleFactor=1.1,
                    minNeighbors=3,
                    minSize=(self.min_face_size, self.min_face_size),
                    maxSize=(self.max_face_size, self.max_face_size)
                )
                
                for (x, y, w, h) in profile_faces_flipped:
                    # Flip x coordinate back
                    x_flipped = gray.shape[1] - x - w
                    quality = self._validate_face_quality_enhanced(
                        detection_frame, x_flipped, y, w, h, 0.70
                    )
                    
                    if quality > 0.35:
                        all_faces.append({
                            'box': [int(x_flipped), int(y), int(w), int(h)],
                            'confidence': 0.70,
                            'quality': quality,
                            'embedding': None,
                            'type': 'profile_right'
                        })
            
            # NMS untuk remove duplicates
            faces = self._non_max_suppression(all_faces)
            
            return faces
            
        except Exception as e:
            print(f"Haar multi-scale detection error: {e}")
            return []
    
    def _non_max_suppression(self, faces, overlap_thresh=0.5):
        """
        Non-maximum suppression untuk remove duplicate detections
        """
        if len(faces) == 0:
            return []
        
        # Convert to numpy array
        boxes = np.array([[f['box'][0], f['box'][1], 
                          f['box'][0] + f['box'][2], 
                          f['box'][1] + f['box'][3]] for f in faces])
        
        scores = np.array([f['confidence'] * f['quality'] for f in faces])
        
        # Sort by score
        idxs = np.argsort(scores)[::-1]
        
        keep = []
        while len(idxs) > 0:
            i = idxs[0]
            keep.append(i)
            
            if len(idxs) == 1:
                break
            
            # Calculate IoU
            xx1 = np.maximum(boxes[i, 0], boxes[idxs[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[idxs[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[idxs[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[idxs[1:], 3])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            intersection = w * h
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_others = (boxes[idxs[1:], 2] - boxes[idxs[1:], 0]) * (boxes[idxs[1:], 3] - boxes[idxs[1:], 1])
            union = area_i + area_others - intersection
            
            iou = intersection / union
            
            # Keep only non-overlapping boxes
            idxs = idxs[np.concatenate(([0], np.where(iou <= overlap_thresh)[0] + 1))[1:]]
        
        return [faces[i] for i in keep]
    
    def _is_valid_face_region(self, frame, box):
        """
        ENHANCED: Relaxed validation untuk allow partial/occluded faces
        - Tidak memerlukan kedua mata terdeteksi
        - Allow wajah dengan topi/jilbab
        - Allow wajah samping
        """
        x, y, w, h = box
        
        # Validate bounds
        if x < 0 or y < 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
            return False
        
        if w < self.min_face_size or h < self.min_face_size:
            return False
        
        try:
            # Extract face region
            face_roi = frame[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                return False
            
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            validation_score = 0
            max_score = 3
            
            # CHECK 1: Aspect ratio (lebih permisif)
            aspect_ratio = w / float(h)
            if 0.5 < aspect_ratio < 1.8:  # Lebih wide range untuk profile
                validation_score += 1
            
            # CHECK 2: Edge detection (ada struktur wajah)
            edges = cv2.Canny(gray_face, 50, 150)
            edge_density = np.sum(edges > 0) / (w * h)
            if edge_density > 0.05:  # Ada edges yang cukup
                validation_score += 1
            
            # CHECK 3: Skin tone detection (optional, rough)
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_ratio = np.sum(skin_mask > 0) / (w * h)
            
            if skin_ratio > 0.15:  # At least some skin visible
                validation_score += 1
            
            # Require at least 2/3 checks to pass (lebih permisif)
            return validation_score >= 2
            
        except Exception as e:
            print(f"Validation error: {e}")
            return False
    
    def _validate_face_quality_enhanced(self, frame, x, y, w, h, confidence):
        """
        ENHANCED: Relaxed quality validation
        - Lower thresholds untuk allow occlusion
        - More permissive scoring
        """
        score = 0.0
        checks_passed = 0
        total_checks = 0
        
        # CHECK 1: Confidence (relaxed)
        total_checks += 1
        if confidence > 0.80:
            score += 0.25
            checks_passed += 1
        elif confidence > 0.70:
            score += 0.20
            checks_passed += 1
        elif confidence > 0.60:
            score += 0.15
            checks_passed += 1
        
        # CHECK 2: Aspect Ratio (more permissive)
        total_checks += 1
        aspect_ratio = w / float(h)
        if 0.5 < aspect_ratio < 1.8:  # Wider range
            score += 0.20
            checks_passed += 1
        
        # CHECK 3: Size validation (relaxed)
        total_checks += 1
        face_area = w * h
        frame_area = frame.shape[0] * frame.shape[1]
        relative_size = face_area / frame_area
        if 0.01 < relative_size < 0.6:  # 1% to 60% of frame (lebih wide)
            score += 0.20
            checks_passed += 1
        
        # CHECK 4: Position validation (less strict)
        total_checks += 1
        margin = 5  # Smaller margin
        if (x > margin and y > margin and 
            x+w < frame.shape[1]-margin and y+h < frame.shape[0]-margin):
            score += 0.15
            checks_passed += 1
        
        # CHECK 5: Brightness check (more permissive)
        total_checks += 1
        try:
            if x >= 0 and y >= 0 and x+w <= frame.shape[1] and y+h <= frame.shape[0]:
                face_roi = frame[y:y+h, x:x+w]
                gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray_roi)
                
                # Wider brightness range
                if 30 < brightness < 230:
                    score += 0.15
                    checks_passed += 1
        except:
            pass
        
        # CHECK 6: Sharpness (optional, low weight)
        total_checks += 1
        try:
            if x >= 0 and y >= 0 and x+w <= frame.shape[1] and y+h <= frame.shape[0]:
                face_roi = frame[y:y+h, x:x+w]
                gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                laplacian_var = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
                
                if laplacian_var > 50:  # Not too blurry
                    score += 0.10
                    checks_passed += 1
        except:
            pass
        
        # Normalize score
        normalized_score = min(1.0, score)
        
        # ENHANCED: Require only 50% of checks to pass (was 70%)
        if checks_passed < total_checks * 0.5:
            return 0.0
        
        return normalized_score
    
    def _extract_embedding(self, frame, box):
        """Extract face embedding menggunakan FaceNet"""
        if not self.use_embeddings:
            return None
        
        try:
            x, y, w, h = box
            
            # Validate bounds
            if x < 0 or y < 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
                return None
            
            # Crop face dengan margin
            margin = int(min(w, h) * 0.2)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)
            
            face_img = frame[y1:y2, x1:x2]
            
            if face_img.size == 0:
                return None
            
            # Convert to RGB and PIL
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            
            # Resize to 160x160 (FaceNet input size)
            face_pil = face_pil.resize((160, 160), Image.LANCZOS)
            
            # Convert to tensor
            face_array = np.array(face_pil).astype(np.float32)
            face_array = (face_array - 127.5) / 128.0
            face_tensor = torch.from_numpy(face_array).permute(2, 0, 1).to(self.device)
            
            # Extract embedding
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
                
                # Track faces dengan database integration
                tracked_faces = self._track_faces(faces, current_time)
                self.current_faces = tracked_faces
                
                # Draw detections dengan color indicator
                for box, face_id, quality, confidence, color_indicator in tracked_faces:
                    self._draw_detection(frame, box, face_id, quality, confidence, color_indicator)
                    self._draw_trail(frame, face_id)
                
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
        """
        Advanced face tracking dengan persistent database
        ENHANCED: More permissive matching untuk handle occlusion
        """
        MAX_DISTANCE = 150  # Increased from 100
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
            
            # === STEP 1: Check database untuk known faces ===
            is_known_face = False
            db_match_id = None
            db_similarity = 0
            
            if embedding is not None:
                db_match_id, db_similarity = self.face_db.find_matching_face(embedding)
                
                if db_match_id is not None:
                    is_known_face = True
                    print(f"🔍 Recognized known face: {db_match_id} (similarity: {db_similarity:.2f})")
            
            # === STEP 2: Find best matching active tracker ===
            best_id = None
            best_score = float('inf')
            
            for fid, tracker_data in self.face_trackers.items():
                if fid in used_ids:
                    continue
                
                track_x, track_y, _, prev_quality, prev_embedding = tracker_data
                
                # Position distance
                position_distance = np.sqrt((cx - track_x)**2 + (cy - track_y)**2)
                
                # Embedding similarity
                embedding_similarity = 0.0
                if embedding is not None and prev_embedding is not None:
                    try:
                        # Normalize embeddings
                        emb1 = embedding / np.linalg.norm(embedding)
                        emb2 = prev_embedding / np.linalg.norm(prev_embedding)
                        embedding_similarity = np.dot(emb1, emb2)
                    except:
                        embedding_similarity = 0.0
                
                # Combined score (more weight on position for robustness)
                if self.use_embeddings and embedding is not None and prev_embedding is not None:
                    if embedding_similarity > EMBEDDING_THRESHOLD:
                        combined_score = position_distance * (1 - embedding_similarity * 0.5)
                    else:
                        combined_score = float('inf')
                else:
                    combined_score = position_distance
                
                if combined_score < best_score:
                    best_score = combined_score
                    best_id = fid
            
            # === STEP 3: Assign ID ===
            if best_id is not None and best_score < MAX_DISTANCE:
                face_id = best_id
                color_indicator = 'tracking'
            else:
                face_id = self.next_id
                self.next_id += 1
                color_indicator = 'new'
            
            # === STEP 4: Update quality history ===
            self.face_quality_history[face_id].append(quality)
            avg_quality = np.mean(list(self.face_quality_history[face_id]))
            
            # Store embedding
            if embedding is not None:
                self.face_embeddings[face_id] = embedding
            
            # === STEP 5: Track with relaxed threshold ===
            if avg_quality > 0.40 and confidence > self.confidence_threshold:  # Lowered from 0.6
                
                # === STEP 6: Database check - add only if truly new ===
                if embedding is not None:
                    is_new_face, matched_db_id, similarity = self.face_db.add_or_update_face(
                        str(face_id),
                        embedding
                    )
                    
                    if is_new_face:
                        # NEW FACE - add to unique counter
                        if face_id not in self.detected_ids:
                            self.detected_ids.add(face_id)
                            self.stats_manager.add_unique_person()
                            color_indicator = 'new'
                            print(f"✨ NEW UNIQUE FACE! ID: {face_id} | Confidence: {confidence:.2f} | Quality: {avg_quality:.2f} | Total: {self.stats_manager.total_detected}")
                    else:
                        # KNOWN FACE - don't add to counter
                        color_indicator = 'known'
                        if face_id not in self.detected_ids:
                            self.detected_ids.add(face_id)
                        print(f"👤 Known face detected: {matched_db_id} (similarity: {similarity:.2f})")
                else:
                    # No embedding - fallback tracking
                    if face_id not in self.detected_ids:
                        self.detected_ids.add(face_id)
                        self.stats_manager.add_unique_person()
                        color_indicator = 'new'
                
                # Update tracker
                self.face_trackers[face_id] = (
                    cx, cy, current_time, quality,
                    self.face_embeddings.get(face_id)
                )
                used_ids.add(face_id)
                
                tracked_faces.append((box, face_id, quality, confidence, color_indicator))
                
                # Update track history
                self.track_history[face_id].append((float(cx), float(cy)))
        
        return tracked_faces
    
    def _draw_detection(self, frame, box, face_id, quality, confidence, color_indicator):
        """Draw detection box dengan color indicator"""
        x, y, w, h = box
        
        # Color based on status
        if color_indicator == 'new':
            color = (0, 255, 0)  # GREEN - New unique face
            label_text = "NEW FACE"
        elif color_indicator == 'known':
            color = (255, 165, 0)  # ORANGE - Known face
            label_text = "KNOWN"
        else:
            color = (200, 200, 200)  # GRAY - Tracking
            label_text = "TRACK"
        
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
        # Label
        label = f"Face #{face_id} | {label_text} ({confidence:.0%})"
        
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_y = max(label_h + 10, y)
        cv2.rectangle(frame, (x, label_y-label_h-10), (x+label_w+10, label_y), color, -1)
        cv2.putText(frame, label, (x+5, label_y-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Center point
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
            'database_size': len(self.face_db.faces),
            'enhanced_features': {
                'multi_scale': True,
                'occlusion_handling': True,
                'profile_detection': True,
                'min_face_size': self.min_face_size,
                'confidence_threshold': self.confidence_threshold
            }
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
        
        # Save database
        self.face_db.save_database()
        
        print("🔄 Daily stats reset (face database preserved)")

# Alias
FaceCounter = OpenVINOFaceCounter