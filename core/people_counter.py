import cv2
import numpy as np
from collections import defaultdict, deque
from datetime import datetime
import threading
import time
import pickle
import os
from queue import Queue
import logging

# Deep Learning Libraries
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# Suppress warnings
logging.getLogger('libav').setLevel(logging.ERROR)

from core.config import Config
from utils.video_utils import VideoStreamHandler
from utils.stats_manager import StatisticsManager

class MTCNNFaceCounter:
    def __init__(self, cctv_urls, user, password):
        print("🔄 Initializing MTCNN Face Counter...")
        
        # Initialize components
        self.video_handler = VideoStreamHandler(cctv_urls, user, password)
        self.stats_manager = StatisticsManager()
        
        # Setup device (GPU if available)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Device: {self.device}")
        
        # Initialize MTCNN (Multi-task Cascaded Convolutional Networks)
        try:
            self.mtcnn = MTCNN(
                image_size=160,           # Face image size for recognition
                margin=40,                # Margin around detected face
                min_face_size=40,         # Minimum face size to detect
                thresholds=[0.6, 0.7, 0.7],  # Detection thresholds for 3 stages
                factor=0.709,             # Scale factor between pyramid layers
                post_process=True,        # Apply post-processing
                keep_all=True,            # Keep all detected faces
                device=self.device
            )
            
            # Initialize FaceNet for embeddings (optional, for better tracking)
            self.use_embeddings = True
            try:
                self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
                print("✅ MTCNN + FaceNet loaded (embedding-based tracking)")
            except:
                self.use_embeddings = False
                print("✅ MTCNN loaded (position-based tracking)")
            
        except Exception as e:
            print(f"❌ Error loading MTCNN: {e}")
            raise
        
        # Haar Cascade as fallback (optional)
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            self.use_fallback = not self.face_cascade.empty()
            if self.use_fallback:
                print("✅ Haar Cascade loaded as fallback")
        except:
            self.use_fallback = False
        
        # Multi-threading for real-time processing
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)
        
        # Enhanced tracking with quality scores
        self.track_history = defaultdict(lambda: deque(maxlen=Config.TRACK_HISTORY_LENGTH))
        self.detected_ids = set()
        self.current_faces = []
        self.next_id = 0
        self.face_trackers = {}  # {id: (center_x, center_y, timestamp, quality_score, embedding)}
        self.id_timeout = 1.5  # 1.5 second timeout
        
        # Quality tracking per ID
        self.face_quality_history = defaultdict(lambda: deque(maxlen=10))
        self.face_embeddings = {}  # Store embeddings for better tracking
        
        # State
        self.frame = None
        self.is_running = False
        self.cap = None
        
        # Enhanced FPS calculation
        self.fps = 0
        self.processing_fps = 0
        self.frame_times = deque(maxlen=20)
        self.last_frame_time = time.time()
        
        # Frame skip for optimization
        self.frame_skip = Config.FRAME_SKIP if hasattr(Config, 'FRAME_SKIP') else 0
        self.frame_count = 0
        
        # Detection confidence threshold
        self.confidence_threshold = 0.90  # MTCNN confidence threshold
        
        # Load saved data
        self.stats_manager.load_statistics()
        Config.init_directories()
        
        print("✅ MTCNN validation system initialized")
        print("   - Multi-stage cascaded detection")
        print("   - Facial landmark detection")
        print("   - High-accuracy face alignment")
        if self.use_embeddings:
            print("   - FaceNet embeddings for tracking")
        print("   - Quality-based validation")
    
    def start(self):
        """Start detection with multi-threading"""
        if self.is_running:
            return
        
        self.cap = self.video_handler.connect()
        self.is_running = True
        
        # Start multiple threads for parallel processing
        threading.Thread(target=self._frame_capture_loop, daemon=True).start()
        threading.Thread(target=self._detection_loop, daemon=True).start()
        threading.Thread(target=self._render_loop, daemon=True).start()
        
        print("✅ MTCNN face detection started")
    
    def stop(self):
        """Stop detection"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.stats_manager.save_statistics()
        print("⏸️  Detection stopped")
    
    def _frame_capture_loop(self):
        """Dedicated thread for frame capture"""
        while self.is_running:
            try:
                if self.cap is None:
                    time.sleep(0.1)
                    continue
                
                ret = self.cap.grab()
                if not ret:
                    self._reconnect_stream()
                    continue
                
                if not self.frame_queue.full():
                    ret, frame = self.cap.retrieve()
                    if ret and frame is not None and frame.size > 0:
                        frame_width = getattr(Config, 'FRAME_WIDTH', 640)
                        frame_height = getattr(Config, 'FRAME_HEIGHT', 480)
                        frame = cv2.resize(frame, (frame_width, frame_height), 
                                         interpolation=cv2.INTER_LINEAR)
                        self.frame_queue.put(frame)
                
                time.sleep(0.001)
                
            except Exception as e:
                print(f"❌ Capture error: {e}")
                time.sleep(1)
    
    def _detection_loop(self):
        """Dedicated thread for MTCNN face detection"""
        while self.is_running:
            try:
                if self.frame_queue.empty():
                    time.sleep(0.001)
                    continue
                
                frame = self.frame_queue.get(timeout=0.1)
                self.frame_count += 1
                
                # Frame skipping
                if self.frame_count % (self.frame_skip + 1) != 0:
                    if not self.result_queue.full():
                        self.result_queue.put((frame, None, time.time()))
                    continue
                
                start_time = time.time()
                
                # Run MTCNN face detection
                faces = self._detect_faces_mtcnn(frame)
                
                # Calculate processing FPS
                process_time = time.time() - start_time
                self.processing_fps = 1.0 / process_time if process_time > 0 else 0
                
                # Put result in queue
                if not self.result_queue.full():
                    self.result_queue.put((frame, faces, time.time()))
                
            except Exception as e:
                print(f"❌ Detection error: {e}")
                time.sleep(0.1)
    
    def _detect_faces_mtcnn(self, frame):
        """
        MTCNN face detection with validation
        Returns: list of [x, y, w, h, confidence, landmarks, embedding]
        """
        try:
            # Convert BGR to RGB for MTCNN
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            # Detect faces with MTCNN
            # Returns: boxes, probs, landmarks
            boxes, probs, landmarks = self.mtcnn.detect(img_pil, landmarks=True)
            
            faces = []
            
            if boxes is not None and probs is not None:
                for i, (box, prob) in enumerate(zip(boxes, probs)):
                    # Filter by confidence threshold
                    if prob < self.confidence_threshold:
                        continue
                    
                    try:
                        # Convert box format: [x1, y1, x2, y2] -> [x, y, w, h]
                        x1, y1, x2, y2 = [int(coord) for coord in box]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2 = min(frame.shape[1], x2)
                        y2 = min(frame.shape[0], y2)
                        
                        w = x2 - x1
                        h = y2 - y1
                        
                        # Validate face size
                        if w < 40 or h < 40:
                            continue
                        
                        # Additional validation
                        quality_score = self._validate_face_region_mtcnn(
                            frame, x1, y1, w, h, prob, 
                            landmarks[i] if landmarks is not None else None
                        )
                        
                        if quality_score < 0.5:  # Minimum quality threshold
                            continue
                        
                        # Extract embedding if available
                        embedding = None
                        if self.use_embeddings:
                            try:
                                embedding = self._extract_embedding(img_pil, box)
                            except:
                                pass
                        
                        # Store face data
                        face_data = {
                            'box': [x1, y1, w, h],
                            'confidence': float(prob),
                            'quality': quality_score,
                            'landmarks': landmarks[i].tolist() if landmarks is not None else None,
                            'embedding': embedding
                        }
                        
                        faces.append(face_data)
                        
                    except Exception as e:
                        print(f"Error processing face {i}: {e}")
                        continue
            
            return faces
            
        except Exception as e:
            print(f"MTCNN detection error: {e}")
            
            # Fallback to Haar Cascade if MTCNN fails
            if self.use_fallback:
                return self._detect_faces_haar_fallback(frame)
            
            return []
    
    def _extract_embedding(self, img_pil, box):
        """Extract face embedding using FaceNet"""
        try:
            x1, y1, x2, y2 = [int(coord) for coord in box]
            
            # Crop and resize face
            face_img = img_pil.crop((x1, y1, x2, y2))
            face_img = face_img.resize((160, 160), Image.LANCZOS)
            
            # Convert to tensor
            face_array = np.array(face_img).astype(np.float32)
            face_array = (face_array - 127.5) / 128.0
            face_tensor = torch.from_numpy(face_array).permute(2, 0, 1).to(self.device)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.resnet(face_tensor.unsqueeze(0))
            
            return embedding.cpu().numpy().flatten()
            
        except Exception as e:
            return None
    
    def _validate_face_region_mtcnn(self, frame, x, y, w, h, confidence, landmarks):
        """
        Advanced validation for MTCNN detected face
        Returns quality score (0.0 - 1.0)
        """
        score = 0.0
        checks_passed = 0
        total_checks = 0
        
        # CHECK 1: MTCNN Confidence (most important)
        total_checks += 1
        if confidence > 0.95:
            score += 0.4
            checks_passed += 1
        elif confidence > 0.90:
            score += 0.3
            checks_passed += 1
        
        # CHECK 2: Aspect Ratio
        total_checks += 1
        aspect_ratio = w / float(h)
        if 0.7 < aspect_ratio < 1.3:  # Face proportions
            score += 0.15
            checks_passed += 1
        
        # CHECK 3: Size validation
        total_checks += 1
        face_area = w * h
        frame_area = frame.shape[0] * frame.shape[1]
        relative_size = face_area / frame_area
        if 0.01 < relative_size < 0.5:  # 1% to 50% of frame
            score += 0.15
            checks_passed += 1
        
        # CHECK 4: Landmark validation (if available)
        if landmarks is not None:
            total_checks += 1
            try:
                # Check if landmarks are within face bounding box
                landmarks_valid = all(
                    x <= lm[0] <= x+w and y <= lm[1] <= y+h 
                    for lm in landmarks
                )
                if landmarks_valid:
                    score += 0.2
                    checks_passed += 1
            except:
                pass
        
        # CHECK 5: Position validation (not at edge)
        total_checks += 1
        margin = 10
        if (x > margin and y > margin and 
            x+w < frame.shape[1]-margin and y+h < frame.shape[0]-margin):
            score += 0.1
            checks_passed += 1
        
        # Normalize score
        normalized_score = min(1.0, score)
        
        # Require at least 60% of checks to pass
        if checks_passed < total_checks * 0.6:
            return 0.0
        
        return normalized_score
    
    def _detect_faces_haar_fallback(self, frame):
        """Fallback Haar Cascade detection"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            
            boxes = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(50, 50),
                maxSize=(400, 400)
            )
            
            faces = []
            for (x, y, w, h) in boxes:
                face_data = {
                    'box': [int(x), int(y), int(w), int(h)],
                    'confidence': 0.85,  # Default confidence for Haar
                    'quality': 0.7,
                    'landmarks': None,
                    'embedding': None
                }
                faces.append(face_data)
            
            return faces
            
        except:
            return []
    
    def _track_faces(self, faces, current_time):
        """
        Advanced face tracking with embeddings and quality-based validation
        """
        MAX_DISTANCE = 100  # Maximum pixel distance for position-based tracking
        EMBEDDING_THRESHOLD = 0.6  # Cosine similarity threshold for embedding-based tracking
        
        # Clean up old trackers
        ids_to_remove = []
        for face_id, tracker_data in self.face_trackers.items():
            timestamp = tracker_data[2]
            if current_time - timestamp > self.id_timeout:
                ids_to_remove.append(face_id)
        
        for face_id in ids_to_remove:
            del self.face_trackers[face_id]
            if face_id in self.face_quality_history:
                del self.face_quality_history[face_id]
            if face_id in self.face_embeddings:
                del self.face_embeddings[face_id]
        
        # Match new faces with existing trackers
        tracked_faces = []
        used_ids = set()
        
        for face in faces:
            box = face['box']
            x, y, w, h = box
            confidence = face['confidence']
            quality = face['quality']
            embedding = face['embedding']
            
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Find best matching tracker
            best_id = None
            best_score = float('inf')
            
            for face_id, tracker_data in self.face_trackers.items():
                if face_id in used_ids:
                    continue
                
                track_x, track_y, _, prev_quality, prev_embedding = tracker_data
                
                # Position-based distance
                position_distance = np.sqrt((center_x - track_x)**2 + (center_y - track_y)**2)
                
                # Embedding-based similarity (if available)
                embedding_similarity = 0.0
                if embedding is not None and prev_embedding is not None:
                    try:
                        # Cosine similarity
                        embedding = embedding / np.linalg.norm(embedding)
                        prev_embedding = prev_embedding / np.linalg.norm(prev_embedding)
                        embedding_similarity = np.dot(embedding, prev_embedding)
                    except:
                        embedding_similarity = 0.0
                
                # Combined score
                if self.use_embeddings and embedding is not None and prev_embedding is not None:
                    # Prioritize embedding similarity
                    if embedding_similarity > EMBEDDING_THRESHOLD:
                        combined_score = position_distance * (1 - embedding_similarity)
                    else:
                        combined_score = float('inf')  # Different person
                else:
                    # Use only position
                    combined_score = position_distance
                
                # Quality consistency bonus
                quality_diff = abs(quality - prev_quality)
                combined_score *= (1 + quality_diff * 0.5)
                
                if combined_score < best_score:
                    best_score = combined_score
                    best_id = face_id
            
            # Assign ID
            if best_id is not None and best_score < MAX_DISTANCE:
                face_id = best_id
            else:
                face_id = self.next_id
                self.next_id += 1
            
            # Update quality history
            self.face_quality_history[face_id].append(quality)
            avg_quality = np.mean(list(self.face_quality_history[face_id]))
            
            # Store embedding
            if embedding is not None:
                self.face_embeddings[face_id] = embedding
            
            # Only track faces with consistently high quality
            if avg_quality > 0.5 and confidence > self.confidence_threshold:
                # Track as new unique face
                if face_id not in self.detected_ids:
                    self.detected_ids.add(face_id)
                    self.stats_manager.add_unique_person()
                    print(f"✨ New face detected! ID: {face_id} | Confidence: {confidence:.2f} | Quality: {avg_quality:.2f} | Total: {self.stats_manager.total_detected}")
                
                # Update tracker
                self.face_trackers[face_id] = (
                    center_x, center_y, current_time, quality,
                    self.face_embeddings.get(face_id)
                )
                used_ids.add(face_id)
                
                tracked_faces.append((box, face_id, quality, confidence))
                
                # Update track history
                self.track_history[face_id].append((float(center_x), float(center_y)))
        
        return tracked_faces
    
    def _render_loop(self):
        """Dedicated thread for rendering"""
        while self.is_running:
            try:
                if self.result_queue.empty():
                    time.sleep(0.001)
                    continue
                
                frame, faces, timestamp = self.result_queue.get(timeout=0.1)
                
                if faces is None:
                   # self._draw_dashboard(frame)
                    self.frame = frame
                    continue
                
                current_time = time.time()
                
                # Track faces with advanced validation
                tracked_faces = self._track_faces(faces, current_time)
                self.current_faces = tracked_faces
                
                # Draw detections with quality indicators
                for face_box, face_id, quality, confidence in tracked_faces:
                    self._draw_detection(frame, face_box, face_id, quality, confidence)
                    self._draw_trail(frame, face_id)
                
                # Update statistics
                self.stats_manager.update(len(tracked_faces))
                
                # Draw dashboard
                #self._draw_dashboard(frame)
                
                self.frame = frame
                
                # Calculate FPS
                elapsed = time.time() - self.last_frame_time
                self.last_frame_time = time.time()
                self.frame_times.append(elapsed)
                if len(self.frame_times) > 5:
                    avg_time = sum(self.frame_times) / len(self.frame_times)
                    self.fps = 1.0 / avg_time if avg_time > 0 else 0
                
            except Exception as e:
                print(f"❌ Render error: {e}")
                time.sleep(0.1)
    
    def _reconnect_stream(self):
        """Reconnect to stream"""
        print("⚠️  Stream disconnected, reconnecting...")
        if self.cap:
            self.cap.release()
        time.sleep(2)
        self.cap = self.video_handler.connect()
    
    def _draw_detection(self, frame, face_box, face_id, quality, confidence):
        """Draw face bounding box with quality indicator"""
        x, y, w, h = face_box
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        # Color based on quality and confidence
        combined_score = (quality + confidence) / 2
        
        if combined_score > 0.9:
            color = (0, 255, 0)  # Green - Excellent
            label_text = "Terdeteksi"
        elif combined_score > 0.8:
            color = (0, 255, 255)  # Yellow - Good
            label_text = "Terdeteksi Baik"
        else:
            color = (0, 165, 255)  # Orange - Fair
            label_text = "Terverifikasi"
        
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
        # Draw label
        label = f"Face #{face_id} | {label_text} ({confidence:.0%})"
        
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_y = max(label_h + 10, y)
        cv2.rectangle(frame, (x, label_y-label_h-10), (x+label_w+10, label_y), color, -1)
        cv2.putText(frame, label, (x+5, label_y-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Draw center point
        center_x = int(x + w // 2)
        center_y = int(y + h // 2)
        cv2.circle(frame, (center_x, center_y), 5, color, -1)
        
        # Draw quality bar
        bar_width = w
        bar_height = 6
        bar_x = x
        bar_y = y + h + 5
        
        if bar_y + bar_height < frame.shape[0]:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         (50, 50, 50), -1)
            fill_width = int(bar_width * combined_score)
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
    
    """def _draw_dashboard(self, frame):
       Draw statistics overlay
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (480, 270), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        # Border
        cv2.rectangle(frame, (10, 10), (480, 270), (0, 255, 0), 2)
        
        # Title
        cv2.putText(frame, "MTCNN FACE COUNTER", (20, 35),
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
        
        # Statistics
        y_offset = 70
        stats = [
            (f"Current Faces: {len(self.current_faces)}", (0, 255, 0)),
            (f"Max Today: {self.stats_manager.max_count}", (0, 255, 255)),
            (f"Total Verified: {self.stats_manager.total_detected}", (255, 165, 0)),
            (f"Display FPS: {self.fps:.1f} | Process: {self.processing_fps:.1f}", (255, 255, 255)),
            (f"Active Trackers: {len(self.face_trackers)}", (200, 200, 200)),
            (f"Detection: MTCNN", (100, 255, 100)),
            (f"Embeddings: {'Enabled' if self.use_embeddings else 'Disabled'}", (150, 200, 255)),
            (f"Time: {datetime.now().strftime('%H:%M:%S')}", (150, 150, 150))
        ]
        
        for text, color in stats:
            cv2.putText(frame, text, (25, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            y_offset += 30"""
    
    def get_frame(self):
        """Get current frame"""
        if self.frame is None:
            frame_width = getattr(Config, 'FRAME_WIDTH', 640)
            frame_height = getattr(Config, 'FRAME_HEIGHT', 480)
            return np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        return self.frame.copy()
    
    def get_frame_jpeg(self):
        """Get JPEG encoded frame"""
        frame = self.get_frame()
        jpeg_quality = getattr(Config, 'JPEG_QUALITY', 80)
        _, buffer = cv2.imencode('.jpg', frame, [
            cv2.IMWRITE_JPEG_QUALITY, jpeg_quality,
            cv2.IMWRITE_JPEG_OPTIMIZE, 1
        ])
        return buffer.tobytes()
    
    def get_statistics(self):
        """Get current statistics"""
        stats = self.stats_manager.get_stats()
        stats['fps'] = round(self.fps, 1)
        stats['processing_fps'] = round(self.processing_fps, 1)
        stats['active_trackers'] = len(self.face_trackers)
        stats['current_faces'] = len(self.current_faces)
        stats['timestamp'] = datetime.now().isoformat()
        stats['detection_method'] = 'MTCNN'
        stats['embedding_tracking'] = self.use_embeddings
        return stats
    
    def get_historical_data(self):
        """Get historical data"""
        return self.stats_manager.get_historical_data()
    
    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.stats_manager.reset_daily()
        self.detected_ids.clear()
        self.face_trackers.clear()
        self.current_faces = []
        self.face_quality_history.clear()
        self.face_embeddings.clear()
        self.next_id = 0
        print("🔄 Daily stats reset")

# Backward compatibility
FaceCounter = MTCNNFaceCounter
ImprovedFaceCounter = MTCNNFaceCounter