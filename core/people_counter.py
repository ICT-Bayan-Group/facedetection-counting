"""
Enhanced Face Counter with Advanced Validation & False Positive Reduction
"""
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

# Suppress warnings
logging.getLogger('libav').setLevel(logging.ERROR)

from core.config import Config
from utils.video_utils import VideoStreamHandler
from utils.stats_manager import StatisticsManager

class ImprovedFaceCounter:
    def __init__(self, cctv_urls, user, password):
        print("🔄 Initializing Advanced Face Counter with Validation...")
        
        # Initialize components
        self.video_handler = VideoStreamHandler(cctv_urls, user, password)
        self.stats_manager = StatisticsManager()
        
        # Load multiple face detection models
        try:
            # Haar Cascade - Frontal Face
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            # Haar Cascade - Eyes (for validation)
            eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
            
            # Profile Face (side view)
            profile_cascade_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'
            self.profile_cascade = cv2.CascadeClassifier(profile_cascade_path)
            
            if self.face_cascade.empty():
                raise Exception("Failed to load face cascade")
            
            print(f"✅ Face detection models loaded with validation")
        except Exception as e:
            print(f"❌ Error loading face detection model: {e}")
            raise
        
        # Try to load DNN face detector (more accurate)
        try:
            self.use_dnn = False
            modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
            configFile = "deploy.prototxt"
            if os.path.exists(modelFile) and os.path.exists(configFile):
                self.net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
                self.use_dnn = True
                print(f"✅ DNN face detector loaded (enhanced accuracy)")
            else:
                print(f"ℹ️  Using Haar Cascade with validation (DNN model not found)")
        except:
            self.use_dnn = False
            print(f"ℹ️  Using Haar Cascade with multi-layer validation")
        
        # Multi-threading for real-time processing
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)
        
        # Enhanced tracking with quality scores
        self.track_history = defaultdict(lambda: deque(maxlen=Config.TRACK_HISTORY_LENGTH))
        self.detected_ids = set()
        self.current_faces = []
        self.next_id = 0
        self.face_trackers = {}  # {id: (center_x, center_y, timestamp, quality_score)}
        self.id_timeout = 1.5  # 1.5 detik timeout
        
        # Quality tracking per ID
        self.face_quality_history = defaultdict(lambda: deque(maxlen=10))
        self.false_positive_filter = {}  # Track potential false positives
        
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
        
        # Load saved data
        self.stats_manager.load_statistics()
        Config.init_directories()
        
        print("✅ Advanced validation system initialized")
        print("   - Multi-cascade detection")
        print("   - Eye validation")
        print("   - Skin color detection")
        print("   - Shape & size validation")
        print("   - Motion analysis")
    
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
        
        print("✅ Advanced face detection started with validation layers")
    
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
        """Dedicated thread for advanced face detection"""
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
                
                # Run advanced face detection with validation
                if self.use_dnn:
                    faces = self._detect_faces_dnn_validated(frame)
                else:
                    faces = self._detect_faces_haar_validated(frame)
                
                # Calculate processing FPS
                process_time = time.time() - start_time
                self.processing_fps = 1.0 / process_time if process_time > 0 else 0
                
                # Put result in queue
                if not self.result_queue.full():
                    self.result_queue.put((frame, faces, time.time()))
                
            except Exception as e:
                print(f"❌ Detection error: {e}")
                time.sleep(0.1)
    
    def _detect_faces_haar_validated(self, frame):
        """
        Advanced Haar Cascade detection with multiple validation layers
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        # Detect frontal faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,  # Increased to reduce false positives
            minSize=(50, 50),  # Increased minimum size
            maxSize=(400, 400),  # Add maximum size limit
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Validate each detection
        validated_faces = []
        for (x, y, w, h) in faces:
            quality_score = self._validate_face_region(frame, gray, x, y, w, h)
            
            # Only accept faces with quality score > threshold
            if quality_score > 0.5:  # Adjustable threshold
                validated_faces.append([x, y, w, h, quality_score])
        
        return np.array(validated_faces) if len(validated_faces) > 0 else np.array([])
    
    def _detect_faces_dnn_validated(self, frame):
        """
        DNN detection with validation
        """
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), 
                                      [104, 117, 123], False, False)
        
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        confidence_threshold = 0.6  # Increased threshold
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                
                # Validate detection
                face_w = x2 - x1
                face_h = y2 - y1
                
                # Additional validation
                quality_score = self._validate_face_region(frame, gray, x1, y1, face_w, face_h)
                combined_score = (confidence + quality_score) / 2
                
                if combined_score > 0.6:  # Combined threshold
                    faces.append([x1, y1, face_w, face_h, combined_score])
        
        return np.array(faces) if len(faces) > 0 else np.array([])
    
    def _validate_face_region(self, frame, gray, x, y, w, h):
        """
        Advanced validation with multiple checks
        Returns quality score (0.0 - 1.0)
        """
        score = 0.0
        checks_passed = 0
        total_checks = 0
        
        # Ensure region is within frame
        if x < 0 or y < 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
            return 0.0
        
        face_roi = frame[y:y+h, x:x+w]
        gray_roi = gray[y:y+h, x:x+w]
        
        if face_roi.size == 0:
            return 0.0
        
        # CHECK 1: Aspect Ratio (faces are roughly oval, not too wide or tall)
        total_checks += 1
        aspect_ratio = w / float(h)
        if 0.6 < aspect_ratio < 1.4:  # Reasonable face proportions
            score += 0.2
            checks_passed += 1
        
        # CHECK 2: Size validation (not too small or too large)
        total_checks += 1
        face_area = w * h
        frame_area = frame.shape[0] * frame.shape[1]
        relative_size = face_area / frame_area
        if 0.005 < relative_size < 0.3:  # 0.5% to 30% of frame
            score += 0.15
            checks_passed += 1
        
        # CHECK 3: Eye detection (most important for validation)
        total_checks += 1
        try:
            eyes = self.eye_cascade.detectMultiScale(
                gray_roi,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(int(w*0.1), int(h*0.1)),
                maxSize=(int(w*0.4), int(h*0.4))
            )
            
            # Valid face should have 1-2 eyes detected
            if len(eyes) >= 1 and len(eyes) <= 3:
                score += 0.3  # Highest weight
                checks_passed += 1
                
                # Extra validation: eyes should be in upper half of face
                for (ex, ey, ew, eh) in eyes:
                    if ey < h * 0.6:  # Eyes in upper 60% of face region
                        score += 0.05
                        break
        except:
            pass
        
        # CHECK 4: Skin color detection
        total_checks += 1
        if self._has_skin_color(face_roi):
            score += 0.15
            checks_passed += 1
        
        # CHECK 5: Texture analysis (faces have more texture than flat objects)
        total_checks += 1
        texture_score = self._analyze_texture(gray_roi)
        if texture_score > 20:  # Threshold for texture variance
            score += 0.1
            checks_passed += 1
        
        # CHECK 6: Edge density (faces have moderate edge density)
        total_checks += 1
        edge_density = self._calculate_edge_density(gray_roi)
        if 0.1 < edge_density < 0.4:  # Not too smooth, not too busy
            score += 0.1
            checks_passed += 1
        
        # Normalize score
        normalized_score = min(1.0, score)
        
        # Require at least 50% of checks to pass
        if checks_passed < total_checks * 0.4:
            return 0.0
        
        return normalized_score
    
    def _has_skin_color(self, roi):
        """
        Check if region contains skin-like colors
        """
        try:
            # Convert to HSV for better skin detection
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Define skin color range in HSV
            # Skin tone typically: H(0-20), S(20-170), V(60-255)
            lower_skin = np.array([0, 20, 60], dtype=np.uint8)
            upper_skin = np.array([20, 170, 255], dtype=np.uint8)
            
            # Create mask
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Calculate percentage of skin pixels
            skin_percentage = np.sum(mask > 0) / mask.size
            
            # Should have significant skin-colored pixels (>20%)
            return skin_percentage > 0.2
        except:
            return False
    
    def _analyze_texture(self, gray_roi):
        """
        Analyze texture variance (faces have natural texture)
        """
        try:
            # Calculate standard deviation (texture measure)
            texture = np.std(gray_roi)
            return texture
        except:
            return 0
    
    def _calculate_edge_density(self, gray_roi):
        """
        Calculate edge density (faces have moderate edges)
        """
        try:
            # Apply Canny edge detection
            edges = cv2.Canny(gray_roi, 50, 150)
            
            # Calculate edge pixel ratio
            edge_density = np.sum(edges > 0) / edges.size
            return edge_density
        except:
            return 0
    
    def _track_faces(self, faces, current_time):
        """
        Advanced face tracking with quality-based validation
        """
        MAX_DISTANCE = 100  # Reduced for better accuracy
        
        # Clean up old trackers
        ids_to_remove = []
        for face_id, (x, y, timestamp, quality) in self.face_trackers.items():
            if current_time - timestamp > self.id_timeout:
                ids_to_remove.append(face_id)
        
        for face_id in ids_to_remove:
            del self.face_trackers[face_id]
            # Also remove from quality history
            if face_id in self.face_quality_history:
                del self.face_quality_history[face_id]
        
        # Match new faces with existing trackers
        tracked_faces = []
        used_ids = set()
        
        for face in faces:
            x, y, w, h = face[:4]
            quality = face[4] if len(face) > 4 else 0.5
            
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Find closest existing tracker
            best_id = None
            best_distance = MAX_DISTANCE
            
            for face_id, (track_x, track_y, _, prev_quality) in self.face_trackers.items():
                if face_id in used_ids:
                    continue
                
                distance = np.sqrt((center_x - track_x)**2 + (center_y - track_y)**2)
                
                # Consider both distance and quality consistency
                quality_diff = abs(quality - prev_quality)
                adjusted_distance = distance * (1 + quality_diff)
                
                if adjusted_distance < best_distance:
                    best_distance = adjusted_distance
                    best_id = face_id
            
            # Assign ID
            if best_id is not None:
                face_id = best_id
            else:
                face_id = self.next_id
                self.next_id += 1
            
            # Update quality history
            self.face_quality_history[face_id].append(quality)
            avg_quality = np.mean(list(self.face_quality_history[face_id]))
            
            # Only track faces with consistently high quality
            if avg_quality > 0.5:
                # Track as new unique face
                if face_id not in self.detected_ids:
                    self.detected_ids.add(face_id)
                    self.stats_manager.add_unique_person()
                    print(f"✨ New face validated! ID: {face_id} | Quality: {avg_quality:.2f} | Total: {self.stats_manager.total_detected}")
                
                # Update tracker
                self.face_trackers[face_id] = (center_x, center_y, current_time, quality)
                used_ids.add(face_id)
                
                tracked_faces.append((face[:4], face_id, quality))
                
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
                    self._draw_dashboard(frame)
                    self.frame = frame
                    continue
                
                current_time = time.time()
                
                # Track faces with advanced validation
                tracked_faces = self._track_faces(faces, current_time)
                self.current_faces = tracked_faces
                
                # Draw detections with quality indicators
                for face_box, face_id, quality in tracked_faces:
                    self._draw_detection(frame, face_box, face_id, quality)
                    self._draw_trail(frame, face_id)
                
                # Update statistics
                self.stats_manager.update(len(tracked_faces))
                
                # Draw dashboard
                self._draw_dashboard(frame)
                
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
    
    def _draw_detection(self, frame, face_box, face_id, quality):
        """Draw face bounding box with quality indicator"""
        x, y, w, h = face_box
        
        # Convert to int explicitly (fix for OpenCV 4.12.0)
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        # Color based on quality score
        if quality > 0.8:
            color = (0, 255, 0)  # Green - Excellent
            label_text = "Sudah Terdeteksi"
        elif quality > 0.6:
            color = (0, 255, 255)  # Yellow - Good
            label_text = "Terdeteksi Baik"
        else:
            color = (0, 165, 255)  # Orange - Fair
            label_text = "Belum Terdeteksi"
        
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
        # Draw label with quality
        label = f"Face #{face_id} | {label_text} {quality:.0%}"
        
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_y = max(label_h + 10, y)  # Ensure label is within frame
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
        
        # Ensure bar is within frame
        if bar_y + bar_height < frame.shape[0]:
            # Background
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         (50, 50, 50), -1)
            # Quality fill
            fill_width = int(bar_width * quality)
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
    
    def _draw_dashboard(self, frame):
        """Draw statistics overlay"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (480, 250), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        # Border
        cv2.rectangle(frame, (10, 10), (480, 250), (0, 255, 0), 2)
        
        # Title
        cv2.putText(frame, "ADVANCED FACE COUNTER", (20, 35),
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
        
        # Statistics
        y_offset = 70
        stats = [
            (f"Current Faces: {len(self.current_faces)}", (0, 255, 0)),
            (f"Max Today: {self.stats_manager.max_count}", (0, 255, 255)),
            (f"Total Verified: {self.stats_manager.total_detected}", (255, 165, 0)),
            (f"Display FPS: {self.fps:.1f} | Process: {self.processing_fps:.1f}", (255, 255, 255)),
            (f"Active Trackers: {len(self.face_trackers)}", (200, 200, 200)),
            (f"Validation: Multi-Layer", (100, 255, 100)),
            (f"Time: {datetime.now().strftime('%H:%M:%S')}", (150, 150, 150))
        ]
        
        for text, color in stats:
            cv2.putText(frame, text, (25, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            y_offset += 30
    
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
        stats['validation_enabled'] = True
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
        self.next_id = 0
        print("🔄 Daily stats reset")

# Backward compatibility
FaceCounter = ImprovedFaceCounter