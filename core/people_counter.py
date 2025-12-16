"""
Enhanced People Counter with Optimized Detection & Real-time Streaming
"""
import cv2
import numpy as np
from ultralytics import YOLO
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
logging.getLogger('h264').setLevel(logging.ERROR)

from core.config import Config
from utils.video_utils import VideoStreamHandler
from utils.stats_manager import StatisticsManager

class PeopleCounter:
    def __init__(self, cctv_urls, user, password):
        print("🔄 Initializing Enhanced People Counter...")
        
        # Initialize components
        self.video_handler = VideoStreamHandler(cctv_urls, user, password)
        self.stats_manager = StatisticsManager()
        
        # Load optimized YOLO model with GPU support
        try:
            self.model = YOLO(Config.YOLO_MODEL)
            self.model.overrides['verbose'] = False
            
            # Enable GPU if available
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(self.device)
            
            # Half precision for faster inference on GPU
            if self.device == 'cuda':
                self.model.model.half()
                print(f"✅ YOLO model loaded on GPU with FP16")
            else:
                print(f"✅ YOLO model loaded on CPU")
        except Exception as e:
            print(f"⚠️  Downloading YOLOv8 model: {e}")
            self.model = YOLO(Config.YOLO_MODEL)
            self.device = 'cpu'
        
        # Multi-threading for real-time processing
        self.frame_queue = Queue(maxsize=2)  # Small queue to reduce latency
        self.result_queue = Queue(maxsize=2)
        
        # Tracking with enhanced ID persistence
        self.track_history = defaultdict(lambda: deque(maxlen=Config.TRACK_HISTORY_LENGTH))
        self.detected_ids = set()
        self.current_ids = set()
        self.id_confidence = defaultdict(float)  # Track confidence per ID
        self.id_last_seen = defaultdict(float)  # Track when ID was last seen
        
        # State
        self.frame = None
        self.is_running = False
        self.cap = None
        
        # Enhanced FPS calculation
        self.fps = 0
        self.processing_fps = 0
        self.frame_times = deque(maxlen=20)  # Smaller window for more responsive FPS
        self.last_frame_time = time.time()
        
        # Frame skip for optimization
        self.frame_skip = Config.FRAME_SKIP
        self.frame_count = 0
        
        # Load saved data
        self.stats_manager.load_statistics()
        Config.init_directories()
        
        # Background subtractor for motion detection (optional optimization)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )
        
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
        
        print("✅ Enhanced detection started with multi-threading")
    
    def stop(self):
        """Stop detection"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.stats_manager.save_statistics()
        print("⏸️  Detection stopped")
    
    def _frame_capture_loop(self):
        """Dedicated thread for frame capture - reduces latency"""
        while self.is_running:
            try:
                if self.cap is None:
                    time.sleep(0.1)
                    continue
                
                # Fast frame grab
                ret = self.cap.grab()
                if not ret:
                    self._reconnect_stream()
                    continue
                
                # Retrieve frame only if queue not full
                if not self.frame_queue.full():
                    ret, frame = self.cap.retrieve()
                    if ret and frame is not None and frame.size > 0:
                        # Quick resize for faster processing
                        frame = cv2.resize(frame, (Config.FRAME_WIDTH, Config.FRAME_HEIGHT), 
                                         interpolation=cv2.INTER_LINEAR)
                        self.frame_queue.put(frame)
                
                time.sleep(0.001)  # Minimal delay
                
            except Exception as e:
                print(f"❌ Capture error: {e}")
                time.sleep(1)
    
    def _detection_loop(self):
        """Dedicated thread for AI detection"""
        while self.is_running:
            try:
                if self.frame_queue.empty():
                    time.sleep(0.001)
                    continue
                
                frame = self.frame_queue.get(timeout=0.1)
                self.frame_count += 1
                
                # Frame skipping for performance optimization
                if self.frame_count % (self.frame_skip + 1) != 0:
                    if not self.result_queue.full():
                        self.result_queue.put((frame, None, time.time()))
                    continue
                
                start_time = time.time()
                
                # Run optimized YOLO detection
                results = self.model.track(
                    frame,
                    persist=True,
                    classes=[0],  # Person class only
                    conf=Config.CONFIDENCE_THRESHOLD,
                    iou=Config.IOU_THRESHOLD,
                    tracker=Config.TRACKER,
                    verbose=False,
                    half=True if self.device == 'cuda' else False,  # FP16 on GPU
                    device=self.device,
                    max_det=50  # Limit max detections for performance
                )
                
                # Calculate processing FPS
                process_time = time.time() - start_time
                self.processing_fps = 1.0 / process_time if process_time > 0 else 0
                
                # Put result in queue
                if not self.result_queue.full():
                    self.result_queue.put((frame, results[0], time.time()))
                
            except Exception as e:
                print(f"❌ Detection error: {e}")
                time.sleep(0.1)
    
    def _render_loop(self):
        """Dedicated thread for rendering and statistics"""
        while self.is_running:
            try:
                if self.result_queue.empty():
                    time.sleep(0.001)
                    continue
                
                frame, results, timestamp = self.result_queue.get(timeout=0.1)
                
                if results is None:
                    # Frame was skipped, just use previous detections overlay
                    self._draw_dashboard(frame)
                    self.frame = frame
                    continue
                
                # Process detections with enhanced tracking
                current_people = set()
                current_time = time.time()
                
                if results.boxes.id is not None:
                    boxes = results.boxes.xywh.cpu().numpy()
                    track_ids = results.boxes.id.int().cpu().tolist()
                    confidences = results.boxes.conf.cpu().tolist()
                    
                    for box, track_id, conf in zip(boxes, track_ids, confidences):
                        x, y, w, h = box
                        
                        # Enhanced tracking with confidence filtering
                        if conf >= Config.MIN_TRACKING_CONFIDENCE:
                            current_people.add(track_id)
                            self.id_confidence[track_id] = conf
                            self.id_last_seen[track_id] = current_time
                            
                            # Track unique people with confidence threshold
                            if track_id not in self.detected_ids and conf >= Config.NEW_ID_CONFIDENCE:
                                self.detected_ids.add(track_id)
                                self.stats_manager.add_unique_person()
                                print(f"✨ New person detected! ID: {track_id} | Confidence: {conf:.2f} | Total: {self.stats_manager.total_detected}")
                            
                            # Draw detection
                            self._draw_detection(frame, box, track_id, conf)
                            
                            # Update track history
                            self.track_history[track_id].append((float(x), float(y)))
                            
                            # Draw trail
                            self._draw_trail(frame, track_id)
                
                # Clean up old IDs (not seen for 5 seconds)
                ids_to_remove = [
                    tid for tid, last_seen in self.id_last_seen.items()
                    if current_time - last_seen > 5.0
                ]
                for tid in ids_to_remove:
                    self.track_history.pop(tid, None)
                    self.id_confidence.pop(tid, None)
                    self.id_last_seen.pop(tid, None)
                
                # Update statistics
                self.current_ids = current_people
                self.stats_manager.update(len(current_people))
                
                # Draw dashboard overlay
                self._draw_dashboard(frame)
                
                self.frame = frame
                
                # Calculate display FPS
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
        """Reconnect to stream with backoff"""
        print("⚠️  Stream disconnected, reconnecting...")
        if self.cap:
            self.cap.release()
        time.sleep(2)
        self.cap = self.video_handler.connect()
    
    def _draw_detection(self, frame, box, track_id, conf):
        """Draw enhanced bounding box with confidence"""
        x, y, w, h = box
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        
        # Dynamic color based on confidence
        if conf > 0.8:
            color = (0, 255, 0)  # Green - High confidence
        elif conf > 0.6:
            color = (0, 255, 255)  # Yellow - Medium confidence
        else:
            color = (0, 165, 255)  # Orange - Low confidence
        
        # Draw thick border for better visibility
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Draw filled label background
        label = f"ID:{track_id} {conf:.0%}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1-label_h-10), (x1+label_w+10, y1), color, -1)
        cv2.putText(frame, label, (x1+5, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw center point
        cv2.circle(frame, (int(x), int(y)), 5, color, -1)
    
    def _draw_trail(self, frame, track_id):
        """Draw smooth tracking trail with fade effect"""
        points = self.track_history[track_id]
        if len(points) > 1:
            color = (0, 255, 0) if track_id in self.current_ids else (100, 100, 100)
            
            # Draw trail with thickness variation
            for i in range(1, len(points)):
                thickness = max(1, int(3 * (i / len(points))))  # Fade effect
                cv2.line(frame,
                        (int(points[i-1][0]), int(points[i-1][1])),
                        (int(points[i][0]), int(points[i][1])),
                        color, thickness)
    
    def _draw_dashboard(self, frame):
        """Draw enhanced statistics overlay"""
        h, w = frame.shape[:2]
        
        # Modern semi-transparent dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 220), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        # Border
        cv2.rectangle(frame, (10, 10), (450, 220), (0, 255, 0), 2)
        
        # Title
        cv2.putText(frame, "PEOPLE COUNTER SYSTEM", (20, 35),
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
        
        # Stats with icons
        y_offset = 70
        stats = [
            (f"Current People: {self.stats_manager.people_count}", (0, 255, 0)),
            (f"Max Today: {self.stats_manager.max_count}", (0, 255, 255)),
            (f"Total Unique: {self.stats_manager.total_detected}", (255, 165, 0)),
            (f"Display FPS: {self.fps:.1f} | Process: {self.processing_fps:.1f}", (255, 255, 255)),
            (f"Active IDs: {len(self.current_ids)}", (200, 200, 200)),
            (f"Time: {datetime.now().strftime('%H:%M:%S')}", (150, 150, 150))
        ]
        
        for text, color in stats:
            cv2.putText(frame, text, (25, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 30
    
    def get_frame(self):
        """Get current frame for streaming with JPEG optimization"""
        if self.frame is None:
            return np.zeros((Config.FRAME_HEIGHT, Config.FRAME_WIDTH, 3), dtype=np.uint8)
        return self.frame.copy()
    
    def get_frame_jpeg(self):
        """Get JPEG encoded frame for faster streaming"""
        frame = self.get_frame()
        _, buffer = cv2.imencode('.jpg', frame, [
            cv2.IMWRITE_JPEG_QUALITY, Config.JPEG_QUALITY,
            cv2.IMWRITE_JPEG_OPTIMIZE, 1
        ])
        return buffer.tobytes()
    
    def get_statistics(self):
        """Get current statistics"""
        stats = self.stats_manager.get_stats()
        stats['fps'] = round(self.fps, 1)
        stats['processing_fps'] = round(self.processing_fps, 1)
        stats['active_ids'] = len(self.current_ids)
        stats['timestamp'] = datetime.now().isoformat()
        return stats
    
    def get_historical_data(self):
        """Get historical data for charts"""
        return self.stats_manager.get_historical_data()
    
    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.stats_manager.reset_daily()
        self.detected_ids.clear()
        self.current_ids.clear()
        self.id_confidence.clear()
        self.id_last_seen.clear()
        print("🔄 Daily stats reset")