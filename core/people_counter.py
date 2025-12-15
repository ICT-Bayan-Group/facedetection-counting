"""
Main People Counter class with YOLO detection and tracking
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

from core.config import Config
from utils.video_utils import VideoStreamHandler
from utils.stats_manager import StatisticsManager
import sys
import logging

# Suppress OpenCV warnings
logging.getLogger('libav').setLevel(logging.ERROR)
logging.getLogger('h264').setLevel(logging.ERROR)

class PeopleCounter:
    def __init__(self, cctv_urls, user, password):
        print("🔄 Initializing People Counter...")
        
        # Initialize components
        self.video_handler = VideoStreamHandler(cctv_urls, user, password)
        self.stats_manager = StatisticsManager()
        
        # Load YOLOv8 model
        try:
            self.model = YOLO(Config.YOLO_MODEL)
            self.model.overrides['verbose'] = False
            print("✅ YOLO model loaded")
        except:
            print("⚠️  Downloading YOLOv8 model...")
            self.model = YOLO(Config.YOLO_MODEL)
            self.model.overrides['verbose'] = False
        
        # Tracking data
        self.track_history = defaultdict(lambda: deque(maxlen=Config.TRACK_HISTORY_LENGTH))
        self.detected_ids = set()
        self.current_ids = set()
        
        # State
        self.frame = None
        self.is_running = False
        self.cap = None
        
        # FPS calculation
        self.fps = 0
        self.frame_times = deque(maxlen=Config.FPS_WINDOW_SIZE)
        
        # Load saved data
        self.stats_manager.load_statistics()
        Config.init_directories()
        
    def start(self):
        """Start detection thread"""
        if self.is_running:
            return
        
        self.cap = self.video_handler.connect()
        self.is_running = True
        
        thread = threading.Thread(target=self.detection_loop, daemon=True)
        thread.start()
        print("✅ Detection started")
    
    def stop(self):
        """Stop detection"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.stats_manager.save_statistics()
        print("⏸️  Detection stopped")
    
    def detection_loop(self):
        """Main detection loop"""
        failed_reads = 0
        max_failed_reads = 5
        
        while self.is_running:
            try:
                start_time = time.time()
                
                # Get frame
                if self.cap is None:
                    self.frame = self._generate_demo_frame()
                    time.sleep(0.1)
                    continue
                
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    failed_reads += 1
                    
                    # Skip corrupt frame, don't reconnect immediately
                    if failed_reads < max_failed_reads:
                        # Clear buffer by reading multiple frames
                        for _ in range(3):
                            self.cap.grab()
                        continue
                    
                    # Reconnect only after multiple consecutive failures
                    print("⚠️  Multiple frame read failures, reconnecting...")
                    self.cap.release()
                    time.sleep(1)
                    self.cap = self.video_handler.connect()
                    failed_reads = 0
                    continue
                
                # Reset failed reads counter on successful read
                failed_reads = 0
                
                # Validate frame
                if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
                    continue
                
                # Process frame
                try:
                    frame = cv2.resize(frame, (Config.FRAME_WIDTH, Config.FRAME_HEIGHT))
                except:
                    continue
                
                # Run YOLO detection
                results = self.model.track(
                    frame,
                    persist=True,
                    classes=[0],  # Person class
                    conf=Config.CONFIDENCE_THRESHOLD,
                    iou=Config.IOU_THRESHOLD,
                    tracker=Config.TRACKER,
                    verbose=False
                )
                
                # Process detections
                current_people = set()
                
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xywh.cpu()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    confidences = results[0].boxes.conf.cpu().tolist()
                    
                    for box, track_id, conf in zip(boxes, track_ids, confidences):
                        x, y, w, h = box
                        current_people.add(track_id)
                        
                        # Track unique people
                        if track_id not in self.detected_ids:
                            self.detected_ids.add(track_id)
                            self.stats_manager.add_unique_person()
                            print(f"✨ New person detected! ID: {track_id} | Total unique: {self.stats_manager.total_detected}")
                        
                        # Draw detection
                        self._draw_detection(frame, box, track_id, conf)
                        
                        # Update track history
                        self.track_history[track_id].append((float(x), float(y)))
                        
                        # Draw trail
                        self._draw_trail(frame, track_id)
                
                # Update statistics
                self.current_ids = current_people
                self.stats_manager.update(len(current_people))
                
                # Draw dashboard overlay
                self._draw_dashboard(frame)
                
                self.frame = frame
                
                # Calculate FPS
                elapsed = time.time() - start_time
                self.frame_times.append(elapsed)
                if len(self.frame_times) > 0:
                    self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
                
            except Exception as e:
                print(f"❌ Detection error: {str(e)}")
                time.sleep(1)
    
    def _draw_detection(self, frame, box, track_id, conf):
        """Draw bounding box and label"""
        x, y, w, h = box
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        
        # Color based on detection status
        if track_id in self.detected_ids and track_id not in self.current_ids:
            color = (0, 255, 255)  # Yellow for returning
        else:
            color = (0, 255, 0)    # Green for current/new
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"ID:{track_id} ({conf:.2f})"
        cv2.putText(frame, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def _draw_trail(self, frame, track_id):
        """Draw tracking trail"""
        points = self.track_history[track_id]
        if len(points) > 1:
            color = (0, 255, 0) if track_id in self.current_ids else (0, 255, 255)
            for i in range(1, len(points)):
                cv2.line(frame,
                        (int(points[i-1][0]), int(points[i-1][1])),
                        (int(points[i][0]), int(points[i][1])),
                        color, 2)
    
    def _draw_dashboard(self, frame):
        """Draw statistics overlay on frame"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Text info
        y_offset = 40
        info = [
            f"Current People: {self.stats_manager.people_count}",
            f"Max Today: {self.stats_manager.max_count}",
            f"Total Unique: {self.stats_manager.total_detected}",
            f"FPS: {self.fps:.1f}",
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        ]
        
        for text in info:
            cv2.putText(frame, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 30
    
    def _generate_demo_frame(self):
        """Generate demo frame when CCTV not available"""
        frame = np.zeros((Config.FRAME_HEIGHT, Config.FRAME_WIDTH, 3), dtype=np.uint8)
        
        cv2.putText(frame, "DEMO MODE - CCTV NOT CONNECTED",
                   (300, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # Simulate random count
        import random
        self.stats_manager.people_count = random.randint(0, 10)
        self._draw_dashboard(frame)
        
        return frame
    
    def get_frame(self):
        """Get current frame for streaming"""
        if self.frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return self.frame.copy()
    
    def get_statistics(self):
        """Get current statistics"""
        stats = self.stats_manager.get_stats()
        stats['fps'] = round(self.fps, 1)
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
        print("🔄 Daily stats reset - Unique people counter cleared")