"""
OpenVINO-Optimized Flask App
Menghindari CPU Freeze dengan OpenVINO inference
"""
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
import time
import os
import sys
import logging
import threading
import numpy as np

# Disable verbose logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

os.environ['OPENCV_LOG_LEVEL'] = 'FATAL'

import warnings
warnings.filterwarnings('ignore')

# Import OpenVINO optimized components
from core.config import Config
from core.face_counter import OpenVINOFaceCounter
from utils.session_manager import SessionManager

app = Flask(__name__)
CORS(app)

# Global state
counter = None
session_manager = None
detection_enabled = True
stream_lock = threading.Lock()
current_session_active = False
session_start_time = None

def get_counter():
    """Get or initialize OpenVINO face counter"""
    global counter
    if counter is None:
        print("🔄 Initializing OpenVINO face counter...")
        Config.print_config()
        
        counter = OpenVINOFaceCounter(
            Config.CCTV_URLS, 
            Config.CCTV_USER, 
            Config.CCTV_PASS,
            Config
        )
        counter.start()
        print("✅ OpenVINO face counter started")
    return counter

def get_session_manager():
    """Get or initialize session manager"""
    global session_manager
    if session_manager is None:
        session_manager = SessionManager()
    return session_manager

def auto_start_session():
    """Auto-start session"""
    global detection_enabled, current_session_active, session_start_time
    
    if not current_session_active and detection_enabled:
        sess_mgr = get_session_manager()
        session = sess_mgr.start_session("CCTV Hall A")
        current_session_active = True
        session_start_time = time.time()
        print(f"🚀 Auto-started session: {session['id']}")

def ensure_stream_alive():
    """Background task untuk ensure stream tetap hidup"""
    global counter
    
    while True:
        try:
            time.sleep(15)  # Check every 15 seconds
            
            if counter and counter.is_running:
                if counter.cap is None or not counter.cap.isOpened():
                    print("⚠️ Stream not healthy, reconnecting...")
                    with stream_lock:
                        try:
                            counter._reconnect_stream()
                            print("✅ Stream reconnected")
                        except Exception as e:
                            print(f"❌ Reconnect failed: {e}")
                            time.sleep(5)
                            
        except Exception as e:
            print(f"⚠️ Monitor error: {e}")
            time.sleep(5)

# Start monitoring thread
monitor_thread = threading.Thread(target=ensure_stream_alive, daemon=True)
monitor_thread.start()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/data-pengunjung')
def data_pengunjung():
    """Halaman Data Pengunjung"""
    return render_template('data_pengunjung.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming - FIXED FPS CONTROL"""
    def generate():
        global detection_enabled, counter
        
        counter = get_counter()
        frame_count = 0
        error_count = 0
        max_errors = 10

        # FIXED: Proper FPS timing
        target_fps = Config.STREAM_FPS
        target_frame_time = 1.0 / target_fps
        last_log_time = time.time()
        last_frame_time = time.time()
        
        print(f"📹 Client connected (target: {target_fps} FPS)")
        
        while True:
            try:
                # CRITICAL: Wait for proper frame timing
                current_time = time.time()
                elapsed = current_time - last_frame_time
                
                if elapsed < target_frame_time:
                    time.sleep(target_frame_time - elapsed)
                    continue
                
                last_frame_time = time.time()
                frame_count += 1
                
                if not detection_enabled:
                    # Blank frame
                    blank = np.zeros((Config.FRAME_HEIGHT, Config.FRAME_WIDTH, 3), dtype=np.uint8)
                    
                    cv2.putText(blank, "DETECTION PAUSED", 
                              (180, 180), cv2.FONT_HERSHEY_DUPLEX, 
                              1.0, (255, 200, 0), 2)
                    
                    ret, buffer = cv2.imencode('.jpg', blank, [
                        cv2.IMWRITE_JPEG_QUALITY, Config.JPEG_QUALITY
                    ])
                    
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + 
                               buffer.tobytes() + b'\r\n')
                    continue
                
                # Get frame
                with stream_lock:
                    frame = counter.get_frame()
                
                if frame is None or frame.size == 0:
                    error_count += 1
                    if error_count > max_errors:
                        with stream_lock:
                            try:
                                counter._reconnect_stream()
                                error_count = 0
                            except:
                                pass
                    continue
                
                error_count = 0
                
                # OPTIMIZED JPEG ENCODING
                ret, buffer = cv2.imencode('.jpg', frame, [
                    cv2.IMWRITE_JPEG_QUALITY, Config.JPEG_QUALITY,
                    cv2.IMWRITE_JPEG_OPTIMIZE, 1
                ])
                
                if not ret:
                    continue
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + 
                       buffer.tobytes() + b'\r\n')
                
                # Log FPS periodically
                if time.time() - last_log_time > 10:
                    actual_fps = frame_count / (time.time() - last_log_time + frame_count * target_frame_time)
                    print(f"📊 Stream FPS: {actual_fps:.1f} (target: {target_fps})")
                    last_log_time = time.time()
                    frame_count = 0
                
            except GeneratorExit:
                print("📹 Client disconnected")
                break
            except Exception as e:
                error_count += 1
                print(f"❌ Stream error: {e}")
                time.sleep(0.1)
    
    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0',
            'Connection': 'close'
        }
    )

@app.route('/api/detection/toggle', methods=['POST'])
def toggle_detection():
    """Toggle detection"""
    global detection_enabled, current_session_active, session_start_time
    
    data = request.get_json()
    if data and 'enabled' in data:
        detection_enabled = data['enabled']
    else:
        detection_enabled = not detection_enabled
    
    counter = get_counter()
    sess_mgr = get_session_manager()
    
    if detection_enabled and not current_session_active:
        camera_location = data.get('camera_location', 'CCTV Hall A') if data else 'CCTV Hall A'
        session = sess_mgr.start_session(camera_location)
        current_session_active = True
        session_start_time = time.time()
    
    elif not detection_enabled and current_session_active:
        stats = counter.get_statistics()
        sess_mgr.end_session(
            total_visitors=stats.get('daily_total', 0),
            max_concurrent=stats.get('max_count', 0),
            status='Selesai',
            notes='Manual stop'
        )
        current_session_active = False
        session_start_time = None
    
    return jsonify({
        'success': True,
        'detection_enabled': detection_enabled,
        'session_active': current_session_active
    })

@app.route('/api/stats')
def get_stats():
    """Get statistics"""
    counter = get_counter()
    stats = counter.get_statistics()
    
    return jsonify({
        'current_count': stats.get('current_count', 0),
        'max_count': stats.get('max_count', 0),
        'daily_total': stats.get('daily_total', 0),
        'database_faces': stats.get('database_size', 0),
        'fps': stats.get('fps', 0),
        'processing_fps': stats.get('processing_fps', 0),
        'active_ids': stats.get('active_trackers', 0),
        'detection_enabled': detection_enabled
    })

@app.route('/api/health')
def health_check():
    """Health check"""
    counter = get_counter()
    
    return jsonify({
        'status': 'ok',
        'service': f'OpenVINO Face Counter - {Config.STREAM_FPS} FPS',
        'running': counter.is_running,
        'detector': counter.detector_type,
        'fps': round(counter.fps, 1),
        'detection_enabled': detection_enabled
    })

@app.route('/api/history')
def get_history():
    """Get historical data"""
    counter = get_counter()
    return jsonify(counter.get_historical_data())

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    sess_mgr = get_session_manager()
    sessions = sess_mgr.get_all_sessions()
    return jsonify({'success': True, 'sessions': sessions})

@app.route('/api/reset', methods=['POST'])
def reset_stats():
    """Reset daily statistics"""
    counter = get_counter()
    counter.reset_daily_stats()
    return jsonify({'success': True, 'message': 'Statistics reset successfully'})

if __name__ == '__main__':
    cli = sys.modules['flask.cli']
    cli.show_server_banner = lambda *x: None
    
    print("\n" + "="*70)
    print("🚀 OPENVINO FACE COUNTER")
    print("="*70)
    print(f"📹 CCTV: {Config.CCTV_IP}")
    print(f"🎯 Detector: OpenVINO on {Config.OPENVINO_DEVICE}")
    print(f"📐 Resolution: {Config.FRAME_WIDTH}x{Config.FRAME_HEIGHT}")
    print(f"🎬 Stream FPS: {Config.STREAM_FPS}")
    print(f"🔍 Detection FPS: {Config.DETECTION_FPS}")
    print("="*70)
    print(f"\n🌐 Dashboard: http://localhost:{Config.PORT}")
    print("="*70)
    print("\n⚠️  Press Ctrl+C to stop\n")
    
    counter = get_counter()
    time.sleep(2)
    auto_start_session()
    
    try:
        app.run(
            host=Config.HOST,
            port=Config.PORT,
            debug=False,
            threaded=True,
            use_reloader=False
        )
    except KeyboardInterrupt:
        print("\n\n⏸️  Stopping...")
        if counter:
            counter.stop()
        print("✅ Stopped\n")