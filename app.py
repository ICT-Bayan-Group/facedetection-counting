"""
Optimized Flask Application for Face Counter - Compatible with Frontend
"""
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
import time
import os
import sys
import logging

# Disable verbose logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Suppress OpenCV/FFmpeg errors
os.environ['OPENCV_LOG_LEVEL'] = 'FATAL'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'

import warnings
warnings.filterwarnings('ignore')

# Redirect stderr to suppress FFmpeg errors
if sys.platform == 'win32':
    try:
        sys.stderr = open(os.devnull, 'w')
    except:
        pass
else:
    sys.stderr = open(os.devnull, 'w')

# Import modules - CHANGED: Import FaceCounter instead of PeopleCounter
from core.people_counter import FaceCounter
from core.config import Config

app = Flask(__name__)
CORS(app)

# Initialize counter
counter = None

def get_counter():
    """Get or initialize face counter"""
    global counter
    if counter is None:
        counter = FaceCounter(Config.CCTV_URLS, Config.CCTV_USER, Config.CCTV_PASS)
        counter.start()
    return counter

# ==================== ROUTES - COMPATIBLE WITH FRONTEND ====================

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    """
    Video streaming route - Optimized for low latency
    Compatible with original frontend
    """
    def generate():
        counter = get_counter()
        last_frame_time = time.time()
        target_interval = 1.0 / 30  # 30 FPS target
        
        while True:
            try:
                # Get frame from optimized counter
                frame = counter.get_frame()
                
                if frame is None or frame.size == 0:
                    time.sleep(0.01)
                    continue
                
                # Optimized JPEG encoding
                jpeg_quality = getattr(Config, 'JPEG_QUALITY', 80)
                encode_params = [
                    int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality,
                    int(cv2.IMWRITE_JPEG_OPTIMIZE), 1
                ]
                
                ret, buffer = cv2.imencode('.jpg', frame, encode_params)
                
                if not ret:
                    continue
                
                frame_bytes = buffer.tobytes()
                
                # Yield frame in multipart format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n'
                       b'\r\n' + frame_bytes + b'\r\n')
                
                # Frame rate limiting (optional - remove for max speed)
                elapsed = time.time() - last_frame_time
                if elapsed < target_interval:
                    time.sleep(target_interval - elapsed)
                last_frame_time = time.time()
                
            except GeneratorExit:
                break
            except Exception as e:
                print(f"Stream error: {e}")
                time.sleep(0.1)
    
    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )

@app.route('/api/database/stats')
def get_database_stats():
    """Get face database statistics"""
    counter = get_counter()
    db_stats = counter.get_database_stats()
    return jsonify(db_stats)

@app.route('/api/database/save', methods=['POST'])
def save_database():
    """Manual save face database"""
    counter = get_counter()
    counter.save_face_database()
    return jsonify({
        'success': True,
        'message': 'Face database saved successfully',
        'total_faces': len(counter.face_db.faces)
    })

@app.route('/api/database/reset', methods=['POST'])
def reset_database():
    """Reset face database - HATI-HATI!"""
    if request.json and request.json.get('confirm') == 'yes':
        counter = get_counter()
        counter.reset_face_database()
        return jsonify({
            'success': True,
            'message': 'Face database has been completely reset'
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Confirmation required. Send {"confirm": "yes"}'
        }), 400

@app.route('/api/database/info')
def database_info():
    """Get detailed database information"""
    counter = get_counter()
    
    return jsonify({
        'total_faces': len(counter.face_db.faces),
        'similarity_threshold': counter.face_db.similarity_threshold,
        'database_path': counter.face_db.db_path,
        'database_exists': os.path.exists(counter.face_db.db_path)
    })

@app.route('/api/stats')
def get_stats():
    """Get current statistics - UPDATED VERSION"""
    counter = get_counter()
    stats = counter.get_statistics()
    
    # Tambahkan info database
    stats['database_faces'] = len(counter.face_db.faces)
    
    return jsonify({
        'current_count': stats.get('current_count', 0),
        'max_count': stats.get('max_count', 0),
        'daily_total': stats.get('daily_total', 0),
        'database_faces': stats.get('database_faces', 0),  # NEW
        'fps': stats.get('fps', 0),
        'hourly_stats': stats.get('hourly_stats', {}),
        'timestamp': stats.get('timestamp', ''),
        'processing_fps': stats.get('processing_fps', 0),
        'active_ids': stats.get('active_trackers', 0),
        'current_faces': stats.get('current_faces', 0)
    })

@app.route('/api/reset', methods=['POST'])
def reset_stats():
    """Reset daily statistics"""
    counter = get_counter()
    counter.reset_daily_stats()
    return jsonify({
        'success': True, 
        'message': 'Daily statistics have been reset successfully'
    })

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    counter = get_counter()
    
    # Detect which model is being used
    model_type = 'Haar Cascade'
    if hasattr(counter, 'use_dnn') and counter.use_dnn:
        model_type = 'DNN Face Detector'
    
    return jsonify({
        'status': 'ok',
        'service': 'Face Counter CCTV - Optimized',
        'running': counter.is_running,
        'cctv_connected': counter.cap is not None,
        'model': model_type,
        'fps': round(counter.fps, 1),
        'processing_fps': round(counter.processing_fps, 1),
        'active_trackers': len(counter.face_trackers)
    })

@app.route('/api/history')
def get_history():
    """Get historical data for charts"""
    counter = get_counter()
    history = counter.get_historical_data()
    
    # Ensure peak_hours format is correct
    if 'peak_hours' not in history:
        history['peak_hours'] = []
    
    return jsonify(history)

@app.route('/stream_info')
def stream_info():
    """Get detailed stream information"""
    counter = get_counter()
    
    model_type = 'Haar Cascade'
    if hasattr(counter, 'use_dnn') and counter.use_dnn:
        model_type = 'DNN Face Detector'
    
    frame_width = getattr(Config, 'FRAME_WIDTH', 640)
    frame_height = getattr(Config, 'FRAME_HEIGHT', 480)
    
    return jsonify({
        'status': 'running' if counter.is_running else 'stopped',
        'fps': round(counter.fps, 1),
        'processing_fps': round(counter.processing_fps, 1),
        'model': model_type,
        'resolution': f"{frame_width}x{frame_height}",
        'active_tracks': len(counter.face_trackers),
        'total_detected': counter.stats_manager.total_detected,
        'current_faces': len(counter.current_faces)
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found', 'message': str(error)}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error', 'message': str(error)}), 500

# ==================== MAIN ====================

if __name__ == '__main__':
    # Disable Flask development server banner
    cli = sys.modules['flask.cli']
    cli.show_server_banner = lambda *x: None
    
    print("\n" + "="*70)
    print("🚀 FACE COUNTER - CCTV MONITORING SYSTEM")
    print("="*70)
    print(f"📹 CCTV IP: {Config.CCTV_IP}")
    print(f"👤 User: {Config.CCTV_USER}")
    print(f"🎯 Model: Haar Cascade + DNN (if available)")
    frame_width = getattr(Config, 'FRAME_WIDTH', 640)
    frame_height = getattr(Config, 'FRAME_HEIGHT', 480)
    print(f"📐 Resolution: {frame_width}x{frame_height}")
    print("="*70)
    print(f"\n🌐 Starting web server...")
    print(f"📊 Dashboard: http://localhost:{Config.PORT}")
    print(f"📊 Dashboard (Network): http://{Config.HOST}:{Config.PORT}")
    print(f"🔗 API Health: http://localhost:{Config.PORT}/api/health")
    print(f"📹 Video Feed: http://localhost:{Config.PORT}/video_feed")
    print("="*70)
    print("\n⚠️  Press Ctrl+C to stop the server\n")
    
    try:
        app.run(
            host=Config.HOST,
            port=Config.PORT,
            debug=Config.DEBUG,
            threaded=True,
            use_reloader=False
        )
    except KeyboardInterrupt:
        print("\n\n⏸️  Stopping server...")
        if counter:
            counter.stop()
        print("✅ Server stopped successfully")
        print("👋 Goodbye!\n")