"""
Optimized Flask Application - Compatible with Original Frontend
"""
from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
import cv2
import time
import os
import sys
import logging

# Disable verbose logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
os.environ['YOLO_VERBOSE'] = 'False'

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

# Import modules
from core.people_counter import PeopleCounter
from core.config import Config

app = Flask(__name__)
CORS(app)

# Initialize counter
counter = None

def get_counter():
    """Get or initialize counter"""
    global counter
    if counter is None:
        counter = PeopleCounter(Config.CCTV_URLS, Config.CCTV_USER, Config.CCTV_PASS)
        counter.start()
    return counter

# ==================== ROUTES - COMPATIBLE WITH ORIGINAL FRONTEND ====================

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
                encode_params = [
                    int(cv2.IMWRITE_JPEG_QUALITY), Config.JPEG_QUALITY,
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

@app.route('/api/stats')
def get_stats():
    """
    Get current statistics
    Returns format compatible with original frontend
    """
    counter = get_counter()
    stats = counter.get_statistics()
    
    # Ensure all required fields are present
    return jsonify({
        'current_count': stats.get('current_count', 0),
        'max_count': stats.get('max_count', 0),
        'daily_total': stats.get('daily_total', 0),
        'fps': stats.get('fps', 0),
        'hourly_stats': stats.get('hourly_stats', {}),
        'timestamp': stats.get('timestamp', ''),
        'processing_fps': stats.get('processing_fps', 0),
        'active_ids': stats.get('active_ids', 0)
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
    return jsonify({
        'status': 'ok',
        'service': 'People Counter CCTV - Optimized',
        'running': counter.is_running,
        'cctv_connected': counter.cap is not None,
        'device': counter.device,
        'model': Config.YOLO_MODEL,
        'fps': round(counter.fps, 1),
        'processing_fps': round(counter.processing_fps, 1)
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
    return jsonify({
        'status': 'running' if counter.is_running else 'stopped',
        'fps': round(counter.fps, 1),
        'processing_fps': round(counter.processing_fps, 1),
        'device': counter.device,
        'model': Config.YOLO_MODEL,
        'resolution': f"{Config.FRAME_WIDTH}x{Config.FRAME_HEIGHT}",
        'active_tracks': len(counter.current_ids),
        'total_detected': counter.stats_manager.total_detected
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
    print("🚀 OPTIMIZED PEOPLE COUNTER - CCTV MONITORING SYSTEM")
    print("="*70)
    print(f"📹 CCTV IP: {Config.CCTV_IP}")
    print(f"👤 User: {Config.CCTV_USER}")
    print(f"🎯 Model: {Config.YOLO_MODEL}")
    print(f"📐 Resolution: {Config.FRAME_WIDTH}x{Config.FRAME_HEIGHT}")
    print(f"⚡ GPU Enabled: {Config.USE_GPU}")
    print(f"🔥 FP16 Mode: {Config.USE_FP16}")
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