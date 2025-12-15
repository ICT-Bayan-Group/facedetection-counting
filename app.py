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
os.environ['YOLO_VERBOSE'] = 'False'

# Suppress OpenCV/FFmpeg errors
os.environ['OPENCV_LOG_LEVEL'] = 'FATAL'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'  # Quiet mode

import warnings
warnings.filterwarnings('ignore')

# Redirect stderr to suppress FFmpeg errors
if sys.platform == 'win32':
    # Windows
    try:
        sys.stderr = open(os.devnull, 'w')
    except:
        pass
else:
    # Linux/Mac
    import contextlib
    sys.stderr = open(os.devnull, 'w')

# Import modules
from core.people_counter import PeopleCounter
from core.config import Config

app = Flask(__name__)
CORS(app)

# Initialize counter
counter = PeopleCounter(Config.CCTV_URLS, Config.CCTV_USER, Config.CCTV_PASS)

# ==================== ROUTES ====================

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    def generate():
        while True:
            frame = counter.get_frame()
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.03)  # ~30 FPS
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def get_stats():
    """Get current statistics"""
    return jsonify(counter.get_statistics())

@app.route('/api/reset', methods=['POST'])
def reset_stats():
    """Reset daily statistics"""
    counter.reset_daily_stats()
    return jsonify({'success': True, 'message': 'Statistics reset successfully'})

@app.route('/api/health')
def health_check():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'service': 'People Counter CCTV',
        'running': counter.is_running,
        'cctv_connected': counter.cap is not None
    })

@app.route('/api/history')
def get_history():
    """Get historical data"""
    return jsonify(counter.get_historical_data())

# ==================== MAIN ====================

if __name__ == '__main__':
    # Disable Flask development server messages
    cli = sys.modules['flask.cli']
    cli.show_server_banner = lambda *x: None
    
    print("\n" + "="*70)
    print("🎥 PEOPLE COUNTER - CCTV MONITORING SYSTEM")
    print("="*70)
    print(f"📹 CCTV IP: {Config.CCTV_IP}")
    print(f"👤 User: {Config.CCTV_USER}")
    print("="*70 + "\n")
    
    # Start detection
    counter.start()
    
    print("🌐 Starting web server...")
    print(f"📊 Dashboard: http://localhost:{Config.PORT}")
    print(f"🔗 API Health: http://localhost:{Config.PORT}/api/health")
    print("="*70 + "\n")
    
    try:
        app.run(
            host=Config.HOST,
            port=Config.PORT,
            debug=Config.DEBUG,
            threaded=True,
            use_reloader=False
        )
    except KeyboardInterrupt:
        print("\n⏸️  Stopping...")
        counter.stop()
        print("✅ Stopped successfully")