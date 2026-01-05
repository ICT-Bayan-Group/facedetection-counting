from flask import Flask, render_template, Response, jsonify, request, send_file
from flask_cors import CORS
import cv2
import time
import os
import sys
import logging
import threading
import io

# Disable verbose logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

os.environ['OPENCV_LOG_LEVEL'] = 'FATAL'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'

import warnings
warnings.filterwarnings('ignore')

from core.people_counter import FaceCounter
from core.config import Config
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
    """Get or initialize face counter"""
    global counter
    if counter is None:
        print("🔄 Initializing face counter...")
        counter = FaceCounter(Config.CCTV_URLS, Config.CCTV_USER, Config.CCTV_PASS)
        counter.start()
        print("✅ Face counter started")
    return counter

def get_session_manager():
    """Get or initialize session manager"""
    global session_manager
    if session_manager is None:
        session_manager = SessionManager()
    return session_manager

def auto_start_session():
    """Auto-start session ketika aplikasi dimulai"""
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
            time.sleep(10)
            
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
    """Video streaming - PERSISTENT & STABLE"""
    def generate():
        global detection_enabled, counter
        
        counter = get_counter()
        frame_count = 0
        error_count = 0
        max_errors = 20
        
        print("📹 Client connected to video stream")
        
        while True:
            try:
                frame_count += 1
                
                if not detection_enabled:
                    import numpy as np
                    blank = np.zeros((480, 640, 3), dtype=np.uint8)
                    
                    for i in range(480):
                        color = int(40 + (i / 480) * 30)
                        blank[i, :] = (color, color, color)
                    
                    cv2.putText(blank, "DETECTION PAUSED", 
                              (120, 220), cv2.FONT_HERSHEY_DUPLEX, 
                              1.5, (255, 200, 0), 3)
                    cv2.putText(blank, "Click 'Start Detection' to resume", 
                              (80, 270), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.9, (200, 200, 200), 2)
                    
                    ret, buffer = cv2.imencode('.jpg', blank, [
                        cv2.IMWRITE_JPEG_QUALITY, 85
                    ])
                    
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + 
                               frame_bytes + b'\r\n')
                    
                    time.sleep(0.1)
                    continue
                
                with stream_lock:
                    frame = counter.get_frame()
                
                if frame is None or frame.size == 0:
                    error_count += 1
                    
                    if error_count > max_errors:
                        print(f"⚠️ Too many frame errors ({error_count}), reconnecting...")
                        with stream_lock:
                            try:
                                counter._reconnect_stream()
                                error_count = 0
                            except:
                                pass
                    
                    time.sleep(0.05)
                    continue
                
                error_count = 0
                
                ret, buffer = cv2.imencode('.jpg', frame, [
                    cv2.IMWRITE_JPEG_QUALITY, 85,
                    cv2.IMWRITE_JPEG_OPTIMIZE, 1
                ])
                
                if not ret:
                    continue
                
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + 
                       frame_bytes + b'\r\n')
                
                time.sleep(0.033)
                
            except GeneratorExit:
                print("📹 Client disconnected from video stream")
                break
            except Exception as e:
                error_count += 1
                print(f"❌ Stream error #{error_count}: {e}")
                
                if error_count > max_errors:
                    with stream_lock:
                        try:
                            counter._reconnect_stream()
                            error_count = 0
                        except:
                            pass
                
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

@app.route('/api/detection/toggle', methods=['POST'])
def toggle_detection():
    """Toggle face detection ON/OFF dengan session management"""
    global detection_enabled, current_session_active, session_start_time
    
    data = request.get_json()
    if data and 'enabled' in data:
        detection_enabled = data['enabled']
    else:
        detection_enabled = not detection_enabled
    
    counter = get_counter()
    sess_mgr = get_session_manager()
    
    # Jika detection DIAKTIFKAN -> Start session
    if detection_enabled and not current_session_active:
        camera_location = data.get('camera_location', 'CCTV Hall A') if data else 'CCTV Hall A'
        session = sess_mgr.start_session(camera_location)
        current_session_active = True
        session_start_time = time.time()
        print(f"🚀 Session started: {session['id']}")
    
    # Jika detection DINONAKTIFKAN -> End session
    elif not detection_enabled and current_session_active:
        stats = counter.get_statistics()
        total_visitors = stats.get('daily_total', 0)
        max_concurrent = stats.get('max_count', 0)
        
        ended_session = sess_mgr.end_session(
            total_visitors=total_visitors,
            max_concurrent=max_concurrent,
            status='Selesai',
            notes='Manual stop by user'
        )
        current_session_active = False
        session_start_time = None
        
        if ended_session:
            print(f"✅ Session ended: {ended_session['id']}")
            print(f"   Total visitors: {total_visitors}")
    
    status = "ENABLED" if detection_enabled else "DISABLED"
    print(f"🎯 Face detection {status}")
    
    return jsonify({
        'success': True,
        'detection_enabled': detection_enabled,
        'session_active': current_session_active,
        'message': f'Face detection {status.lower()}'
    })

@app.route('/api/detection/status')
def detection_status():
    """Get current detection status"""
    global detection_enabled, current_session_active, session_start_time
    counter = get_counter()
    sess_mgr = get_session_manager()
    
    current_session = sess_mgr.get_current_session()
    
    # Calculate session duration
    session_duration_seconds = 0
    if session_start_time and current_session_active:
        session_duration_seconds = int(time.time() - session_start_time)
    
    return jsonify({
        'detection_enabled': detection_enabled,
        'is_running': counter.is_running,
        'stream_connected': counter.cap is not None and counter.cap.isOpened(),
        'session_active': current_session_active,
        'current_session': current_session,
        'session_duration_seconds': session_duration_seconds
    })

@app.route('/api/stream/reconnect', methods=['POST'])
def force_reconnect():
    """Force reconnect stream"""
    global counter
    
    try:
        counter = get_counter()
        print("🔄 Force reconnecting stream...")
        
        with stream_lock:
            counter._reconnect_stream()
        
        return jsonify({
            'success': True,
            'message': 'Stream reconnected successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Reconnect failed: {str(e)}'
        }), 500

@app.route('/api/stats')
def get_stats():
    """Get current statistics - INTEGRATED WITH SESSIONS"""
    counter = get_counter()
    sess_mgr = get_session_manager()
    
    stats = counter.get_statistics()
    
    # Get current session info
    current_session = sess_mgr.get_current_session()
    session_visitors = current_session['total_visitors'] if current_session else 0
    
    # Update session visitors jika ada session aktif
    if current_session_active and current_session:
        # Update total visitors dari counter ke session (realtime)
        session_visitors = stats.get('daily_total', 0)
    
    stats['database_faces'] = len(counter.face_db.faces)
    stats['detection_enabled'] = detection_enabled
    stats['session_active'] = current_session_active
    stats['session_visitors'] = session_visitors
    
    return jsonify({
        'current_count': stats.get('current_count', 0),
        'max_count': stats.get('max_count', 0),
        'daily_total': stats.get('daily_total', 0),
        'database_faces': stats.get('database_faces', 0),
        'fps': stats.get('fps', 0),
        'hourly_stats': stats.get('hourly_stats', {}),
        'timestamp': stats.get('timestamp', ''),
        'processing_fps': stats.get('processing_fps', 0),
        'active_ids': stats.get('active_trackers', 0),
        'current_faces': stats.get('current_faces', 0),
        'detection_enabled': detection_enabled,
        'session_active': current_session_active,
        'session_visitors': session_visitors
    })

@app.route('/api/reset', methods=['POST'])
def reset_stats():
    """Reset daily statistics"""
    counter = get_counter()
    counter.reset_daily_stats()
    return jsonify({
        'success': True, 
        'message': 'Daily statistics reset successfully'
    })

# ==================== SESSION MANAGEMENT ENDPOINTS ====================

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """Get all sessions"""
    try:
        sess_mgr = get_session_manager()
        counter = get_counter()
        
        # Get query parameters
        limit = request.args.get('limit', type=int)
        date_filter = request.args.get('date')
        
        sessions = sess_mgr.get_all_sessions(limit=limit, date_filter=date_filter)
        statistics = sess_mgr.get_statistics()
        
        # Update total visitors dengan data dari database faces
        database_faces = len(counter.face_db.faces)
        statistics['total_visitors_database'] = database_faces
        
        return jsonify({
            'success': True,
            'sessions': sessions,
            'statistics': statistics,
            'total': len(sessions),
            'database_faces': database_faces
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/sessions/<session_id>', methods=['GET'])
def get_session(session_id):
    """Get specific session by ID"""
    try:
        sess_mgr = get_session_manager()
        session = sess_mgr.get_session_by_id(session_id)
        
        if session:
            return jsonify({
                'success': True,
                'session': session
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Session not found'
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete session by ID"""
    try:
        sess_mgr = get_session_manager()
        success = sess_mgr.delete_session(session_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Session deleted successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Session not found'
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/sessions/export', methods=['GET'])
def export_sessions():
    """Export sessions to CSV"""
    try:
        sess_mgr = get_session_manager()
        
        # Create CSV in memory
        import csv
        from io import StringIO
        
        output = StringIO()
        sessions = sess_mgr.get_all_sessions()
        
        if not sessions:
            return jsonify({
                'success': False,
                'message': 'No sessions to export'
            }), 404
        
        fieldnames = ['No', 'Tanggal', 'Waktu Mulai', 'Waktu Selesai', 
                     'Total Pengunjung', 'Lokasi/Kamera', 'Durasi', 'Status', 'Catatan']
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for idx, session in enumerate(sessions, 1):
            writer.writerow({
                'No': idx,
                'Tanggal': session.get('date', ''),
                'Waktu Mulai': session.get('start_time', ''),
                'Waktu Selesai': session.get('end_time', ''),
                'Total Pengunjung': f"{session.get('total_visitors', 0)} Orang",
                'Lokasi/Kamera': session.get('camera_location', ''),
                'Durasi': session.get('duration_formatted', ''),
                'Status': session.get('status', ''),
                'Catatan': session.get('notes', '')
            })
        
        # Convert to bytes for download
        output.seek(0)
        
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename=data-pengunjung-{time.strftime("%Y%m%d")}.csv'
            }
        )
        
    except Exception as e:
        print(f"Export error: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/sessions/statistics', methods=['GET'])
def get_session_statistics():
    """Get session statistics"""
    try:
        sess_mgr = get_session_manager()
        counter = get_counter()
        
        statistics = sess_mgr.get_statistics()
        statistics['database_faces'] = len(counter.face_db.faces)
        
        return jsonify({
            'success': True,
            'statistics': statistics
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

# ==================== OTHER ENDPOINTS ====================

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    counter = get_counter()
    sess_mgr = get_session_manager()
    
    return jsonify({
        'status': 'ok',
        'service': 'Face Counter CCTV - Session Management',
        'running': counter.is_running,
        'cctv_connected': counter.cap is not None and counter.cap.isOpened(),
        'model': 'MTCNN + FaceNet',
        'fps': round(counter.fps, 1),
        'processing_fps': round(counter.processing_fps, 1),
        'active_trackers': len(counter.face_trackers),
        'detection_enabled': detection_enabled,
        'session_active': current_session_active,
        'total_sessions': len(sess_mgr.sessions),
        'database_faces': len(counter.face_db.faces)
    })

@app.route('/api/history')
def get_history():
    """Get historical data"""
    counter = get_counter()
    history = counter.get_historical_data()
    
    if 'peak_hours' not in history:
        history['peak_hours'] = []
    
    return jsonify(history)

@app.route('/api/database/stats')
def get_database_stats():
    """Get face database statistics"""
    counter = get_counter()
    return jsonify(counter.get_database_stats())

@app.route('/api/database/save', methods=['POST'])
def save_database():
    """Save face database"""
    counter = get_counter()
    counter.save_face_database()
    return jsonify({
        'success': True,
        'message': 'Database saved',
        'total_faces': len(counter.face_db.faces)
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    cli = sys.modules['flask.cli']
    cli.show_server_banner = lambda *x: None
    
    print("\n" + "="*70)
    print("🚀 FACE COUNTER - SESSION MANAGEMENT SYSTEM")
    print("="*70)
    print(f"📹 CCTV: {Config.CCTV_IP}")
    print(f"🎯 Model: MTCNN + FaceNet")
    print(f"📐 Resolution: {Config.FRAME_WIDTH}x{Config.FRAME_HEIGHT}")
    print("✅ Auto-reconnect: ENABLED")
    print("✅ Session tracking: ENABLED")
    print("✅ Data Pengunjung: ENABLED")
    print("✅ Auto-start session: ENABLED")
    print("="*70)
    print(f"\n🌐 Dashboard: http://localhost:{Config.PORT}")
    print(f"🌐 Data Pengunjung: http://localhost:{Config.PORT}/data-pengunjung")
    print(f"🌐 Network: http://{Config.HOST}:{Config.PORT}")
    print("="*70)
    print("\n⚠️  Press Ctrl+C to stop\n")
    
    # Initialize counter dan auto-start session
    counter = get_counter()
    time.sleep(2)  # Wait for counter to initialize
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
        print("\n\n⏸️  Stopping server...")
        
        # End current session if active
        if current_session_active and counter:
            sess_mgr = get_session_manager()
            stats = counter.get_statistics()
            sess_mgr.end_session(
                total_visitors=stats.get('daily_total', 0),
                max_concurrent=stats.get('max_count', 0),
                status='Selesai',
                notes='System shutdown'
            )
            print("💾 Current session saved")
        
        if counter:
            counter.stop()
        
        print("✅ Stopped")
        print("👋 Goodbye!\n")