import json
import os
from datetime import datetime
from pathlib import Path
import threading

class SessionManager:

    def __init__(self, sessions_file='data/sessions.json'):
        self.sessions_file = sessions_file
        self.sessions = []
        self.current_session = None
        self._lock = threading.Lock()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.sessions_file), exist_ok=True)
        
        # Load existing sessions
        self.load_sessions()
        
        print(f"✅ Session Manager initialized")
        print(f"📁 Sessions file: {os.path.abspath(self.sessions_file)}")
        print(f"📊 Loaded {len(self.sessions)} previous sessions")
    
    def load_sessions(self):
        """Load sessions dari JSON file"""
        try:
            if os.path.exists(self.sessions_file):
                with open(self.sessions_file, 'r') as f:
                    self.sessions = json.load(f)
                print(f"✅ Loaded {len(self.sessions)} sessions from database")
            else:
                self.sessions = []
                self.save_sessions()
                print("📁 Created new sessions database")
        except Exception as e:
            print(f"❌ Error loading sessions: {e}")
            self.sessions = []
    
    def save_sessions(self):
        """Save sessions ke JSON file"""
        try:
            with self._lock:
                # Write to temporary file first
                temp_path = self.sessions_file + '.tmp'
                with open(temp_path, 'w') as f:
                    json.dump(self.sessions, f, indent=2)
                
                # Atomic replace
                os.replace(temp_path, self.sessions_file)
                
                return True
        except Exception as e:
            print(f"❌ Error saving sessions: {e}")
            return False
    
    def start_session(self, camera_location="CCTV Hall A"):
        """
        Mulai sesi deteksi baru
        """
        with self._lock:
            self.current_session = {
                'id': self._generate_session_id(),
                'date': datetime.now().strftime('%Y-%m-%d'),
                'start_time': datetime.now().strftime('%H:%M:%S'),
                'start_timestamp': datetime.now().isoformat(),
                'end_time': None,
                'end_timestamp': None,
                'total_visitors': 0,
                'camera_location': camera_location,
                'duration_seconds': 0,
                'duration_formatted': '0 menit',
                'status': 'Running',
                'max_concurrent': 0,
                'notes': ''
            }
            
            print(f"🚀 Session started: {self.current_session['id']}")
            print(f"   Location: {camera_location}")
            print(f"   Start time: {self.current_session['start_time']}")
            
            return self.current_session
    
    def end_session(self, total_visitors, max_concurrent=0, status='Selesai', notes=''):
        """
        Akhiri sesi deteksi dan simpan ke database
        
        Args:
            total_visitors: Jumlah unique visitors yang terdeteksi
            max_concurrent: Maksimum concurrent faces
            status: 'Selesai' atau 'Gagal'
            notes: Catatan tambahan
        """
        if self.current_session is None:
            print("⚠️ No active session to end")
            return None
        
        with self._lock:
            end_time = datetime.now()
            start_timestamp = datetime.fromisoformat(self.current_session['start_timestamp'])
            
            # Calculate duration
            duration = end_time - start_timestamp
            duration_seconds = int(duration.total_seconds())
            
            # Format duration
            hours = duration_seconds // 3600
            minutes = (duration_seconds % 3600) // 60
            
            if hours > 0:
                duration_formatted = f"{hours} Jam {minutes} Menit"
            else:
                duration_formatted = f"{minutes} Menit"
            
            # Update session data
            self.current_session.update({
                'end_time': end_time.strftime('%H:%M:%S'),
                'end_timestamp': end_time.isoformat(),
                'total_visitors': total_visitors,
                'max_concurrent': max_concurrent,
                'duration_seconds': duration_seconds,
                'duration_formatted': duration_formatted,
                'status': status,
                'notes': notes
            })
            
            # Add to sessions list (newest first)
            self.sessions.insert(0, self.current_session)
            
            # Save to file
            self.save_sessions()
            
            print(f"✅ Session ended: {self.current_session['id']}")
            print(f"   Duration: {duration_formatted}")
            print(f"   Total visitors: {total_visitors}")
            print(f"   Status: {status}")
            
            ended_session = self.current_session.copy()
            self.current_session = None
            
            return ended_session
    
    def get_current_session(self):
        """Get current active session"""
        return self.current_session
    
    def get_all_sessions(self, limit=None, date_filter=None):
        """
        Get all sessions
        
        Args:
            limit: Limit number of results
            date_filter: Filter by date (YYYY-MM-DD)
        """
        with self._lock:
            sessions = self.sessions.copy()
            
            # Filter by date if provided
            if date_filter:
                sessions = [s for s in sessions if s['date'] == date_filter]
            
            # Apply limit
            if limit:
                sessions = sessions[:limit]
            
            return sessions
    
    def get_session_by_id(self, session_id):
        """Get session by ID"""
        with self._lock:
            for session in self.sessions:
                if session['id'] == session_id:
                    return session
            return None
    
    def delete_session(self, session_id):
        """Delete session by ID"""
        with self._lock:
            original_count = len(self.sessions)
            self.sessions = [s for s in self.sessions if s['id'] != session_id]
            
            if len(self.sessions) < original_count:
                self.save_sessions()
                print(f"🗑️ Session deleted: {session_id}")
                return True
            
            return False
    
    def get_statistics(self):
        """Get session statistics"""
        with self._lock:
            if not self.sessions:
                return {
                    'total_sessions': 0,
                    'total_visitors_all_time': 0,
                    'average_visitors_per_session': 0,
                    'average_duration_minutes': 0,
                    'success_rate': 0
                }
            
            completed_sessions = [s for s in self.sessions if s['status'] == 'Selesai']
            
            total_visitors = sum(s['total_visitors'] for s in self.sessions)
            total_duration = sum(s['duration_seconds'] for s in self.sessions)
            
            return {
                'total_sessions': len(self.sessions),
                'completed_sessions': len(completed_sessions),
                'total_visitors_all_time': total_visitors,
                'average_visitors_per_session': round(total_visitors / len(self.sessions), 1) if self.sessions else 0,
                'average_duration_minutes': round(total_duration / 60 / len(self.sessions), 1) if self.sessions else 0,
                'success_rate': round(len(completed_sessions) / len(self.sessions) * 100, 1) if self.sessions else 0
            }
    
    def _generate_session_id(self):
        """Generate unique session ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"SESSION_{timestamp}"
    
    def export_to_csv(self, output_path='data/sessions_export.csv'):
        """Export sessions to CSV"""
        try:
            import csv
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                if not self.sessions:
                    return False
                
                fieldnames = ['id', 'date', 'start_time', 'end_time', 'total_visitors', 
                             'camera_location', 'duration_formatted', 'status', 'notes']
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for session in self.sessions:
                    writer.writerow({k: session.get(k, '') for k in fieldnames})
            
            print(f"📊 Sessions exported to: {os.path.abspath(output_path)}")
            return True
            
        except Exception as e:
            print(f"❌ Error exporting sessions: {e}")
            return False