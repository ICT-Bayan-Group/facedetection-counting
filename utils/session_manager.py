"""
Session Manager
"""
import json
import os
from datetime import datetime
import uuid

class SessionManager:
    """Manages detection sessions"""
    
    def __init__(self, sessions_file='data/sessions.json'):
        self.sessions_file = sessions_file
        self.sessions = []
        self.current_session = None
        self.load_sessions()
    
    def load_sessions(self):
        """Load sessions from file"""
        try:
            if os.path.exists(self.sessions_file):
                with open(self.sessions_file, 'r') as f:
                    self.sessions = json.load(f)
        except Exception as e:
            print(f"Failed to load sessions: {e}")
            self.sessions = []
    
    def save_sessions(self):
        """Save sessions to file"""
        try:
            os.makedirs('data', exist_ok=True)
            with open(self.sessions_file, 'w') as f:
                json.dump(self.sessions, f, indent=2)
        except Exception as e:
            print(f"Failed to save sessions: {e}")
    
    def start_session(self, camera_location='CCTV Hall A'):
        """Start new session"""
        self.current_session = {
            'id': str(uuid.uuid4())[:8],
            'camera_location': camera_location,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'total_visitors': 0,
            'max_concurrent': 0,
            'status': 'Active'
        }
        return self.current_session
    
    def end_session(self, total_visitors=0, max_concurrent=0, status='Completed', notes=''):
        """End current session"""
        if self.current_session:
            self.current_session.update({
                'end_time': datetime.now().isoformat(),
                'total_visitors': total_visitors,
                'max_concurrent': max_concurrent,
                'status': status,
                'notes': notes
            })
            self.sessions.append(self.current_session)
            self.save_sessions()
            session = self.current_session
            self.current_session = None
            return session
        return None
    
    def get_all_sessions(self):
        """Get all sessions"""
        return self.sessions
    
    def get_current_session(self):
        """Get current active session"""
        return self.current_session