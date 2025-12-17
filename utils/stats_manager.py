"""
Statistics management utilities for Face Counter
"""
import pickle
import os
from collections import defaultdict
from datetime import datetime, timedelta

from core.config import Config

class StatisticsManager:
    """Manages face detection statistics and historical data"""
    
    def __init__(self):
        self.people_count = 0  # Current faces in frame
        self.max_count = 0
        self.total_detected = 0  # Total unique faces
        self.hourly_stats = defaultdict(int)
        self.daily_history = []
        self.entry_times = []
    
    def update(self, current_count):
        """Update current face count statistics"""
        self.people_count = current_count
        
        if self.people_count > self.max_count:
            self.max_count = self.people_count
        
        # Update hourly stats
        current_hour = datetime.now().hour
        if self.people_count > 0:
            self.hourly_stats[current_hour] = max(
                self.hourly_stats[current_hour],
                self.people_count
            )
    
    def add_unique_person(self):
        """Increment unique face counter"""
        self.total_detected += 1
        self.entry_times.append(datetime.now())
    
    def get_stats(self):
        """Get current statistics"""
        return {
            'current_count': self.people_count,
            'max_count': self.max_count,
            'daily_total': self.total_detected,
            'hourly_stats': dict(self.hourly_stats)
        }
    
    def get_historical_data(self):
        """Get historical data for charts"""
        # Calculate average by hour
        avg_by_hour = defaultdict(list)
        
        # Peak times
        peak_hours = sorted(
            self.hourly_stats.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            'hourly_stats': dict(self.hourly_stats),
            'peak_hours': [{'hour': h, 'count': c} for h, c in peak_hours],
            'entry_distribution': self._get_entry_distribution(),
            'daily_trend': self.daily_history[-7:] if len(self.daily_history) > 0 else []
        }
    
    def _get_entry_distribution(self):
        """Get face detection time distribution"""
        if not self.entry_times:
            return {}
        
        distribution = defaultdict(int)
        for entry_time in self.entry_times:
            hour = entry_time.hour
            distribution[hour] += 1
        
        return dict(distribution)
    
    def reset_daily(self):
        """Reset daily statistics"""
        # Save to history before reset
        if self.total_detected > 0:
            self.daily_history.append({
                'date': datetime.now().strftime('%Y-%m-%d'),
                'total': self.total_detected,
                'max': self.max_count,
                'hourly': dict(self.hourly_stats)
            })
        
        self.max_count = 0
        self.hourly_stats.clear()
        self.total_detected = 0
        self.entry_times.clear()
        
        self.save_statistics()
    
    def save_statistics(self):
        """Save statistics to file"""
        data = {
            'max_count': self.max_count,
            'hourly_stats': dict(self.hourly_stats),
            'total_detected': self.total_detected,
            'daily_history': self.daily_history,
            'entry_times': [t.isoformat() for t in self.entry_times],
            'last_update': datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(Config.STATS_FILE), exist_ok=True)
        with open(Config.STATS_FILE, 'wb') as f:
            pickle.dump(data, f)
    
    def load_statistics(self):
        """Load statistics from file"""
        try:
            if os.path.exists(Config.STATS_FILE):
                with open(Config.STATS_FILE, 'rb') as f:
                    data = pickle.load(f)
                    self.max_count = data.get('max_count', 0)
                    self.hourly_stats = defaultdict(int, data.get('hourly_stats', {}))
                    self.total_detected = data.get('total_detected', 0)
                    self.daily_history = data.get('daily_history', [])
                    
                    # Load entry times
                    entry_times_iso = data.get('entry_times', [])
                    self.entry_times = [
                        datetime.fromisoformat(t) for t in entry_times_iso
                    ]
                    
                    print(f"✅ Statistics loaded - {self.total_detected} unique faces recorded")
        except Exception as e:
            print(f"⚠️  Could not load statistics: {e}")