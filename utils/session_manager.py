"""
SessionManager — SQLite backend, API lengkap untuk endpoint publik.
"""
import os, sqlite3, json, uuid
from datetime import datetime
from typing import Optional

DB_PATH = os.environ.get('SESSION_DB_PATH', 'data/sessions.db')


class SessionManager:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path         = db_path
        self.current_session = None
        os.makedirs(os.path.dirname(self.db_path) or '.', exist_ok=True)
        self._init_db()
        print(f"✅ SessionManager (SQLite): {self._count()} sesi")

    def _init_db(self):
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id              TEXT PRIMARY KEY,
                    camera_location TEXT,
                    start_time      TEXT NOT NULL,
                    end_time        TEXT,
                    total_visitors  INTEGER DEFAULT 0,
                    max_concurrent  INTEGER DEFAULT 0,
                    status          TEXT DEFAULT 'Active',
                    notes           TEXT DEFAULT ''
                )""")
            c.execute("CREATE INDEX IF NOT EXISTS idx_start  ON sessions(start_time)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_status ON sessions(status)")
            c.commit()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ── lifecycle ─────────────────────────────────────────────────────────
    def start_session(self, camera_location='CCTV Hall A') -> dict:
        sid = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()
        self.current_session = {
            'id': sid, 'camera_location': camera_location,
            'start_time': now, 'end_time': None,
            'total_visitors': 0, 'max_concurrent': 0,
            'status': 'Active', 'notes': '',
        }
        with self._conn() as c:
            c.execute(
                "INSERT INTO sessions (id,camera_location,start_time,status) VALUES (?,?,?,'Active')",
                (sid, camera_location, now))
            c.commit()
        print(f"🚀 Session started: {sid} | {camera_location}")
        return self.current_session

    def end_all_running_sessions(self, total_visitors=0, status='Interrupted', notes='') -> int:
        now = datetime.now().isoformat()
        with self._conn() as c:
            n = c.execute(
                "UPDATE sessions SET end_time=?, total_visitors=?, status=?, notes=? "
                "WHERE status='Active' AND end_time IS NULL",
                (now, total_visitors, status, notes)
            ).rowcount
            c.commit()
        if n > 0:
            print(f"⏹️  Closed {n} stale running session(s)")
        return n

    def end_session(self, total_visitors=0, max_concurrent=0,
                    status='Selesai', notes='') -> Optional[dict]:
        if not self.current_session: return None
        now = datetime.now().isoformat()
        sid = self.current_session['id']
        self.current_session.update({
            'end_time': now, 'total_visitors': total_visitors,
            'max_concurrent': max_concurrent, 'status': status, 'notes': notes,
        })
        with self._conn() as c:
            c.execute(
                "UPDATE sessions SET end_time=?,total_visitors=?,max_concurrent=?,status=?,notes=? WHERE id=?",
                (now, total_visitors, max_concurrent, status, notes, sid))
            c.commit()
        print(f"⏹️  Session ended: {sid} | visitors={total_visitors}")
        s = self.current_session
        self.current_session = None
        return s

    # ── queries ───────────────────────────────────────────────────────────
    def get_all_sessions(self, limit=500) -> list:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM sessions ORDER BY start_time DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_sessions_by_date(self, date_str: str) -> list:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM sessions WHERE start_time LIKE ? ORDER BY start_time DESC",
                (f"{date_str}%",)).fetchall()
        return [dict(r) for r in rows]

    def get_sessions_by_location(self, location: str) -> list:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM sessions WHERE camera_location=? ORDER BY start_time DESC",
                (location,)).fetchall()
        return [dict(r) for r in rows]

    def delete_session(self, session_id: str) -> bool:
        with self._conn() as c:
            n = c.execute("DELETE FROM sessions WHERE id=?", (session_id,)).rowcount
            c.commit()
        return n > 0

    def get_summary(self) -> dict:
        today = datetime.now().strftime('%Y-%m-%d')
        with self._conn() as c:
            total    = c.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            finished = c.execute(
                "SELECT SUM(total_visitors), MAX(max_concurrent), COUNT(*) "
                "FROM sessions WHERE status != 'Active'").fetchone()
            today_ct = c.execute(
                "SELECT COUNT(*), SUM(total_visitors) FROM sessions WHERE start_time LIKE ?",
                (f"{today}%",)).fetchone()
            avg_dur  = c.execute("""
                SELECT AVG((julianday(end_time) - julianday(start_time)) * 1440)
                FROM sessions WHERE end_time IS NOT NULL
            """).fetchone()[0]
        total_vis = finished[0] or 0
        total_ses = finished[2] or 0
        return {
            'total_sessions':              total,
            'total_visitors_all_time':     total_vis,
            'max_concurrent_ever':         finished[1] or 0,
            'average_visitors_per_session': round(total_vis / total_ses, 1) if total_ses else 0,
            'average_duration_minutes':    round(avg_dur, 1) if avg_dur else 0,
            'sessions_today':              today_ct[0] or 0,
            'visitors_today':              today_ct[1] or 0,
        }

    def get_current_session(self): return self.current_session

    def export_csv(self) -> str:
        """Return CSV string untuk semua sessions."""
        rows = self.get_all_sessions(limit=10000)
        lines = ['id,camera_location,start_time,end_time,total_visitors,max_concurrent,status,notes']
        for r in rows:
            lines.append(','.join(str(r.get(k,'')) for k in
                ['id','camera_location','start_time','end_time',
                 'total_visitors','max_concurrent','status','notes']))
        return '\n'.join(lines)

    def _count(self) -> int:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

    def migrate_from_json(self, json_path='data/sessions.json') -> int:
        if not os.path.exists(json_path):
            print(f"⚠️  JSON tidak ditemukan: {json_path}"); return 0
        with open(json_path) as f: sessions = json.load(f)
        migrated = skipped = 0
        with self._conn() as c:
            for s in sessions:
                sid = s.get('id', str(uuid.uuid4())[:8])
                if c.execute("SELECT 1 FROM sessions WHERE id=?", (sid,)).fetchone():
                    skipped += 1; continue
                try:
                    c.execute(
                        "INSERT INTO sessions (id,camera_location,start_time,end_time,"
                        "total_visitors,max_concurrent,status,notes) VALUES (?,?,?,?,?,?,?,?)",
                        (sid, s.get('camera_location',''),
                         s.get('start_time', datetime.now().isoformat()),
                         s.get('end_time'), s.get('total_visitors',0),
                         s.get('max_concurrent',0), s.get('status','Selesai'),
                         s.get('notes','')))
                    migrated += 1
                except Exception as e: print(f"⚠️  Skip {sid}: {e}")
            c.commit()
        print(f"✅ Migrasi sessions: {migrated} diimpor, {skipped} sudah ada")
        return migrated

    # ── compat shims ──────────────────────────────────────────────────────
    def load_sessions(self): pass
    def save_sessions(self): pass