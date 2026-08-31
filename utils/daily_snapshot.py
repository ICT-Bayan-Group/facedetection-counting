import os
import sqlite3
import threading
from typing import Optional

from utils.wita_time import now_wita_iso  # TZ FIX — jangan pakai datetime.now() polos

SNAPSHOT_DB_PATH = os.environ.get('SNAPSHOT_DB_PATH', 'data/daily_snapshots.db')


class DailySnapshotDB:
    """Thread-safe SQLite store untuk snapshot harian."""

    def __init__(self, db_path: str = SNAPSHOT_DB_PATH):
        self.db_path = db_path
        self._lock   = threading.Lock()
        os.makedirs(os.path.dirname(self.db_path) or '.', exist_ok=True)
        self._init_db()
        print(f"✅ DailySnapshotDB: {os.path.abspath(self.db_path)}")

    # ── setup ──────────────────────────────────────────────────────────────

    def _init_db(self):
        with self._conn() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS daily_summary (
                    date              TEXT PRIMARY KEY,   -- YYYY-MM-DD
                    total_visitors    INTEGER NOT NULL DEFAULT 0,
                    max_concurrent    INTEGER NOT NULL DEFAULT 0,
                    raw_detections    INTEGER NOT NULL DEFAULT 0,  -- total "terdeteksi manusia" (raw) hari itu
                    session_count     INTEGER NOT NULL DEFAULT 0,
                    first_detection   TEXT,               -- ISO datetime
                    last_detection    TEXT,               -- ISO datetime
                    detection_method  TEXT,
                    notes             TEXT,
                    created_at        TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS daily_face_log (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    log_date        TEXT NOT NULL,        -- YYYY-MM-DD
                    face_id         TEXT NOT NULL,
                    first_seen      TEXT NOT NULL,        -- ISO datetime
                    last_seen       TEXT NOT NULL,        -- ISO datetime
                    detection_count INTEGER NOT NULL DEFAULT 1,
                    camera_label    TEXT,
                    thumbnail_b64   TEXT,
                    UNIQUE(log_date, face_id)
                );

                CREATE INDEX IF NOT EXISTS idx_dfl_date ON daily_face_log(log_date);
                CREATE INDEX IF NOT EXISTS idx_ds_date  ON daily_summary(date);
            """)

            # MIGRATION: DB lama (sebelum kolom raw_detections ada) —
            # tambahkan kolomnya biar gak error di install existing.
            cols = [row[1] for row in c.execute("PRAGMA table_info(daily_summary)").fetchall()]
            if 'raw_detections' not in cols:
                c.execute("ALTER TABLE daily_summary ADD COLUMN raw_detections INTEGER NOT NULL DEFAULT 0")
                c.commit()
                print("🔧 Migrasi DB: kolom raw_detections ditambahkan ke daily_summary")

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    # ── write ──────────────────────────────────────────────────────────────

    def save_daily_snapshot(
        self,
        snapshot_date:    str,          # "YYYY-MM-DD"
        total_visitors:   int,
        max_concurrent:   int,
        raw_detections:   int           = 0,   # BARU — total "terdeteksi manusia" (raw, sebelum dedup) hari itu
        session_count:    int           = 0,
        first_detection:  Optional[str] = None,
        last_detection:   Optional[str] = None,
        detection_method: Optional[str] = None,
        notes:            Optional[str] = None,
    ) -> bool:
        """
        INSERT OR REPLACE ringkasan harian.
        Jika tanggal sudah ada, nilai total_visitors/max_concurrent/raw_detections
        diambil yang LEBIH BESAR (supaya re-save di hari yang sama tidak
        menghapus/mengecilkan angka yang sudah lebih tinggi).
        """
        now = now_wita_iso()  # TZ FIX — sebelumnya datetime.now().isoformat() (naive)
        try:
            with self._lock:
                with self._conn() as c:
                    existing = c.execute(
                        "SELECT total_visitors, max_concurrent, raw_detections "
                        "FROM daily_summary WHERE date=?",
                        (snapshot_date,)
                    ).fetchone()

                    if existing:
                        tv = max(existing['total_visitors'], total_visitors)
                        mc = max(existing['max_concurrent'],  max_concurrent)
                        rd = max(existing['raw_detections'] or 0, raw_detections or 0)
                        c.execute("""
                            UPDATE daily_summary
                               SET total_visitors  = ?,
                                   max_concurrent  = ?,
                                   raw_detections  = ?,
                                   session_count   = ?,
                                   last_detection  = ?,
                                   detection_method= ?,
                                   notes           = ?
                             WHERE date = ?
                        """, (tv, mc, rd, session_count, last_detection or now,
                              detection_method, notes, snapshot_date))
                    else:
                        c.execute("""
                            INSERT INTO daily_summary
                              (date, total_visitors, max_concurrent, raw_detections, session_count,
                               first_detection, last_detection, detection_method, notes, created_at)
                            VALUES (?,?,?,?,?,?,?,?,?,?)
                        """, (snapshot_date, total_visitors, max_concurrent, raw_detections or 0,
                              session_count, first_detection or now,
                              last_detection or now, detection_method, notes, now))
                    c.commit()
            return True
        except Exception as e:
            print(f"❌ save_daily_snapshot error: {e}")
            return False

    def log_faces_for_date(
        self,
        log_date:  str,           # "YYYY-MM-DD"
        faces:     list[dict],    # list of face metadata dicts
        camera_label: str = "",
    ) -> int:
        """
        Bulk-INSERT wajah unik ke daily_face_log.
        faces: [{'id', 'first_seen', 'last_seen', 'detection_count', 'thumbnail_b64'}, ...]
        Returns: jumlah baris baru yang dimasukkan.
        """
        if not faces:
            return 0
        inserted = 0
        try:
            with self._lock:
                with self._conn() as c:
                    for f in faces:
                        try:
                            c.execute("""
                                INSERT OR IGNORE INTO daily_face_log
                                  (log_date, face_id, first_seen, last_seen,
                                   detection_count, camera_label, thumbnail_b64)
                                VALUES (?,?,?,?,?,?,?)
                            """, (
                                log_date,
                                f.get('id', ''),
                                f.get('first_seen', now_wita_iso()),  # TZ FIX
                                f.get('last_seen',  now_wita_iso()),  # TZ FIX
                                f.get('detection_count', 1),
                                camera_label,
                                f.get('thumbnail_b64'),
                            ))
                            if c.execute("SELECT changes()").fetchone()[0]:
                                inserted += 1
                        except Exception as inner_e:
                            print(f"⚠️  log_face skip {f.get('id')}: {inner_e}")
                    c.commit()
        except Exception as e:
            print(f"❌ log_faces_for_date error: {e}")
        return inserted

    # ── read ───────────────────────────────────────────────────────────────

    def get_all_daily_summaries(self, limit: int = 90) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("""
                SELECT ds.*,
                       (SELECT COUNT(*) FROM daily_face_log dfl WHERE dfl.log_date = ds.date) AS face_count
                  FROM daily_summary ds
                 ORDER BY ds.date DESC
                 LIMIT ?
            """, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_summary_for_date(self, query_date: str) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM daily_summary WHERE date=?", (query_date,)
            ).fetchone()
        return dict(row) if row else None

    def get_faces_for_date(self, query_date: str, limit: int = 1000) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("""
                SELECT * FROM daily_face_log
                 WHERE log_date = ?
                 ORDER BY first_seen ASC
                 LIMIT ?
            """, (query_date, limit)).fetchall()
        return [dict(r) for r in rows]

    def get_date_range_summary(self, start_date: str, end_date: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("""
                SELECT * FROM daily_summary
                 WHERE date BETWEEN ? AND ?
                 ORDER BY date DESC
            """, (start_date, end_date)).fetchall()
        return [dict(r) for r in rows]

    def get_overall_stats(self) -> dict:
        with self._conn() as c:
            row = c.execute("""
                SELECT COUNT(*)               AS total_days,
                       SUM(total_visitors)    AS grand_total,
                       MAX(max_concurrent)    AS all_time_max,
                       AVG(total_visitors)    AS avg_per_day,
                       SUM(raw_detections)    AS grand_total_raw_detections,
                       AVG(raw_detections)    AS avg_raw_detections_per_day,
                       MIN(date)              AS oldest_date,
                       MAX(date)              AS newest_date
                  FROM daily_summary
            """).fetchone()
            face_count = c.execute("SELECT COUNT(*) FROM daily_face_log").fetchone()[0]
        d = dict(row) if row else {}
        d['total_face_logs'] = face_count
        return d

    def get_available_dates(self) -> list[str]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT date FROM daily_summary ORDER BY date DESC"
            ).fetchall()
        return [r['date'] for r in rows]