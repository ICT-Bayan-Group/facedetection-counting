"""
FaceDatabase — SQLite, real-time insert, event callbacks.
Wajah baru langsung INSERT ke DB dan trigger callback ke Flask SSE.
"""
import os, sqlite3, threading, json, numpy as np
from datetime import timedelta
from typing import Callable, Optional

from utils.wita_time import now_wita, now_wita_iso  # TZ FIX — jangan pakai datetime.now() polos

DB_PATH = os.environ.get('FACE_DB_PATH', 'data/face_database.db')

def _normalize(a):
    n = np.linalg.norm(a); return a / n if n > 0 else a
def _emb_to_blob(a): return a.astype(np.float32).tobytes()
def _blob_to_emb(b): return np.frombuffer(b, dtype=np.float32)

class _FacesDictCompat:
    def __init__(self, db): self._db = db
    def __len__(self): return self._db._count()
    def __iter__(self):
        with self._db._conn() as c:
            for r in c.execute("SELECT id FROM faces"): yield r[0]

class FaceDatabase:
    SIMILARITY_THRESHOLD = 0.72

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.similarity_threshold = self.SIMILARITY_THRESHOLD
        self._lock = threading.Lock()
        self._running = True
        self._on_new_face_cbs:     list[Callable] = []
        self._on_updated_face_cbs: list[Callable] = []
        os.makedirs(os.path.dirname(self.db_path) or '.', exist_ok=True)
        self._init_db()
        print(f"✅ FaceDatabase (SQLite): {self._count()} wajah | {os.path.abspath(self.db_path)}")

    # ── setup ─────────────────────────────────────────────────────────────
    def _init_db(self):
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS faces (
                    id              TEXT PRIMARY KEY,
                    embedding       BLOB NOT NULL,
                    first_seen      TEXT NOT NULL,
                    last_seen       TEXT NOT NULL,
                    detection_count INTEGER NOT NULL DEFAULT 1,
                    thumbnail_b64   TEXT
                )""")
            c.execute("CREATE INDEX IF NOT EXISTS idx_fs ON faces(first_seen)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_ls ON faces(last_seen)")
            c.commit()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    # ── callbacks ─────────────────────────────────────────────────────────
    def register_callback(self, event: str, fn: Callable):
        """event: 'new_face' | 'face_updated'"""
        if event == 'new_face':      self._on_new_face_cbs.append(fn)
        elif event == 'face_updated': self._on_updated_face_cbs.append(fn)

    def _fire(self, cbs, face_id, meta):
        for fn in cbs:
            try: fn(face_id, meta)
            except Exception as e: print(f"⚠️  Callback error: {e}")

    # ── compat ────────────────────────────────────────────────────────────
    @property
    def faces(self): return _FacesDictCompat(self)

    # ── find ──────────────────────────────────────────────────────────────
    def find_matching_face(self, embedding) -> tuple:
        if embedding is None: return None, 0
        try: q = _normalize(np.array(embedding, dtype=np.float32))
        except: return None, 0
        best_id, best_sim = None, 0.0
        with self._lock:
            with self._conn() as c:
                for row in c.execute("SELECT id, embedding FROM faces"):
                    try:
                        sim = float(np.dot(q, _normalize(_blob_to_emb(row['embedding']))))
                        if sim > best_sim: best_sim, best_id = sim, row['id']
                    except: continue
        return (best_id, best_sim) if best_sim >= self.similarity_threshold else (None, 0)

    # ── add / update ──────────────────────────────────────────────────────
    def add_or_update_face(self, face_id: str, embedding,
                        thumbnail_b64: Optional[str] = None) -> tuple:
        if embedding is None: return False, None, 0
        try:
            emb = np.array(embedding, dtype=np.float32)
            if emb.size == 0 or np.isnan(emb).any(): return False, None, 0
        except: return False, None, 0

        matched_id, similarity = self.find_matching_face(emb)
        now = now_wita_iso()  # TZ FIX — sebelumnya datetime.now().isoformat() (naive, ikut TZ OS)

        with self._lock:
            with self._conn() as c:
                if matched_id:
                    # Wajah sudah ada di DB — update embedding
                    row = c.execute(
                        "SELECT embedding, detection_count FROM faces WHERE id=?",
                        (matched_id,)).fetchone()
                    if row:
                        merged = _normalize(0.9 * _blob_to_emb(row['embedding']) + 0.1 * emb)
                        c.execute(
                            "UPDATE faces SET embedding=?, last_seen=?, detection_count=? WHERE id=?",
                            (_emb_to_blob(merged), now, row['detection_count']+1, matched_id))
                        c.commit()
                    meta = self._row_to_meta(c, matched_id)
                    threading.Thread(target=self._fire,
                        args=(self._on_updated_face_cbs, matched_id, meta), daemon=True).start()
                    return False, matched_id, similarity

                else:
                    # Wajah baru — INSERT OR IGNORE untuk mencegah race condition
                    norm = _normalize(emb)
                    c.execute(
                        "INSERT OR IGNORE INTO faces "
                        "(id,embedding,first_seen,last_seen,detection_count,thumbnail_b64) "
                        "VALUES (?,?,?,?,1,?)",
                        (face_id, _emb_to_blob(norm), now, now, thumbnail_b64))
                    c.commit()

                    if c.execute("SELECT changes()").fetchone()[0] == 0:
                        # ID bentrok (race condition) — perlakukan sebagai update
                        row = c.execute(
                            "SELECT embedding, detection_count FROM faces WHERE id=?",
                            (face_id,)).fetchone()
                        if row:
                            merged = _normalize(0.9 * _blob_to_emb(row['embedding']) + 0.1 * norm)
                            c.execute(
                                "UPDATE faces SET embedding=?, last_seen=?, detection_count=? WHERE id=?",
                                (_emb_to_blob(merged), now, row['detection_count']+1, face_id))
                            c.commit()
                        meta = self._row_to_meta(c, face_id)
                        threading.Thread(target=self._fire,
                            args=(self._on_updated_face_cbs, face_id, meta), daemon=True).start()
                        return False, face_id, 1.0

                    total = c.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
                    meta = {
                        'id': face_id, 'first_seen': now, 'last_seen': now,
                        'detection_count': 1, 'thumbnail_b64': thumbnail_b64,
                        'total_in_db': total,
                    }
                    print(f"✨ WAJAH BARU → DB: {face_id} (total={total})")
                    threading.Thread(target=self._fire,
                        args=(self._on_new_face_cbs, face_id, meta), daemon=True).start()
                    return True, face_id, 1.0

    def _row_to_meta(self, conn, face_id):
        r = conn.execute(
            "SELECT id,first_seen,last_seen,detection_count FROM faces WHERE id=?",
            (face_id,)).fetchone()
        return dict(r) if r else {'id': face_id}

    # ── public read ───────────────────────────────────────────────────────
    def get_all_faces_meta(self, limit=1000) -> list:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id,first_seen,last_seen,detection_count,thumbnail_b64 "
                "FROM faces ORDER BY first_seen DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_recent_faces(self, n=10) -> list:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id,first_seen,last_seen,detection_count,thumbnail_b64 "
                "FROM faces ORDER BY first_seen DESC LIMIT ?", (n,)).fetchall()
        return [dict(r) for r in rows]

    def get_face_info(self, face_id) -> Optional[dict]:
        with self._conn() as c:
            r = c.execute(
                "SELECT id,first_seen,last_seen,detection_count FROM faces WHERE id=?",
                (face_id,)).fetchone()
        return dict(r) if r else None

    def get_statistics(self) -> dict:
        with self._conn() as c:
            total = c.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
            if total == 0: return {'total_faces': 0}
            oldest = dict(c.execute(
                "SELECT id,first_seen,detection_count FROM faces ORDER BY first_seen ASC LIMIT 1"
            ).fetchone())
            newest = dict(c.execute(
                "SELECT id,first_seen FROM faces ORDER BY first_seen DESC LIMIT 1"
            ).fetchone())
            most = dict(c.execute(
                "SELECT id,detection_count FROM faces ORDER BY detection_count DESC LIMIT 1"
            ).fetchone())
        return {'total_faces': total, 'oldest_face': oldest,
                'newest_face': newest, 'most_detected': most}

    # ── management ────────────────────────────────────────────────────────
    def remove_old_faces(self, days=30) -> int:
        # TZ FIX — sebelumnya datetime.now() - timedelta(days=days) (naive)
        cutoff = (now_wita() - timedelta(days=days)).isoformat()
        with self._lock:
            with self._conn() as c:
                n = c.execute("DELETE FROM faces WHERE last_seen < ?", (cutoff,)).rowcount
                c.commit()
        if n: print(f"🗑️  Removed {n} old faces")
        return n

    def reset_database(self):
        with self._lock:
            with self._conn() as c:
                c.execute("DELETE FROM faces"); c.commit()
        print("🔄 Face database reset")

    def _count(self) -> int:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM faces").fetchone()[0]

    def migrate_from_json(self, json_path='data/face_database.json') -> int:
        if not os.path.exists(json_path):
            print(f"⚠️  JSON tidak ditemukan: {json_path}"); return 0
        with open(json_path) as f: data = json.load(f)
        migrated = skipped = 0
        with self._lock:
            with self._conn() as c:
                for fid, fd in data.items():
                    if c.execute("SELECT 1 FROM faces WHERE id=?", (fid,)).fetchone():
                        skipped += 1; continue
                    try:
                        norm = _normalize(np.array(fd['embedding'], dtype=np.float32))
                        c.execute(
                            "INSERT INTO faces (id,embedding,first_seen,last_seen,detection_count) "
                            "VALUES (?,?,?,?,?)",
                            (fid, _emb_to_blob(norm),
                             fd.get('first_seen', now_wita_iso()),
                             fd.get('last_seen',  now_wita_iso()),
                             fd.get('detection_count', 1)))
                        migrated += 1
                    except Exception as e: print(f"⚠️  Skip {fid}: {e}")
                c.commit()
        print(f"✅ Migrasi: {migrated} diimpor, {skipped} sudah ada")
        return migrated

    # ── compat shims ──────────────────────────────────────────────────────
    def save_database(self, force=False): pass
    def load_database(self): pass
    def shutdown(self): self._running = False; print("💾 FaceDatabase shutdown")