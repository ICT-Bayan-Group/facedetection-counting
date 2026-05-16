"""
MidnightResetScheduler
======================
- Memantau waktu sistem (zona waktu WITA = UTC+8)
- Tepat pukul 00:00:00 WITA → simpan snapshot harian → reset counter
- Mendaftarkan signal-handler (SIGINT / SIGTERM) agar data tersimpan
  ketika server dimatikan kapan saja

FIX v2:
- Reset counter via method reset_daily_stats() yang thread-safe
- Tidak lagi akses sm.* langsung dari luar (race condition)
- Tidak reset face_db — wajah dipertahankan untuk anti-duplicate lintas hari
"""

import signal
import sys
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Callable

WITA = timezone(timedelta(hours=8))


def now_wita() -> datetime:
    return datetime.now(WITA)


class MidnightResetScheduler:
    """
    Parameters
    ----------
    get_counters       : callable → list[OpenVINOFaceCounter | None]
    snapshot_db        : DailySnapshotDB instance
    get_session_manager: callable → SessionManager instance
    on_reset_done      : optional callback dipanggil setelah reset selesai
    """

    def __init__(
        self,
        get_counters:        Callable,
        snapshot_db,
        get_session_manager: Callable,
        on_reset_done:       Callable | None = None,
    ):
        self._get_counters        = get_counters
        self._snapshot_db         = snapshot_db
        self._get_session_manager = get_session_manager
        self._on_reset_done       = on_reset_done
        self._running             = False
        self._thread              = None
        self._shutdown_lock       = threading.Lock()
        self._shutdown_done       = False

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="MidnightReset"
        )
        self._thread.start()
        self._register_signals()
        print("⏰ MidnightResetScheduler started (WITA timezone)")

    def stop(self):
        self._running = False

    # ── main loop ──────────────────────────────────────────────────────────

    def _loop(self):
        """
        Tidur sampai 1 menit sebelum tengah malam,
        lalu polling setiap detik hingga jam 00:00.
        """
        while self._running:
            try:
                now = now_wita()
                seconds_until_midnight = self._seconds_until_midnight(now)

                if seconds_until_midnight > 60:
                    sleep_time = seconds_until_midnight - 60
                    time.sleep(min(sleep_time, 300))
                    continue

                now = now_wita()
                h, m, s = now.hour, now.minute, now.second

                if h == 0 and m == 0 and s <= 5:
                    print(f"\n🕛 MIDNIGHT RESET TRIGGERED — {now.strftime('%Y-%m-%d %H:%M:%S')} WITA")
                    self._save_snapshot_and_reset()
                    time.sleep(10)
                else:
                    time.sleep(1)

            except Exception as e:
                print(f"❌ MidnightResetScheduler error: {e}")
                time.sleep(5)

    @staticmethod
    def _seconds_until_midnight(now: datetime) -> float:
        tomorrow_midnight = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return (tomorrow_midnight - now).total_seconds()

    # ── core: simpan snapshot lalu reset ──────────────────────────────────

    def _save_snapshot_and_reset(self, notes: str = "Auto midnight reset"):
        """
        1. Kumpulkan stats dari semua counter yang aktif
        2. Simpan daily_summary ke snapshot DB
        3. Simpan semua wajah unik hari ini ke daily_face_log
        4. End sesi yang masih Running
        5. Reset counter via reset_daily_stats() yang thread-safe
        """
        snap_date = (now_wita() - timedelta(days=1)).strftime('%Y-%m-%d')

        counters = [c for c in self._get_counters() if c is not None]
        if not counters:
            print("⚠️  No active counters, skip snapshot")
            return

        # ── 1. Kumpulkan statistik ─────────────────────────────────────
        total_visitors   = 0
        max_concurrent   = 0
        detection_method = ""
        first_detection  = None
        last_detection   = None

        for c in counters:
            try:
                stats = c.get_statistics()
                total_visitors  += stats.get('daily_total',  0)
                max_concurrent   = max(max_concurrent, stats.get('max_count', 0))
                if not detection_method:
                    detection_method = stats.get('detection_method', '')
                ts = stats.get('timestamp', '')
                if ts:
                    if not last_detection or ts > last_detection:
                        last_detection = ts
            except Exception as e:
                print(f"⚠️  stats error: {e}")

        print(f"   📊 Snapshot {snap_date}: visitors={total_visitors}, max={max_concurrent}")

        # ── 2. Simpan daily_summary ────────────────────────────────────
        try:
            mgr = self._get_session_manager()
            sc  = mgr.get_summary().get('total_sessions', 0) if mgr else 0
        except Exception:
            sc = 0

        self._snapshot_db.save_daily_snapshot(
            snapshot_date    = snap_date,
            total_visitors   = total_visitors,
            max_concurrent   = max_concurrent,
            session_count    = sc,
            first_detection  = first_detection,
            last_detection   = last_detection,
            detection_method = detection_method,
            notes            = notes,
        )

        # ── 3. Simpan wajah unik ──────────────────────────────────────
        try:
            face_db = counters[0].face_db
            faces   = face_db.get_all_faces_meta(limit=10000)
            if faces:
                n = self._snapshot_db.log_faces_for_date(
                    log_date     = snap_date,
                    faces        = faces,
                    camera_label = "All Cameras",
                )
                print(f"   👤 Logged {n} unique faces → {snap_date}")
        except Exception as e:
            print(f"⚠️  face log error: {e}")

        # ── 4. End running sessions ────────────────────────────────────
        try:
            mgr = self._get_session_manager()
            if mgr:
                mgr.end_session(
                    total_visitors = total_visitors,
                    max_concurrent = max_concurrent,
                    status         = 'Selesai',
                    notes          = f'Auto midnight reset — {snap_date}',
                )
        except Exception as e:
            print(f"⚠️  session end error: {e}")

        # ── 5. Reset counter via method thread-safe ────────────────────
        # FIX: gunakan reset_daily_stats() bukan akses langsung ke sm.*
        # Wajah di face_db TIDAK direset — tetap ada untuk anti-duplicate
        for c in counters:
            try:
                c.reset_daily_stats()
                print(f"   ✅ Counter {getattr(c, 'label', '')} reset (stats cleared, face DB preserved)")
            except Exception as e:
                print(f"⚠️  counter reset error: {e}")
                # Fallback: reset manual jika method belum ada
                self._reset_counter_fallback(c)

        print(f"✅ Midnight reset complete — data saved for {snap_date}")

        if self._on_reset_done:
            try:
                self._on_reset_done(snap_date, total_visitors)
            except Exception as e:
                print(f"⚠️  on_reset_done callback error: {e}")

    @staticmethod
    def _reset_counter_fallback(c):
        """
        Fallback jika reset_daily_stats() belum diimplementasikan di counter.
        Diproteksi dengan lock internal stats_manager.
        """
        try:
            sm = c.stats_manager
            # Gunakan lock jika ada, kalau tidak pakai threading.Lock dummy
            lock = getattr(sm, '_lock', threading.Lock())
            with lock:
                sm.max_count      = 0
                sm.total_detected = 0
                sm.hourly_stats.clear()
                sm.entry_times.clear()
            sm.save_statistics()

            # Reset tracker state
            c.trackers.clear()
            c.current_faces = []
            c.next_id       = 0
            print(f"   ✅ Fallback reset done")
        except Exception as e:
            print(f"⚠️  fallback reset error: {e}")

    # ── shutdown handler ──────────────────────────────────────────────────

    def _register_signals(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._shutdown_handler)
            except (ValueError, OSError):
                pass

    def _shutdown_handler(self, signum, frame):
        print(f"\n🛑 Signal {signum} received — saving snapshot before shutdown...")
        self.save_shutdown_snapshot()
        sys.exit(0)

    def save_shutdown_snapshot(self, notes: str = "Server shutdown"):
        """
        Dipanggil ketika server dimatikan (SIGINT/SIGTERM atau manual).
        Idempoten — hanya berjalan sekali.
        """
        with self._shutdown_lock:
            if self._shutdown_done:
                return
            self._shutdown_done = True

        today_str = now_wita().strftime('%Y-%m-%d')
        print(f"💾 Saving shutdown snapshot for {today_str}...")

        counters = [c for c in self._get_counters() if c is not None]
        if not counters:
            print("⚠️  No active counters")
            return

        total_visitors   = 0
        max_concurrent   = 0
        detection_method = ""

        for c in counters:
            try:
                stats            = c.get_statistics()
                total_visitors  += stats.get('daily_total',  0)
                max_concurrent   = max(max_concurrent, stats.get('max_count', 0))
                if not detection_method:
                    detection_method = stats.get('detection_method', '')
                c.stop()
            except Exception as e:
                print(f"⚠️  shutdown stats error: {e}")

        print(f"   📊 Shutdown snapshot: visitors={total_visitors}, max={max_concurrent}")

        self._snapshot_db.save_daily_snapshot(
            snapshot_date    = today_str,
            total_visitors   = total_visitors,
            max_concurrent   = max_concurrent,
            detection_method = detection_method,
            notes            = notes,
        )

        try:
            face_db = counters[0].face_db
            faces   = face_db.get_all_faces_meta(limit=10000)
            if faces:
                n = self._snapshot_db.log_faces_for_date(
                    log_date     = today_str,
                    faces        = faces,
                    camera_label = "All Cameras",
                )
                print(f"   👤 Shutdown: logged {n} faces → {today_str}")
        except Exception as e:
            print(f"⚠️  shutdown face log: {e}")

        try:
            mgr = self._get_session_manager()
            if mgr:
                mgr.end_session(
                    total_visitors = total_visitors,
                    max_concurrent = max_concurrent,
                    status         = 'Selesai',
                    notes          = 'Server shutdown',
                )
        except Exception as e:
            print(f"⚠️  shutdown session end: {e}")

        print("✅ Shutdown snapshot saved")