import signal
import sys
import threading
import time
from datetime import datetime, timedelta
from typing import Callable

# TZ FIX: import dari utils/wita_time.py — single source of truth, bukan
# definisi WITA lokal lagi. Lihat utils/wita_time.py untuk penjelasan kenapa
# ini penting (tidak boleh ada drift antar file kalau TZ Linux berubah).
from utils.wita_time import now_wita


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

        # flag agar tidak double-reset di hari yang sama.
        # Diisi dengan string 'YYYY-MM-DD' setelah reset berhasil.
        # Diinisialisasi dengan tanggal hari ini saat startup agar
        # restart server tidak memicu reset ulang untuk hari yang sama.
        self._last_reset_date: str = now_wita().strftime('%Y-%m-%d')

        # BUG #2 FIX: heartbeat, biar kelihatan di log kalau thread scheduler
        # masih hidup/jalan (requirement PRD: Heartbeat Logging)
        self._last_heartbeat_log = 0.0

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="MidnightReset"
        )
        self._thread.start()
        self._register_signals()
        print(f"⏰ MidnightResetScheduler started (WITA) | last_reset={self._last_reset_date}")

    def stop(self):
        self._running = False

    def is_alive(self) -> bool:
        """Dipakai health-check kalau mau expose status thread scheduler."""
        return bool(self._thread and self._thread.is_alive())

    # ── main loop ──────────────────────────────────────────────────────────

    def _loop(self):
        """
        BUG #2 FIX (root cause):
        Versi sebelumnya menunggu window sempit `h==0 and m==0 and s<=30`.
        Di production, 2 kamera + OpenVINO inference bikin GIL contention
        yang cukup berat sehingga thread scheduler ini bisa ke-starve lebih
        dari 30 detik. Kalau itu terjadi persis pas melewati window
        tersebut, reset ke-skip total untuk hari itu (baru kedeteksi lagi
        besoknya, di mana window juga bisa ke-skip lagi -> data histori
        hilang berhari-hari, sesuai laporan bug).

        Fix: jangan sandarkan trigger ke window detik yang sempit. Sandarkan
        ke **perubahan tanggal WITA** dibanding `_last_reset_date`. Dengan
        begitu, kapan pun thread ini akhirnya sempat jalan lagi setelah
        starvation berapa lama pun (asal masih di hari yang baru), reset
        tetap akan terpicu — tidak ada window yang bisa "terlewat".
        """
        while self._running:
            try:
                now       = now_wita()
                today_str = now.strftime('%Y-%m-%d')

                # ── Trigger utama: tanggal WITA sudah berubah dari
                #    tanggal terakhir kali kita reset ─────────────────────
                if today_str != self._last_reset_date:
                    snap_date = self._last_reset_date  # hari yang baru saja berakhir
                    print(f"\n🕛 MIDNIGHT RESET terdeteksi — tanggal berubah "
                          f"{snap_date} → {today_str} (WITA {now.strftime('%H:%M:%S')})")
                    self._save_snapshot_and_reset(snap_date=snap_date)
                    self._last_reset_date = today_str
                    self._last_heartbeat_log = 0.0  # biar heartbeat langsung log lagi
                    time.sleep(5)
                    continue

                seconds_until_midnight = self._seconds_until_midnight(now)

                # Heartbeat logging (requirement PRD) — cukup tiap ~60 detik
                # biar log tidak banjir, dan makin sering saat mendekati midnight.
                nowt = time.time()
                if seconds_until_midnight <= 120:
                    if nowt - self._last_heartbeat_log >= 15:
                        print(f"⏳ Waiting midnight... {int(seconds_until_midnight)}s left "
                              f"(WITA {now.strftime('%H:%M:%S')})")
                        self._last_heartbeat_log = nowt
                elif nowt - self._last_heartbeat_log >= 300:
                    print(f"💓 MidnightResetScheduler alive | last_reset={self._last_reset_date} "
                          f"| next reset in ~{int(seconds_until_midnight)}s")
                    self._last_heartbeat_log = nowt

                # Masih jauh dari midnight → tidur pendek, jangan pernah lebih dari 20 detik
                # sekali cek, supaya kalaupun ada delay/starvation, kita tetap sering-sering
                # cek ulang kondisi tanggal di atas.
                if seconds_until_midnight > 30:
                    sleep_time = min(seconds_until_midnight - 20, 20)
                    time.sleep(max(sleep_time, 1))
                else:
                    # Mendekati midnight — polling ketat tiap 1 detik
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

    def _save_snapshot_and_reset(
        self,
        notes:     str = "Auto midnight reset",
        snap_date: str | None = None,
    ):
        """
        1. Kumpulkan stats dari semua counter yang aktif
        2. Simpan daily_summary ke snapshot DB
        3. Simpan semua wajah unik ke daily_face_log
        4. End sesi yang masih Running
        5. Reset counter via reset_daily_stats() yang thread-safe
        6. Buat sesi baru untuk hari yang baru (FR-05)
        7. Trigger broadcast callback (dashboard SSE)
        """
        if snap_date is None:
            # Fallback: ambil kemarin relatif terhadap sekarang
            snap_date = (now_wita() - timedelta(days=1)).strftime('%Y-%m-%d')

        counters = [c for c in self._get_counters() if c is not None]
        if not counters:
            print("⚠️  No active counters, skip snapshot")
            return

        # ── 1. Kumpulkan statistik ─────────────────────────────────────
        total_visitors   = 0
        max_concurrent   = 0
        raw_detections   = 0  # BARU — total "terdeteksi manusia" (raw) hari ini, semua kamera
        detection_method = ""
        first_detection  = None
        last_detection   = None

        for c in counters:
            try:
                stats            = c.get_statistics()
                total_visitors  += stats.get('daily_total',  0)
                max_concurrent   = max(max_concurrent, stats.get('max_count', 0))
                raw_detections  += stats.get('raw_detections', 0)
                if not detection_method:
                    detection_method = stats.get('detection_method', '')
                ts = stats.get('timestamp', '')
                if ts:
                    if not last_detection or ts > last_detection:
                        last_detection = ts
            except Exception as e:
                print(f"⚠️  stats error: {e}")

        print(f"   📊 Snapshot {snap_date}: visitors={total_visitors}, max={max_concurrent}, "
              f"raw_detections={raw_detections}")

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
            raw_detections   = raw_detections,
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
        mgr = None
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
        for c in counters:
            try:
                c.reset_daily_stats()
                print(f"   ✅ Counter {getattr(c, 'label', '')} reset (stats cleared, face DB preserved)")
            except Exception as e:
                print(f"⚠️  counter reset error: {e}")
                self._reset_counter_fallback(c)

        # ── 6. Buat sesi baru untuk hari berikutnya (FR-05) ─────────────
        try:
            if mgr:
                mgr.start_session("CCTV Hall A (2 Kamera)")
                print("   🚀 Sesi baru dibuat untuk hari berikutnya")
        except Exception as e:
            print(f"⚠️  start new session error: {e}")

        print(f"✅ Midnight reset complete — data saved for {snap_date}")

        # ── 7. Broadcast ke dashboard (SSE) ─────────────────────────────
        if self._on_reset_done:
            try:
                self._on_reset_done(snap_date, total_visitors)
            except Exception as e:
                print(f"⚠️  on_reset_done callback error: {e}")

    @staticmethod
    def _reset_counter_fallback(c):
        """
        Fallback jika reset_daily_stats() belum diimplementasikan di counter.
        """
        try:
            sm   = c.stats_manager
            lock = getattr(sm, '_lock', threading.Lock())
            with lock:
                sm.max_count      = 0
                sm.total_detected = 0
                sm.hourly_stats.clear()
                sm.entry_times.clear()
            sm.save_statistics()

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
        Pakai tanggal HARI INI (bukan kemarin) karena data belum direset.
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
        raw_detections   = 0  # BARU
        detection_method = ""

        for c in counters:
            try:
                stats            = c.get_statistics()
                total_visitors  += stats.get('daily_total',  0)
                max_concurrent   = max(max_concurrent, stats.get('max_count', 0))
                raw_detections  += stats.get('raw_detections', 0)
                if not detection_method:
                    detection_method = stats.get('detection_method', '')
                c.stop()
            except Exception as e:
                print(f"⚠️  shutdown stats error: {e}")

        print(f"   📊 Shutdown snapshot: visitors={total_visitors}, max={max_concurrent}, "
              f"raw_detections={raw_detections}")

        self._snapshot_db.save_daily_snapshot(
            snapshot_date    = today_str,
            total_visitors   = total_visitors,
            max_concurrent   = max_concurrent,
            raw_detections   = raw_detections,
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