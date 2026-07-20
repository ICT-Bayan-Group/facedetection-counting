"""
Statistics management utilities for Face Counter
"""
import pickle
import os
from collections import defaultdict

from core.config import Config
from utils.wita_time import now_wita, now_wita_iso, today_wita  # TZ FIX


class StatisticsManager:
    """Manages face detection statistics and historical data"""

    def __init__(self, stats_file: str = None):
        self.people_count = 0  # Current faces in frame
        self.max_count = 0
        self.total_detected = 0  # Total unique faces
        self.hourly_stats = defaultdict(int)
        self.daily_history = []
        self.entry_times = []

        # BUG FIX (data cam0/cam1 saling timpa): sebelumnya semua instance
        # StatisticsManager() selalu pakai Config.STATS_FILE yang sama persis
        # — jadi kalau dipanggil untuk 2 kamera, kamera terakhir yang
        # save_statistics() akan menimpa punya kamera lainnya di disk, dan
        # pas restart, KEDUA kamera bakal load angka yang sama (bukan
        # historinya masing-masing). Sekarang tiap OpenVINOFaceCounter wajib
        # kasih path unik per kamera lewat parameter stats_file. Kalau nggak
        # dikasih (backward-compat), tetap fallback ke Config.STATS_FILE.
        self.stats_file = stats_file or Config.STATS_FILE

        # TZ / carryover FIX: tanggal (WITA) yang direpresentasikan oleh
        # counter di atas. Dipakai buat mendeteksi kalau file statistik
        # yang di-load ternyata milik hari SEBELUMNYA (misal proses mati
        # sebelum midnight reset sempat jalan) — supaya angka kemarin
        # nggak numpuk jadi hitungan hari ini.
        self.date = today_wita()

        # Kalau load_statistics() mendeteksi data yang di-load itu milik
        # hari lain (bukan hari ini), sisa data hari itu ditaruh di sini
        # dulu (bukan langsung dibuang, bukan juga digabung ke hari ini).
        # Caller (app.py saat startup) yang tanggung jawab nyimpen ini ke
        # daily_snapshot dengan tanggal yang BENAR, lewat
        # pop_pending_carryover().
        self.pending_carryover = None

    def update(self, current_count):
        """Update current face count statistics"""
        self.people_count = current_count

        if self.people_count > self.max_count:
            self.max_count = self.people_count

        # Update hourly stats
        current_hour = now_wita().hour  # TZ FIX — sebelumnya datetime.now().hour (naive)
        if self.people_count > 0:
            self.hourly_stats[current_hour] = max(
                self.hourly_stats[current_hour],
                self.people_count
            )

    def add_unique_person(self):
        """Increment unique face counter"""
        self.total_detected += 1
        self.entry_times.append(now_wita())  # TZ FIX

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
        """Reset daily statistics (dipanggil manual snapshot / midnight scheduler)."""
        # Simpan ke histori lokal sebelum reset — pakai self.date (tanggal
        # yang baru saja "berakhir"), bukan today_wita() (yang mungkin
        # sudah hari baru di titik ini).
        if self.total_detected > 0:
            self.daily_history.append({
                'date':   self.date,
                'total':  self.total_detected,
                'max':    self.max_count,
                'hourly': dict(self.hourly_stats)
            })

        self.max_count = 0
        self.hourly_stats.clear()
        self.total_detected = 0
        self.entry_times.clear()
        self.date = today_wita()  # TZ FIX — mulai "hari baru" resmi dari sini

        self.save_statistics()

    def save_statistics(self):
        """
        Simpan statistik ke file, secara ATOMIC.

        FIX: sebelumnya file ditulis langsung (open + dump), jadi kalau
        proses mati persis di tengah penulisan (listrik mati, OOM-kill),
        file jadi setengah-nulis / corrupt, dan load_statistics() berikutnya
        akan gagal total (fallback ke 0, semua histori ilang).

        Sekarang ditulis ke file sementara dulu, baru di-rename ke nama
        aslinya. os.replace() atomic di Linux — nggak akan pernah ada
        kondisi file target dalam keadaan "setengah nulis".
        """
        data = {
            'date':           self.date,  # TZ FIX — buat deteksi carryover pas load
            'max_count':      self.max_count,
            'hourly_stats':   dict(self.hourly_stats),
            'total_detected': self.total_detected,
            'daily_history':  self.daily_history,
            'entry_times':    [t.isoformat() for t in self.entry_times],
            'last_update':    now_wita_iso(),  # TZ FIX
        }

        os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)
        tmp_path = f"{self.stats_file}.tmp"
        try:
            with open(tmp_path, 'wb') as f:
                pickle.dump(data, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.stats_file)  # atomic rename
        except Exception as e:
            print(f"⚠️  Could not save statistics: {e}")
            # Bersihkan file tmp kalau sempat kebuat tapi gagal di tengah
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def load_statistics(self):
        """
        Load statistik dari file.

        FIX (carryover hari): sebelumnya tidak ada pengecekan tanggal sama
        sekali — kalau proses restart setelah lewat tengah malam WITA tapi
        SEBELUM midnight scheduler sempat mereset file ini, total_detected
        kemarin bakal langsung dianggap sebagai hitungan hari ini (numpuk).

        Sekarang: kalau tanggal di file beda dari hari ini (WITA), data
        lama itu TIDAK digabung ke hari ini. Ditaruh dulu di
        `pending_carryover` (dengan tanggal aslinya) supaya caller bisa
        menyimpannya ke daily_snapshot dengan tanggal yang benar, lalu
        counter di sini di-reset bersih untuk hari yang baru.

        Kalau file lama belum punya field 'date' (dibuat sebelum fix ini
        dipasang), diperlakukan sebagai data hari ini juga — supaya proses
        upgrade nggak tiba-tiba membuang data yang sah cuma karena field
        barunya belum ada.
        """
        try:
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'rb') as f:
                    data = pickle.load(f)

                saved_date = data.get('date', today_wita())  # backward-compat default
                today      = today_wita()

                loaded_total_detected = data.get('total_detected', 0)
                loaded_max_count      = data.get('max_count', 0)
                loaded_hourly_stats   = data.get('hourly_stats', {})

                if saved_date == today:
                    # Data memang milik hari ini — restore normal.
                    self.date            = today
                    self.max_count       = loaded_max_count
                    self.hourly_stats    = defaultdict(int, loaded_hourly_stats)
                    self.total_detected  = loaded_total_detected
                    self.daily_history   = data.get('daily_history', [])

                    entry_times_iso = data.get('entry_times', [])
                    self.entry_times = [
                        datetime_fromisoformat_safe(t) for t in entry_times_iso
                    ]

                    print(f"✅ Statistics loaded - {self.total_detected} unique faces recorded ({today})")

                else:
                    # Data ini milik hari SEBELUMNYA (proses mati sebelum
                    # midnight reset sempat jalan). Selamatkan sebagai
                    # carryover, JANGAN digabung ke hari ini.
                    self.pending_carryover = {
                        'date':           saved_date,
                        'total_detected': loaded_total_detected,
                        'max_count':      loaded_max_count,
                        'hourly_stats':   loaded_hourly_stats,
                    }
                    self.daily_history = data.get('daily_history', [])

                    # Mulai hari ini bersih dari 0.
                    self.date           = today
                    self.max_count      = 0
                    self.hourly_stats   = defaultdict(int)
                    self.total_detected = 0
                    self.entry_times    = []

                    print(f"⚠️  Statistics file berisi data tanggal {saved_date}, "
                          f"tapi hari ini {today} — carryover disimpan terpisah "
                          f"({loaded_total_detected} visitors), counter direset bersih.")

                    # Langsung flush state bersih ke disk, supaya kalau crash
                    # lagi sebelum caller sempat proses carryover, kita nggak
                    # baca ulang data basi yang sama berkali-kali.
                    self.save_statistics()
            else:
                self.date = today_wita()

        except Exception as e:
            print(f"⚠️  Could not load statistics: {e}")
            self.date = today_wita()

    def pop_pending_carryover(self):
        """
        Ambil dan hapus carryover (kalau ada). Dipanggil sekali oleh
        startup logic (app.py) untuk menyimpan sisa data hari sebelumnya
        ke daily_snapshot dengan tanggal yang benar. Return None kalau
        tidak ada carryover.
        """
        carry = self.pending_carryover
        self.pending_carryover = None
        return carry


def datetime_fromisoformat_safe(iso_str: str):
    """Helper kecil biar entry_times yang korup/format aneh nggak bikin load gagal total."""
    from datetime import datetime
    try:
        return datetime.fromisoformat(iso_str)
    except Exception:
        return now_wita()