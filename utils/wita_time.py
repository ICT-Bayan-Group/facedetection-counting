from datetime import datetime, timezone, timedelta

WITA = timezone(timedelta(hours=8))


def now_wita() -> datetime:
    """datetime object waktu sekarang di WITA (UTC+8), timezone-aware."""
    return datetime.now(WITA)


def today_wita() -> str:
    """Tanggal hari ini di WITA, format 'YYYY-MM-DD'."""
    return now_wita().strftime('%Y-%m-%d')


def now_wita_iso() -> str:
    """
    Pengganti langsung untuk `datetime.now().isoformat()` yang lama.
    Isinya ISO string tapi berdasarkan WITA, bukan TZ OS.
    """
    return now_wita().isoformat()


def seconds_until_midnight_wita() -> int:
    """Jumlah detik sampai 00:00:00 WITA berikutnya."""
    now = now_wita()
    midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return int((midnight - now).total_seconds())