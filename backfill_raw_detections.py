"""
backfill_raw_detections.py
───────────────────────────────────────────────────────────────────────────
Rekonstruksi kolom `raw_detections` di daily_summary (daily_snapshots.db)
untuk tanggal-tanggal LAMA, diambil dari data historis di tabel
`raw_detections` (face_database.db) yang sudah punya timestamp per baris.

INI BUKAN pembagian rata / estimasi — tiap baris di raw_detections punya
`detected_at` sendiri, jadi hitungannya sama persis dengan COUNT(*) yang
sebenarnya terjadi di tanggal itu.

CARA PAKAI (jalankan di server, di folder yang sama dengan app.py, atau
sesuaikan --face-db / --snapshot-db ke path aslinya):

    # 1) Lihat dulu apa yang AKAN diubah, tanpa nulis apa-apa:
    python backfill_raw_detections.py --dry-run

    # 2) Kalau sudah oke, jalankan beneran:
    python backfill_raw_detections.py

    # 3) Kalau ada tanggal di log raw_detections yang BELUM punya baris
    #    di daily_summary sama sekali (misal hari itu server sempat mati
    #    total sebelum sempat snapshot), sertakan --create-missing supaya
    #    dibuatkan baris baru (total_visitors=0, karena memang tidak ada
    #    datanya, cuma raw_detections yang terselamatkan):
    python backfill_raw_detections.py --create-missing

Default path DB ikut konvensi yang sama dengan app.py / face_database.py /
daily_snapshot.py (env var FACE_DB_PATH / SNAPSHOT_DB_PATH, fallback ke
'data/face_database.db' dan 'data/daily_snapshots.db').
"""
import argparse
import os
import sqlite3
import sys
from datetime import datetime


def parse_args():
    p = argparse.ArgumentParser(description="Backfill raw_detections harian ke daily_summary")
    p.add_argument('--face-db', default=os.environ.get('FACE_DB_PATH', 'data/face_database.db'),
                    help="Path ke face_database.db (sumber log raw_detections dengan timestamp)")
    p.add_argument('--snapshot-db', default=os.environ.get('SNAPSHOT_DB_PATH', 'data/daily_snapshots.db'),
                    help="Path ke daily_snapshots.db (tujuan, kolom daily_summary.raw_detections)")
    p.add_argument('--dry-run', action='store_true',
                    help="Cuma tampilkan apa yang akan diubah, tidak nulis ke DB")
    p.add_argument('--create-missing', action='store_true',
                    help="Buat baris daily_summary baru untuk tanggal yang ada di raw_detections "
                         "tapi belum ada snapshot-nya sama sekali (total_visitors akan 0)")
    p.add_argument('--overwrite', action='store_true',
                    help="Timpa langsung nilai raw_detections (default: ambil MAX antara nilai "
                         "lama & hasil hitung, lebih aman kalau script ini dijalankan berkali-kali)")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.face_db):
        print(f"❌ File tidak ditemukan: {args.face_db}")
        sys.exit(1)
    if not os.path.exists(args.snapshot_db):
        print(f"❌ File tidak ditemukan: {args.snapshot_db}")
        sys.exit(1)

    print(f"📂 Face DB      : {os.path.abspath(args.face_db)}")
    print(f"📂 Snapshot DB  : {os.path.abspath(args.snapshot_db)}")
    print(f"🔍 Mode         : {'DRY RUN (tidak nulis apa-apa)' if args.dry_run else 'WRITE'}")
    print("─" * 65)

    # ── 1. Hitung raw_detections per tanggal dari log historis ─────────────
    face_conn = sqlite3.connect(args.face_db)
    face_conn.row_factory = sqlite3.Row
    rows = face_conn.execute("""
        SELECT substr(detected_at, 1, 10) AS d, COUNT(*) AS cnt
          FROM raw_detections
         GROUP BY d
         ORDER BY d ASC
    """).fetchall()
    face_conn.close()

    if not rows:
        print("⚠️  Tabel raw_detections kosong — tidak ada apa pun untuk di-backfill.")
        return

    per_date = {r['d']: r['cnt'] for r in rows}
    print(f"📊 Ditemukan log raw_detections untuk {len(per_date)} tanggal "
          f"(total {sum(per_date.values())} baris)\n")

    # ── 2. Cocokkan dengan daily_summary yang sudah ada ─────────────────────
    snap_conn = sqlite3.connect(args.snapshot_db)
    snap_conn.row_factory = sqlite3.Row

    existing_dates = {
        r['date']: r['raw_detections']
        for r in snap_conn.execute("SELECT date, raw_detections FROM daily_summary")
    }

    to_update  = []   # (date, old_val, new_val)
    to_create  = []   # date, cnt
    unchanged  = []

    for date_str, cnt in per_date.items():
        if date_str in existing_dates:
            old_val = existing_dates[date_str] or 0
            new_val = cnt if args.overwrite else max(old_val, cnt)
            if new_val != old_val:
                to_update.append((date_str, old_val, new_val))
            else:
                unchanged.append(date_str)
        else:
            to_create.append((date_str, cnt))

    # ── 3. Laporan ────────────────────────────────────────────────────────
    if to_update:
        print(f"✏️  Akan di-UPDATE ({len(to_update)} tanggal):")
        for d, old, new in to_update:
            print(f"   {d}  :  {old:>6}  →  {new:>6}")
    else:
        print("✏️  Tidak ada tanggal existing yang perlu di-update.")

    print()
    if to_create:
        if args.create_missing:
            print(f"🆕 Akan dibuat baris BARU ({len(to_create)} tanggal, --create-missing aktif):")
        else:
            print(f"🆕 Ditemukan {len(to_create)} tanggal di log TAPI belum punya snapshot sama sekali "
                  f"(pakai --create-missing untuk membuatnya):")
        for d, cnt in to_create:
            print(f"   {d}  :  raw_detections={cnt}  (total_visitors akan 0 — data pengunjung hari itu memang tidak ada)")
    else:
        print("🆕 Tidak ada tanggal yang hilang dari daily_summary.")

    if unchanged:
        print(f"\n⏭️  {len(unchanged)} tanggal sudah sesuai, tidak diubah.")

    if args.dry_run:
        print("\n🔍 DRY RUN — tidak ada perubahan yang ditulis. Jalankan ulang tanpa --dry-run untuk eksekusi.")
        snap_conn.close()
        return

    # ── 4. Eksekusi ───────────────────────────────────────────────────────
    n_updated = 0
    for d, old, new in to_update:
        snap_conn.execute("UPDATE daily_summary SET raw_detections=? WHERE date=?", (new, d))
        n_updated += 1

    n_created = 0
    if args.create_missing:
        now_iso = datetime.now().isoformat()
        for d, cnt in to_create:
            snap_conn.execute("""
                INSERT INTO daily_summary
                  (date, total_visitors, max_concurrent, raw_detections, session_count,
                   first_detection, last_detection, detection_method, notes, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (d, 0, 0, cnt, 0, None, None, None,
                  'Backfilled dari raw_detections log (tidak ada snapshot pengunjung untuk hari ini)',
                  now_iso))
            n_created += 1

    snap_conn.commit()
    snap_conn.close()

    print(f"\n✅ Selesai — {n_updated} baris di-update, {n_created} baris baru dibuat.")
    print("   Buka halaman /data-pengunjung atau Export Rekap untuk lihat hasilnya.")


if __name__ == '__main__':
    main()