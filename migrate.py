"""
migrate.py — Jalankan SEKALI untuk pindahkan data lama ke SQLite.

Usage:
    python migrate.py

Setelah selesai, file JSON lama TIDAK dihapus (backup otomatis).
"""

import os
import sys
import importlib.util


# ── Import langsung ke file, bypass utils/__init__.py ────────────────────────
# Tujuan: hindari circular import karena __init__.py memuat face_counter.py

def _load(module_name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

_base = os.path.dirname(os.path.abspath(__file__))

_face_db_mod   = _load("_face_db",  os.path.join(_base, "utils", "face_database.py"))
_session_mod   = _load("_sess_mgr", os.path.join(_base, "utils", "session_manager.py"))

FaceDatabase   = _face_db_mod.FaceDatabase
SessionManager = _session_mod.SessionManager


# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  MIGRASI JSON → SQLite")
    print("=" * 55)

    # ── Face Database ─────────────────────────────────────────────────────
    print("\n[1/2] Face Database...")
    face_db  = FaceDatabase(db_path='data/face_database.db')
    migrated = face_db.migrate_from_json('data/face_database.json')

    if migrated > 0:
        print(f"      {migrated} wajah berhasil diimpor")
    else:
        print("      Tidak ada data baru (mungkin sudah dimigrasi sebelumnya)")

    # ── Sessions ──────────────────────────────────────────────────────────
    print("\n[2/2] Sessions...")
    sess_mgr = SessionManager(db_path='data/sessions.db')
    migrated = sess_mgr.migrate_from_json('data/sessions.json')

    if migrated > 0:
        print(f"      {migrated} sesi berhasil diimpor")
    else:
        print("      Tidak ada data baru")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("✅ Migrasi selesai!")
    print(f"   Face DB  : data/face_database.db  ({face_db._count()} wajah)")
    print(f"   Sessions : data/sessions.db        ({sess_mgr._count()} sesi)")
    print()
    print("📌 File JSON lama TIDAK dihapus (tersimpan sebagai backup).")
    print("   Setelah yakin OK, hapus .json lama secara manual.")
    print("=" * 55)


if __name__ == '__main__':
    main()