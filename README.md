# People Counter — CCTV Face Detection & Counting

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0.0-000000?style=for-the-badge&logo=flask&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.0-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white)
![Chart.js](https://img.shields.io/badge/Chart.js-3.9-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white)

## Ringkasan singkat

Aplikasi monitoring CCTV real-time yang mendeteksi dan menghitung orang menggunakan YOLOv8 + tracker. Menyajikan live stream, statistik (current, max, total unique), grafik hourly, dan API ringan untuk integrasi.

## Fitur utama

- Deteksi orang real-time dengan `ultralytics.YOLO` (YOLOv8)
- Tracking ID untuk menghitung total unik
- Dashboard web modern (Tailwind + Chart.js)
- API endpoints untuk stats, reset, health, history
- Fallback demo mode jika CCTV tidak terhubung

## Tech stack

- Bahasa: Python
- Web: Flask
- Detection: Ultralytics YOLOv8
- Video: OpenCV
- Frontend: Tailwind CSS, Chart.js

## Persyaratan

- Python 3.8+ (direkomendasikan 3.11)
- Dependencies ada di `requirements.txt` (lihat juga file `yolov8n.pt` sudah termasuk)

## Instalasi (Windows)

1. Buat virtual environment:

   python -m venv .venv

2. Aktifkan venv:

   .\.venv\Scripts\activate

3. Pasang dependencies:

   pip install -r requirements.txt

4. (Opsional) Pastikan file model `yolov8n.pt` ada di root proyek.

## Konfigurasi

Edit pengaturan CCTV dan port pada `core/config.py` jika perlu:

- `CCTV_IP`, `CCTV_USER`, `CCTV_PASS` — untuk koneksi RTSP/MJPEG
- `YOLO_MODEL` — nama file model (default `yolov8n.pt`)
- `PORT`, `HOST`, `DEBUG` — pengaturan server Flask

## Menjalankan aplikasi

Jalankan dari root proyek:

```powershell
python app.py
```

Kemudian buka dashboard di browser:

http://localhost:5000

## Endpoint API

- `GET /api/stats` — current stats (current_count, max_count, daily_total, hourly_stats, fps)
- `POST /api/reset` — reset daily statistics
- `GET /api/health` — health check (service status, CCTV connected)
- `GET /api/history` — historical data untuk grafik dan peak hours

## Arsitektur & file penting

- `app.py` — entrypoint + Flask routes
- `core/people_counter.py` — deteksi, tracking, overlay, loop utama
- `core/config.py` — konfigurasi aplikasi
- `utils/video_utils.py` — handler koneksi CCTV (mencoba berbagai URL)
- `utils/stats_manager.py` — simpan / load statistik, history
- `templates/dashboard.html` — frontend dashboard (Tailwind + Chart.js)

## On Development

Tim ICT-Bayan-Group
