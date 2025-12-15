# People Counter — CCTV Face Detection & Counting

![Python](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQCTdlxpN40oRq28d7owUaaoj4y37IjSn5RNA&s)
![Flask](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQOL9HihGRlCubXlGV_FBX6B6y-pK2KAx6O4Q&s)
![OpenCV](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQfRklXyWQy1ditXPl8oBPdbcdjxuiVU3Z3VA&s)
![YOLOv8](https://miro.medium.com/v2/resize:fit:1200/1*YQWYPi4uoT8RcG6BPbUoVw.png)
![Tailwind CSS](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQNhoXisDruJMDAq3Ltd-wuaMW2lGxck9wAKw&s)
![Chart.js](https://miro.medium.com/v2/1*W3-xgZUKr4ruD1FNL-xaMQ.png)

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
