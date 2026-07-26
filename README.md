# Streamlit and AWS Web Deployment — Credit Score Prediction

Project end-to-end machine learning untuk memprediksi kategori credit score (Good/Standard/Poor), mulai dari eksplorasi data, pipeline training terstruktur, sampai deployment ke AWS (SageMaker + EC2) dengan Streamlit sebagai frontend.

## Struktur Project

- `1A/` — Notebook eksplorasi awal (EDA & eksperimen model)
- `1B/` — Pipeline training terstruktur (lokal), menggunakan MLflow untuk tracking eksperimen
- `1C/` — Aplikasi lokal: FastAPI backend + Streamlit frontend
- `2/2A/` — Pipeline training terstruktur (versi untuk deployment cloud)
- `2/2B/` — Deployment ke AWS: SageMaker endpoint + FastAPI + Streamlit di EC2

## Alur Kerja

1. **Eksplorasi (1A)** — Eksperimen awal di notebook untuk memahami data dan mencoba beberapa model.
2. **Training Pipeline (1B)** — Pipeline modular (data loader, preprocessor, trainer, evaluator) yang melatih beberapa model (Random Forest, XGBoost, LightGBM, MLP) beserta versi tuning-nya, lalu memilih model terbaik berdasarkan F1 macro. Semua eksperimen dicatat via MLflow.
3. **Aplikasi Lokal (1C)** — Model terbaik disajikan lewat FastAPI (`backend.py` + `inference.py`), dikonsumsi oleh Streamlit (`frontend.py`) sebagai antarmuka input & hasil prediksi.
4. **Deployment Cloud (2A & 2B)** — Model dilatih ulang lewat pipeline yang sama, kemudian di-deploy sebagai endpoint di AWS SageMaker. FastAPI (`api.py`) meneruskan request ke SageMaker endpoint, dan Streamlit (`frontend.py`) berjalan sebagai UI. Semua service dijalankan otomatis di EC2 lewat `user-data.sh` (systemd service untuk FastAPI di port 8000 dan Streamlit di port 8501).

## Model & Hasil

Model terbaik: **LightGBM (Tuned)**
- Macro F1: 0.7228
- Test Accuracy: 0.7404

Perbandingan model yang dicoba: Random Forest, XGBoost, LightGBM, MLP (masing-masing versi default & tuned lewat randomized search).

## Tech Stack

- Python
- Scikit-learn, XGBoost, LightGBM
- MLflow (experiment tracking)
- FastAPI (serving model)
- Streamlit (frontend)
- AWS SageMaker & EC2 (cloud deployment)
- boto3

## Cara Menjalankan (Lokal — folder 1C)

1. Clone repository ini:

   git clone https://github.com/Adhikaxx88/Streamlit-and-AWS-web-deployment.git

   cd Streamlit-and-AWS-web-deployment/1C

2. Install dependencies:

   pip install -r requirements.txt

3. Jalankan backend (FastAPI):

   uvicorn backend:app --host 0.0.0.0 --port 8000

4. Jalankan frontend (Streamlit) di terminal terpisah:

   streamlit run frontend.py

5. Buka browser di `http://localhost:8501`

## Cara Menjalankan Training Pipeline (folder 1B)

1. Masuk ke folder `1B/`
2. Install dependencies:

   pip install -r requirements.txt

3. Jalankan pipeline:

   python pipeline.py

Output berupa model (`.pkl`), label encoder, dan laporan evaluasi akan tersimpan di folder `outputs/`.

## Deployment ke AWS (folder 2/2B)

Deployment cloud memakai dua tahap:

1. **SageMaker Endpoint** — `deploy_endpoint.py` mendeploy model (`inference.py` di folder `src/`) sebagai SageMaker endpoint, lengkap dengan smoke test contoh prediksi (Good/Standard/Poor).
2. **EC2 (FastAPI + Streamlit)** — `user-data.sh` dijalankan sebagai bootstrap script saat instance EC2 dibuat. Script ini otomatis clone repo, install dependencies, lalu menjalankan FastAPI (`api.py`) dan Streamlit (`frontend.py`) sebagai systemd service, sehingga otomatis restart kalau instance reboot.

Catatan: konfigurasi seperti nama bucket S3, region, dan endpoint name di `deploy_endpoint.py` dan `user-data.sh` perlu disesuaikan dengan environment AWS masing-masing.
