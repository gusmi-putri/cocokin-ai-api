# Cocokin AI API

Cocokin AI adalah service rekomendasi pekerjaan berbasis FastAPI yang mencocokkan profil kandidat dengan katalog lowongan. API ini dapat membaca CV PDF, mengekstrak profil kandidat, menghitung skor kecocokan menggunakan model SBERT + Keras, dan menambahkan penjelasan rekomendasi dengan Gemini.

## Fitur Utama

- Parsing CV PDF menjadi profil kandidat.
- Rekomendasi pekerjaan berdasarkan skill, sektor industri, pengalaman, dan pendidikan.
- Analisis kecocokan kandidat terhadap target role tertentu.
- Penjelasan "why you match" dengan Google Gemini.
- Fallback scoring jika model TensorFlow/SBERT tidak dapat dimuat sepenuhnya.
- Endpoint dokumentasi otomatis melalui Swagger UI di `/docs`.
- Dockerfile siap deploy ke Hugging Face Spaces atau container host lain.
- Demo Gradio sederhana untuk prediksi fit score manual.

## Tech Stack

- Python 3.11
- FastAPI
- Uvicorn
- TensorFlow / Keras
- Sentence Transformers
- scikit-learn
- pandas / NumPy
- PyMuPDF
- Google Generative AI SDK
- Gradio

## Struktur Project

```text
.
|-- app/
|   |-- core/
|   |   `-- config.py
|   |-- services/
|   |   |-- cv_parser.py
|   |   |-- gemini_service.py
|   |   `-- recommender.py
|   `-- schemas.py
|-- artifacts/
|   |-- finetuned_sbert/
|   |-- cocokin_sbert_keras_model.keras
|   |-- job_catalog.csv
|   |-- numeric_features_sbert.json
|   `-- numeric_scaler_sbert.pkl
|-- scripts/
|   `-- deploy_hf_space.py
|-- app.py
|-- main.py
|-- Dockerfile
|-- requirements.txt
`-- README.md
```

## Prasyarat

Pastikan file artifacts berikut tersedia di folder `artifacts/`:

- `finetuned_sbert/`
- `cocokin_sbert_keras_model.keras`
- `numeric_scaler_sbert.pkl`
- `numeric_features_sbert.json`
- `job_catalog.csv`

Tanpa artifacts tersebut, recommender tidak dapat berjalan dengan penuh.

## Setup Lokal

1. Clone repository:

```bash
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>
```

2. Buat dan aktifkan virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Untuk macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependency:

```bash
pip install -r requirements.txt
```

4. Buat file `.env`:

```bash
copy .env.example .env
```

Untuk macOS/Linux:

```bash
cp .env.example .env
```

5. Isi konfigurasi Gemini jika ingin menggunakan fitur ekstraksi dan penjelasan berbasis LLM:

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

6. Jalankan API:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 7860
```

API akan tersedia di:

- `http://localhost:7860`
- `http://localhost:7860/docs`
- `http://localhost:7860/health`

## Menjalankan Demo Gradio

Selain API FastAPI, project ini juga memiliki demo Gradio di `app.py`.

```bash
python app.py
```

Demo akan berjalan di port `7860`.

## Environment Variables

| Variable | Default | Keterangan |
| --- | --- | --- |
| `GEMINI_API_KEY` | kosong | API key Gemini. Jika kosong, fitur Gemini tidak aktif. |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model Gemini yang digunakan untuk ekstraksi profil dan explanation. |
| `ARTIFACT_DIR` | `artifacts` | Lokasi folder artifacts model. |

## API Endpoints

### `GET /`

Redirect ke Swagger UI.

### `GET /health`

Mengecek status service, Gemini, lokasi artifacts, dan status model recommender.

Contoh:

```bash
curl http://localhost:7860/health
```

### `POST /recommend-from-cv`

Mengunggah CV PDF, mengekstrak profil kandidat, lalu mengembalikan 3 rekomendasi pekerjaan teratas.

Query parameter:

- `use_gemini`: `true` atau `false`, default `true`.

Contoh:

```bash
curl -X POST "http://localhost:7860/recommend-from-cv?use_gemini=true" \
  -F "file=@cv.pdf"
```

### `POST /analyze-target-role`

Mengunggah CV PDF dan menganalisis kecocokan kandidat terhadap role target.

Query parameter:

- `target_role`: nama posisi yang ingin dianalisis.

Contoh:

```bash
curl -X POST "http://localhost:7860/analyze-target-role?target_role=Data%20Scientist" \
  -F "file=@cv.pdf"
```

## Docker

Build image:

```bash
docker build -t cocokin-ai-api .
```

Jalankan container:

```bash
docker run --rm -p 7860:7860 --env-file .env cocokin-ai-api
```

## Deployment ke Hugging Face Spaces

Project ini menggunakan Docker dan menjalankan Uvicorn di port `7860`, sesuai standar Hugging Face Spaces.

Pastikan secrets berikut diatur di Hugging Face Spaces:

- `GEMINI_API_KEY`
- `GEMINI_MODEL`

Jika artifacts berukuran besar, simpan dengan Git LFS atau mekanisme storage yang sesuai.

## Catatan Model

Recommender menggunakan kombinasi:

- Sentence-BERT untuk embedding teks kandidat dan lowongan.
- Keras model untuk memprediksi skor kecocokan.
- Numeric scaler untuk fitur numerik seperti pengalaman, salary, dan fitur tambahan dari katalog.
- Rule-based fallback untuk menjaga API tetap dapat memberi hasil saat model utama gagal dimuat.

## Lisensi

Tambahkan lisensi sesuai kebutuhan project, misalnya MIT, Apache-2.0, atau lisensi internal.
