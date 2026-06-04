import gradio as gr
import tensorflow as tf
import pickle
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# Direktori artifacts
artifacts_dir = "./artifacts"

print("Memuat model SBERT...")
sbert_model = SentenceTransformer(os.path.join(artifacts_dir, 'finetuned_sbert'))

print("Memuat model Keras...")
keras_model = tf.keras.models.load_model(os.path.join(artifacts_dir, 'cocokin_sbert_keras_model.keras'))

print("Memuat scaler...")
with open(os.path.join(artifacts_dir, 'numeric_scaler_sbert.pkl'), 'rb') as f:
    scaler = pickle.load(f)

def predict_fit(candidate_text, job_text, exp_years, min_exp_years, skill_count, remote_allowed):
    try:
        # Embed text menggunakan SBERT
        cand_emb = sbert_model.encode([candidate_text])
        job_emb = sbert_model.encode([job_text])
        
        # Proses fitur numerik sesuai urutan: 
        # ["experience_years", "minimum_experience_years", "skill_count", "remote_allowed"]
        nums = np.array([[exp_years, min_exp_years, skill_count, remote_allowed]])
        nums_scaled = scaler.transform(nums)
        
        # Mencoba variasi input ke dalam Keras model (tergantung bagaimana modelnya di-compile saat training)
        try:
            # Asumsi 1: Model menerima 3 input terpisah (Candidate Emb, Job Emb, Numeric)
            prediction = keras_model.predict([cand_emb, job_emb, nums_scaled])
        except ValueError:
            try:
                # Asumsi 2: Model menerima 2 input (Gabungan text emb, dan Numeric)
                prediction = keras_model.predict([np.concatenate([cand_emb, job_emb], axis=1), nums_scaled])
            except ValueError:
                # Asumsi 3: Semua fitur digabung menjadi 1 tensor (Concatenated)
                combined_input = np.concatenate([cand_emb, job_emb, nums_scaled], axis=1)
                prediction = keras_model.predict(combined_input)
            
        score = float(prediction[0][0])
        return f"Cocok-in Fit Score: {score:.4f}"
    except Exception as e:
        return f"Error saat melakukan prediksi: {str(e)}"

# Antarmuka UI menggunakan Gradio
with gr.Blocks(title="Cocok-in AI Fit Score", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤝 Cocok-in AI - Prediksi Fit Score Kandidat & Lowongan")
    gr.Markdown("Aplikasi ini menggunakan **Sentence-BERT** dan **Keras (Neural Network)** untuk menghitung tingkat kecocokan antara profil kandidat dengan kriteria pekerjaan.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 👤 Informasi Kandidat")
            candidate_text = gr.Textbox(
                label="Profil / Keahlian Kandidat", 
                placeholder="Contoh: Lulusan Ilmu Komputer dengan pengalaman Python, ReactJS, dan SQL.",
                lines=4
            )
            exp_years = gr.Number(label="Pengalaman Kandidat (Tahun)", value=0)
            skill_count = gr.Number(label="Jumlah Keahlian (Skill Count)", value=0)
            
        with gr.Column():
            gr.Markdown("### 🏢 Informasi Pekerjaan")
            job_text = gr.Textbox(
                label="Deskripsi / Syarat Pekerjaan", 
                placeholder="Contoh: Dicari Backend Engineer yang menguasai Python dan Database SQL.",
                lines=4
            )
            min_exp_years = gr.Number(label="Syarat Minimal Pengalaman (Tahun)", value=0)
            remote_allowed = gr.Radio(choices=[0, 1], label="Remote Allowed? (0=Tidak, 1=Ya)", value=0)
            
    predict_btn = gr.Button("🔮 Hitung Fit Score", variant="primary")
    output_score = gr.Textbox(label="Hasil Prediksi Score", text_align="center")
    
    predict_btn.click(
        fn=predict_fit,
        inputs=[candidate_text, job_text, exp_years, min_exp_years, skill_count, remote_allowed],
        outputs=output_score
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
