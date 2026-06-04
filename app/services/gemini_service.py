from __future__ import annotations

import json
import google.generativeai as genai
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

class GeminiExplanationService:
    def __init__(self, api_key: str | None, model_name: str = "gemini-1.5-flash"):
        self.enabled = bool(api_key)
        self.model = None
        if self.enabled:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(
                model_name,
                system_instruction="Anda adalah Cocokin AI, asisten rekrutmen cerdas, konsultan karir, dan pakar HR. Berikan jawaban yang profesional, ramah, dan sangat akurat."
            )

    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(4))
    async def _generate_content_with_retry(self, prompt: str, is_json: bool = False):
        if not self.enabled or self.model is None:
            raise Exception("Model not enabled")
        
        config = {"response_mime_type": "application/json"} if is_json else {}
        response = await self.model.generate_content_async(prompt, generation_config=config)
        return getattr(response, "text", "").strip()

    async def explain(self, candidate_profile: dict, job: dict) -> str:
        fallback = self._fallback(candidate_profile, job)
        if not self.enabled or self.model is None:
            return fallback

        prompt = f"""
Tugas:
Jelaskan mengapa kandidat cocok atau kurang cocok untuk pekerjaan berikut berdasarkan data yang diberikan.
Aturan:
- Gunakan Bahasa Indonesia yang profesional dan mudah dipahami.
- Tulis dalam satu paragraf utuh.
- Panjang 80-120 kata.
- Berikan analisis yang spesifik dan berbasis data.
- Fokus pada kecocokan skill, pengalaman, pendidikan, dan sektor industri.
- Gunakan skor cocok hanya sebagai informasi pendukung, bukan alasan utama.
- Soroti kekuatan kandidat yang paling relevan dengan pekerjaan.
- Jika terdapat skill yang belum sesuai, sebutkan maksimal 1-2 gap terpenting dan berikan saran singkat yang membangun.
- Jangan menggunakan bullet point, numbering, heading, atau markdown.
- Jangan membuat asumsi atau menambahkan informasi yang tidak tersedia pada data.
- Jangan mengklaim pengalaman, proyek, sertifikasi, atau kemampuan yang tidak disebutkan.
Data kandidat:
- Sektor: {candidate_profile.get('industry_sector_cand')}
- Skill: {candidate_profile.get('cand_tech_skills')}
- Pengalaman: {candidate_profile.get('experience_years')} tahun
- Pendidikan: {candidate_profile.get('education_level_cand')}
Data pekerjaan:
- Judul: {job.get('job_title')}
- Sektor: {job.get('industry_sector_job')}
- Skill dibutuhkan: {job.get('req_tech_skills')}
- Skill kandidat yang match: {job.get('matched_skills')}
- Skill yang belum match: {job.get('missing_skills')}
- Minimum pengalaman: {job.get('minimum_experience_years')}
- Skor cocok: {job.get('match_score_percent')}%
"""
        try:
            text = await self._generate_content_with_retry(prompt)
            return text or fallback
        except Exception:
            return fallback

    async def explain_batch(self, candidate_profile: dict, jobs: list[dict]) -> dict[str, str]:
        if not self.enabled or self.model is None or not jobs:
            return {str(j.get("job_id", i)): self._fallback(candidate_profile, j) for i, j in enumerate(jobs)}

        jobs_text = ""
        for i, job in enumerate(jobs[:3]):
            job_id = str(job.get("job_id", i))
            experience_gap = max(
                0,
                float(job.get("minimum_experience_years", 0)) - float(candidate_profile.get("experience_years", 0))
            )
            jobs_text += f"\n[Pekerjaan {i+1}]\n"
            jobs_text += f"ID: {job_id}\n"
            jobs_text += f"Role: {job.get('job_title')}\n"
            jobs_text += f"Skor: {job.get('match_score_percent')}%\n"
            jobs_text += f"Skill cocok: {job.get('matched_skills')}\n"
            jobs_text += f"Skill gap: {job.get('missing_skills')}\n"
            jobs_text += f"Gap pengalaman: {experience_gap} tahun\n"

        prompt = f"""
Tugas:
Jelaskan alasan kecocokan kandidat untuk SETIAP pekerjaan berdasarkan hasil pencocokan yang telah dihitung sistem.
ATURAN WAJIB:
* Gunakan Bahasa Indonesia.
* Buat 2-3 kalimat untuk setiap pekerjaan.
* Fokus pada skill yang cocok, pengalaman kandidat, dan kebutuhan pekerjaan.
* Jika terdapat skill gap atau gap pengalaman, sebutkan maksimal 1 area pengembangan yang paling penting.
* Gunakan informasi yang tersedia saja.
* Jangan membuat asumsi, sertifikasi, proyek, pengalaman, atau kemampuan yang tidak disebutkan dalam data.
* Jangan mengubah, mengoreksi, atau mempertanyakan skor kecocokan.
* Setiap value harus berupa satu string pendek yang natural dan profesional.
FORMAT OUTPUT WAJIB:
* Gunakan ID pekerjaan sebagai key.
* Value harus berupa string penjelasan.
Data Kandidat:
* Pengalaman: {candidate_profile.get('experience_years')} tahun
* Pendidikan: {candidate_profile.get('education_level_cand')}
Data Pekerjaan:
{jobs_text}
"""
        try:
            out_text = await self._generate_content_with_retry(prompt, is_json=True)
            return json.loads(out_text)
        except Exception:
            return {str(j.get("job_id", i)): self._fallback(candidate_profile, j) for i, j in enumerate(jobs)}

    @staticmethod
    def _fallback(candidate_profile: dict, job: dict) -> str:
        matched_skills = job.get("matched_skills") or []
        missing_skills = job.get("missing_skills") or []
        try:
            score = float(job.get("match_score_percent", 0))
        except (TypeError, ValueError):
            score = 0.0
    
        if score >= 75:
            fit_label = "Sangat Cocok"
        elif score >= 55:
            fit_label = "Cocok"
        elif score >= 35:
            fit_label = "Lumayan Cocok"
        else:
            fit_label = "Tidak Cocok"
    
        job_title = job.get("job_title") or "role ini"
        if isinstance(matched_skills, list) and matched_skills:
            skill_text = ", ".join(map(str, matched_skills[:3]))
            explanation = (
                f"Kandidat berada pada kategori {fit_label} untuk posisi {job_title} "
                f"dengan skor {score:.0f}%, terutama karena memiliki skill relevan seperti {skill_text}."
            )
            if isinstance(missing_skills, list) and missing_skills:
                explanation += (
                    f" Penguatan pada {missing_skills[0]} dapat membantu meningkatkan kesiapan kandidat untuk role ini."
                )
            return explanation
    
        return (
            f"Kandidat berada pada kategori {fit_label} untuk posisi {job_title} "
            f"dengan skor {score:.0f}%. Hasil ini dapat digunakan sebagai prioritas awal untuk evaluasi lebih lanjut."
        )

    async def analyze_target(self, candidate_profile: dict, job: dict) -> str:
        fallback = self._fallback(candidate_profile, job)
        if not self.enabled or self.model is None:
            return fallback

        prompt = f"""
Tugas:
Jelaskan seberapa cocok kandidat untuk posisi "{job.get('job_title')}" berdasarkan hasil pencocokan sistem.
Aturan:
- Gunakan Bahasa Indonesia.
- Maksimal 3 kalimat.
- Ramah, profesional, dan praktis.
- Fokus pada skor kecocokan, skill yang cocok, skill gap, dan gap pengalaman.
- Jika ada gap, berikan saran pengembangan yang singkat dan membangun.
- Jangan membuat asumsi, pengalaman, proyek, sertifikasi, atau kemampuan yang tidak ada pada data.
- Jangan mengubah, mengoreksi, atau mempertanyakan skor kecocokan.
- Jangan gunakan bullet point, numbering, heading, markdown, atau JSON.
Data:
Role: {job.get('job_title')}
Skor Kecocokan: {job.get('match_score_percent')}%
Skill Cocok: {job.get('matched_skills')}
Skill Gap: {job.get('missing_skills')}
Pengalaman Kandidat: {candidate_profile.get('experience_years')} tahun
Minimum Pengalaman: {job.get('minimum_experience_years')} tahun
Experience Gap: {job.get('experience_gap_years')} tahun
Pendidikan: {candidate_profile.get('education_level_cand')}
"""
        try:
            text = await self._generate_content_with_retry(prompt)
            return text or fallback
        except Exception:
            return fallback

    async def extract_profile(self, text: str) -> dict | None:
        if not self.enabled or self.model is None:
            return None

        prompt = f"""
Tugas Anda adalah mengekstrak profil kandidat dari teks CV berikut.
Teks CV mungkin memiliki spasi yang tidak beraturan akibat ekstraksi PDF. Abaikan kesalahan spasi/ketik tersebut.
Format JSON yang diharapkan:
{{
  "candidate_name": "Nama Lengkap Kandidat",
  "cand_tech_skills": ["skill teknis 1", "skill teknis 2"],
  "cand_soft_skills": ["soft skill 1", "soft skill 2"],
  "experience_years": 2.5,
  "education_level_cand": "Bachelor's"
}}
Panduan:
- "candidate_name": Ekstrak nama lengkap kandidat jika ada di teks CV. Kembalikan null jika tidak ditemukan.
- "cand_tech_skills": Ekstrak skill teknis secara natural seperti yang tertulis di CV.
- "cand_soft_skills": Ekstrak soft skill secara natural.
- "experience_years": Angka desimal mewakili total lama pengalaman.
- "education_level_cand": Pilih salah satu: "High School / Diploma", "Associate's", "Bachelor's", "Master's", "PhD".
Teks CV:
{text[:4000]}
"""
        try:
            out_text = await self._generate_content_with_retry(prompt, is_json=True)
            return json.loads(out_text)
        except Exception:
            return None

    async def analyze_and_mock_target_role(self, target_role: str) -> dict | None:
        if not self.enabled or self.model is None:
            return None

        prompt = f"""
Pengguna menargetkan peran "{target_role}".
Berikan JSON dengan dua kunci utama:
1. "synonyms": Array berisi maksimal 8 variasi istilah profesional / sinonim jabatan untuk peran tersebut.
2. "mock_profile": Spesifikasi standar industri untuk peran tersebut.

Format JSON yang diharapkan:
{{
  "synonyms": ["Role 1", "Role 2"],
  "mock_profile": {{
    "job_title": "{target_role}",
    "industry_sector_job": "Sektor industri relevan",
    "req_tech_skills": ["skill1", "skill2", "skill3"],
    "req_soft_skills": ["skill1", "skill2"],
    "minimum_experience_years": 2,
    "education_level_job": "Bachelor's"
  }}
}}
"""
        try:
            text = await self._generate_content_with_retry(prompt, is_json=True)
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                return None
            return parsed
        except Exception:
            return None
