import re
from typing import Any

import fitz
import pandas as pd
from rapidfuzz import process, fuzz


EDU_PATTERNS = [
    (r"\b(s3|phd|doctor|doctoral|doktor)\b", "PhD"),
    (r"\b(s2|master|magister)\b", "Master's"),
    (r"\b(s1|bachelor|sarjana)\b", "Bachelor's"),
    (r"\b(d3|diploma|associate)\b", "Associate"),
    (r"\b(sma|smk|high school)\b", "High School / Diploma"),
]


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join(page.get_text("text") for page in doc)
        doc.close()
    except Exception as exc:
        raise ValueError(f"Gagal membaca PDF: {exc}") from exc

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise ValueError("PDF berhasil dibaca, tetapi teks kosong. CV kemungkinan berupa scan gambar.")
    return text


def parse_skill_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v).lower().strip() for v in value if str(v).strip()]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    value = str(value)
    try:
        import ast
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(v).lower().strip() for v in parsed if str(v).strip()]
    except Exception:
        pass
    return [item.lower().strip() for item in re.split(r",|;|\|", value) if item.strip()]


def build_known_skills_split(job_catalog: pd.DataFrame) -> tuple[list[str], list[str]]:
    known_tech = set()
    known_soft = set()
    
    if "req_tech_skills" in job_catalog.columns:
        for value in job_catalog["req_tech_skills"].dropna().tolist():
            known_tech.update(parse_skill_list(value))
            
    if "req_soft_skills" in job_catalog.columns:
        for value in job_catalog["req_soft_skills"].dropna().tolist():
            known_soft.update(parse_skill_list(value))

    tech_list = sorted([s for s in known_tech if s], key=len, reverse=True)
    soft_list = sorted([s for s in known_soft if s], key=len, reverse=True)
    return tech_list, soft_list


def standardize_skills(extracted_skills: list[str], known_skills: list[str], threshold: float = 80.0) -> list[str]:
    if not extracted_skills or not known_skills:
        return []
    
    standardized = []
    for skill in extracted_skills:
        skill = str(skill).strip()
        if not skill:
            continue
        match = process.extractOne(skill, known_skills, scorer=fuzz.token_set_ratio, score_cutoff=threshold)
        if match:
            standardized.append(match[0])
        else:
            standardized.append(skill.lower())
    return sorted(list(set(standardized)))


def extract_skills_from_text(text: str, known_tech: list[str], known_soft: list[str]) -> tuple[list[str], list[str]]:
    def extract_from_list(vocab: list[str]):
        if not vocab: return []
        escaped = [re.escape(skill) for skill in vocab if skill]
        pattern = re.compile(r"(?<!\w)(?:" + "|".join(escaped) + r")(?!\w)", re.IGNORECASE)
        found = {match.group(0).lower().strip() for match in pattern.finditer(text)}
        return sorted(found)
        
    return extract_from_list(known_tech), extract_from_list(known_soft)


def extract_experience_years(text: str) -> float:
    patterns = [
        r"(\d+(?:\.\d+)?)\s*(?:\+)?\s*(?:years?|yrs?)\s+(?:of\s+)?experience",
        r"experience\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(?:\+)?\s*(?:years?|yrs?)",
        r"work\s+experience\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(?:\+)?\s*(?:years?|yrs?)",
        r"pengalaman\s+(\d+(?:\.\d+)?)\s*tahun",
        r"pengalaman\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*tahun",
        r"(\d+(?:\.\d+)?)\s*tahun\s+pengalaman",
    ]
    values = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            try:
                values.append(float(match.group(1)))
            except Exception:
                pass
    return max(values) if values else 0.0


def extract_education(text: str) -> str:
    lowered = text.lower()
    for pattern, label in EDU_PATTERNS:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            return label
    return "Bachelor's"


def infer_candidate_sector(tech_skills: list[str], soft_skills: list[str]) -> str:
    skill_text = " ".join(tech_skills + soft_skills).lower()
    sector_keywords = {
        "Technology & IT": [
            "python", "sql", "machine learning", "tensorflow", "backend",
            "frontend", "cloud", "cybersecurity", "software", "database",
        ],
        "Marketing & PR": [
            "marketing", "digital marketing", "seo", "campaign",
            "google analytics", "meta ads", "content planning",
            "copywriting", "social media marketing",
        ],
        "Sales & Retail": [
            "sales", "b2b sales", "crm", "lead generation", "sales pipeline",
            "customer outreach", "sales reporting", "negotiation",
        ],
        "Design, Media & Creative": [
            "design", "ui/ux", "poster", "canva", "creative messaging",
            "content scheduling",
        ],
        "Finance & Accounting": [
            "finance", "accounting", "banking", "risk management",
        ],
        "Operations & Supply Chain": [
            "logistics", "supply chain", "warehouse", "inventory",
        ],
        "Human Resources": [
            "recruitment", "hris", "payroll", "employee relations",
        ],
    }

    scores = {
        sector: sum(1 for keyword in keywords if keyword in skill_text)
        for sector, keywords in sector_keywords.items()
    }
    best_sector, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score > 0:
        return best_sector

    if any(token in skill_text for token in ["finance", "accounting", "banking", "risk management"]):
        return "Finance & Accounting"
    return "Other"


def build_candidate_profile_from_cv(text: str, known_tech: list[str], known_soft: list[str]) -> dict:
    tech_skills, soft_skills = extract_skills_from_text(text, known_tech, known_soft)
    profile = {
        "industry_sector_cand": infer_candidate_sector(tech_skills, soft_skills),
        "cand_tech_skills": tech_skills,
        "cand_soft_skills": soft_skills,
        "experience_years": extract_experience_years(text),
        "education_level_cand": extract_education(text),
    }
    return profile
