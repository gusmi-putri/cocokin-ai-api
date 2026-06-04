from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.schemas import CandidateProfile, CVRecommendationResponse, JobRecommendation, TargetedRoleResponse
from app.services.cv_parser import (
    build_candidate_profile_from_cv,
    build_known_skills_split,
    extract_text_from_pdf_bytes,
    infer_candidate_sector,
    standardize_skills,
)
from app.services.gemini_service import GeminiExplanationService

# Try to import recommender, but allow failure for testing
try:
    from app.services.recommender import CocokinRecommender
    RECOMMENDER_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    print(f"Warning: Recommender not available: {e}")
    RECOMMENDER_AVAILABLE = False
    CocokinRecommender = None

settings = get_settings()
app = FastAPI(
    title="Cocokin AI Recommendation API",
    description="Two-Tower job matching API with CV PDF parsing and Gemini why-you-match explanation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

recommender: CocokinRecommender | None = None
gemini_service: GeminiExplanationService | None = None


@app.on_event("startup")
def startup_event():
    global recommender, gemini_service
    if RECOMMENDER_AVAILABLE:
        try:
            recommender = CocokinRecommender(settings.resolved_artifact_dir())
            if getattr(recommender, "model_load_error", None):
                print(f"Warning: Recommender running in fallback mode: {recommender.model_load_error}")
        except Exception as e:
            print(f"Warning: Failed to load recommender: {e}")
    
    try:
        gemini_service = GeminiExplanationService(
            api_key=settings.gemini_api_key,
            model_name=settings.gemini_model or "gemini-1.5-flash",
        )
    except Exception as e:
        print(f"Warning: Failed to load Gemini service: {e}")


@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health_check():
    gemini_enabled = gemini_service is not None and gemini_service.enabled
    return {
        "status": "ok",
        "message": "Cocokin AI API is running",
        "gemini_enabled": gemini_enabled,
        "artifact_dir": settings.resolved_artifact_dir(),
        "recommender_ready": recommender is not None,
        "recommender_model_status": getattr(recommender, "model_status", None),
        "recommender_model_load_error": getattr(recommender, "model_load_error", None),
    }


@app.post("/recommend", response_model=list[JobRecommendation], include_in_schema=False)
def recommend(profile: CandidateProfile, top_n: int = Query(default=3, ge=1, le=20)):
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender belum siap.")
    return recommender.recommend(profile.model_dump(), top_n=top_n)


@app.post("/recommend-from-cv", response_model=CVRecommendationResponse)
async def recommend_from_cv(
    file: UploadFile = File(...),
    use_gemini: bool = Query(default=True),
    use_vertex: bool | None = Query(default=None, include_in_schema=False),
):
    top_n = 3
    enable_gemini = use_gemini if use_vertex is None else use_vertex
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender belum siap.")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File harus berformat PDF.")

    pdf_bytes = await file.read()
    try:
        text = extract_text_from_pdf_bytes(pdf_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    known_tech, known_soft = build_known_skills_split(recommender.job_catalog)
    
    profile_dict = None
    if enable_gemini and gemini_service is not None:
        extracted = await gemini_service.extract_profile(text)
        if extracted:
            tech = standardize_skills(extracted.get("cand_tech_skills", []), known_tech)
            soft = standardize_skills(extracted.get("cand_soft_skills", []), known_soft)
            sector = infer_candidate_sector(tech, soft)
            profile_dict = {
                "candidate_name": extracted.get("candidate_name"),
                "industry_sector_cand": sector,
                "cand_tech_skills": tech,
                "cand_soft_skills": soft,
                "experience_years": extracted.get("experience_years", 0.0),
                "education_level_cand": extracted.get("education_level_cand", "Bachelor's")
            }
            
    if profile_dict is None:
        profile_dict = build_candidate_profile_from_cv(text, known_tech, known_soft)
        profile_dict["candidate_name"] = None

    recommendations = recommender.recommend(profile_dict, top_n=top_n)

    if enable_gemini and gemini_service is not None:
        batch_explanations = await gemini_service.explain_batch(profile_dict, recommendations)
        for item in recommendations:
            job_id_str = str(item.get("job_id", ""))
            item["why_you_match"] = batch_explanations.get(job_id_str) or await gemini_service.explain(profile_dict, item)

    return {
        "filename": file.filename,
        "extracted_profile": profile_dict,
        "extracted_text_preview": text[:700],
        "top_recommendations": recommendations,
    }


@app.post("/analyze-target-role", response_model=TargetedRoleResponse)
async def analyze_target_role(
    target_role: str = Query(..., description="The role the candidate wants to apply for (e.g., 'Data Scientist')"),
    file: UploadFile = File(...),
):
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender belum siap.")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File harus berformat PDF.")

    pdf_bytes = await file.read()
    try:
        text = extract_text_from_pdf_bytes(pdf_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    known_tech, known_soft = build_known_skills_split(recommender.job_catalog)
    
    profile_dict = None
    if gemini_service is not None:
        extracted = await gemini_service.extract_profile(text)
        if extracted:
            tech = standardize_skills(extracted.get("cand_tech_skills", []), known_tech)
            soft = standardize_skills(extracted.get("cand_soft_skills", []), known_soft)
            sector = infer_candidate_sector(tech, soft)
            profile_dict = {
                "candidate_name": extracted.get("candidate_name"),
                "industry_sector_cand": sector,
                "cand_tech_skills": tech,
                "cand_soft_skills": soft,
                "experience_years": extracted.get("experience_years", 0.0),
                "education_level_cand": extracted.get("education_level_cand", "Bachelor's")
            }
            
    if profile_dict is None:
        profile_dict = build_candidate_profile_from_cv(text, known_tech, known_soft)
        profile_dict["candidate_name"] = None
    
    expanded_terms = []
    mock_job = None
    if gemini_service is not None:
        mock_result = await gemini_service.analyze_and_mock_target_role(target_role)
        if mock_result:
            expanded_terms = mock_result.get("synonyms", [])
            mock_job = mock_result.get("mock_profile")

    target_job = recommender.analyze_target_role(profile_dict, target_role, expanded_terms=expanded_terms)
    if not target_job:
        if not mock_job:
            raise HTTPException(status_code=404, detail=f"Tidak ditemukan posisi yang mirip dengan '{target_role}' di katalog.")
            
        target_job = recommender.score_mock_job(profile_dict, mock_job)
        
        all_jobs_scored = recommender.recommend(profile_dict, top_n=20)
        same_sector_jobs = [j for j in all_jobs_scored if j.get("industry_sector_job") == mock_job.get("industry_sector_job")]
        if len(same_sector_jobs) >= 5:
            target_job["similar_jobs"] = same_sector_jobs[:5]
        else:
            target_job["similar_jobs"] = all_jobs_scored[:5]
        
    reasoning = None
    if gemini_service is not None:
        reasoning = await gemini_service.analyze_target(profile_dict, target_job)

    return {
        "filename": file.filename,
        "target_role": target_job.get("job_title", target_role),
        "extracted_profile": profile_dict,
        "match_score_percent": target_job.get("match_score_percent", 0.0),
        "fit_category": target_job.get("fit_category", "no_fit"),
        "missing_skills": target_job.get("missing_skills", []),
        "matched_skills": target_job.get("matched_skills", []),
        "experience_gap_years": target_job.get("experience_gap_years", 0.0),
        "edu_gap": target_job.get("edu_gap", 0.0),
        "reasoning": reasoning,
        "similar_jobs": target_job.get("similar_jobs", []),
    }
