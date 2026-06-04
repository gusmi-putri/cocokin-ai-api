from pydantic import BaseModel, Field
from typing import Any


class CandidateProfile(BaseModel):
    candidate_name: str | None = Field(default=None, examples=["Budi"])
    industry_sector_cand: str = Field(default="Technology", examples=["Technology"])
    cand_tech_skills: list[str] = Field(default_factory=list, examples=[["python", "sql"]])
    cand_soft_skills: list[str] = Field(default_factory=list, examples=[["leadership", "communication"]])
    experience_years: float = Field(default=0, ge=0, examples=[2])
    education_level_cand: str = Field(default="Bachelor's", examples=["Bachelor's"])


class JobRecommendation(BaseModel):
    job_id: Any
    job_title: str | None = None
    industry_sector_job: str | None = None
    req_tech_skills: Any | None = None
    req_soft_skills: Any | None = None
    minimum_experience_years: float | None = None
    model_score: float | None = None
    final_rank_score: float | None = None
    match_score_percent: float
    fit_category: str
    user_fit_label: str | None = None
    weighted_skill_score: float | None = None
    sector_similarity_score: float | None = None
    skill_completeness_factor: float | None = None
    matched_skills: list[str] | None = None
    missing_skills: list[str] | None = None
    why_you_match: str | None = None


class CVRecommendationResponse(BaseModel):
    filename: str
    extracted_profile: CandidateProfile
    extracted_text_preview: str
    top_recommendations: list[JobRecommendation]


class TargetedRoleResponse(BaseModel):
    filename: str
    target_role: str
    extracted_profile: CandidateProfile
    match_score_percent: float
    fit_category: str
    missing_skills: list[str] | None = None
    matched_skills: list[str] | None = None
    experience_gap_years: float | None = None
    edu_gap: float | None = None
    reasoning: str | None = None
    similar_jobs: list[JobRecommendation] | None = None
