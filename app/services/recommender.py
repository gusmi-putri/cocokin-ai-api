import ast
from difflib import SequenceMatcher
import json
import os
import re
from typing import Any

# SentenceTransformer runs on PyTorch here. This prevents Transformers from
# importing TensorFlow adapters that are incompatible with Keras 3.
os.environ.setdefault("USE_TF", "0")

import joblib
import numpy as np
import pandas as pd


try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    TENSORFLOW_AVAILABLE = True
    TENSORFLOW_IMPORT_ERROR = None
except Exception as exc:
    tf = None
    keras = None
    layers = None
    TENSORFLOW_AVAILABLE = False
    TENSORFLOW_IMPORT_ERROR = exc


try:
    from sentence_transformers import SentenceTransformer

    SBERT_AVAILABLE = True
    SBERT_IMPORT_ERROR = None
except Exception as exc:
    SentenceTransformer = None
    SBERT_AVAILABLE = False
    SBERT_IMPORT_ERROR = exc


if TENSORFLOW_AVAILABLE:

    @tf.keras.utils.register_keras_serializable(package="Cocokin")
    class MatchingFeatureLayer(layers.Layer):
        def call(self, inputs):
            candidate_embedding, job_embedding = inputs

            abs_diff = tf.abs(candidate_embedding - job_embedding)
            multiply = candidate_embedding * job_embedding

            candidate_norm = tf.nn.l2_normalize(candidate_embedding, axis=-1)
            job_norm = tf.nn.l2_normalize(job_embedding, axis=-1)
            cosine_similarity = tf.reduce_sum(
                candidate_norm * job_norm,
                axis=-1,
                keepdims=True,
            )

            return tf.concat([abs_diff, multiply, cosine_similarity], axis=-1)

        def get_config(self):
            return super().get_config()


    @tf.keras.utils.register_keras_serializable(package="Cocokin")
    class WeightedHuberMAELoss(tf.keras.losses.Loss):
        def __init__(
            self,
            delta=0.10,
            alpha=0.70,
            name="weighted_huber_mae_loss",
            **kwargs,
        ):
            super().__init__(name=name, **kwargs)
            self.delta = delta
            self.alpha = alpha
            self.huber = tf.keras.losses.Huber(delta=delta, reduction="none")
            self.mae = tf.keras.losses.MeanAbsoluteError(reduction="none")

        def call(self, y_true, y_pred):
            return (
                self.alpha * self.huber(y_true, y_pred)
                + (1.0 - self.alpha) * self.mae(y_true, y_pred)
            )

        def get_config(self):
            config = super().get_config()
            config.update({"delta": self.delta, "alpha": self.alpha})
            return config

else:
    MatchingFeatureLayer = None
    WeightedHuberMAELoss = None


EDU_RANK = {
    "": 0,
    "High School": 1,
    "High School / Diploma": 1,
    "SMA": 1,
    "SMK": 1,
    "Diploma": 2,
    "D3": 2,
    "Associate's": 2,
    "Associate": 2,
    "Bachelor's": 3,
    "Bachelor": 3,
    "S1": 3,
    "Sarjana": 3,
    "Master's": 4,
    "Master": 4,
    "S2": 4,
    "Magister": 4,
    "PhD": 5,
    "Doctorate": 5,
    "S3": 5,
}


RELATED_SECTORS = {
    ("Finance & Accounting", "Finance & Banking"): 0.8,
    ("Finance & Banking", "Finance & Accounting"): 0.8,
    ("Banking & Insurance", "Finance & Accounting"): 0.7,
    ("Finance & Accounting", "Banking & Insurance"): 0.7,
    ("Marketing", "Sales"): 0.55,
    ("Sales", "Marketing"): 0.55,
    ("Marketing & PR", "Sales & Retail"): 0.65,
    ("Sales & Retail", "Marketing & PR"): 0.65,
    ("Customer Service", "Sales"): 0.5,
    ("Sales", "Customer Service"): 0.5,
    ("Customer Service", "Sales & Retail"): 0.5,
    ("Sales & Retail", "Customer Service"): 0.5,
    ("Technology", "Project / Product Management"): 0.55,
    ("Project / Product Management", "Technology"): 0.55,
    ("Technology", "Engineering"): 0.45,
    ("Engineering", "Technology"): 0.45,
    ("Technology & IT", "Engineering"): 0.45,
    ("Engineering", "Technology & IT"): 0.45,
    ("Operations & Logistics", "Engineering"): 0.45,
    ("Engineering", "Operations & Logistics"): 0.45,
    ("Operations & Supply Chain", "Engineering"): 0.45,
    ("Engineering", "Operations & Supply Chain"): 0.45,
    ("Design, Media & Creative", "Marketing & PR"): 0.45,
    ("Marketing & PR", "Design, Media & Creative"): 0.45,
}


ROLE_STOPWORDS = {
    "a", "an", "and", "as", "for", "in", "of", "on", "or", "the", "to", "with",
    "remote", "full", "time", "part", "senior", "sr", "junior", "jr", "staff",
    "associate", "assistant",
}


class CocokinRecommender:
    def __init__(self, artifact_dir: str):
        self.artifact_dir = artifact_dir

        self.model = None
        self.sbert_model = None
        self.numeric_scaler = None

        self.model_status = "not_loaded"
        self.model_load_error = None

        self.numeric_features: list[str] = []
        self.median_salary = 0.0

        self.job_catalog: pd.DataFrame | None = None
        self.cached_job_tech_emb = None
        self.cached_job_soft_emb = None

        self.load_artifacts()

    # =========================================================
    # Artifact loading
    # =========================================================

    def load_artifacts(self):
        required_files = [
            "finetuned_sbert",
            "cocokin_sbert_keras_model.keras",
            "numeric_scaler_sbert.pkl",
            "numeric_features_sbert.json",
            "job_catalog.csv",
        ]

        missing = [
            name
            for name in required_files
            if not os.path.exists(os.path.join(self.artifact_dir, name))
        ]

        if missing:
            raise FileNotFoundError(
                f"Artifacts belum lengkap di folder '{self.artifact_dir}'. Missing: {missing}"
            )

        with open(
            os.path.join(self.artifact_dir, "numeric_features_sbert.json"),
            "r",
            encoding="utf-8",
        ) as f:
            config_data = json.load(f)
            self.numeric_features = config_data.get("numeric_features", [])
            self.median_salary = self.safe_float(
                config_data.get("median_salary", 0.0),
                0.0,
            )

        self.job_catalog = pd.read_csv(os.path.join(self.artifact_dir, "job_catalog.csv"))
        self._ensure_job_catalog_columns()

        if not SBERT_AVAILABLE:
            self.model_status = "fallback_no_sbert"
            self.model_load_error = f"SentenceTransformer tidak tersedia: {SBERT_IMPORT_ERROR}"
            return

        try:
            self.sbert_model = SentenceTransformer(
                os.path.join(self.artifact_dir, "finetuned_sbert")
            )
        except Exception as exc:
            self.model_status = "fallback_sbert_failed"
            self.model_load_error = f"Gagal memuat SBERT: {exc}"
            return

        if not TENSORFLOW_AVAILABLE:
            self.model_status = "fallback_no_tensorflow"
            self.model_load_error = f"TensorFlow tidak tersedia: {TENSORFLOW_IMPORT_ERROR}"
            return

        try:
            self.numeric_scaler = joblib.load(
                os.path.join(self.artifact_dir, "numeric_scaler_sbert.pkl")
            )

            model_path = os.path.join(
                self.artifact_dir,
                "cocokin_sbert_keras_model.keras",
            )

            self.model = keras.models.load_model(
                model_path,
                custom_objects={
                    "MatchingFeatureLayer": MatchingFeatureLayer,
                    "WeightedHuberMAELoss": WeightedHuberMAELoss,
                },
            )

            self.model_status = "tensorflow_sbert_loaded"

        except Exception as exc:
            self.model = None
            self.model_status = "fallback_model_load_failed"
            self.model_load_error = str(exc)

        self._cache_job_embeddings()

    def _ensure_job_catalog_columns(self):
        if self.job_catalog is None:
            return

        default_columns = {
            "job_id": "",
            "job_title": "",
            "industry_sector_job": "Other",
            "education_level_job": "",
            "req_tech_skills": "[]",
            "req_soft_skills": "[]",
            "minimum_experience_years": 0.0,
            "required_skill_count": np.nan,
            "skill_completeness_factor": np.nan,
            "market_demand_score": 0.0,
            "demand_n": np.nan,
            "avg_salary": self.median_salary,
            "salary_percentile": np.nan,
            "remote_allowed": 0.0,
        }

        for col, default_value in default_columns.items():
            if col not in self.job_catalog.columns:
                self.job_catalog[col] = default_value

    def _cache_job_embeddings(self):
        if self.sbert_model is None or self.job_catalog is None:
            return

        job_tech_list = []
        job_soft_list = []

        for _, job_row in self.job_catalog.iterrows():
            job_title = self._normalize_text(job_row.get("job_title", ""))
            job_edu = self._normalize_text(job_row.get("education_level_job", ""))

            job_req_tech = self.skill_list_to_text(job_row.get("req_tech_skills", ""))
            job_req_soft = self.skill_list_to_text(job_row.get("req_soft_skills", ""))

            job_tech_list.append(
                f"Posisi {job_title} pendidikan minimum {job_edu}. "
                f"Dibutuhkan skill teknis: {job_req_tech}"
            )
            job_soft_list.append(
                f"Soft skill yang dibutuhkan: {job_req_soft}"
            )

        self.cached_job_tech_emb = self.sbert_model.encode(
            job_tech_list,
            convert_to_numpy=True,
        ).astype("float32")

        self.cached_job_soft_emb = self.sbert_model.encode(
            job_soft_list,
            convert_to_numpy=True,
        ).astype("float32")

    # =========================================================
    # Label / category helpers
    # =========================================================

    @staticmethod
    def score_to_category(score: float) -> str:
        if score >= 0.75:
            return "strong_fit"
        if score >= 0.55:
            return "good_fit"
        if score >= 0.35:
            return "partial_fit"
        if score >= 0.18:
            return "weak_fit"
        return "no_fit"

    @staticmethod
    def category_to_user_label(category: str) -> str:
        if category in {"good_fit", "strong_fit"}:
            return "Cocok"
        if category == "partial_fit":
            return "Lumayan Cocok"
        return "Tidak Cocok"

    # =========================================================
    # Normalization helpers
    # =========================================================

    @staticmethod
    def safe_float(value: Any, default: float = 0.0) -> float:
        value = pd.to_numeric(value, errors="coerce")
        if pd.isna(value):
            return float(default)
        return float(value)

    @staticmethod
    def _normalize_text(text: Any) -> str:
        if text is None:
            return ""
        try:
            if pd.isna(text):
                return ""
        except Exception:
            pass

        return str(text).replace("\n", " ").strip()

    @staticmethod
    def normalize_skill(skill: Any) -> str:
        return (
            str(skill)
            .lower()
            .strip()
            .replace("_", " ")
            .replace("-", " ")
        )

    @staticmethod
    def parse_skill_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [
                CocokinRecommender.normalize_skill(v)
                for v in value
                if str(v).strip()
            ]

        if isinstance(value, (tuple, set)):
            return [
                CocokinRecommender.normalize_skill(v)
                for v in value
                if str(v).strip()
            ]

        if value is None:
            return []

        if isinstance(value, float) and pd.isna(value):
            return []

        value_str = str(value).strip()
        if not value_str or value_str.lower() in {"nan", "none", "null", "[]"}:
            return []

        try:
            parsed = ast.literal_eval(value_str)

            if isinstance(parsed, list):
                return [
                    CocokinRecommender.normalize_skill(v)
                    for v in parsed
                    if str(v).strip()
                ]

            if isinstance(parsed, (tuple, set)):
                return [
                    CocokinRecommender.normalize_skill(v)
                    for v in parsed
                    if str(v).strip()
                ]

        except Exception:
            pass

        return [
            CocokinRecommender.normalize_skill(item)
            for item in re.split(r",|;|\|", value_str)
            if item.strip()
        ]

    def skill_list_to_text(self, value: Any) -> str:
        return " ".join(self.parse_skill_list(value))

    @staticmethod
    def normalize_education(value: Any) -> str:
        if value is None:
            return ""

        value_str = str(value).strip()
        if not value_str:
            return ""

        key = value_str.lower()

        mapping = {
            "sma": "High School",
            "smk": "High School",
            "high school": "High School",
            "high school / diploma": "High School / Diploma",
            "diploma": "Diploma",
            "d3": "Diploma",
            "associate": "Associate",
            "associate's": "Associate's",
            "s1": "Bachelor's",
            "sarjana": "Bachelor's",
            "bachelor": "Bachelor's",
            "bachelors": "Bachelor's",
            "bachelor's": "Bachelor's",
            "undergraduate": "Bachelor's",
            "s2": "Master's",
            "magister": "Master's",
            "master": "Master's",
            "masters": "Master's",
            "master's": "Master's",
            "s3": "PhD",
            "phd": "PhD",
            "doctorate": "Doctorate",
            "doctoral": "Doctorate",
        }

        return mapping.get(key, value_str)

    @classmethod
    def education_rank(cls, value: Any) -> int:
        normalized = cls.normalize_education(value)
        return EDU_RANK.get(normalized, 0)

    @staticmethod
    def sector_similarity(candidate_sector: str, job_sector: str) -> float:
        candidate_sector = str(candidate_sector or "").strip()
        job_sector = str(job_sector or "").strip()

        if candidate_sector == job_sector:
            return 1.0

        return RELATED_SECTORS.get(
            (candidate_sector, job_sector),
            0.25 if "Other" in {candidate_sector, job_sector} else 0.1,
        )

    # =========================================================
    # Feature engineering for inference
    # =========================================================

    def compute_pair_feature_values(
        self,
        candidate_profile: dict,
        job_row: pd.Series | dict,
    ) -> dict[str, Any]:
        cand_tech = set(self.parse_skill_list(candidate_profile.get("cand_tech_skills", [])))
        cand_soft = set(self.parse_skill_list(candidate_profile.get("cand_soft_skills", [])))
        candidate_skills = cand_tech | cand_soft

        req_tech = set(self.parse_skill_list(job_row.get("req_tech_skills", [])))
        req_soft = set(self.parse_skill_list(job_row.get("req_soft_skills", [])))
        required_set = req_tech | req_soft

        matched_tech = cand_tech & req_tech
        matched_soft = cand_soft & req_soft
        matched_skills = candidate_skills & required_set
        missing_skills = required_set - candidate_skills

        skill_match_ratio = len(matched_skills) / len(required_set) if required_set else 0.0
        tech_score = len(matched_tech) / len(req_tech) if req_tech else 0.0
        soft_score = len(matched_soft) / len(req_soft) if req_soft else 0.0

        if req_tech and req_soft:
            if tech_score == 0.0:
                weighted_skill_score = soft_score * 0.10
            else:
                weighted_skill_score = (tech_score * 0.85) + (soft_score * 0.15)
        elif req_tech:
            weighted_skill_score = tech_score
        elif req_soft:
            weighted_skill_score = soft_score
        else:
            weighted_skill_score = 0.0

        experience_years = self.safe_float(candidate_profile.get("experience_years", 0), 0)
        minimum_experience_years = self.safe_float(
            job_row.get("minimum_experience_years", 0),
            0,
        )

        candidate_edu_rank = self.education_rank(
            candidate_profile.get("education_level_cand", "")
        )
        job_edu_rank = self.education_rank(
            job_row.get("education_level_job", "")
        )

        required_skill_count = int(
            self.safe_float(
                job_row.get("required_skill_count", len(required_set)),
                len(required_set),
            )
        )

        default_skill_completeness = min(1.0, required_skill_count / 5.0)

        skill_completeness_factor = self.safe_float(
            job_row.get("skill_completeness_factor", default_skill_completeness),
            default_skill_completeness,
        )

        avg_salary_value = job_row.get("avg_salary", self.median_salary)
        if pd.isna(avg_salary_value):
            avg_salary_value = self.median_salary

        avg_salary = self.safe_float(avg_salary_value, self.median_salary)

        market_demand_score = self.safe_float(
            job_row.get("market_demand_score", job_row.get("demand_n", 0)),
            0,
        )

        demand_n = self.safe_float(
            job_row.get("demand_n", market_demand_score),
            market_demand_score,
        )

        # Neutral-ish fallback. If salary_percentile was used during training
        # but absent from job_catalog, 0.5 is safer than 0 because 0 can mean
        # "very low salary" to the scaler/model.
        salary_percentile = self.safe_float(
            job_row.get("salary_percentile", 0.5),
            0.5,
        )

        experience_gap_years = max(0, minimum_experience_years - experience_years)
        edu_gap = max(0, job_edu_rank - candidate_edu_rank)

        return {
            "has_req_tech": len(req_tech) > 0,

            "experience_years": experience_years,
            "minimum_experience_years": minimum_experience_years,
            "skill_count": len(candidate_skills),

            "market_demand_score": market_demand_score,
            "demand_n": demand_n,
            "avg_salary": avg_salary,
            "salary_percentile": salary_percentile,
            "remote_allowed": self.safe_float(job_row.get("remote_allowed", 0), 0),

            "candidate_edu_rank": candidate_edu_rank,
            "job_edu_rank": job_edu_rank,

            "skill_match_ratio": skill_match_ratio,
            "weighted_skill_score": weighted_skill_score,
            "hard_skill_score": tech_score,
            "domain_skill_score": tech_score,
            "tool_score": tech_score,
            "soft_skill_score": soft_score,

            "sector_similarity_score": self.sector_similarity(
                candidate_profile.get("industry_sector_cand", ""),
                job_row.get("industry_sector_job", ""),
            ),

            "skill_completeness_factor": skill_completeness_factor,
            "missing_skill_count": len(missing_skills),
            "experience_gap_years": experience_gap_years,
            "edu_gap": edu_gap,

            "matched_skills": sorted(list(matched_skills)),
            "missing_skills": sorted(list(missing_skills)),
        }

    @staticmethod
    def fallback_model_score(feature_values: dict[str, Any]) -> float:
        experience_fit = 1.0 - min(
            float(feature_values.get("experience_gap_years", 0.0)) / 5.0,
            1.0,
        )

        education_fit = 1.0 - min(
            float(feature_values.get("edu_gap", 0.0)) / 3.0,
            1.0,
        )

        score = (
            float(feature_values.get("weighted_skill_score", 0.0)) * 0.55
            + float(feature_values.get("sector_similarity_score", 0.0)) * 0.20
            + experience_fit * 0.15
            + education_fit * 0.10
        )

        return float(np.clip(score, 0, 1))

    # =========================================================
    # Scoring helpers
    # =========================================================

    @staticmethod
    def compute_final_rank_score(
        model_score: float,
        weighted_skill_score: float,
        sector_similarity_score: float,
        skill_completeness_factor: float,
        has_req_tech: bool,
        hard_skill_score: float,
    ) -> float:
        tech_penalty = 0.5 if (hard_skill_score == 0.0 and has_req_tech) else 1.0

        final_score = (
            model_score * 0.40
            + weighted_skill_score * 0.45
            + sector_similarity_score * 0.15
        ) * skill_completeness_factor * tech_penalty

        return float(np.clip(final_score, 0, 1))

    def _build_candidate_embedding_inputs(self, candidate_profile: dict) -> tuple[str, str]:
        cand_tech_str = self.skill_list_to_text(candidate_profile.get("cand_tech_skills", ""))
        cand_soft_str = self.skill_list_to_text(candidate_profile.get("cand_soft_skills", ""))

        cand_edu = self._normalize_text(candidate_profile.get("education_level_cand", ""))
        cand_exp = self._normalize_text(candidate_profile.get("experience_years", "0"))

        cand_tech_input = (
            f"Pendidikan {cand_edu} pengalaman {cand_exp} tahun. "
            f"Skill teknis: {cand_tech_str}"
        )

        cand_soft_input = f"Soft skill kandidat: {cand_soft_str}"

        return cand_tech_input, cand_soft_input

    def _build_job_embedding_inputs(self, job: pd.Series | dict) -> tuple[str, str]:
        job_title = self._normalize_text(job.get("job_title", ""))
        job_edu = self._normalize_text(job.get("education_level_job", ""))

        job_req_tech = self.skill_list_to_text(job.get("req_tech_skills", ""))
        job_req_soft = self.skill_list_to_text(job.get("req_soft_skills", ""))

        job_tech_input = (
            f"Posisi {job_title} pendidikan minimum {job_edu}. "
            f"Dibutuhkan skill teknis: {job_req_tech}"
        )

        job_soft_input = f"Soft skill yang dibutuhkan: {job_req_soft}"

        return job_tech_input, job_soft_input

    # =========================================================
    # Main recommendation
    # =========================================================

    def recommend(self, candidate_profile: dict, top_n: int = 5) -> list[dict]:
        if self.job_catalog is None:
            raise RuntimeError("Job catalog belum dimuat.")

        cand_tech_input, cand_soft_input = self._build_candidate_embedding_inputs(
            candidate_profile
        )

        numeric_rows = []
        feature_rows = []

        for _, job_row in self.job_catalog.iterrows():
            feature_values = self.compute_pair_feature_values(candidate_profile, job_row)
            numeric_rows.append(
                [feature_values.get(col, 0) for col in self.numeric_features]
            )
            feature_rows.append(feature_values)

        use_tensorflow_model = (
            self.model is not None
            and self.sbert_model is not None
            and self.numeric_scaler is not None
            and self.cached_job_tech_emb is not None
            and self.cached_job_soft_emb is not None
        )

        if use_tensorflow_model:
            cand_tech_emb_single = self.sbert_model.encode(
                [cand_tech_input],
                convert_to_numpy=True,
            ).astype("float32")

            cand_soft_emb_single = self.sbert_model.encode(
                [cand_soft_input],
                convert_to_numpy=True,
            ).astype("float32")

            num_jobs = len(self.job_catalog)

            cand_tech_emb = np.repeat(cand_tech_emb_single, num_jobs, axis=0)
            cand_soft_emb = np.repeat(cand_soft_emb_single, num_jobs, axis=0)

            numeric_df = pd.DataFrame(numeric_rows, columns=self.numeric_features)
            numeric_rows_scaled = self.numeric_scaler.transform(numeric_df).astype("float32")

            pred_scores = self.model.predict(
                {
                    "cand_tech_emb": cand_tech_emb,
                    "cand_soft_emb": cand_soft_emb,
                    "job_tech_emb": self.cached_job_tech_emb,
                    "job_soft_emb": self.cached_job_soft_emb,
                    "numeric_features": numeric_rows_scaled,
                },
                batch_size=256,
                verbose=0,
            ).reshape(-1)

        else:
            print(
                "WARNING: Using fallback scoring. "
                f"Status={self.model_status}, Error={self.model_load_error}"
            )

            pred_scores = np.array(
                [self.fallback_model_score(values) for values in feature_rows],
                dtype="float32",
            )

        df = self.job_catalog.copy()
        features_df = pd.DataFrame(feature_rows)

        df["model_score"] = np.clip(pred_scores, 0, 1)

        feature_cols_to_attach = [
            "weighted_skill_score",
            "sector_similarity_score",
            "skill_completeness_factor",
            "matched_skills",
            "missing_skills",
            "has_req_tech",
            "hard_skill_score",
            "soft_skill_score",
            "skill_match_ratio",
            "experience_gap_years",
            "edu_gap",
            "missing_skill_count",
        ]

        for col in feature_cols_to_attach:
            if col in features_df.columns:
                df[col] = features_df[col].values

        df["final_rank_score"] = df.apply(
            lambda row: self.compute_final_rank_score(
                model_score=float(row.get("model_score", 0.0)),
                weighted_skill_score=float(row.get("weighted_skill_score", 0.0)),
                sector_similarity_score=float(row.get("sector_similarity_score", 0.0)),
                skill_completeness_factor=float(row.get("skill_completeness_factor", 1.0)),
                has_req_tech=bool(row.get("has_req_tech", False)),
                hard_skill_score=float(row.get("hard_skill_score", 0.0)),
            ),
            axis=1,
        )

        df["match_score"] = df["final_rank_score"].clip(0, 1)
        df["match_score_percent"] = (df["match_score"] * 100).round(2)
        df["fit_category"] = df["match_score"].apply(self.score_to_category)
        df["user_fit_label"] = df["fit_category"].apply(self.category_to_user_label)

        output_cols = [
            "job_id",
            "job_title",
            "industry_sector_job",
            "req_tech_skills",
            "req_soft_skills",
            "minimum_experience_years",

            "model_score",
            "final_rank_score",
            "match_score_percent",
            "fit_category",
            "user_fit_label",

            "weighted_skill_score",
            "sector_similarity_score",
            "skill_completeness_factor",
            "skill_match_ratio",
            "hard_skill_score",
            "soft_skill_score",

            "experience_gap_years",
            "edu_gap",

            "matched_skills",
            "missing_skills",
            "missing_skill_count",
        ]

        output_cols = [col for col in output_cols if col in df.columns]

        records = (
            df.sort_values("final_rank_score", ascending=False)
            .head(top_n)
            [output_cols]
            .to_dict(orient="records")
        )

        for item in records:
            item["req_tech_skills"] = self.parse_skill_list(
                item.get("req_tech_skills")
            )
            item["req_soft_skills"] = self.parse_skill_list(
                item.get("req_soft_skills")
            )

            item["matched_skills"] = self.parse_skill_list(
                item.get("matched_skills", [])
            )
            item["missing_skills"] = self.parse_skill_list(
                item.get("missing_skills", [])
            )

            item["model_status"] = self.model_status

        return records

    # =========================================================
    # Scoring Gemini mock job for target role fallback
    # =========================================================

    def score_mock_job(self, candidate_profile: dict, mock_job: dict) -> dict:
        mock_job = dict(mock_job)
        mock_job["job_id"] = "mock_job_from_gemini"

        feature_values = self.compute_pair_feature_values(candidate_profile, mock_job)

        use_tensorflow_model = (
            self.model is not None
            and self.sbert_model is not None
            and self.numeric_scaler is not None
        )

        if use_tensorflow_model:
            cand_tech_input, cand_soft_input = self._build_candidate_embedding_inputs(
                candidate_profile
            )
            job_tech_input, job_soft_input = self._build_job_embedding_inputs(mock_job)

            cand_tech_emb = self.sbert_model.encode(
                [cand_tech_input],
                convert_to_numpy=True,
            ).astype("float32")

            cand_soft_emb = self.sbert_model.encode(
                [cand_soft_input],
                convert_to_numpy=True,
            ).astype("float32")

            job_tech_emb = self.sbert_model.encode(
                [job_tech_input],
                convert_to_numpy=True,
            ).astype("float32")

            job_soft_emb = self.sbert_model.encode(
                [job_soft_input],
                convert_to_numpy=True,
            ).astype("float32")

            numeric_row = [
                feature_values.get(col, 0)
                for col in self.numeric_features
            ]

            numeric_df = pd.DataFrame([numeric_row], columns=self.numeric_features)
            numeric_row_scaled = self.numeric_scaler.transform(numeric_df).astype("float32")

            pred_score = self.model.predict(
                {
                    "cand_tech_emb": cand_tech_emb,
                    "cand_soft_emb": cand_soft_emb,
                    "job_tech_emb": job_tech_emb,
                    "job_soft_emb": job_soft_emb,
                    "numeric_features": numeric_row_scaled,
                },
                batch_size=1,
                verbose=0,
            ).reshape(-1)[0]

        else:
            print(
                "WARNING: Using fallback scoring for mock job. "
                f"Status={self.model_status}, Error={self.model_load_error}"
            )
            pred_score = self.fallback_model_score(feature_values)

        model_score = float(np.clip(pred_score, 0, 1))

        weighted_skill_score = float(feature_values.get("weighted_skill_score", 0))
        sector_similarity_score = float(feature_values.get("sector_similarity_score", 0))
        skill_completeness_factor = float(
            feature_values.get("skill_completeness_factor", 1)
        )
        has_req_tech = bool(feature_values.get("has_req_tech", False))
        hard_skill_score = float(feature_values.get("hard_skill_score", 0.0))

        final_rank_score = self.compute_final_rank_score(
            model_score=model_score,
            weighted_skill_score=weighted_skill_score,
            sector_similarity_score=sector_similarity_score,
            skill_completeness_factor=skill_completeness_factor,
            has_req_tech=has_req_tech,
            hard_skill_score=hard_skill_score,
        )

        match_score = float(np.clip(final_rank_score, 0, 1))
        match_score_percent = round(match_score * 100, 2)
        fit_category = self.score_to_category(match_score)

        return {
            **mock_job,

            "req_tech_skills": self.parse_skill_list(mock_job.get("req_tech_skills")),
            "req_soft_skills": self.parse_skill_list(mock_job.get("req_soft_skills")),

            "model_score": model_score,
            "final_rank_score": final_rank_score,
            "match_score_percent": match_score_percent,
            "fit_category": fit_category,
            "user_fit_label": self.category_to_user_label(fit_category),

            "weighted_skill_score": weighted_skill_score,
            "sector_similarity_score": sector_similarity_score,
            "skill_completeness_factor": skill_completeness_factor,
            "skill_match_ratio": float(feature_values.get("skill_match_ratio", 0)),
            "hard_skill_score": hard_skill_score,
            "soft_skill_score": float(feature_values.get("soft_skill_score", 0)),

            "matched_skills": list(feature_values.get("matched_skills", [])),
            "missing_skills": list(feature_values.get("missing_skills", [])),
            "missing_skill_count": int(feature_values.get("missing_skill_count", 0)),

            "experience_gap_years": float(feature_values.get("experience_gap_years", 0)),
            "edu_gap": float(feature_values.get("edu_gap", 0)),

            "model_status": self.model_status,
        }

    # =========================================================
    # Analyze target role
    # =========================================================

    def analyze_target_role(
        self,
        candidate_profile: dict,
        target_role: str,
        expanded_terms: list[str] | None = None,
    ) -> dict | None:
        if self.job_catalog is None:
            raise RuntimeError("Job catalog belum dimuat.")

        all_jobs_scored = self.recommend(
            candidate_profile,
            top_n=len(self.job_catalog),
        )

        def normalize_title(value: str) -> str:
            return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))

        def meaningful_tokens(value: str) -> set[str]:
            return {
                token
                for token in re.findall(r"[a-z0-9]+", str(value).lower())
                if token not in ROLE_STOPWORDS and len(token) > 1
            }

        target_role_normalized = normalize_title(target_role)

        search_terms = [target_role_normalized, *(expanded_terms or [])]
        search_terms = [
            normalize_title(term)
            for term in search_terms
            if normalize_title(term)
        ]

        target_tokens = meaningful_tokens(target_role_normalized)

        def term_match_score(
            term: str,
            title: str,
            title_tokens: set[str],
            index: int,
        ) -> float:
            term_tokens = meaningful_tokens(term)

            if not term_tokens:
                return 0.0

            if len(term_tokens) == 1:
                token = next(iter(term_tokens))
                matched = term_tokens if token in title_tokens else set()

            elif f" {term} " in f" {title} ":
                matched = term_tokens

            else:
                matched = term_tokens & title_tokens
                if len(matched) < len(term_tokens):
                    return 0.0

            if not matched:
                return 0.0

            title_focus = len(matched) / max(len(title_tokens), 1)
            term_coverage = len(matched) / max(len(term_tokens), 1)

            if index == 0:
                return min(
                    1.0,
                    0.82 + (0.12 * term_coverage) + (0.06 * title_focus),
                )

            base = max(0.70, 0.88 - (index * 0.025))

            return min(
                0.96,
                base + (0.08 * title_focus) + (0.04 * term_coverage),
            )

        def title_score(job: dict) -> float:
            title = normalize_title(str(job.get("job_title", "")))
            title_tokens = meaningful_tokens(title)

            if not title_tokens:
                return 0.0

            if target_role_normalized == title:
                return 1.0

            for index, term in enumerate(search_terms):
                score = term_match_score(term, title, title_tokens, index)
                if score > 0:
                    return score

            if target_tokens:
                overlap_count = len(target_tokens & title_tokens)
                coverage = overlap_count / max(len(target_tokens), 1)
                title_focus = overlap_count / max(len(title_tokens), 1)

                if coverage >= 0.67:
                    return 0.72 + (0.18 * coverage) + (0.05 * title_focus)

                if coverage >= 0.50:
                    return 0.62 + (0.15 * coverage) + (0.05 * title_focus)

                if coverage > 0 and len(target_tokens) == 1:
                    return 0.75

            fuzzy_score = 0.0

            if len(target_role_normalized) >= 5:
                fuzzy_score = SequenceMatcher(
                    None,
                    target_role_normalized,
                    title,
                ).ratio()

                if fuzzy_score >= 0.72:
                    return 0.55 + (0.35 * fuzzy_score)

            return fuzzy_score if fuzzy_score >= 0.62 else 0.0

        matched_jobs = []

        for job in all_jobs_scored:
            match_score = title_score(job)
            if match_score > 0:
                matched_jobs.append(
                    {
                        **job,
                        "target_role_match_score": match_score,
                    }
                )

        if not matched_jobs:
            return None

        sorted_matches = sorted(
            matched_jobs,
            key=lambda job: (
                job.get("target_role_match_score", 0),
                job.get("final_rank_score", 0),
            ),
            reverse=True,
        )

        best_match = sorted_matches[0]

        exp_gap = max(
            0,
            self.safe_float(best_match.get("minimum_experience_years", 0))
            - self.safe_float(candidate_profile.get("experience_years", 0)),
        )

        cand_edu_rank = self.education_rank(
            candidate_profile.get("education_level_cand", "")
        )

        matched_rows = self.job_catalog[
            self.job_catalog["job_id"].astype(str) == str(best_match.get("job_id"))
        ]

        if matched_rows.empty:
            best_match["experience_gap_years"] = exp_gap
            best_match["edu_gap"] = float(best_match.get("edu_gap", 0))
            best_match["similar_jobs"] = sorted_matches[1:6]
            return best_match

        job_row = matched_rows.iloc[0]
        job_edu_rank = self.education_rank(job_row.get("education_level_job", ""))

        edu_gap = max(0, job_edu_rank - cand_edu_rank)

        best_match["experience_gap_years"] = exp_gap
        best_match["edu_gap"] = edu_gap
        best_match["similar_jobs"] = sorted_matches[1:6]

        return best_match
