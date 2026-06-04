from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    artifact_dir: str = "artifacts"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    def resolved_artifact_dir(self) -> str:
        configured = Path(self.artifact_dir)
        package_root = Path(__file__).resolve().parents[2]
        project_api_root = Path(__file__).resolve().parents[3]

        candidates = [
            configured,
            Path.cwd() / configured,
            package_root / configured,
            project_api_root / configured,
            project_api_root / "artifacts",
        ]

        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())

        return str((Path.cwd() / configured).resolve())


@lru_cache
def get_settings() -> Settings:
    return Settings()
