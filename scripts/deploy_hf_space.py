import os
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]
SPACE_ID = "gusmiputri/cocokin-ai-api"


def main() -> None:
    api = HfApi()
    api.create_repo(
        repo_id=SPACE_ID,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
    )

    gemini_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    api.add_space_variable(SPACE_ID, "GEMINI_MODEL", gemini_model)

    api.upload_folder(
        repo_id=SPACE_ID,
        repo_type="space",
        folder_path=str(ROOT),
        path_in_repo=".",
        ignore_patterns=[
            ".env",
            ".venv",
            ".venv/",
            ".venv311",
            ".venv311/",
            ".git",
            ".git/",
            "__pycache__",
            "__pycache__/",
            "*.pyc",
            "uvicorn.out.log",
            "uvicorn.err.log",
        ],
    )
    print(f"Deployed https://huggingface.co/spaces/{SPACE_ID}")


if __name__ == "__main__":
    main()
