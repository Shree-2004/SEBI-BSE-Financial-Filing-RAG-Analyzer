"""
push_hf_space.py

Uploads the API to a Hugging Face Space (Docker SDK). Only what the
Dockerfile needs is uploaded; the Space's README.md (its config, incl.
app_port) comes from deploy/hf-space/README.md, not the project README.

One-time setup:
    1. Create a Space at huggingface.co/new-space (SDK: Docker, Blank, CPU basic)
    2. hf auth login            # paste a token with "write" access
    3. In the Space: Settings -> Variables and secrets -> add the secrets
       QDRANT_URL, QDRANT_API_KEY, GROQ_API_KEY, OPENAI_API_KEY,
       CORS_ALLOWED_ORIGINS

Run (from the project root):
    python deploy/push_hf_space.py <hf-username>/<space-name>
"""

import sys

from huggingface_hub import HfApi

ALLOW = [
    "Dockerfile",
    "requirements.txt",
    "src/**/*.py",
    "data/processed/bm25_index.pkl",
]


def main():
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        sys.exit("usage: python deploy/push_hf_space.py <hf-username>/<space-name>")
    repo_id = sys.argv[1]
    api = HfApi()

    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=".",
        allow_patterns=ALLOW,
        commit_message="Deploy API from GitHub repo",
    )
    api.upload_file(
        repo_id=repo_id,
        repo_type="space",
        path_or_fileobj="deploy/hf-space/README.md",
        path_in_repo="README.md",
        commit_message="Space config",
    )
    print(f"✓ Uploaded. Build logs: https://huggingface.co/spaces/{repo_id}?logs=build")


if __name__ == "__main__":
    main()
