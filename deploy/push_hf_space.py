"""
push_hf_space.py

Uploads the API to a Hugging Face Space. Docker Spaces are paid, so it
runs in a free Gradio-SDK Space: deploy/hf-space/app.py starts the FastAPI
app on port 7860, and deploy/hf-space/requirements.txt holds the runtime
deps. The Space's README.md (its config) comes from deploy/hf-space/,
not the project README.

One-time setup:
    1. Create a Space at huggingface.co/new-space (SDK: Gradio, Blank, CPU basic)
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
    "src/**/*.py",
    "data/processed/bm25_index.pkl",
    "data/processed/metrics.sqlite",
]

# deploy/hf-space/<name> -> Space root
SPACE_FILES = ["README.md", "app.py", "requirements.txt"]


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
    for name in SPACE_FILES:
        api.upload_file(
            repo_id=repo_id,
            repo_type="space",
            path_or_fileobj=f"deploy/hf-space/{name}",
            path_in_repo=name,
            commit_message=f"Space {name}",
        )
    print(f"✓ Uploaded. Build logs: https://huggingface.co/spaces/{repo_id}?logs=build")


if __name__ == "__main__":
    main()
