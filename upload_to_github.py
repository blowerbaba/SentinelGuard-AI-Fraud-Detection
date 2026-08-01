"""
SentinelGuard AI - GitHub Uploader Utility.
Uploads the project to GitHub using the GitHub REST API or Git CLI.
"""

import os
import sys
import json
import base64
import subprocess
import urllib.request
import urllib.error
from pathlib import Path

# Ensure UTF-8 output encoding for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


PROJECT_DIR = Path(__file__).resolve().parent

IGNORE_PATTERNS = [
    ".git", "__pycache__", ".pytest_cache", ".venv", "venv", "env",
    ".DS_Store", "Thumbs.db", "*.pyc", "models/*.pkl", "data/*.csv"
]


def should_ignore(path: Path) -> bool:
    for part in path.parts:
        if part in [".git", "__pycache__", ".pytest_cache", "venv", "env"]:
            return True
        if part.endswith(".pyc"):
            return True
    return False


def upload_via_api(repo_name: str, token: str, is_private: bool = False):
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "SentinelGuard-Uploader"
    }

    print(f"🚀 Creating GitHub repository '{repo_name}'...")
    create_url = "https://api.github.com/user/repos"
    payload = json.dumps({
        "name": repo_name,
        "description": "SentinelGuard AI: Real-Time Financial Fraud & Anomaly Detection Platform",
        "private": is_private
    }).encode("utf-8")

    req = urllib.request.Request(create_url, data=payload, headers=headers, method="POST")
    repo_full_name = ""
    html_url = ""

    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            repo_full_name = data["full_name"]
            html_url = data["html_url"]
            print(f"✅ Repository created: {html_url}")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        if "name already exists" in err_body:
            print(f"ℹ️ Repository '{repo_name}' already exists. Syncing files...")
            user_req = urllib.request.Request("https://api.github.com/user", headers=headers)
            with urllib.request.urlopen(user_req) as u_resp:
                username = json.loads(u_resp.read().decode("utf-8"))["login"]
                repo_full_name = f"{username}/{repo_name}"
                html_url = f"https://github.com/{repo_full_name}"
        else:
            print(f"❌ Failed to create repository: {e.code} - {err_body}")
            sys.exit(1)

    print("📦 Uploading project files to GitHub...")
    uploaded_count = 0
    for root, dirs, files in os.walk(PROJECT_DIR):
        root_path = Path(root)
        dirs[:] = [d for d in dirs if not should_ignore(root_path / d)]

        for file in files:
            file_path = root_path / file
            if should_ignore(file_path):
                continue

            relative_path = file_path.relative_to(PROJECT_DIR).as_posix()
            with open(file_path, "rb") as f:
                content_b64 = base64.b64encode(f.read()).decode("utf-8")

            put_url = f"https://api.github.com/repos/{repo_full_name}/contents/{relative_path}"
            
            # Check existing file for SHA update
            sha = None
            try:
                get_req = urllib.request.Request(put_url, headers=headers)
                with urllib.request.urlopen(get_req) as g_resp:
                    sha = json.loads(g_resp.read().decode("utf-8")).get("sha")
            except urllib.error.HTTPError:
                pass

            put_payload = {
                "message": f"Add/Update {relative_path}",
                "content": content_b64
            }
            if sha:
                put_payload["sha"] = sha

            put_data = json.dumps(put_payload).encode("utf-8")
            put_req = urllib.request.Request(put_url, data=put_data, headers=headers, method="PUT")

            try:
                with urllib.request.urlopen(put_req):
                    print(f"  -> Uploaded: {relative_path}")
                    uploaded_count += 1
            except urllib.error.HTTPError as pe:
                print(f"  ❌ Error uploading {relative_path}: {pe.code}")

    print("\n" + "=" * 60)
    print(f"🎉 PROJECT SUCCESSFULLY UPLOADED TO GITHUB!")
    print(f"🔗 Repository URL: {html_url}")
    print(f"📊 Total Files Uploaded: {uploaded_count}")
    print("=" * 60)


def main():
    token = os.environ.get("GITHUB_TOKEN") or (sys.argv[2] if len(sys.argv) > 2 else None)
    repo_name = sys.argv[1] if len(sys.argv) > 1 else "SentinelGuard-AI-Fraud-Detection"

    if token:
        upload_via_api(repo_name, token)
    else:
        print("SentinelGuard AI - GitHub Upload Utility")
        print("---------------------------------------")
        print("To upload this repository automatically via GitHub API:")
        print(f"Run: python upload_to_github.py {repo_name} <YOUR_GITHUB_PERSONAL_ACCESS_TOKEN>")
        print("\nOr set the GITHUB_TOKEN environment variable and re-run.")


if __name__ == "__main__":
    main()
