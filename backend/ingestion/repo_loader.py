"""Load a codebase from a local path or a git URL into a local working dir."""
import os
import shutil
import tempfile
from pathlib import Path
from typing import List

import git  # GitPython

# File extensions we know how to parse meaningfully.
SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rb",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".php", ".rs", ".md",
}

# Directories to skip entirely (build artifacts, deps, vcs internals).
IGNORE_DIRS = {
    ".git", "node_modules", "venv", ".venv", "__pycache__",
    "dist", "build", ".idea", ".vscode", "target", "vendor",
}


def clone_repo(repo_url: str, dest_dir: str = None) -> str:
    """Clone a git repo to a temp/dest directory and return the local path."""
    dest_dir = dest_dir or tempfile.mkdtemp(prefix="codesense_repo_")
    if os.path.exists(dest_dir) and os.listdir(dest_dir):
        shutil.rmtree(dest_dir)
    git.Repo.clone_from(repo_url, dest_dir, depth=1)
    return dest_dir


def collect_source_files(repo_path: str) -> List[str]:
    """Walk the repo and return paths of files worth indexing."""
    collected = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")]
        for fname in files:
            ext = Path(fname).suffix.lower()
            if ext in SUPPORTED_EXTENSIONS:
                collected.append(os.path.join(root, fname))
    return collected


def load_repo(repo_path: str = None, repo_url: str = None) -> tuple[str, List[str]]:
    """Resolve a repo (local path or URL) to a local dir + list of source files."""
    if repo_url:
        local_path = clone_repo(repo_url)
    elif repo_path:
        local_path = repo_path
    else:
        raise ValueError("Either repo_path or repo_url must be provided")

    if not os.path.isdir(local_path):
        raise FileNotFoundError(f"Repo path not found: {local_path}")

    files = collect_source_files(local_path)
    return local_path, files
