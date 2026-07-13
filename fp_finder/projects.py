from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .yolo_dataset import index_records_ready


PROJECTS_PATH = Path("artifacts") / "projects.json"


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def slugify(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", value.strip())
    slug = slug.strip("._-")
    return slug or "project"


def load_projects(path: Path = PROJECTS_PATH) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f) or {}
    projects = data.get("projects", [])
    return list(projects) if isinstance(projects, list) else []


def save_projects(projects: List[Dict], path: Path = PROJECTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"projects": projects}, f, ensure_ascii=False, indent=2)


def upsert_project(project: Dict, path: Path = PROJECTS_PATH) -> Dict:
    projects = load_projects(path)
    name = str(project["name"])
    existing = next((item for item in projects if item.get("name") == name), None)
    timestamp = now_text()
    if existing is None:
        project = dict(project)
        project.setdefault("created_at", timestamp)
        project["updated_at"] = timestamp
        projects.append(project)
    else:
        created_at = existing.get("created_at") or timestamp
        existing.clear()
        existing.update(project)
        existing["created_at"] = created_at
        existing["updated_at"] = timestamp
        project = existing
    save_projects(projects, path)
    return project


def delete_project(name: str, path: Path = PROJECTS_PATH) -> None:
    projects = [project for project in load_projects(path) if project.get("name") != name]
    save_projects(projects, path)


def get_project(name: str, path: Path = PROJECTS_PATH) -> Optional[Dict]:
    for project in load_projects(path):
        if project.get("name") == name:
            return project
    return None


def index_ready(project: Dict) -> bool:
    root = Path(str(project.get("feature_index_dir", "")))
    return (root / "index.faiss").exists() and (root / "config.json").exists() and index_records_ready(root)


def project_log_root(project: Dict) -> Path:
    slug = slugify(str(project.get("name", "project")))
    return Path("artifacts") / "project_build_logs" / slug


def project_records_json(project: Dict) -> Path:
    slug = slugify(str(project.get("name", "project")))
    return Path("artifacts") / "project_records" / f"{slug}.json"


def project_shard_root(project: Dict) -> Path:
    slug = slugify(str(project.get("name", "project")))
    return Path("artifacts") / "project_feature_shards" / slug
