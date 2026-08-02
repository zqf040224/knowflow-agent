"""
Storage path configuration for local disk or mounted NAS.

The application still defaults to the current local folders. To test a NAS,
mount the NAS share on the application server and set:

    STORAGE_BACKEND=nas
    NAS_MOUNT_PATH=/mnt/knowledge-agent
"""

import os
import tempfile
from pathlib import Path
from typing import Dict


PROJECT_ROOT = Path(__file__).resolve().parent


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name, "").strip()
    return Path(value).expanduser() if value else default


def _storage_root() -> Path:
    backend = os.getenv("STORAGE_BACKEND", "local").strip().lower()
    if backend == "nas":
        return _env_path("NAS_MOUNT_PATH", PROJECT_ROOT)
    return _env_path("STORAGE_ROOT", PROJECT_ROOT)


STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local").strip().lower() or "local"
STORAGE_ROOT = _storage_root()

UPLOADS_DIR = _env_path("UPLOADS_DIR", STORAGE_ROOT / "uploads")
KNOWLEDGE_BASE_DIR = _env_path("KNOWLEDGE_BASE_DIR", STORAGE_ROOT / "knowledge_base")
OUTPUTS_DIR = _env_path("OUTPUTS_DIR", STORAGE_ROOT / "outputs")
KNOWLEDGE_SOURCE_DIR = _env_path("KNOWLEDGE_SOURCE_DIR", STORAGE_ROOT / "知识库")

INGESTION_MANIFEST_DB = _env_path(
    "INGESTION_MANIFEST_DB",
    KNOWLEDGE_BASE_DIR / "ingestion_manifest.sqlite",
)
SPREADSHEET_DB_PATH = _env_path(
    "SPREADSHEET_DB_PATH",
    KNOWLEDGE_BASE_DIR / "spreadsheets.sqlite",
)
ADMIN_BACKUP_DIR = _env_path(
    "ADMIN_BACKUP_DIR",
    KNOWLEDGE_BASE_DIR / "admin_backups",
)


def ensure_storage_dirs() -> None:
    """Create directories that the app is allowed to manage."""
    for path in [UPLOADS_DIR, KNOWLEDGE_BASE_DIR, OUTPUTS_DIR, ADMIN_BACKUP_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def storage_summary() -> Dict:
    return {
        "backend": STORAGE_BACKEND,
        "root": str(STORAGE_ROOT),
        "uploads_dir": str(UPLOADS_DIR),
        "knowledge_base_dir": str(KNOWLEDGE_BASE_DIR),
        "outputs_dir": str(OUTPUTS_DIR),
        "knowledge_source_dir": str(KNOWLEDGE_SOURCE_DIR),
        "ingestion_manifest_db": str(INGESTION_MANIFEST_DB),
        "spreadsheet_db_path": str(SPREADSHEET_DB_PATH),
        "admin_backup_dir": str(ADMIN_BACKUP_DIR),
    }


def _writable_check(path: Path) -> Dict:
    result = {
        "path": str(path),
        "exists": path.exists(),
        "is_dir": path.is_dir(),
        "readable": os.access(path, os.R_OK) if path.exists() else False,
        "writable": False,
        "error": "",
    }
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".agent_write_test_", dir=str(path), delete=True) as temp:
            temp.write(b"ok")
            temp.flush()
        result.update({
            "exists": path.exists(),
            "is_dir": path.is_dir(),
            "readable": os.access(path, os.R_OK),
            "writable": True,
        })
    except Exception as exc:
        result["error"] = str(exc)
    return result


def storage_health() -> Dict:
    checks = {
        "uploads": _writable_check(UPLOADS_DIR),
        "knowledge_base": _writable_check(KNOWLEDGE_BASE_DIR),
        "outputs": _writable_check(OUTPUTS_DIR),
        "admin_backups": _writable_check(ADMIN_BACKUP_DIR),
    }
    source_exists = KNOWLEDGE_SOURCE_DIR.exists()
    checks["knowledge_source"] = {
        "path": str(KNOWLEDGE_SOURCE_DIR),
        "exists": source_exists,
        "is_dir": KNOWLEDGE_SOURCE_DIR.is_dir(),
        "readable": os.access(KNOWLEDGE_SOURCE_DIR, os.R_OK) if source_exists else False,
        "writable": os.access(KNOWLEDGE_SOURCE_DIR, os.W_OK) if source_exists else False,
        "error": "" if source_exists else "目录不存在；如未使用原始知识库目录可忽略",
    }
    ok = all(item.get("writable") for key, item in checks.items() if key != "knowledge_source")
    return {
        "ok": ok,
        "summary": storage_summary(),
        "checks": checks,
    }
