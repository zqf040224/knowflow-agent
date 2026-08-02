#!/usr/bin/env python3
"""Ingest curated local source materials into the shared knowledge base.

The script reads files from Agent/资料原件, copies each source file to a
temporary staging path, and then reuses the normal upload ingestion pipeline.
Original files under Agent/资料原件 are never moved.
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_base import KnowledgeBase
from storage_config import KNOWLEDGE_BASE_DIR
from upload_manager import get_upload_manager


DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "资料原件"


def infer_category(path: Path, source_root: Path) -> str:
    return "公共资料"


def iter_supported_files(source_root: Path):
    supported = {".docx", ".doc", ".pdf", ".txt", ".md", ".xlsx", ".xls", ".csv"}
    for path in sorted(source_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in supported:
            yield path


def ingest_files(source_root: Path, user_id: str, dry_run: bool = False) -> int:
    files = list(iter_supported_files(source_root))
    if not files:
        print(f"未找到可入库文件: {source_root}")
        return 1

    print(f"发现 {len(files)} 个可入库文件")
    for path in files:
        category = infer_category(path, source_root)
        print(f"- [{category}] {path.relative_to(source_root)}")

    if dry_run:
        print("dry-run 完成，未写入知识库")
        return 0

    manager = get_upload_manager()
    knowledge_base = KnowledgeBase(str(KNOWLEDGE_BASE_DIR), lazy_load=True)
    failures = []

    with tempfile.TemporaryDirectory(prefix="original_material_ingest_") as tmpdir:
        staging_dir = Path(tmpdir)
        for path in files:
            category = infer_category(path, source_root)
            staged = staging_dir / path.name
            shutil.copy2(path, staged)
            result = manager.process_knowledge_upload(
                file_path=str(staged),
                filename=path.name,
                category=category,
                user_id=user_id,
                knowledge_base=knowledge_base,
                department="",
            )
            status = "OK" if result.get("success") else "FAIL"
            print(f"{status} [{category}] {path.name}: {result.get('message', '')}")
            if not result.get("success") and not result.get("duplicate"):
                failures.append((path, result.get("message", "")))

    if failures:
        print("\n以下文件入库失败:")
        for path, message in failures:
            print(f"- {path.name}: {message}")
        return 1

    print("原件资料入库完成")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest Agent/资料原件 into the knowledge base")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT), help="source materials root")
    parser.add_argument("--user-id", default="system", help="uploaded_by value for manifest/audit metadata")
    parser.add_argument("--dry-run", action="store_true", help="preview files without writing to the knowledge base")
    args = parser.parse_args()

    return ingest_files(Path(args.source_root), args.user_id, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
