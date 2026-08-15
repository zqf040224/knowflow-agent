"""Run-scoped attachment snapshots for durable graph execution.

LangGraph checkpoints should keep compact references rather than uploaded file
contents.  This store copies the parsed text needed by a run into the configured
storage root and enforces the authenticated owner again when it is read.
"""

from __future__ import annotations

import json
import math
import os
import re
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_SPREADSHEET_SUFFIXES = {".xlsx", ".xls", ".csv"}
_OWNER_FILENAME = "owner.json"
_DOCUMENT_RUNTIME_FILENAME = "document_runtime.json"
_DOCUMENT_RUNTIME_VERSION = 1
_LOCK_FILENAME = ".artifacts.lock"


class GraphArtifactStore:
    def __init__(self, root: Path | str, upload_manager: Any):
        self.root = Path(root)
        self.upload_manager = upload_manager
        self.root.mkdir(parents=True, exist_ok=True)

    def snapshot_attachments(
        self,
        *,
        run_id: str,
        user_id: str,
        file_ids: list[str] | None,
    ) -> list[dict[str, Any]]:
        """Copy accessible parsed uploads into an immutable run directory."""
        run_dir = self._ensure_run_owner(run_id=run_id, user_id=user_id)
        # Snapshotting is a read-modify-write operation.  The execution lease
        # normally leaves a single writer, but recovery after a network split
        # must not let two worker processes overwrite each other's manifest
        # entries.  ``flock`` is shared by Gunicorn workers on the same volume.
        with self._run_lock(run_dir):
            manifest = self._read_manifest(run_id)
            manifest_owner = manifest.get("user_id")
            if manifest_owner and manifest_owner != user_id:
                raise PermissionError("graph artifact run belongs to another user")

            existing = {
                str(item.get("file_id")): item
                for item in manifest.get("attachments", [])
                if isinstance(item, dict)
            }
            attachments: list[dict[str, Any]] = []
            for raw_file_id in file_ids or []:
                file_id = str(raw_file_id or "").strip()
                if not file_id:
                    continue
                cached = existing.get(file_id)
                if cached and self._content_path(run_id, cached.get("artifact_id", "")).exists():
                    attachments.append(self._public_ref(cached))
                    continue

                content = self.upload_manager.get_temp_content(file_id, user_id)
                if not content:
                    continue
                info = self.upload_manager.get_temp_file_info(file_id, user_id) or {}
                filename = str(info.get("filename") or file_id)
                artifact_id = uuid.uuid4().hex
                self._atomic_write_text(
                    self._content_path(run_id, artifact_id),
                    str(content),
                )
                item = {
                    "artifact_id": artifact_id,
                    "file_id": file_id,
                    "filename": filename,
                    "char_count": len(str(info.get("content") or content or "")),
                    "is_spreadsheet": Path(filename).suffix.lower() in _SPREADSHEET_SUFFIXES,
                }
                existing[file_id] = item
                attachments.append(self._public_ref(item))

            self._atomic_write_json(run_dir / "manifest.json", {
                "run_id": run_id,
                "user_id": user_id,
                "attachments": list(existing.values()),
            })
            return attachments

    def snapshot_document_runtime_context(
        self,
        *,
        run_id: str,
        user_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist and return the immutable runtime context for one document run.

        The first successful call wins.  Later calls for the same ``run_id``
        return the originally stored payload, even when the caller supplies a
        different payload.  This makes retries reuse the exact memory/profile
        view that the initial attempt observed instead of silently drifting to
        newer session state.
        """

        normalized_payload = self._normalize_json_object(payload, "payload")
        run_dir = self._ensure_run_owner(run_id=run_id, user_id=user_id)
        path = run_dir / _DOCUMENT_RUNTIME_FILENAME
        envelope = {
            "version": _DOCUMENT_RUNTIME_VERSION,
            "run_id": run_id,
            "user_id": user_id,
            "payload": normalized_payload,
        }
        self._atomic_create_json(path, envelope)
        return self._load_document_runtime_envelope(
            run_id=run_id,
            user_id=user_id,
        )["payload"]

    def load_document_runtime_context(
        self,
        *,
        run_id: str,
        user_id: str,
    ) -> dict[str, Any] | None:
        """Load a run's immutable document context after checking ownership."""

        run_dir = self._validated_existing_run_dir(run_id)
        if run_dir is None:
            return None
        self._assert_run_owner(run_id=run_id, user_id=user_id)
        path = run_dir / _DOCUMENT_RUNTIME_FILENAME
        if not path.exists():
            return None
        return self._load_document_runtime_envelope(
            run_id=run_id,
            user_id=user_id,
        )["payload"]

    def load_content(self, *, run_id: str, artifact_id: str, user_id: str) -> str:
        manifest = self._read_manifest(run_id)
        if not manifest or manifest.get("user_id") != user_id:
            return ""
        allowed = {
            str(item.get("artifact_id"))
            for item in manifest.get("attachments", [])
            if isinstance(item, dict)
        }
        if artifact_id not in allowed:
            return ""
        path = self._content_path(run_id, artifact_id)
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def hydrate_message(
        self,
        *,
        run_id: str,
        user_id: str,
        message: str,
        attachments: list[dict[str, Any]] | None,
    ) -> str:
        blocks = []
        for item in attachments or []:
            artifact_id = str(item.get("artifact_id") or "")
            content = self.load_content(
                run_id=run_id,
                artifact_id=artifact_id,
                user_id=user_id,
            )
            if content:
                blocks.append(f"[文件内容]\n{content}\n[/文件内容]")
        if not blocks:
            return message or ""
        return "\n\n".join(blocks) + "\n\n[用户提问]\n" + (message or "")

    def delete_run(self, run_id: str) -> bool:
        run_dir = self._run_dir(run_id)
        if not run_dir.exists() or run_dir.is_symlink() or not run_dir.is_dir():
            return False
        # Keep the manifest until every content file is gone.  Its original
        # mtime remains a retry signal if an unlink fails halfway through.
        paths = sorted(
            run_dir.iterdir(),
            key=lambda path: (
                path.name in {"manifest.json", _OWNER_FILENAME},
                path.name,
            ),
        )
        for path in paths:
            if path.is_file():
                path.unlink()
        run_dir.rmdir()
        return True

    def list_expired_run_ids(self, *, cutoff: datetime) -> list[str]:
        """List safe run directories old enough for orphan cleanup.

        A manifest mtime is preferred because deleting other files changes the
        directory mtime.  Manifest-less orphan directories fall back to their
        own mtime.  Symlinks, non-directories and names outside the run-id
        allowlist are deliberately ignored.
        """

        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)
        cutoff_timestamp = cutoff.astimezone(timezone.utc).timestamp()
        expired: list[str] = []
        try:
            candidates = sorted(self.root.iterdir(), key=lambda path: path.name)
        except OSError:
            return expired

        for run_dir in candidates:
            run_id = run_dir.name
            if not _SAFE_ID.fullmatch(run_id) or run_dir.is_symlink():
                continue
            try:
                if not run_dir.is_dir():
                    continue
                manifest = run_dir / "manifest.json"
                anchor = (
                    manifest
                    if manifest.is_file() and not manifest.is_symlink()
                    else run_dir
                )
                modified_at = anchor.stat().st_mtime
            except OSError:
                continue
            if modified_at <= cutoff_timestamp:
                expired.append(run_id)
        return expired

    def _read_manifest(self, run_id: str) -> dict[str, Any]:
        path = self._run_dir(run_id) / "manifest.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _ensure_run_owner(self, *, run_id: str, user_id: str) -> Path:
        user_id = self._validate_user_id(user_id)
        run_dir = self._ensure_run_dir(run_id)
        existing_owner = self._existing_run_owner(run_id)
        if existing_owner is not None and existing_owner != user_id:
            raise PermissionError("graph artifact run belongs to another user")

        owner_path = run_dir / _OWNER_FILENAME
        self._atomic_create_json(owner_path, {
            "run_id": run_id,
            "user_id": user_id,
        })
        self._assert_run_owner(run_id=run_id, user_id=user_id)
        return run_dir

    def _assert_run_owner(self, *, run_id: str, user_id: str) -> None:
        user_id = self._validate_user_id(user_id)
        owner = self._existing_run_owner(run_id)
        if owner is None:
            raise PermissionError("graph artifact run has no verified owner")
        if owner != user_id:
            raise PermissionError("graph artifact run belongs to another user")

    def _existing_run_owner(self, run_id: str) -> str | None:
        run_dir = self._validated_existing_run_dir(run_id)
        if run_dir is None:
            return None

        owners: list[str] = []
        for filename in (
            _OWNER_FILENAME,
            "manifest.json",
            _DOCUMENT_RUNTIME_FILENAME,
        ):
            path = run_dir / filename
            if not path.exists():
                continue
            payload = self._read_json_file(path, filename)
            owner = payload.get("user_id")
            if not isinstance(owner, str) or not owner.strip():
                raise RuntimeError(f"invalid graph artifact {filename} owner")
            owners.append(owner)
        if len(set(owners)) > 1:
            raise RuntimeError("conflicting graph artifact run owners")
        return owners[0] if owners else None

    def _load_document_runtime_envelope(
        self,
        *,
        run_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        path = self._run_dir(run_id) / _DOCUMENT_RUNTIME_FILENAME
        envelope = self._read_json_file(path, _DOCUMENT_RUNTIME_FILENAME)
        if envelope.get("version") != _DOCUMENT_RUNTIME_VERSION:
            raise RuntimeError("unsupported document runtime snapshot version")
        if envelope.get("run_id") != run_id:
            raise RuntimeError("document runtime snapshot run mismatch")
        if envelope.get("user_id") != user_id:
            raise PermissionError("graph artifact run belongs to another user")
        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            raise RuntimeError("invalid document runtime snapshot payload")
        return {
            **envelope,
            "payload": self._normalize_json_object(payload, "stored payload"),
        }

    def _ensure_run_dir(self, run_id: str) -> Path:
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        if run_dir.is_symlink() or not run_dir.is_dir():
            raise ValueError("invalid graph artifact run directory")
        return run_dir

    def _validated_existing_run_dir(self, run_id: str) -> Path | None:
        run_dir = self._run_dir(run_id)
        if not run_dir.exists():
            return None
        if run_dir.is_symlink() or not run_dir.is_dir():
            raise ValueError("invalid graph artifact run directory")
        return run_dir

    def _run_dir(self, run_id: str) -> Path:
        self._validate_id(run_id, "run_id")
        return self.root / run_id

    def _content_path(self, run_id: str, artifact_id: str) -> Path:
        self._validate_id(artifact_id, "artifact_id")
        return self._run_dir(run_id) / f"{artifact_id}.txt"

    @staticmethod
    @contextmanager
    def _run_lock(run_dir: Path):
        """Serialize manifest updates across processes sharing one run volume."""

        try:
            import fcntl
        except ImportError as exc:  # pragma: no cover - production image is Linux
            raise RuntimeError(
                "graph artifact locking requires an OS with fcntl support"
            ) from exc
        lock_path = run_dir / _LOCK_FILENAME
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            os.chmod(lock_path, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    @staticmethod
    def _validate_id(value: str, label: str) -> None:
        if not _SAFE_ID.fullmatch(str(value or "")):
            raise ValueError(f"invalid {label}")

    @staticmethod
    def _validate_user_id(user_id: str) -> str:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("invalid user_id")
        return user_id

    @classmethod
    def _normalize_json_object(
        cls,
        payload: dict[str, Any],
        label: str,
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise TypeError(f"{label} must be a JSON object")
        cls._validate_json_value(payload, label)
        try:
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{label} must contain only JSON values") from exc
        decoded = json.loads(encoded)
        if not isinstance(decoded, dict):
            raise TypeError(f"{label} must be a JSON object")
        return decoded

    @classmethod
    def _validate_json_value(cls, value: Any, label: str) -> None:
        if value is None or isinstance(value, (str, bool, int)):
            return
        if isinstance(value, float):
            if not math.isfinite(value):
                raise TypeError(f"{label} must contain only finite JSON numbers")
            return
        if isinstance(value, list):
            for item in value:
                cls._validate_json_value(item, label)
            return
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError(f"{label} must use string object keys")
                cls._validate_json_value(item, label)
            return
        raise TypeError(f"{label} must contain only JSON values")

    @staticmethod
    def _read_json_file(path: Path, label: str) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read graph artifact {label}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"invalid graph artifact {label}")
        return payload

    @staticmethod
    def _public_ref(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "artifact_id": item.get("artifact_id", ""),
            "file_id": item.get("file_id", ""),
            "filename": item.get("filename", ""),
            "char_count": int(item.get("char_count", 0) or 0),
            "is_spreadsheet": bool(item.get("is_spreadsheet")),
        }

    @staticmethod
    def _atomic_write_text(path: Path, content: str) -> None:
        temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        descriptor = os.open(temp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
            os.chmod(path, 0o600)
        finally:
            try:
                temp.unlink()
            except FileNotFoundError:
                pass

    @classmethod
    def _atomic_write_json(cls, path: Path, payload: dict[str, Any]) -> None:
        cls._atomic_write_text(
            path,
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        )

    @classmethod
    def _atomic_create_json(cls, path: Path, payload: dict[str, Any]) -> bool:
        """Install a complete JSON file only when the destination is absent."""

        temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        try:
            descriptor = os.open(
                temp,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temp, path)
            except FileExistsError:
                return False
            os.chmod(path, 0o600)
            return True
        finally:
            try:
                temp.unlink()
            except FileNotFoundError:
                pass
