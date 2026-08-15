"""Public-checkpointer adapter that isolates a child workflow namespace.

LangGraph reserves ``checkpoint_ns`` for nested graph execution and clears it
for a separately-invoked root graph.  The document workflow is invoked through
an existing tool boundary, so this adapter maps its root checkpoints into a
stable physical namespace while retaining the parent run's ``thread_id`` and
the parent's saver backend.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Collection, Iterator, Mapping, Sequence
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver


class NamespacedCheckpointSaver(BaseCheckpointSaver):
    """Delegate checkpoint I/O after forcing one controlled namespace."""

    def __init__(self, delegate: BaseCheckpointSaver, namespace: str) -> None:
        namespace = str(namespace or "").strip()
        if not namespace or ":" in namespace:
            raise ValueError("checkpoint namespace must be a non-empty root name")
        super().__init__(serde=delegate.serde)
        self.delegate = delegate
        self.namespace = namespace

    @property
    def config_specs(self) -> list:
        return self.delegate.config_specs

    def _config(self, config: dict[str, Any] | None):
        if config is None:
            return None
        patched = dict(config)
        configurable = dict(patched.get("configurable") or {})
        configurable["checkpoint_ns"] = self.namespace
        patched["configurable"] = configurable
        return patched

    def get_tuple(self, config):
        return self.delegate.get_tuple(self._config(config))

    def list(self, config, *, filter=None, before=None, limit=None) -> Iterator:
        return self.delegate.list(
            self._config(config),
            filter=filter,
            before=self._config(before),
            limit=limit,
        )

    def put(self, config, checkpoint, metadata, new_versions):
        return self.delegate.put(
            self._config(config), checkpoint, metadata, new_versions
        )

    def put_writes(self, config, writes, task_id, task_path="") -> None:
        self.delegate.put_writes(
            self._config(config), writes, task_id, task_path
        )

    def delete_thread(self, thread_id: str) -> None:
        self.delegate.delete_thread(thread_id)

    def delete_for_runs(self, run_ids: Sequence[str]) -> None:
        self.delegate.delete_for_runs(run_ids)

    def copy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        self.delegate.copy_thread(source_thread_id, target_thread_id)

    def prune(self, thread_ids: Sequence[str], *, strategy="keep_latest") -> None:
        self.delegate.prune(thread_ids, strategy=strategy)

    def get_next_version(self, current, channel):
        return self.delegate.get_next_version(current, channel)

    def with_allowlist(
        self,
        extra_allowlist: Collection[tuple[str, ...]],
    ) -> "NamespacedCheckpointSaver":
        return type(self)(
            self.delegate.with_allowlist(extra_allowlist),
            self.namespace,
        )

    async def aget_tuple(self, config):
        return await self.delegate.aget_tuple(self._config(config))

    async def alist(
        self,
        config,
        *,
        filter=None,
        before=None,
        limit=None,
    ) -> AsyncIterator:
        async for item in self.delegate.alist(
            self._config(config),
            filter=filter,
            before=self._config(before),
            limit=limit,
        ):
            yield item

    async def aput(self, config, checkpoint, metadata, new_versions):
        return await self.delegate.aput(
            self._config(config), checkpoint, metadata, new_versions
        )

    async def aput_writes(self, config, writes, task_id, task_path="") -> None:
        await self.delegate.aput_writes(
            self._config(config), writes, task_id, task_path
        )

    async def adelete_thread(self, thread_id: str) -> None:
        await self.delegate.adelete_thread(thread_id)

    async def adelete_for_runs(self, run_ids: Sequence[str]) -> None:
        await self.delegate.adelete_for_runs(run_ids)

    async def acopy_thread(
        self, source_thread_id: str, target_thread_id: str
    ) -> None:
        await self.delegate.acopy_thread(source_thread_id, target_thread_id)

    async def aprune(
        self, thread_ids: Sequence[str], *, strategy="keep_latest"
    ) -> None:
        await self.delegate.aprune(thread_ids, strategy=strategy)

    def get_delta_channel_history(
        self, *, config, channels: Sequence[str]
    ) -> Mapping:
        return self.delegate.get_delta_channel_history(
            config=self._config(config), channels=channels
        )

    async def aget_delta_channel_history(
        self, *, config, channels: Sequence[str]
    ) -> Mapping:
        return await self.delegate.aget_delta_channel_history(
            config=self._config(config), channels=channels
        )
