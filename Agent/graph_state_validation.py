"""Strict JSON boundary for values written to LangGraph State.

LangGraph's default serializer can encode several Python runtime objects that
must never become part of a durable checkpoint.  Graph nodes use this module at
their return boundary so a bad update fails before the checkpointer sees it.
"""

from __future__ import annotations

import math
from typing import Any


class GraphStateSerializationError(TypeError):
    """A graph update contains a value outside the JSON State contract."""


def validate_json_value(
    value: Any,
    *,
    path: str = "$",
    _active_containers: set[int] | None = None,
) -> Any:
    """Return *value* after recursively proving it is strict JSON data.

    Accepted values are ``None``, strings, booleans, integers, finite floats,
    lists, and dictionaries with string keys.  Tuples, bytes, dataclasses,
    generators, agents, database connections, and all other runtime objects are
    rejected rather than converted implicitly.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise GraphStateSerializationError(
                f"{path} contains a non-finite float, which is not valid JSON"
            )
        return value

    active = _active_containers if _active_containers is not None else set()
    if isinstance(value, list):
        identity = id(value)
        if identity in active:
            raise GraphStateSerializationError(f"{path} contains a cyclic list")
        active.add(identity)
        try:
            for index, item in enumerate(value):
                validate_json_value(
                    item,
                    path=f"{path}[{index}]",
                    _active_containers=active,
                )
        finally:
            active.remove(identity)
        return value

    if isinstance(value, dict):
        identity = id(value)
        if identity in active:
            raise GraphStateSerializationError(f"{path} contains a cyclic object")
        active.add(identity)
        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise GraphStateSerializationError(
                        f"{path} contains non-string key {key!r}"
                    )
                child_path = f"{path}.{key}" if key else f"{path}['']"
                validate_json_value(
                    item,
                    path=child_path,
                    _active_containers=active,
                )
        finally:
            active.remove(identity)
        return value

    value_type = f"{type(value).__module__}.{type(value).__qualname__}"
    raise GraphStateSerializationError(
        f"{path} contains unsupported runtime object {value_type}; "
        "LangGraph State must contain strict JSON values only"
    )


def validate_graph_update(update: Any, *, node: str) -> dict[str, Any]:
    """Validate one node update and preserve its original dictionary."""

    if not isinstance(update, dict):
        value_type = f"{type(update).__module__}.{type(update).__qualname__}"
        raise GraphStateSerializationError(
            f"LangGraph node {node!r} returned {value_type}; expected a dict update"
        )
    validate_json_value(update, path=f"node:{node}")
    return update


__all__ = [
    "GraphStateSerializationError",
    "validate_graph_update",
    "validate_json_value",
]
