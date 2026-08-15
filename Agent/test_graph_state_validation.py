from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import pytest

from graph_state_validation import (
    GraphStateSerializationError,
    validate_graph_update,
)


@dataclass
class RuntimeDataclass:
    value: str


@pytest.mark.parametrize(
    "bad_value",
    [
        RuntimeDataclass("secret"),
        ("tuple",),
        b"bytes",
        float("nan"),
        float("inf"),
    ],
)
def test_graph_state_rejects_non_json_runtime_values(bad_value):
    with pytest.raises(GraphStateSerializationError):
        validate_graph_update({"bad": bad_value}, node="test")


def test_graph_state_rejects_generators_connections_and_non_string_keys():
    connection = sqlite3.connect(":memory:")
    try:
        for bad_value in ((item for item in [1]), connection):
            with pytest.raises(GraphStateSerializationError):
                validate_graph_update({"bad": bad_value}, node="test")
        with pytest.raises(GraphStateSerializationError, match="non-string key"):
            validate_graph_update({1: "bad"}, node="test")
    finally:
        connection.close()


def test_graph_state_accepts_nested_strict_json_without_copying():
    update = {"ok": [None, True, 1, 1.5, {"nested": "value"}]}
    assert validate_graph_update(update, node="test") is update
