"""Week 4, Day 3 learner surface for effect receipts."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from .protocol import AgentError


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


@dataclass(frozen=True)
class EffectReceipt:
    """One completed tool call and its observable outcome."""

    tool_call_id: str
    tool: str
    arguments: dict[str, Any]
    exit_state: str
    result: str
    changed_artifacts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (self.tool_call_id, self.tool)
        ):
            raise AgentError("receipt call id and tool must be non-empty strings")
        if self.exit_state not in ("ok", "error") or not isinstance(self.result, str):
            raise AgentError("invalid receipt outcome")
        if not isinstance(self.arguments, dict):
            raise AgentError("receipt arguments must be an object")
        try:
            arguments = json.loads(_canonical(self.arguments))
        except (TypeError, ValueError) as error:
            raise AgentError("receipt arguments must be JSON-compatible") from error
        if not isinstance(self.changed_artifacts, tuple):
            raise AgentError("receipt artifacts must be a tuple")
        for artifact in self.changed_artifacts:
            if (
                not isinstance(artifact, str)
                or not artifact
                or "\0" in artifact
                or PurePosixPath(artifact).is_absolute()
                or ".." in PurePosixPath(artifact).parts
                or PurePosixPath(artifact).as_posix() != artifact
                or artifact == "."
            ):
                raise AgentError("receipt artifacts must be normalized relative paths")
        object.__setattr__(self, "arguments", arguments)

    def __getattribute__(self, name: str) -> Any:
        value = object.__getattribute__(self, name)
        # Return a defensive copy even for nested argument containers.
        return copy.deepcopy(value) if name == "arguments" else value

    def _payload(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "tool": self.tool,
            "arguments": self.arguments,
            "exit_state": self.exit_state,
            "result": self.result,
            "changed_artifacts": list(self.changed_artifacts),
        }

    @property
    def receipt_id(self) -> str:
        return hashlib.sha256(_canonical(self._payload()).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "receipt_id": self.receipt_id}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EffectReceipt:
        fields = {
            "tool_call_id",
            "tool",
            "arguments",
            "exit_state",
            "result",
            "changed_artifacts",
            "receipt_id",
        }
        if not isinstance(payload, dict) or set(payload) != fields:
            raise AgentError("invalid receipt fields")
        if not isinstance(payload["changed_artifacts"], list):
            raise AgentError("invalid receipt artifacts")
        receipt = cls(
            payload["tool_call_id"],
            payload["tool"],
            payload["arguments"],
            payload["exit_state"],
            payload["result"],
            tuple(payload["changed_artifacts"]),
        )
        if receipt.receipt_id != payload["receipt_id"]:
            raise AgentError("receipt digest mismatch")
        return receipt


@dataclass
class ReceiptStore:
    """Keep receipts in memory and optionally append them to one JSONL file."""

    path: Path | None = None
    _by_call: dict[str, EffectReceipt] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.path is None:
            return
        self.path = Path(self.path)
        try:
            if not self.path.exists():
                return
            with self.path.open(encoding="utf-8") as stream:
                for line in stream:
                    receipt = EffectReceipt.from_dict(json.loads(line))
                    previous = self.get(receipt.tool_call_id)
                    if previous is not None and previous != receipt:
                        raise AgentError("tool call id already used")
                    self._by_call[receipt.tool_call_id] = receipt
        except (OSError, ValueError) as error:
            raise AgentError(f"cannot load receipt log: {error}") from error

    def put(self, receipt: EffectReceipt) -> EffectReceipt:
        if not isinstance(receipt, EffectReceipt):
            raise AgentError("expected an effect receipt")
        previous = self.get(receipt.tool_call_id)
        if previous is not None:
            if previous != receipt:
                raise AgentError("tool call id already used")
            return previous
        if self.path is not None:
            try:
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(_canonical(receipt.to_dict()) + "\n")
            except OSError as error:
                raise AgentError(f"cannot append receipt: {error}") from error
        self._by_call[receipt.tool_call_id] = receipt
        return receipt

    def get(self, tool_call_id: str) -> EffectReceipt | None:
        return self._by_call.get(tool_call_id)
