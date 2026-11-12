"""Day 7 Task 5: validate named receipt evidence without replaying effects."""

from dataclasses import replace
import json

import pytest

from tiny_llm.agent import (
    AgentError,
    AgentRun,
    EffectReceipt,
    EvaluationCase,
    ReceiptExpectation,
    ReceiptStore,
    ToolPolicy,
    Workspace,
    evaluate_run,
)


@pytest.fixture
def evidence(tmp_path):
    store = ReceiptStore(tmp_path / "receipts.jsonl")
    receipt = EffectReceipt(
        "id", "edit_file", {}, "ok", "edited app.py", ("app.py", "b.py")
    )
    store.put(receipt)
    expected = ReceiptExpectation("id", "edit_file", "ok", "edited", ("app.py", "b.py"))
    return store, receipt, expected


def evaluate(tmp_path, store, *expected):
    return evaluate_run(
        AgentRun(True, "final", "done", ()),
        Workspace(ToolPolicy(tmp_path)),
        store,
        EvaluationCase("done", receipts=expected),
    ).checks[1:]


@pytest.mark.parametrize("durable", [True, False])
def test_valid_receipts_are_read_only(tmp_path, evidence, durable, monkeypatch):
    store, receipt, expected = evidence
    if not durable:
        store = ReceiptStore()
        store.put(receipt)
    before = evidence[0].path.read_bytes()

    def forbidden(*args):
        pytest.fail("evaluation must not append receipts")

    monkeypatch.setattr(ReceiptStore, "put", forbidden)
    checks = evaluate(tmp_path, store, expected)
    assert checks[0].passed
    assert checks[0].detail == "receipt facts match"
    assert store.get("id") == receipt
    assert evidence[0].path.read_bytes() == before


@pytest.mark.parametrize(
    "field, value",
    [
        ("tool_call_id", "absent"),
        ("tool", "write_file"),
        ("exit_state", "error"),
        ("result_contains", "absent"),
        ("changed_artifacts", ("b.py", "app.py")),
        ("changed_artifacts", ("app.py",)),
    ],
)
def test_each_fact_is_required(tmp_path, evidence, field, value):
    store, _, expected = evidence
    checks = evaluate(tmp_path, store, replace(expected, **{field: value}))
    assert checks[0].passed is False


@pytest.mark.parametrize("tamper", ["digest", "utf8", "json", "deleted", "empty"])
def test_durable_evidence_is_reloaded(tmp_path, evidence, tamper):
    store, _, expected = evidence
    if tamper == "digest":
        payload = json.loads(store.path.read_text())
        payload["tool"] = "write_file"
        store.path.write_text(json.dumps(payload) + "\n")
    elif tamper == "utf8":
        store.path.write_bytes(b"\xff")
    elif tamper == "json":
        store.path.write_text("{")
    elif tamper == "deleted":
        store.path.unlink()
    else:
        store.path.write_text("")
    checks = evaluate(tmp_path, store, expected)
    assert checks[0].passed is False
    if tamper in ("digest", "utf8", "json"):
        assert "receipt evidence unavailable" in checks[0].detail
    assert store.get("id") is not None


def test_lookup_failure_does_not_abort_other_checks(tmp_path, evidence, monkeypatch):
    _, receipt, expected = evidence
    store = ReceiptStore()
    store.put(receipt)
    original = store.get

    def get(call_id):
        if call_id == "broken":
            raise AgentError("lookup broke")
        return original(call_id)

    monkeypatch.setattr(store, "get", get)
    checks = evaluate(
        tmp_path, store, replace(expected, tool_call_id="broken"), expected
    )
    assert [check.name for check in checks] == ["receipt:broken", "receipt:id"]
    assert [check.passed for check in checks] == [False, True]
    assert "receipt lookup failed" in checks[0].detail
