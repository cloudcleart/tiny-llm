"""Day 7 Task 3: exact, read-only workspace file evidence."""

from pathlib import Path

import pytest

from tiny_llm.agent import (
    AgentRun,
    EvaluationCase,
    FileExpectation,
    ReceiptStore,
    ToolPolicy,
    Workspace,
    evaluate_run,
)


def evaluate_files(workspace, *files):
    return evaluate_run(
        AgentRun(True, "final", "done", ()),
        workspace,
        ReceiptStore(),
        EvaluationCase("done", files=files),
    )


@pytest.mark.parametrize(
    "data, expected, passed",
    [
        (b"answer = 2\n", "answer = 2\n", True),
        (b"", "", True),
        ("中文\n".encode("utf-8"), "中文\n", True),
        (b"answer = 2\r\n", "answer = 2\n", False),
        (b"answer = 2\r\n", "answer = 2\r\n", True),
        (b"answer = 2\nextra", "answer = 2\n", False),
        (b"\xff", "", False),
        (b"\xff", "\ufffd", False),
    ],
)
def test_exact_utf8_content(tmp_path, data, expected, passed):
    (tmp_path / "app.py").write_bytes(data)
    report = evaluate_files(
        Workspace(ToolPolicy(tmp_path)), FileExpectation("app.py", expected)
    )
    assert [check.name for check in report.checks] == ["final", "file:app.py"]
    assert report.checks[1].passed is passed
    assert (tmp_path / "app.py").read_bytes() == data


def test_unavailable_files_do_not_abort_later_checks(tmp_path, monkeypatch):
    (tmp_path / "folder").mkdir()
    (tmp_path / "good").write_text("ok")
    (tmp_path / "blocked").write_text("ok")
    (tmp_path / "link").symlink_to(tmp_path / "good")
    (tmp_path / ".env").write_text("ok")
    original = Path.read_bytes

    def read_bytes(path):
        if path.name == "blocked":
            raise PermissionError("unreadable")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    names = ("missing", "folder", "link", ".env", "blocked", "good")
    report = evaluate_files(
        Workspace(ToolPolicy(tmp_path)),
        *(FileExpectation(name, "ok") for name in names),
    )
    assert [check.name for check in report.checks[1:]] == [
        f"file:{name}" for name in names
    ]
    assert [check.passed for check in report.checks] == [
        True,
        False,
        False,
        False,
        False,
        False,
        True,
    ]
    assert all(
        "file evidence unavailable" in check.detail for check in report.checks[1:-1]
    )


def test_evaluation_does_not_dispatch_stateful_read_tool(tmp_path, monkeypatch):
    (tmp_path / "evidence").write_text("ok")
    workspace = Workspace(ToolPolicy(tmp_path))

    def forbidden(*args, **kwargs):
        pytest.fail("evaluation must not dispatch workspace tools")

    monkeypatch.setattr(workspace, "read_file", forbidden)
    monkeypatch.setattr(workspace, "execute", forbidden)
    report = evaluate_files(workspace, FileExpectation("evidence", "ok"))
    assert report.checks[1].passed
    assert workspace.modified_files == ()
    assert workspace._observed == {}
