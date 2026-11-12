"""Day 7 Task 4: match observable results without grading a transcript."""

from itertools import permutations

import pytest

from tiny_llm.agent import (
    AgentEvent,
    AgentRun,
    EvaluationCase,
    FileExpectation,
    FinalAction,
    ReceiptStore,
    ResultExpectation,
    ToolAction,
    ToolPolicy,
    Workspace,
    evaluate_run,
)


def event(tool, result):
    return AgentEvent(1, "validation passed", ToolAction(tool, {}), result)


def evaluate(tmp_path, events, results, files=()):
    return evaluate_run(
        AgentRun(True, "final", "done", tuple(events)),
        Workspace(ToolPolicy(tmp_path)),
        ReceiptStore(),
        EvaluationCase("done", files=files, results=results),
    )


@pytest.mark.parametrize(
    "events",
    [
        (),
        (event("run_command", None),),
        (event("run_command", ""),),
        (event("read_file", "validation passed"),),
        (event("run_command", "VALIDATION PASSED"),),
        (event("run_command", "pending"), event("read_file", "validation passed")),
        (AgentEvent(1, "validation passed", None, "validation passed"),),
        (
            AgentEvent(
                1,
                "validation passed",
                FinalAction("validation passed"),
                "validation passed",
            ),
        ),
    ],
)
def test_missing_or_split_evidence_fails(tmp_path, events):
    report = evaluate(
        tmp_path, events, (ResultExpectation("run_command", "validation passed"),)
    )
    assert report.checks[0].passed
    assert report.checks[1].name == "result:run_command"
    assert report.checks[1].passed is False


@pytest.mark.parametrize(
    "events",
    list(
        permutations(
            (
                event("run_command", "status: 0\nvalidation passed\n"),
                event("edit_file", "edited app.py"),
                event("read_file", "unrelated"),
            )
        )
    ),
)
def test_order_and_extra_observations_do_not_matter(tmp_path, events):
    (tmp_path / "app.py").write_text("ok")
    report = evaluate(
        tmp_path,
        (*events, events[0]),
        (
            ResultExpectation("edit_file", "edited"),
            ResultExpectation("run_command", "validation passed"),
            ResultExpectation("run_command", "not present"),
        ),
        (FileExpectation("app.py", "ok"),),
    )
    assert [check.name for check in report.checks] == [
        "final",
        "file:app.py",
        "result:edit_file",
        "result:run_command",
        "result:run_command",
    ]
    assert [check.passed for check in report.checks] == [True, True, True, True, False]
    assert report.checks[2].detail == "required result observed"
