"""Day 7 Task 2: check only the completed run's public final answer."""

import pytest

from tiny_llm.agent import (
    AgentEvent,
    AgentRun,
    EvaluationCase,
    EvaluationCheck,
    ReceiptStore,
    ToolPolicy,
    Workspace,
    evaluate_run,
)


@pytest.mark.parametrize(
    "completed, final, expected",
    [
        (True, "updated app.py and validated the result", True),
        (True, "The result is validated!", True),
        (False, "validated", False),
        (True, None, False),
        (True, "", False),
        (True, "validation pending", False),
        (True, "VALIDATED", False),
    ],
)
def test_final_requires_completion_and_literal_substring(
    tmp_path, completed, final, expected
):
    # Matching text in events cannot substitute for the public final answer.
    run = AgentRun(
        completed,
        "final" if completed else "step_limit",
        final,
        (AgentEvent(0, "validated", None, "validated"),),
    )
    workspace = Workspace(ToolPolicy(tmp_path))
    report = evaluate_run(run, workspace, ReceiptStore(), EvaluationCase("validated"))
    assert report.checks == (
        EvaluationCheck(
            "final",
            expected,
            "required final observed" if expected else "required final not observed",
        ),
    )


@pytest.mark.parametrize("invalid", ["case", "run"])
def test_invalid_evaluation_inputs(tmp_path, invalid):
    run = AgentRun(True, "final", "done", ())
    case = EvaluationCase("done")
    with pytest.raises(
        ValueError, match="EvaluationCase" if invalid == "case" else "AgentRun"
    ):
        evaluate_run(
            object() if invalid == "run" else run,
            Workspace(ToolPolicy(tmp_path)),
            ReceiptStore(),
            object() if invalid == "case" else case,
        )
