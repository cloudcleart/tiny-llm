"""Day 7 Task 6: aggregate checks and render a stable report."""

import pytest

from tiny_llm.agent import EvaluationCheck, EvaluationReport


@pytest.mark.parametrize("failed_index", [None, 0, 1, 2])
def test_report_requires_every_check_to_pass(failed_index):
    checks = tuple(
        EvaluationCheck(name, i != failed_index, detail)
        for i, (name, detail) in enumerate(
            (
                ("final", "required final observed"),
                ("file:z.py", "content matches"),
                ("file:a.py", "content mismatch"),
            )
        )
    )
    report = EvaluationReport(checks)
    assert report.passed is (failed_index is None)
    expected = "\n".join(
        [
            "evaluation: PASS" if failed_index is None else "evaluation: FAIL",
            *[
                f"- {check.name}: {'PASS' if check.passed else 'FAIL'} ({check.detail})"
                for check in checks
            ],
        ]
    )
    assert report.render() == expected
    assert report.render() == expected
    assert report.checks == checks


def test_exact_success_summary():
    report = EvaluationReport(
        (
            EvaluationCheck("final", True, "required final observed"),
            EvaluationCheck("file:app.py", True, "content matches"),
            EvaluationCheck("result:run_command", True, "required result observed"),
            EvaluationCheck("receipt:call-1", True, "receipt facts match"),
        )
    )
    assert report.render() == (
        "evaluation: PASS\n"
        "- final: PASS (required final observed)\n"
        "- file:app.py: PASS (content matches)\n"
        "- result:run_command: PASS (required result observed)\n"
        "- receipt:call-1: PASS (receipt facts match)"
    )


def test_empty_report_has_no_failed_checks():
    report = EvaluationReport(())
    assert report.passed
    assert report.render() == "evaluation: PASS"
