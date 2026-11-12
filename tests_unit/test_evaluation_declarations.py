"""Day 7 Task 1: validate outcome declarations without running evaluation."""

import pytest

from tiny_llm.agent import (
    EvaluationCase,
    FileExpectation,
    ReceiptExpectation,
    ResultExpectation,
)


def test_valid_declarations_preserve_exact_content():
    file = FileExpectation("./a//b", "")
    receipt = ReceiptExpectation("call-1", "edit_file", "ok", "edited", ("a/b",))
    case = EvaluationCase(
        " done ", (file,), (ResultExpectation("edit_file", "ok"),), (receipt,)
    )
    assert case.files[0].path == "a/b"
    assert case.files[0].content == ""
    assert case.final_contains == " done "


@pytest.mark.parametrize(
    "path", ["", "  ", "/tmp/a", "../a", "a/../b", ".", "a\0b", None]
)
def test_invalid_paths(path):
    with pytest.raises(ValueError, match="file path"):
        FileExpectation(path, "data")


@pytest.mark.parametrize(
    "factory",
    [
        lambda: EvaluationCase(" "),
        lambda: FileExpectation("a", None),
        lambda: ResultExpectation("", "ok"),
        lambda: ResultExpectation("read_file", " "),
        lambda: ReceiptExpectation(" ", "edit_file", "ok", "edited"),
        lambda: ReceiptExpectation("id", "", "ok", "edited"),
        lambda: ReceiptExpectation("id", "edit_file", "maybe", "edited"),
        lambda: ReceiptExpectation("id", "edit_file", "ok", " "),
        lambda: ReceiptExpectation("id", "edit_file", "ok", "edited", ("../a",)),
        lambda: EvaluationCase("done", files=(object(),)),
        lambda: EvaluationCase("done", files=[]),
    ],
)
def test_invalid_declarations(factory):
    with pytest.raises(ValueError):
        factory()


def test_duplicate_normalized_paths():
    with pytest.raises(ValueError, match="file paths must be unique"):
        EvaluationCase(
            "done", files=(FileExpectation("a/b", "1"), FileExpectation("a//b", "2"))
        )


def test_duplicate_receipt_ids():
    with pytest.raises(ValueError, match="receipt call ids must be unique"):
        EvaluationCase(
            "done",
            receipts=(
                ReceiptExpectation("id", "edit_file", "ok", "edited"),
                ReceiptExpectation("id", "run_command", "error", "failed"),
            ),
        )
