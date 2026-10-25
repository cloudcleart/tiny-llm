"""Single-call verification checkpoint; cache reconciliation belongs to Task 4."""

import mlx.core as mx
import pytest

from tiny_llm.generate import _verify_draft, speculative_generate
from tests_refsol.test_week_3_day_7 import (
    EOS,
    PROMPT,
    ScriptedModel,
    _assert_released,
    _tokenizers,
)


@pytest.mark.parametrize(
    "proposal,predictions,matched,next_token",
    [
        ([2, 3, 4], [7, 3, 4, 8], 0, 7),
        ([2, 3, 4], [2, 7, 4, 8], 1, 7),
        ([2, 3, 4], [2, 3, 7, 8], 2, 7),
        ([2, 3, 4], [2, 3, 4, 8], 3, 8),
        ([2, 3], [2, 3, EOS], 2, EOS),
        ([2, EOS], [2, EOS, 8], 2, 8),
        ([EOS], [3, 8], 0, 3),
        ([], [8], 0, 8),
    ],
)
def test_verification_alignment_and_single_target_call(
    proposal, predictions, matched, next_token
):
    target = ScriptedModel([predictions], "target")
    cache = target.create_kv_cache()
    cache[0].offset = 5
    # Mixed signed/unsigned inputs must not become 64-bit embedding indices.
    result = _verify_draft(
        target,
        mx.array([1], dtype=mx.uint32),
        mx.array(proposal, dtype=mx.int32),
        5,
        cache,
    )
    assert result.matched_count == matched
    assert result.next_token.tolist() == [next_token]
    assert result.next_token.dtype == mx.int32
    assert result.offset == cache[0].offset == 6 + len(proposal)
    assert target.calls == [
        {
            "tokens": [1, *proposal],
            "offset": 5,
            "logits_to_keep": len(proposal) + 1,
            "dtype": mx.int32,
        }
    ]
    # Verification reports the speculative offset, leaving rewind to Task 4.
    assert cache[0].rewind_calls == []


def test_verification_is_connected_to_generation_and_releases_caches():
    target = ScriptedModel([[1], [2, 7, 8]], "target")
    draft = ScriptedModel([[9], [2], [3]], "draft")
    draft_tokenizer, tokenizer = _tokenizers()
    assert (
        speculative_generate(
            draft,
            target,
            draft_tokenizer,
            tokenizer,
            "prompt",
            proposal_length=2,
            max_tokens=3,
        )
        == "ABG"
    )
    assert [call["tokens"] for call in target.calls] == [PROMPT, [1, 2, 3]]
    assert tokenizer.created_detokenizers[0].tokens == [1, 2, 7]
    _assert_released(target, draft)
