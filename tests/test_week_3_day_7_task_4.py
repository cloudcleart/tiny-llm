"""Exercise speculative commit with real caches and prefix-dependent logits."""

import mlx.core as mx
import pytest

from tiny_llm.generate import speculative_generate
from tiny_llm.kv_cache import TinyKvFullCache
from tiny_llm.paged_kv_cache import TinyKvPagedCache, TinyKvPagedPool
from tests_refsol.test_week_3_day_7 import _tokenizers


class PrefixModel:
    def __init__(self, paged, draft=False, fail=False):
        self.paged, self.draft, self.fail = paged, draft, fail
        self.pools = [TinyKvPagedPool(page_size=2) for _ in range(2)]
        self.caches = []
        self.calls = 0

    def create_kv_cache(self):
        self.caches = [
            TinyKvPagedCache(pool) if self.paged else TinyKvFullCache()
            for pool in self.pools
        ]
        return self.caches

    def __call__(self, tokens, offset, caches, logits_to_keep=1):
        self.calls += 1
        for cache in caches:
            assert cache.offset == offset
        # Store actual input IDs in K/V, so bad rollback changes predictions.
        values = tokens.astype(mx.float32).reshape(1, 1, -1, 1)
        histories = []
        for cache in caches:
            keys, _, length, _ = cache.update_and_fetch(values, values)
            assert length == offset + tokens.size
            histories.append([int(x) for x in keys.reshape(-1).tolist()])
        assert histories[0] == histories[1]
        if self.fail and self.calls == 2:
            raise RuntimeError("verification failed")
        predictions = []
        for position in range(offset, len(histories[0])):
            prediction = sum(histories[0][: position + 1]) % 9 + 1
            # Alternate matched prefixes and rejections, including across pages.
            if self.draft and position % 3 == 0:
                prediction = prediction % 9 + 1
            predictions.append(prediction)
        logits = [
            [10.0 if token == prediction else -10.0 for token in range(32)]
            for prediction in predictions[-logits_to_keep:]
        ]
        return mx.array([logits])


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("budget", [1, 2, 5, 13])
@pytest.mark.parametrize("proposal_length", [1, 4])
def test_commit_matches_target_only_with_real_caches(paged, budget, proposal_length):
    draft_tokenizer, tokenizer = _tokenizers()
    target, draft = PrefixModel(paged), PrefixModel(paged, draft=True)
    actual = speculative_generate(
        draft,
        target,
        draft_tokenizer,
        tokenizer,
        "prompt",
        proposal_length=proposal_length,
        max_tokens=budget,
    )
    expected = speculative_generate(
        PrefixModel(paged),
        PrefixModel(paged),
        draft_tokenizer,
        tokenizer,
        "prompt",
        proposal_length=0,
        max_tokens=budget,
    )
    assert actual == expected
    assert len(actual) == budget
    if paged:
        assert all(
            not pool.used_page_ids for model in (draft, target) for pool in model.pools
        )
        assert all(
            cache.offset == 0 for model in (draft, target) for cache in model.caches
        )


def test_verification_exception_returns_all_pages():
    draft_tokenizer, tokenizer = _tokenizers()
    target, draft = PrefixModel(True, fail=True), PrefixModel(True, draft=True)
    with pytest.raises(RuntimeError, match="verification failed"):
        speculative_generate(draft, target, draft_tokenizer, tokenizer, "prompt")
    assert all(
        not pool.used_page_ids for model in (draft, target) for pool in model.pools
    )
