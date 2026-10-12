"""Day 4 edge cases: query boundaries, poisoned tails, idle slots, and batching."""

import mlx.core as mx
import numpy as np
import pytest

from tiny_llm.attention import paged_attention
from tiny_llm.embedding import QuantizedEmbedding
from tiny_llm.kv_cache import TinyKvFullCache
from tiny_llm.quantize import QuantizedWeights


def _assert_paged_attention_matches_numpy(
    query_length, head_dim, page_size, dtype, mask, prefix_lengths=(5, 13)
):
    # Use two active requests and one idle slot, with deliberately shuffled physical pages.
    # Build metadata independently so matching cache bugs cannot hide attention errors.
    mx.random.seed(42)
    lengths = [query_length + prefix_lengths[0], 0, query_length + prefix_lengths[1]]
    counts = [(s + page_size - 1) // page_size for s in lengths]
    width = max(counts)
    total = sum(counts)
    physical = np.random.default_rng(17).permutation(total).tolist()
    tables, cursor = [], 0
    for count in counts:
        tables.append(physical[cursor : cursor + count] + [-1] * (width - count))
        cursor += count
    pages = mx.full((total, 2, page_size, head_dim), mx.nan, dtype=dtype)
    values = mx.full(pages.shape, mx.nan, dtype=dtype)
    query = mx.random.normal((3, 4, query_length, head_dim)).astype(dtype)
    expected = []
    for b, length in enumerate(lengths):
        if not length:
            expected.append(np.zeros((4, query_length, head_dim), dtype=np.float32))
            continue
        key = mx.random.normal((2, length, head_dim)).astype(dtype)
        value = mx.random.normal(key.shape).astype(dtype)
        for logical, page in enumerate(tables[b][: counts[b]]):
            start = logical * page_size
            count = min(page_size, length - start)
            pages[page, :, :count] = key[:, start : start + count]
            values[page, :, :count] = value[:, start : start + count]
        # Use an independent NumPy float64 oracle on valid data already rounded to the target dtype.
        q = np.array(query[b].astype(mx.float32)).astype(np.float64)
        k = np.repeat(np.array(key.astype(mx.float32)), 2, axis=0).astype(np.float64)
        v = np.repeat(np.array(value.astype(mx.float32)), 2, axis=0).astype(np.float64)
        scores = (q @ k.swapaxes(-1, -2)) * head_dim**-0.5
        if mask:
            visible = (
                np.arange(length)[None, :]
                <= (length - query_length + np.arange(query_length))[:, None]
            )
            scores = np.where(visible, scores, -np.inf)
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True)
        expected.append(weights @ v)
    output = paged_attention(
        query,
        pages,
        values,
        mx.array(tables, mx.int32),
        mx.array(lengths, mx.int32),
        page_size,
        mask=mask,
    )
    assert output.shape == query.shape
    assert output.dtype == dtype
    actual = np.array(output.astype(mx.float32))
    tolerance = 0.008 if dtype == mx.bfloat16 else 2e-5
    np.testing.assert_allclose(
        actual, np.stack(expected), atol=tolerance, rtol=tolerance
    )
    np.testing.assert_array_equal(actual[1], 0)


@pytest.mark.parametrize(
    "query_length,head_dim,page_size", [(9, 128, 4), (65, 128, 32), (17, 37, 7)]
)
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("mask", [None, "causal"])
def test_task_2_bfloat16_long_prefill(query_length, head_dim, page_size, dtype, mask):
    _assert_paged_attention_matches_numpy(
        query_length, head_dim, page_size, dtype, mask
    )


@pytest.mark.parametrize("query_length", [1, 8], ids=["decode", "short-query-limit"])
@pytest.mark.parametrize(
    "head_dim,page_size",
    [(32, 4), (37, 7), (128, 33)],
    ids=["generic-d32", "generic-d37", "d128-page33"],
)
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("mask", [None, "causal"])
def test_task_2_short_queries_match_numpy(
    query_length, head_dim, page_size, dtype, mask
):
    # A longer prefix spans several physical pages and leaves a partial tail page.
    # Page size 33 covers all 32 partitions and makes the first one revisit slot 32.
    _assert_paged_attention_matches_numpy(
        query_length, head_dim, page_size, dtype, mask, prefix_lengths=(37, 73)
    )


@pytest.mark.parametrize(
    "head_dim,page_size",
    [(32, 4), (37, 7), (128, 33)],
    ids=["generic-d32", "generic-d37", "d128-page33"],
)
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("mask", [None, "causal"])
def test_task_2_first_long_query_matches_numpy(head_dim, page_size, dtype, mask):
    # L=9 is the first long query. Its context crosses both page and 32-token
    # tile boundaries, while an idle request and NaN tails remain in the batch.
    _assert_paged_attention_matches_numpy(
        9, head_dim, page_size, dtype, mask, prefix_lengths=(37, 73)
    )


@pytest.mark.parametrize(
    "query_length,head_dim",
    [(1, 129), (9, 129), (1, 257), (9, 257), (1, 1025), (1, 2046)],
    ids=[
        "decode-d129",
        "prefill-d129",
        "decode-d257",
        "prefill-d257",
        "decode-d1025",
        "decode-d2046",
    ],
)
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_task_2_wide_heads_match_numpy(query_length, head_dim, dtype):
    # Non-divisible dimensions exercise the final lane and final dimension chunk.
    # Keep contexts small so the widest supported heads test resource handling cheaply.
    _assert_paged_attention_matches_numpy(query_length, head_dim, 7, dtype, "causal")


def test_dense_cache_rejects_paged_contract():
    cache = TinyKvFullCache()
    with pytest.raises(ValueError, match="paged runtime metadata"):
        cache.update_and_fetch_paged(mx.zeros((1, 1, 1, 4)), mx.zeros((1, 1, 1, 4)))
    assert cache.offset == 0


@pytest.mark.parametrize("shape", [(), (2, 2), (0,)])
@pytest.mark.parametrize("has_bias", [False, True])
def test_custom_embedding_token_shapes(shape, has_bias):
    mx.random.seed(7)
    packed, scales, biases = mx.quantize(
        mx.random.normal((11, 256)).astype(mx.bfloat16), group_size=128, bits=4
    )
    weights = QuantizedWeights(scales, biases if has_bias else None, 128, 4, packed)
    indices = mx.zeros(shape, dtype=mx.uint32)
    actual = QuantizedEmbedding(11, 256, weights, use_custom_kernel=True)(indices)
    expected = QuantizedEmbedding(11, 256, weights)(indices)
    assert actual.shape == (*shape, 256)
    np.testing.assert_array_equal(
        np.array(actual.astype(mx.float32)), np.array(expected.astype(mx.float32))
    )


def test_continuous_batch_uses_pages_and_releases_every_layer(monkeypatch):
    from tiny_llm.batch import batch_generate
    from tiny_llm.paged_kv_cache import TinyKvPagedCache
    from tiny_llm.qwen3_week3 import Qwen3ModelWeek3
    from .test_week_3_day_2_task_3 import Tokenizer
    from .utils import tiny_qwen3_mlx_model

    mx.random.seed(13)
    model = Qwen3ModelWeek3(tiny_qwen3_mlx_model(num_hidden_layers=2), page_size=2)

    def reject_dense(*args, **kwargs):
        pytest.fail("completed Week 3 route must never gather dense KV")

    monkeypatch.setattr(TinyKvPagedCache, "gather_dense", reject_dense)
    results = batch_generate(
        model,
        Tokenizer(),
        ["abc", "d", "efgh", "ij"],
        max_seq_len=7,
        batch_size=2,
        prefill_step=2,
    )
    assert sorted(index for index, _ in results) == [0, 1, 2, 3]
    for pool in model.page_pools:
        assert not pool.used_page_ids
        assert pool.num_free_pages == pool.num_pages
        assert pool.reused_page_allocations > 0
