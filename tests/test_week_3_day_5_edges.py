"""Day 5 numerical boundaries and compatibility with the Day 4 schedules."""

import mlx.core as mx
import numpy as np
import pytest

from tiny_llm.attention import paged_attention
from tiny_llm.paged_kv_cache import TinyKvPagedCache, TinyKvPagedPool


def _as_numpy(array):
    return np.array(array.astype(mx.float32)).astype(np.float64)


def _dense_attention(query, key, value, mask, scale):
    """Independent dense oracle; no page walking, tiling, or online recurrence."""
    repeats = query.shape[0] // key.shape[0]
    key = np.repeat(key, repeats, axis=0)
    value = np.repeat(value, repeats, axis=0)
    scores = (query @ key.swapaxes(-1, -2)) * scale
    if mask == "causal":
        query_length, context_length = query.shape[1], key.shape[1]
        visible = (
            np.arange(context_length)[None, :]
            <= (context_length - query_length + np.arange(query_length))[:, None]
        )
        scores = np.where(visible, scores, -np.inf)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ value


def _assert_output(actual, expected, query):
    # Synchronize at the operator boundary so dispatch/address errors stay local.
    mx.eval(actual)
    assert actual.shape == query.shape
    assert actual.dtype == query.dtype
    tolerance = 1e-2 if query.dtype == mx.bfloat16 else 2e-5
    np.testing.assert_allclose(
        _as_numpy(actual), expected, atol=tolerance, rtol=tolerance
    )


def _assert_shuffled_pages(
    query_length,
    head_dim,
    page_size,
    dtype,
    prefix_length,
    query_heads,
    kv_heads,
    mask,
    scale=None,
):
    rng = np.random.default_rng(193)
    # Two different active context lengths plus an idle batch slot. Every live
    # logical page maps to a nonconsecutive physical page; all other bytes are NaN.
    lengths = [query_length + prefix_length, 0, query_length + prefix_length + 7]
    page_counts = [(length + page_size - 1) // page_size for length in lengths]
    total_live_pages = sum(page_counts)
    physical_ids = list(range(1, 2 * total_live_pages, 2))[::-1]
    table = np.full((3, max(page_counts) + 1), -1, dtype=np.int32)
    page_shape = (2 * total_live_pages + 1, kv_heads, page_size, head_dim)
    keys = np.full(page_shape, np.nan, dtype=np.float32)
    values = np.full(page_shape, np.nan, dtype=np.float32)
    query = mx.array(
        rng.normal(size=(3, query_heads, query_length, head_dim)).astype(np.float32),
        dtype=dtype,
    )
    rounded_query = _as_numpy(query)
    expected = []
    cursor = 0
    score_scale = head_dim**-0.5 if scale is None else scale
    for batch, length in enumerate(lengths):
        if length == 0:
            expected.append(np.zeros((query_heads, query_length, head_dim)))
            continue
        # Round the oracle inputs to exactly the same dtype as the page buffers.
        key = _as_numpy(
            mx.array(rng.normal(size=(kv_heads, length, head_dim)), dtype=dtype)
        )
        value = _as_numpy(
            mx.array(rng.normal(size=(kv_heads, length, head_dim)), dtype=dtype)
        )
        for logical_page in range(page_counts[batch]):
            physical_page = physical_ids[cursor]
            cursor += 1
            table[batch, logical_page] = physical_page
            start = logical_page * page_size
            live = min(page_size, length - start)
            keys[physical_page, :, :live] = key[:, start : start + live]
            values[physical_page, :, :live] = value[:, start : start + live]
        expected.append(
            _dense_attention(rounded_query[batch], key, value, mask, score_scale)
        )
    actual = paged_attention(
        query,
        mx.array(keys, dtype=dtype),
        mx.array(values, dtype=dtype),
        mx.array(table, dtype=mx.int32),
        mx.array(lengths, dtype=mx.int32),
        page_size,
        mask=mask,
        scale=scale,
    )
    _assert_output(actual, np.stack(expected), query)
    np.testing.assert_array_equal(_as_numpy(actual)[1], 0)


@pytest.mark.parametrize("mask", [None, "causal"], ids=["unmasked", "causal"])
@pytest.mark.parametrize(
    "query_length,prefix_length,page_size,query_heads,kv_heads",
    [
        (9, 0, 128, 2, 2),
        (9, 32, 16, 8, 2),
        (65, 5, 33, 4, 1),
        (65, 64, 128, 4, 2),
    ],
    ids=["single-page", "cross-page-gqa", "odd-pages-q-tail", "aligned-kv-tail"],
)
def test_paged_flash_attention_page_and_tile_tails(
    query_length, prefix_length, page_size, query_heads, kv_heads, mask
):
    _assert_shuffled_pages(
        query_length,
        128,
        page_size,
        mx.bfloat16,
        prefix_length,
        query_heads,
        kv_heads,
        mask,
        scale=0.19,
    )


@pytest.mark.parametrize("mask", [None, "causal"], ids=["unmasked", "causal"])
@pytest.mark.parametrize(
    "query_length,head_dim,dtype",
    [
        (1, 128, mx.bfloat16),
        (8, 128, mx.bfloat16),
        (9, 128, mx.float32),
        (65, 128, mx.float32),
        (9, 37, mx.bfloat16),
        (65, 129, mx.bfloat16),
        (9, 37, mx.float32),
        (8, 257, mx.bfloat16),
    ],
    ids=[
        "decode",
        "last-short-query",
        "first-f32-prefill",
        "f32-q-tail",
        "bf16-small-generic-head",
        "bf16-wide-generic-head",
        "f32-generic-head",
        "wide-short-query",
    ],
)
def test_day5_preserves_generic_paged_attention(query_length, head_dim, dtype, mask):
    _assert_shuffled_pages(query_length, head_dim, 7, dtype, 13, 4, 2, mask)


def test_late_tile_maximum_rescales_history_and_respects_causal_tail():
    # Only the final query can see the very large score in the final K/V tile.
    # Its output must discard almost all historical weight; earlier rows retain 1.
    query_length, context, page_size, head_dim = 65, 129, 16, 128
    query_data = np.zeros((1, 2, query_length, head_dim), dtype=np.float32)
    query_data[..., 0] = 1
    key = np.zeros((1, context, head_dim), dtype=np.float32)
    key[:, -1, 0] = 20
    value = np.ones_like(key)
    value[:, -1] = -7
    table = list(range(17, 0, -2))
    pages = np.full((18, 1, page_size, head_dim), np.nan, dtype=np.float32)
    value_pages = np.full_like(pages, np.nan)
    for logical, physical in enumerate(table):
        start = logical * page_size
        live = min(page_size, context - start)
        pages[physical, :, :live] = key[:, start : start + live]
        value_pages[physical, :, :live] = value[:, start : start + live]
    query = mx.array(query_data, dtype=mx.bfloat16)
    actual = paged_attention(
        query,
        mx.array(pages, dtype=mx.bfloat16),
        mx.array(value_pages, dtype=mx.bfloat16),
        mx.array([table], dtype=mx.int32),
        mx.array([context], dtype=mx.int32),
        page_size,
        scale=1.0,
        mask="causal",
    )
    expected = _dense_attention(query_data[0], key, value, "causal", 1.0)[None]
    _assert_output(actual, expected, query)


def test_paged_prefill_then_decode_without_dense_gather(monkeypatch):
    rng = np.random.default_rng(41)
    page_size, head_dim, query_heads, kv_heads = 16, 128, 4, 2
    pool = TinyKvPagedPool(page_size=page_size)
    cache = TinyKvPagedCache(pool=pool)
    blocker = TinyKvPagedCache(pool=pool)

    def chunk(length):
        shape = (1, kv_heads, length, head_dim)
        return (
            mx.array(rng.normal(size=shape), dtype=mx.bfloat16),
            mx.array(rng.normal(size=shape), dtype=mx.bfloat16),
        )

    prefix_key, prefix_value = chunk(64)
    cache.update_and_fetch_paged(prefix_key, prefix_value, mask="causal")
    initial_pages = len(cache.page_ids)
    blocker.update_and_fetch_paged(*chunk(page_size), mask="causal")
    dense_keys = [_as_numpy(prefix_key)[0]]
    dense_values = [_as_numpy(prefix_value)[0]]

    def reject_gather(*args, **kwargs):
        pytest.fail("paged prefill and decode must not gather dense KV")

    monkeypatch.setattr(TinyKvPagedCache, "gather_dense", reject_gather)
    for query_length in (65, 1, 8):
        key, value = chunk(query_length)
        metadata = cache.update_and_fetch_paged(key, value, mask="causal")
        dense_keys.append(_as_numpy(key)[0])
        dense_values.append(_as_numpy(value)[0])
        query = mx.array(
            rng.normal(size=(1, query_heads, query_length, head_dim)),
            dtype=mx.bfloat16,
        )
        expected = _dense_attention(
            _as_numpy(query)[0],
            np.concatenate(dense_keys, axis=1),
            np.concatenate(dense_values, axis=1),
            "causal",
            head_dim**-0.5,
        )[None]
        actual = paged_attention(
            query,
            metadata.key_pages,
            metadata.value_pages,
            metadata.block_table,
            metadata.context_lens,
            metadata.page_size,
            mask=metadata.mask,
        )
        _assert_output(actual, expected, query)
        assert metadata.context_lens.tolist() == [sum(x.shape[1] for x in dense_keys)]

    assert cache.page_ids[initial_pages] == initial_pages + 1
