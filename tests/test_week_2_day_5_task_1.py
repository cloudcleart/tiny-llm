"""Python attention oracle tests independent of the later Metal tasks."""

import mlx.core as mx
import numpy as np
import pytest

from tiny_llm.week2_kernels import scaled_dot_product_attention


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("gqa_ratio", [1, 4])
@pytest.mark.parametrize(
    "query_length,context_length", [(1, 1), (1, 31), (8, 32), (3, 257)]
)
@pytest.mark.parametrize("mask_kind", ["none", "causal", "shared", "per_head"])
def test_task_1_matches_numpy(
    dtype, gqa_ratio, query_length, context_length, mask_kind
):
    rng = np.random.default_rng(5)
    batch_size, query_heads, head_dim = 2, 4, 128
    kv_heads = query_heads // gqa_ratio
    query = mx.array(
        rng.normal(size=(batch_size, query_heads, query_length, head_dim))
    ).astype(dtype)
    key = mx.array(
        rng.normal(size=(batch_size, kv_heads, context_length, head_dim))
    ).astype(dtype)
    value = mx.array(rng.normal(size=key.shape)).astype(dtype)

    mask = None
    if mask_kind == "causal":
        mask = "causal"
    elif mask_kind in ("shared", "per_head"):
        mask_shape = (
            (query_length, context_length)
            if mask_kind == "shared"
            else (batch_size, query_heads, query_length, context_length)
        )
        bias = rng.normal(size=mask_shape).astype(np.float32)
        if context_length > 1:
            bias[..., -1] = -np.inf
        mask = mx.array(bias)

    # Explicit head repetition is intentionally confined to this independent oracle.
    q = np.array(query.astype(mx.float32))
    k = np.repeat(np.array(key.astype(mx.float32)), gqa_ratio, axis=1)
    v = np.repeat(np.array(value.astype(mx.float32)), gqa_ratio, axis=1)
    scale = 0.17
    scores = (q @ k.swapaxes(-2, -1)) * scale
    if mask_kind == "causal":
        positions = np.arange(query_length) + context_length - query_length
        allowed = np.arange(context_length)[None, :] <= positions[:, None]
        scores = np.where(allowed, scores, -np.inf)
    elif mask is not None:
        scores += np.array(mask)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    expected = weights @ v

    result = scaled_dot_product_attention(query, key, value, scale, mask)
    assert result.shape == query.shape
    assert result.dtype == dtype
    tolerance = 2e-2 if dtype == mx.bfloat16 else 2e-5
    np.testing.assert_allclose(
        np.array(result.astype(mx.float32)), expected, atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize(
    "query_shape,key_shape,value_shape",
    [
        ((4, 1, 8), (1, 2, 5, 8), (1, 2, 5, 8)),
        ((2, 4, 1, 8), (1, 2, 5, 8), (1, 2, 5, 8)),
        ((1, 4, 1, 8), (1, 2, 5, 8), (1, 2, 4, 8)),
        ((1, 4, 1, 8), (1, 2, 5, 16), (1, 2, 5, 16)),
        ((1, 3, 1, 8), (1, 2, 5, 8), (1, 2, 5, 8)),
    ],
)
def test_task_1_rejects_incompatible_shapes(query_shape, key_shape, value_shape):
    with pytest.raises(ValueError):
        scaled_dot_product_attention(
            mx.zeros(query_shape), mx.zeros(key_shape), mx.zeros(value_shape), 0.5
        )


def test_task_1_rejects_unknown_string_mask():
    query = mx.zeros((1, 4, 1, 8))
    key = mx.zeros((1, 2, 5, 8))
    with pytest.raises(ValueError, match="unsupported attention mask"):
        scaled_dot_product_attention(query, key, key, 0.5, "sliding")
