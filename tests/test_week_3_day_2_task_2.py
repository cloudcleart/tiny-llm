"""Chunked prefill mask and absolute-position regression tests, without downloads."""

import mlx.core as mx
import numpy as np
import pytest

from tiny_llm.attention import causal_mask, scaled_dot_product_attention_grouped
from tiny_llm.qwen3_week2 import Qwen3ModelWeek2
from tiny_llm.positional_encoding import RoPE
from .utils import tiny_qwen3_mlx_model


@pytest.mark.parametrize("traditional", [False, True])
def test_task_2_rope_uses_each_requests_absolute_position(traditional):
    mx.random.seed(7)
    rope = RoPE(8, 32, traditional=traditional)
    x = mx.random.normal((2, 3, 2, 8))
    offsets = [slice(5, 8), slice(11, 14)]
    actual = rope(x, offsets)
    expected = mx.concatenate(
        [rope(x[i : i + 1], offset) for i, offset in enumerate(offsets)]
    )
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-6)


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("prefix,length", [(0, 3), (5, 3), (5, 1)])
def test_task_2_rectangular_mask(prefix, length, dtype):
    mask = causal_mask(length, prefix + length, dtype)
    expected = [
        [0 if j <= prefix + i else -np.inf for j in range(prefix + length)]
        for i in range(length)
    ]
    assert mask.dtype == dtype
    np.testing.assert_array_equal(np.array(mask.astype(mx.float32)), expected)


def test_task_2_attention_reads_prefix_without_seeing_future():
    # Equal scores make every visible value contribute equally. The large
    # final value exposes any accidental access to a future chunk position.
    query = mx.zeros((1, 2, 3, 4))
    key = mx.zeros((1, 1, 8, 4))
    value = mx.broadcast_to(
        mx.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 800.0])[None, None, :, None],
        (1, 1, 8, 4),
    )
    result = scaled_dot_product_attention_grouped(query, key, value, mask="causal")
    expected = np.broadcast_to(
        np.array([3.5, 4.0, 103.5])[None, None, :, None], (1, 2, 3, 4)
    )
    np.testing.assert_allclose(np.array(result), expected, rtol=1e-6)


@pytest.mark.parametrize("checkpoint", ["kv-cache", "rope", "decode-attention"])
@pytest.mark.parametrize("chunk_size", [1, 3, 8])
def test_task_2_chunked_logits_and_cache_match_full_prefill(checkpoint, chunk_size):
    mx.random.seed(42)
    model = Qwen3ModelWeek2(
        tiny_qwen3_mlx_model(num_hidden_layers=2), checkpoint=checkpoint
    )
    tokens = mx.array([[1, 9, 2, 8, 3, 7, 4, 6]], dtype=mx.int32)
    full_cache = model.create_kv_cache()
    full_logits = model(tokens, [0], full_cache)
    chunk_cache = model.create_kv_cache()
    chunks = []
    for start in range(0, tokens.shape[1], chunk_size):
        chunks.append(
            model(tokens[:, start : start + chunk_size], [start], chunk_cache)
        )
    chunk_logits = mx.concatenate(chunks, axis=1)
    np.testing.assert_allclose(
        np.array(chunk_logits.astype(mx.float32)),
        np.array(full_logits.astype(mx.float32)),
        rtol=0.05,
        atol=0.1,
    )
    for full, chunked in zip(full_cache, chunk_cache):
        assert full.offset == chunked.offset == tokens.shape[1]
        for full_tensor, chunk_tensor in zip(full.key_values, chunked.key_values):
            np.testing.assert_allclose(
                np.array(chunk_tensor.astype(mx.float32)),
                np.array(full_tensor.astype(mx.float32)),
                rtol=0.05,
                atol=0.1,
            )
