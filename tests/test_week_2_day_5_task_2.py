"""Online-softmax primitive coverage beyond the chapter's Qwen boundary sweep."""

import mlx.core as mx
import numpy as np
import pytest

from extensions import tiny_llm_ext
from tiny_llm.week2_kernels import decode_attention_custom, scaled_dot_product_attention


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("dim", [16, 33, 128, 256])
@pytest.mark.parametrize("mask_kind", ["none", "causal", "explicit"])
def test_task_2_matches_oracle_with_strided_inputs(dtype, dim, mask_kind):
    rng = np.random.default_rng(2)
    # Transposes exercise the adapter; distinct batches and KV heads expose mapping errors.
    query = mx.array(rng.normal(size=(2, 3, 8, dim))).astype(dtype).swapaxes(1, 2)
    key = mx.array(rng.normal(size=(2, 65, 2, dim))).astype(dtype).swapaxes(1, 2)
    value = mx.array(rng.normal(size=(2, 65, 2, dim))).astype(dtype).swapaxes(1, 2)
    mask = None
    if mask_kind == "causal":
        mask = "causal"
    elif mask_kind == "explicit":
        bias = rng.normal(size=(2, 8, 3, 65)).astype(np.float32)
        # Several SIMD groups see only masked positions; others initialize late.
        bias[..., :40] = -np.inf
        mask = mx.array(bias)
    result = decode_attention_custom(query, key, value, dim**-0.5, mask)
    expected = scaled_dot_product_attention(query, key, value, dim**-0.5, mask)
    assert result.shape == query.shape
    assert result.dtype == dtype
    tolerance = 2e-2 if dtype == mx.bfloat16 else 2e-3 if dtype == mx.float16 else 2e-5
    np.testing.assert_allclose(
        np.array(result.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
        atol=tolerance,
        rtol=tolerance,
    )


def test_task_2_large_scores_remain_finite():
    query = mx.full((1, 4, 1, 128), 100.0)
    key = mx.full((1, 1, 129, 128), 100.0)
    value = mx.arange(129 * 128, dtype=mx.float32).reshape(key.shape) / 1000
    result = decode_attention_custom(query, key, value, 1.0)
    expected = mx.broadcast_to(mx.mean(value, axis=2, keepdims=True), query.shape)
    np.testing.assert_allclose(
        np.array(result), np.array(expected), atol=2e-5, rtol=2e-5
    )


def test_task_2_all_masked_row_matches_softmax_nan():
    query = mx.ones((1, 4, 1, 128))
    key = mx.ones((1, 1, 1, 128))
    result = decode_attention_custom(query, key, key, 1.0, mx.array([-mx.inf]))
    assert mx.all(mx.isnan(result)).item()


@pytest.mark.parametrize("heads,kv_heads", [(4, 0), (0, 1), (3, 2), (8, 1)])
def test_task_2_primitive_rejects_invalid_head_mapping(heads, kv_heads):
    q = mx.zeros((4, 1, 128))
    k = mx.zeros((1, 1, 128))
    with pytest.raises(RuntimeError, match="incompatible attention shapes"):
        tiny_llm_ext.decode_attention(
            q, k, k, mx.array(0.0), 1.0, False, False, heads, kv_heads
        )


def test_task_2_primitive_rejects_invalid_mask_shape():
    q = mx.zeros((4, 1, 128))
    k = mx.zeros((1, 5, 128))
    with pytest.raises(RuntimeError, match="mask must have shape"):
        tiny_llm_ext.decode_attention(
            q, k, k, mx.zeros((1, 1, 5)), 1.0, False, True, 4, 1
        )


def test_task_2_primitive_rejects_mixed_dtypes():
    q = mx.zeros((4, 1, 128), dtype=mx.bfloat16)
    k = mx.zeros((1, 5, 128))
    with pytest.raises(RuntimeError, match="dtypes must match"):
        tiny_llm_ext.decode_attention(q, k, k, mx.array(0.0), 1.0, False, False, 4, 1)


def test_task_2_cpu_reports_gpu_only():
    with mx.stream(mx.cpu):
        q = mx.zeros((4, 1, 128))
        k = mx.zeros((1, 5, 128))
        with pytest.raises(RuntimeError, match="GPU-only"):
            mx.eval(
                tiny_llm_ext.decode_attention(
                    q, k, k, mx.array(0.0), 1.0, False, False, 4, 1
                )
            )
