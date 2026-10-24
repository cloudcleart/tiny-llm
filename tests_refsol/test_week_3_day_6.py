"""Optional Week 3 Day 6 MoE tests."""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3_moe import (
    Model as MlxQwen3MoeModel,
    ModelArgs as MlxQwen3MoeArgs,
    Qwen3MoeSparseMoeBlock as MlxQwen3MoeSparseMoeBlock,
)
from mlx_lm.models.switch_layers import SwitchLinear

from .tiny_llm_base import (
    Moe,
    QuantizedWeights,
    grouped_expert_linear,
    route_topk,
    dispatch_model,
)
from .utils import assert_allclose


def test_task_1_grouped_expert_linear():
    mx.random.seed(1)
    scale = 0.25
    x = mx.random.normal(shape=(2, 3, 128), dtype=mx.bfloat16) * scale
    w_experts = mx.random.normal(shape=(3, 64, 128), dtype=mx.bfloat16) * scale
    expert_ids = mx.array(
        [
            [2, 0, 1],
            [1, 2, 0],
        ],
        dtype=mx.uint32,
    )

    ref = SwitchLinear(
        input_dims=w_experts.shape[-1],
        output_dims=w_experts.shape[-2],
        num_experts=w_experts.shape[0],
        bias=False,
    )
    ref.weight = w_experts
    ref = ref.to_quantized(group_size=128, bits=4)

    out = grouped_expert_linear(
        x,
        QuantizedWeights.from_mlx_layer(ref),
        expert_ids,
    )
    expected = ref(mx.expand_dims(x, -2), expert_ids).squeeze(-2)

    assert out.shape == (2, 3, 64)
    assert_allclose(out, expected, precision=mx.bfloat16, atol=2e-2)


def test_task_2_router_topk():
    mx.random.seed(2)
    scale = 0.25
    x = mx.random.normal(shape=(2, 2, 128), dtype=mx.bfloat16) * scale
    ref = nn.Linear(128, 4, bias=False)
    ref.weight = mx.random.normal(shape=(4, 128), dtype=mx.bfloat16) * scale
    ref = ref.to_quantized(group_size=128, bits=4)

    router_probs, expert_ids, expert_scores = route_topk(
        x,
        QuantizedWeights.from_mlx_layer(ref),
        top_k=2,
    )
    _, _, normalized_scores = route_topk(
        x,
        QuantizedWeights.from_mlx_layer(ref),
        top_k=2,
        norm_topk_prob=True,
    )

    expected_probs = mx.softmax(ref(x), axis=-1, precise=True)
    expected_ids = mx.argpartition(-expected_probs, kth=1, axis=-1)[..., :2]
    expected_scores = mx.take_along_axis(expected_probs, expected_ids, axis=-1)
    expected_normalized_scores = expected_scores / expected_scores.sum(
        axis=-1,
        keepdims=True,
    )

    assert router_probs.shape == (2, 2, 4)
    assert expert_ids.shape == (2, 2, 2)
    assert expert_scores.shape == (2, 2, 2)
    assert expert_ids.tolist() == expected_ids.tolist()
    assert_allclose(router_probs, expected_probs, precision=mx.bfloat16)
    assert_allclose(expert_scores, expected_scores, precision=mx.bfloat16)
    assert_allclose(
        normalized_scores,
        expected_normalized_scores,
        precision=mx.bfloat16,
    )


def test_task_3_moe():
    mx.random.seed(3)
    scale = 0.25
    x = mx.random.normal(shape=(2, 3, 128), dtype=mx.bfloat16) * scale
    ref = MlxQwen3MoeSparseMoeBlock(
        SimpleNamespace(
            hidden_size=128,
            moe_intermediate_size=128,
            num_experts=3,
            num_experts_per_tok=2,
            norm_topk_prob=True,
        )
    )
    ref.gate.weight = mx.random.normal(shape=(3, 128), dtype=mx.bfloat16) * scale
    ref.switch_mlp.gate_proj.weight = (
        mx.random.normal(shape=(3, 128, 128), dtype=mx.bfloat16) * scale
    )
    ref.switch_mlp.up_proj.weight = (
        mx.random.normal(shape=(3, 128, 128), dtype=mx.bfloat16) * scale
    )
    ref.switch_mlp.down_proj.weight = (
        mx.random.normal(shape=(3, 128, 128), dtype=mx.bfloat16) * scale
    )
    nn.quantize(ref, group_size=128, bits=4)

    moe = Moe(
        w_router=QuantizedWeights.from_mlx_layer(ref.gate),
        w_gate=QuantizedWeights.from_mlx_layer(ref.switch_mlp.gate_proj),
        w_up=QuantizedWeights.from_mlx_layer(ref.switch_mlp.up_proj),
        w_down=QuantizedWeights.from_mlx_layer(ref.switch_mlp.down_proj),
        num_experts_per_tok=2,
        norm_topk_prob=True,
    )

    out = moe(x)
    expected = ref(x)

    assert out.shape == x.shape
    assert_allclose(out, expected, precision=mx.bfloat16, atol=2e-2)


@pytest.mark.parametrize("sparse_step,mlp_only_layers", [(1, [0]), (2, [])])
@pytest.mark.parametrize("tie_word_embeddings", [False, True])
def test_task_4_mixed_moe_model(sparse_step, mlp_only_layers, tie_word_embeddings):
    """Exercise public dispatch and paged prefill/decode against MLX."""
    mx.random.seed(4)
    args = MlxQwen3MoeArgs(
        model_type="qwen3_moe",
        hidden_size=128,
        num_hidden_layers=2,
        intermediate_size=256,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=sparse_step,
        mlp_only_layers=mlp_only_layers,
        moe_intermediate_size=128,
        rms_norm_eps=1e-6,
        vocab_size=128,
        rope_theta=10000.0,
        tie_word_embeddings=tie_word_embeddings,
        max_position_embeddings=256,
        norm_topk_prob=True,
    )
    ref = MlxQwen3MoeModel(args)
    ref.set_dtype(mx.bfloat16)
    nn.quantize(ref, group_size=128, bits=4)
    model = dispatch_model("qwen3-30b-a3b", ref, week=3, page_size=4)
    assert not isinstance(model.layers_inner[0].mlp, Moe)
    assert isinstance(model.layers_inner[1].mlp, Moe)

    cache = model.create_kv_cache()
    ref_cache = [KVCache() for _ in range(args.num_hidden_layers)]
    offset = 0
    # Cross a page boundary during decode, retaining both layers' KV state.
    for chunk in ([[1, 2, 3]], [[4]], [[5]], [[6]]):
        tokens = mx.array(chunk, dtype=mx.int32)
        actual = model(tokens, offset, cache).astype(mx.float32)
        expected = ref(tokens, cache=ref_cache).astype(mx.float32)
        assert actual.shape == (1, len(chunk[0]), args.vocab_size)
        actual = actual - mx.logsumexp(actual, axis=-1, keepdims=True)
        expected = expected - mx.logsumexp(expected, axis=-1, keepdims=True)
        assert_allclose(actual, expected, precision=mx.bfloat16, rtol=1e-2, atol=2e-2)
        offset += len(chunk[0])
