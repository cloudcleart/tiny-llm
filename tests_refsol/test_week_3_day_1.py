"""Week 3 Day 1 continuous-batching tests."""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm import load

if Path(__file__).parent.name == "tests_refsol":
    import tiny_llm.batch as batch_runtime
    import tiny_llm.models as model_dispatch
else:
    import tiny_llm.batch as batch_runtime
    import tiny_llm.models as model_dispatch

from .tiny_llm_base import *
from .utils import *


def rope_helper(stream: mx.Stream, traditional: bool, precision: mx.Dtype):
    BATCH_SIZE = 16
    NUM_HEADS = 8
    HEAD_DIM = 4
    MAX_SEQ_LEN = 14
    SEQ_LEN = 9
    BASE = 10000
    with mx.stream(stream):
        for _ in range(100):
            user_layer = FastRoPE(HEAD_DIM, MAX_SEQ_LEN, BASE, traditional=traditional)
            x = mx.random.uniform(
                shape=(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM), dtype=precision
            )

            input_pos = np.random.randint(0, MAX_SEQ_LEN - SEQ_LEN, size=BATCH_SIZE)
            input_pos_mx = mx.array(input_pos, dtype=mx.int32)
            input_pos_user = input_pos.tolist()

            reference_output = mx.fast.rope(
                x.transpose(0, 2, 1, 3),
                dims=HEAD_DIM,
                traditional=traditional,
                base=BASE,
                scale=1.0,
                offset=input_pos_mx,
            ).transpose(0, 2, 1, 3)
            user_output = user_layer(x, input_pos_user)
            assert_allclose(
                user_output,
                reference_output,
                precision,
                atol=5e-6 if precision == mx.float32 else 1e-3,
            )


@pytest.mark.parametrize("traditional", [False, True], ids=["default", "traditional"])
def test_task_1_rope_multiple_offsets(traditional: bool):
    rope_helper(mx.gpu, traditional, mx.bfloat16)


@pytest.mark.parametrize("precision", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    "query_length,source_length,allowed",
    [
        (3, 3, [[True, False, False], [True, True, False], [True, True, True]]),
        (2, 4, [[True, True, True, False], [True, True, True, True]]),
        (1, 4, [[True, True, True, True]]),
    ],
    ids=["prefill", "cached-prefill", "decode"],
)
def test_task_1_rectangular_causal_mask(
    precision, query_length, source_length, allowed
):
    mask = causal_mask(query_length, source_length, precision)

    assert mask.shape == (query_length, source_length)
    assert mask.dtype == precision
    expected = np.where(np.array(allowed), 0.0, -np.inf)
    np.testing.assert_array_equal(np.array(mask.astype(mx.float32)), expected)


def test_task_2_batching_kv_cache():
    cache = BatchingKvCache(max_active_requests=3)
    assert cache.max_seq_len is None

    slot0 = TinyKvFullCache()
    slot0.update_and_fetch(
        mx.array([[[[10.0]]]], dtype=mx.float32),
        mx.array([[[[110.0]]]], dtype=mx.float32),
    )

    slot2 = TinyKvFullCache()
    slot2.update_and_fetch(
        mx.array([[[[20.0], [21.0]]]], dtype=mx.float32),
        mx.array([[[[120.0], [121.0]]]], dtype=mx.float32),
    )

    cache.add_request(slot0, 0)
    cache.add_request(slot2, 2)

    keys = mx.array(
        [
            [[[12.0], [13.0]]],
            [[[0.0], [0.0]]],
            [[[22.0], [23.0]]],
        ],
        dtype=mx.float32,
    )
    values = mx.array(
        [
            [[[112.0], [113.0]]],
            [[[0.0], [0.0]]],
            [[[122.0], [123.0]]],
        ],
        dtype=mx.float32,
    )

    batched_keys, batched_values, seq_len, mask = cache.update_and_fetch(
        keys, values, mask_length=2
    )

    expected_keys = mx.array(
        [
            [[[0.0], [10.0], [12.0], [13.0]]],
            [[[0.0], [0.0], [0.0], [0.0]]],
            [[[20.0], [21.0], [22.0], [23.0]]],
        ],
        dtype=mx.float32,
    )
    expected_values = mx.array(
        [
            [[[0.0], [110.0], [112.0], [113.0]]],
            [[[0.0], [0.0], [0.0], [0.0]]],
            [[[120.0], [121.0], [122.0], [123.0]]],
        ],
        dtype=mx.float32,
    )
    expected_mask = mx.array(
        [
            [[[-mx.inf, 0.0, 0.0, -mx.inf], [-mx.inf, 0.0, 0.0, 0.0]]],
            [
                [
                    [-mx.inf, -mx.inf, -mx.inf, -mx.inf],
                    [-mx.inf, -mx.inf, -mx.inf, -mx.inf],
                ]
            ],
            [[[0.0, 0.0, 0.0, -mx.inf], [0.0, 0.0, 0.0, 0.0]]],
        ],
        dtype=mx.float32,
    ).reshape(3, 1, 2, 4)

    assert seq_len is None
    assert_allclose(batched_keys, expected_keys, mx.float32)
    assert_allclose(batched_values, expected_values, mx.float32)
    assert_allclose(mask, expected_mask, mx.float32)
    assert cache.last_batch_bytes == 96
    assert cache.staging_copy_bytes == 56


def test_task_2_batching_kv_cache_reuses_slots_without_stale_history():
    cache = BatchingKvCache(max_active_requests=2)
    old_request = TinyKvFullCache()
    history = mx.array([10.0, 11.0]).reshape(1, 1, 2, 1)
    old_request.update_and_fetch(history, history + 100)
    survivor = TinyKvFullCache()
    cache.add_request(old_request, 0)
    cache.add_request(survivor, 1)

    tokens = mx.array([12.0, 20.0]).reshape(2, 1, 1, 1)
    cache.update_and_fetch(tokens, tokens + 100)
    cache.remove_request(0)

    # Removing the longest request shrinks the batch; its inactive row stays masked.
    tokens = mx.array([999.0, 21.0]).reshape(2, 1, 1, 1)
    keys, values, _, mask = cache.update_and_fetch(tokens, tokens + 100)
    assert keys.reshape(2, 2).tolist() == [[0.0, 0.0], [20.0, 21.0]]
    assert values.reshape(2, 2).tolist() == [[0.0, 0.0], [120.0, 121.0]]
    assert mask.reshape(2, 2).tolist() == [[-float("inf"), -float("inf")], [0, 0]]
    assert old_request.offset == 3

    replacement = TinyKvFullCache()
    cache.add_request(replacement, 0)
    tokens = mx.array([30.0, 22.0]).reshape(2, 1, 1, 1)
    keys, values, _, mask = cache.update_and_fetch(tokens, tokens + 100)
    assert keys.reshape(2, 3).tolist() == [[0.0, 0.0, 30.0], [20.0, 21.0, 22.0]]
    assert values.reshape(2, 3).tolist() == [[0.0, 0.0, 130.0], [120.0, 121.0, 122.0]]
    assert mask.reshape(2, 3).tolist() == [
        [-float("inf"), -float("inf"), 0],
        [0, 0, 0],
    ]
    assert replacement.offset == 1
    assert survivor.offset == 3
    assert cache.last_batch_bytes == 48
    assert cache.staging_copy_bytes == 80


def helper_test_task_3(
    model_name: str,
    seq_len: int,
    iters: int = 1,
):
    """Tests for continuous batching of decode requests."""
    requests = 4
    max_seq_len = seq_len

    mlx_model, tokenizer = load(model_name)
    model = model_dispatch.dispatch_week3_batch_model(model_name, mlx_model)
    for _ in range(iters):
        cache = [
            BatchingKvCache(requests, max_seq_len)
            for _ in range(model.num_hidden_layers)
        ]
        # Start each request at a staggered token index.
        staggered_start = [seq_len * i // requests for i in range(requests)]
        inputs = (
            mx.arange(requests * seq_len, dtype=mx.int32).reshape(requests, seq_len)
            % tokenizer.vocab_size
        )
        ref_outputs = mlx_model(inputs)
        for offset in range(seq_len + staggered_start[-1]):
            seq_idx = [offset - start for start in staggered_start]

            # Requests join at the staggered start, and leave when they reach seq_len.
            for request_id, sidx in enumerate(seq_idx):
                if sidx == 0:
                    for c in cache:
                        c.add_request(TinyKvFullCache(), request_id)
                elif sidx == seq_len:
                    for c in cache:
                        c.remove_request(request_id)

            next_tokens = []
            next_offsets = []
            for request_id, sidx in enumerate(seq_idx):
                if 0 <= sidx < seq_len:
                    next_tokens.append(inputs[request_id, sidx].item())
                    next_offsets.append(sidx)
                else:
                    next_tokens.append(0)
                    next_offsets.append(0)

            user_out = model(
                inputs=mx.array(next_tokens, dtype=mx.int32).reshape(-1, 1),
                offset=mx.array(next_offsets, dtype=mx.int32),
                cache=cache,
            )

            for request_id, sidx in enumerate(seq_idx):
                if 0 <= sidx < seq_len:
                    user_out_r = user_out[request_id, 0, :]
                    ref_out_r = ref_outputs[request_id, sidx, :]
                    user_out_r = user_out_r - mx.logsumexp(user_out_r, keepdims=True)
                    ref_out_r = ref_out_r - mx.logsumexp(ref_out_r, keepdims=True)
                    assert_allclose(
                        user_out_r,
                        ref_out_r,
                        precision=mx.bfloat16,
                        rtol=0.1,
                        atol=2.0,
                    )


@pytest.mark.skipif(
    not qwen3_0_6b_model_exists(), reason="Qwen3-0.6B-4bit model not found"
)
def test_task_3_qwen3_0_6b():
    helper_test_task_3("Qwen/Qwen3-0.6B-MLX-4bit", seq_len=3)


@pytest.mark.skipif(not qwen3_4b_model_exists(), reason="Qwen3-4B-4bit model not found")
def test_task_3_qwen3_4b():
    helper_test_task_3(
        "Qwen/Qwen3-4B-MLX-4bit",
        seq_len=3,
    )


def test_task_3_week3_batch_factory_selects_mlx_projections(monkeypatch):
    mlx_model = object()
    captured = {}

    def fake_dispatch(model_name, model, week, **kwargs):
        captured.update(
            model_name=model_name,
            model=model,
            week=week,
            kwargs=kwargs,
        )
        return "week3-batch-model"

    monkeypatch.setattr(model_dispatch, "dispatch_model", fake_dispatch)

    result = model_dispatch.dispatch_week3_batch_model("qwen3-0.6b", mlx_model)

    assert result == "week3-batch-model"
    assert captured == {
        "model_name": "qwen3-0.6b",
        "model": mlx_model,
        "week": 2,
        "kwargs": {"use_mlx_quantized_linear": True},
    }


def test_task_3_week2_model_keeps_course_projections_by_default():
    mlx_model = tiny_qwen3_mlx_model()
    course_model = Qwen3ModelWeek2(mlx_model)
    week3_batch_model = Qwen3ModelWeek2(
        mlx_model,
        use_mlx_quantized_linear=True,
    )

    assert not course_model.layers_inner[0].self_attn.wq.use_mlx_quantized_linear
    assert week3_batch_model.layers_inner[0].self_attn.wq.use_mlx_quantized_linear

    for model, enabled in [(course_model, False), (week3_batch_model, True)]:
        weights = [model.embedding.weight]
        if model.w_lm_head is not None:
            weights.append(model.w_lm_head)
        for layer in model.layers_inner:
            weights.extend(
                [
                    layer.self_attn.wq,
                    layer.self_attn.wk,
                    layer.self_attn.wv,
                    layer.self_attn.wo,
                    layer.mlp.w_gate,
                    layer.mlp.w_up,
                    layer.mlp.w_down,
                ]
            )
        assert all(w.use_mlx_quantized_linear == enabled for w in weights)


@pytest.mark.parametrize("with_bias", [False, True])
def test_task_3_mlx_projection_matches_dequantized_batch(with_bias):
    dense = mx.sin(mx.arange(8 * 64, dtype=mx.float32)).reshape(8, 64)
    packed, scales, biases = mx.quantize(dense, group_size=32, bits=4)
    weights = QuantizedWeights(
        scales=scales,
        biases=biases,
        group_size=32,
        bits=4,
        weight=packed,
        use_mlx_quantized_linear=True,
    )
    x = mx.cos(mx.arange(2 * 3 * 64, dtype=mx.float32)).reshape(2, 3, 64)
    bias = mx.arange(8, dtype=mx.float32) if with_bias else None
    expected = x @ mx.dequantize(packed, scales, biases, group_size=32, bits=4).T
    if bias is not None:
        expected = expected + bias

    actual = quantized_linear(x, weights, bias)
    assert actual.shape == (2, 3, 8)
    np.testing.assert_allclose(
        np.array(actual), np.array(expected), rtol=1e-5, atol=1e-5
    )


def test_task_3_mlx_projection_selector_is_causal(monkeypatch):
    calls = []

    def fake_quantized_matmul(x, weight, **kwargs):
        calls.append((x, weight, kwargs))
        return mx.full((*x.shape[:-1], weight.shape[0]), 7, dtype=mx.float32)

    monkeypatch.setattr(mx, "quantized_matmul", fake_quantized_matmul)
    weights = QuantizedWeights(
        scales=mx.ones((3, 1)),
        biases=mx.zeros((3, 1)),
        group_size=4,
        bits=4,
        weight=mx.zeros((3, 1), dtype=mx.uint32),
        use_mlx_quantized_linear=True,
    )
    x = mx.ones((2, 4))

    result = quantized_linear(x, weights)

    assert result.tolist() == [[7.0, 7.0, 7.0], [7.0, 7.0, 7.0]]
    assert len(calls) == 1
    assert calls[0][2]["scales"] is weights.scales
    assert calls[0][2]["biases"] is weights.biases
    assert calls[0][2]["transpose"] is True
    assert calls[0][2]["group_size"] == 4
    assert calls[0][2]["bits"] == 4


class SchedulerFakeDetokenizer:
    def __init__(self, _):
        self.text = ""

    def add_token(self, token):
        self.text += str(token)


class SchedulerFakeTokenizer:
    eos_token_id = 99
    _tokenizer = object()
    detokenizer = SchedulerFakeDetokenizer(_tokenizer)

    def encode(self, prompt, add_special_tokens=False):
        assert not add_special_tokens
        return {"a": [10], "bb": [20, 21], "c": [30]}[prompt]


class SchedulerFakeModel:
    num_hidden_layers = 1

    def __init__(self):
        self.cache_request_ids = {}
        self.prefill_records = []
        self.active_slots = []
        self.next_request_id = 0

    def create_kv_cache(self):
        cache = TinyKvFullCache()
        self.cache_request_ids[id(cache)] = self.next_request_id
        self.next_request_id += 1
        return [cache]

    def __call__(self, inputs, offsets, cache, logits_to_keep=1):
        assert logits_to_keep == 1
        batch_cache = isinstance(cache[0], BatchingKvCache)
        if batch_cache:
            self.active_slots.append(
                tuple(
                    None
                    if request_cache is None
                    else self.cache_request_ids[id(request_cache)]
                    for request_cache in cache[0].kv_caches
                )
            )
        else:
            request_id = self.cache_request_ids[id(cache[0])]
            self.prefill_records.append((request_id, list(offsets), inputs.tolist()))

        sequence_length = 1 if batch_cache else inputs.shape[1]
        keys = mx.zeros((inputs.shape[0], 1, sequence_length, 1), dtype=mx.float32)
        if batch_cache:
            cache[0].update_and_fetch(keys, keys, mask_length=sequence_length)
        else:
            cache[0].update_and_fetch(keys, keys)

        # The first chunk of "bb" must not emit a completion, even if its
        # prediction is EOS. Only the final prompt chunk predicts token 2.
        next_by_token = {0: 99, 1: 99, 2: 3, 3: 99, 4: 99, 10: 1, 20: 99, 21: 2, 30: 4}
        logits = mx.zeros((inputs.shape[0], 1, 100), dtype=mx.float32)
        for row, token in enumerate(inputs[:, -1].tolist()):
            logits[row, 0, next_by_token[token]] = 1
        return logits


def test_task_4_batch_generate_prefills_reuses_slots_and_orders_results(monkeypatch):
    monkeypatch.setattr(batch_runtime, "_print_progress", lambda *args: None)

    probe_model = SchedulerFakeModel()
    probe = batch_runtime.Request(
        probe_model,
        SchedulerFakeTokenizer(),
        "a",
        prompt_idx=0,
    )
    probe.try_prefill()
    assert probe.is_prefill_done
    assert probe.offset == 1
    assert probe.next_token == 1

    model = SchedulerFakeModel()
    result = batch_runtime.batch_generate(
        model,
        SchedulerFakeTokenizer(),
        ["a", "bb", "c"],
        batch_size=2,
    )

    assert result == [(0, "1"), (1, "23"), (2, "4")]
    assert model.prefill_records == [
        (0, [0], [[10]]),
        (1, [0], [[20, 21]]),
        (2, [0], [[30]]),
    ]
    assert model.active_slots == [(0, None), (1, None), (1, 2)]


@pytest.mark.skipif(
    not qwen3_1_7b_model_exists(), reason="Qwen3-1.7B-4bit model not found"
)
def test_task_3_qwen3_1_7b():
    helper_test_task_3(
        "Qwen/Qwen3-1.7B-MLX-4bit",
        seq_len=3,
    )


@pytest.mark.parametrize("limit,expected", [(2, ""), (3, "2"), (4, "23")])
@pytest.mark.parametrize("prefill_step", [1, 2], ids=["chunked", "whole-prompt"])
def test_task_4_length_limit_includes_prompt(
    monkeypatch, limit, expected, prefill_step
):
    monkeypatch.setattr(batch_runtime, "_print_progress", lambda *args: None)
    model = SchedulerFakeModel()
    prompts = ["bb"]
    result = batch_runtime.batch_generate(
        model,
        SchedulerFakeTokenizer(),
        prompts,
        max_seq_len=limit,
        prefill_step=prefill_step,
    )
    assert result == [(0, expected)]
    assert prompts == ["bb"]
    expected_prefill = (
        [(0, [0], [[20]]), (0, [1], [[21]])]
        if prefill_step == 1
        else [(0, [0], [[20, 21]])]
    )
    assert model.prefill_records == expected_prefill


def test_task_4_rejects_long_prompt_before_cache_creation(monkeypatch):
    monkeypatch.setattr(batch_runtime, "_print_progress", lambda *args: None)
    model = SchedulerFakeModel()
    with pytest.raises(ValueError, match="max_seq_len"):
        batch_runtime.batch_generate(
            model, SchedulerFakeTokenizer(), ["bb"], max_seq_len=1
        )
    assert model.next_request_id == 0
    assert model.prefill_records == []


def test_task_4_immediate_eos_returns_completion_order(monkeypatch):
    monkeypatch.setattr(batch_runtime, "_print_progress", lambda *args: None)

    class Tokenizer(SchedulerFakeTokenizer):
        def encode(self, prompt, add_special_tokens=False):
            return (
                [0] if prompt == "eos" else super().encode(prompt, add_special_tokens)
            )

    model = SchedulerFakeModel()
    result = batch_runtime.batch_generate(
        model, Tokenizer(), ["bb", "eos"], batch_size=1
    )
    assert result == [(1, ""), (0, "23")]
    assert model.active_slots == [(0,), (0,)]
