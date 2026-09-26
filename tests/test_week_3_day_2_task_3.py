"""Cache materialization boundaries, independent of the later page allocator."""

import mlx.core as mx
import pytest

from tiny_llm.batch import Request
from tiny_llm.kv_cache import TinyKvFullCache
from tiny_llm.paged_kv_cache import TinyKvPagedCache, TinyKvPagedPool


class Tokenizer:
    eos_token_id = 99
    _tokenizer = None

    class Detokenizer:
        def __init__(self, _):
            self.text = ""

        def add_token(self, token):
            self.text += str(token)

    detokenizer = Detokenizer(None)

    def encode(self, prompt, add_special_tokens=False):
        return [1] * len(prompt)


@pytest.mark.parametrize("length", [2, 3, 4, 7])
def test_task_3_materializes_every_layer_after_every_chunk(length):
    events = []

    class Cache:
        # Deliberately exposes no storage: Request must use the lifecycle hook.
        def __init__(self, layer):
            self.layer = layer

        def materialize(self):
            events.append(("materialize", self.layer))

    class Model:
        def create_kv_cache(self):
            return [Cache(0), Cache(1)]

        def __call__(self, inputs, offsets, cache, logits_to_keep=1):
            events.append(("model", offsets[0], inputs.shape[1]))
            return mx.array([[[0.0, 1.0]]])

    request = Request(Model(), Tokenizer(), "x" * length, prefill_max_step=3)
    for start in range(0, length, 3):
        events.clear()
        request.try_prefill()
        assert events == [
            ("model", start, min(3, length - start)),
            ("materialize", 0),
            ("materialize", 1),
        ]
    assert request.is_prefill_done
    assert request.text() == "1"


def test_task_3_dense_cache_evaluates_owned_tensors(monkeypatch):
    cache = TinyKvFullCache()
    calls = []
    monkeypatch.setattr(mx, "eval", lambda *arrays: calls.append(arrays))
    cache.materialize()
    assert calls == []
    cache.update_and_fetch(mx.ones((1, 1, 3, 4)), mx.zeros((1, 1, 3, 4)))
    storage = cache.key_values
    cache.materialize()
    assert len(calls) == 1
    assert calls[0][0] is storage[0]
    assert calls[0][1] is storage[1]
    assert cache.key_values is storage
    assert cache.offset == 3


def test_task_3_paged_cache_evaluates_pool_without_dense_gather(monkeypatch):
    pool = TinyKvPagedPool(page_size=4)
    cache = TinyKvPagedCache(pool)
    calls = []
    monkeypatch.setattr(mx, "eval", lambda *arrays: calls.append(arrays))
    cache.materialize()
    assert calls == []
    # Supply page storage directly; allocation belongs to Week 3 Day 3.
    pool.key_pages = mx.ones((2, 1, 4, 8))
    pool.value_pages = mx.zeros((2, 1, 4, 8))

    def reject_gather():
        pytest.fail("materialize must not gather dense K/V")

    monkeypatch.setattr(cache, "gather_dense", reject_gather)
    cache.materialize()
    assert len(calls) == 1
    assert calls[0][0] is pool.key_pages
    assert calls[0][1] is pool.value_pages
