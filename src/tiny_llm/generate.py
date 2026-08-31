import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .qwen3_week1 import Qwen3ModelWeek1
from .qwen3_week2 import Qwen3ModelWeek2
from typing import Callable


def _release_kv_cache(kv_cache):
    if kv_cache is None:
        return
    for layer in kv_cache:
        layer.release()


def simple_generate(
    model: Qwen3ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
) -> None:
    def _step(model, y):
        pass


def simple_generate_with_kv_cache(
    model: Qwen3ModelWeek2,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None = None,
) -> str:
    """
    Generate text while reusing one request-scoped KV cache.

    Args:
        model: Week 2 Qwen3 model that accepts an offset and layer caches.
        tokenizer: Tokenizer and incremental detokenizer used for generation.
        prompt: Input text to prefill into the KV cache.
        sampler: Optional token sampler. None uses greedy decoding.

    Returns:
        The generated text, excluding the input prompt.
    """

    def _step(model, y, offset, kv_cache):
        logits = model(y[None], offset, kv_cache, logits_to_keep=1)
        logits = logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        token = mx.argmax(logprobs, axis=-1) if sampler is None else sampler(logprobs)
        token = token.reshape(-1)
        if token.size != 1:
            raise ValueError(
                "simple generation expects the sampler to return one token"
            )
        return token.astype(mx.int32), logprobs.squeeze(0)

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_tokens:
        raise ValueError("prompt must encode to at least one token")

    tokens = mx.array(prompt_tokens, dtype=mx.int32)
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    eos_token_ids = getattr(tokenizer, "eos_token_ids", None)
    if eos_token_ids is None:
        eos_token_ids = {tokenizer.eos_token_id}
    eos_token_ids = {int(token_id) for token_id in eos_token_ids}

    kv_cache = model.create_kv_cache()
    offset = 0
    try:
        while True:
            # The first call prefills the whole prompt; later calls contain one token.
            token, _ = _step(model, tokens, offset, kv_cache)
            mx.eval(token)
            token_id = int(token.item())

            if token_id in eos_token_ids:
                break

            detokenizer.add_token(token_id)
            print(detokenizer.last_segment, end="", flush=True)

            # The model cached every input token from this call.
            offset += tokens.size
            tokens = token

        finalize = getattr(detokenizer, "finalize", None)
        if callable(finalize):
            finalize()
        return detokenizer.text
    finally:
        _release_kv_cache(kv_cache)


def speculative_generate(
    draft_model: Qwen3ModelWeek2,
    model: Qwen3ModelWeek2,
    draft_tokenizer: TokenizerWrapper,
    tokenizer: TokenizerWrapper,
    prompt: str,
    proposal_length: int = 4,
) -> str:
    pass
