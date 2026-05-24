import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .qwen3_week1 import Qwen3ModelWeek1
from .qwen3_week2 import Qwen3ModelWeek2
from .kv_cache import TinyKvFullCache
from typing import Callable


def simple_generate(
    model: Qwen3ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
) -> str:
    """Generate tokens from a prompt using the Week 1 model.

    Args:
        model: Qwen3 model that accepts token ids with shape ``1 x S`` and
            returns logits with shape ``1 x S x vocab_size``.
        tokenizer: Tokenizer used to encode the prompt and detokenize generated
            token ids for streaming output.
        prompt: User input text used as the initial context.
        sampler: Optional decoding function that maps log-probabilities with
            shape ``1 x vocab_size`` to the next token id. When omitted, greedy
            decoding is used.
    """

    def _step(model, y):
        """Run one generation step: current context tokens -> next token id."""
        # The Week 1 model has no batch generation, so add a batch dimension.
        logits = model(y[None])

        # Only the final sequence position predicts the next generated token.
        logits = logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        if sampler is None:
            return mx.argmax(logprobs, axis=-1)
        return sampler(logprobs)

    # Prefill: encode the whole prompt and use it as the initial context.
    tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()

    while True:
        # Decode without KV cache: feed the full context on every step.
        token = _step(model, tokens)
        mx.eval(token)
        tokens = mx.concat([tokens, token])
        if token.item() == tokenizer.eos_token_id:
            break
        detokenizer.add_token(token.item())
        print(detokenizer.last_segment, end="", flush=True)


def simple_generate_with_kv_cache(
    model: Qwen3ModelWeek2,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None = None,
) -> str:
    def _step(model, y, offset, kv_cache):
        """Run one cached generation step: new tokens + offset -> next token id."""
        logits = model(y[None], offset, kv_cache)
        logits = logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        if sampler is None:
            return mx.argmax(logprobs, axis=-1)
        return sampler(logprobs)

    tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
    kv_cache = [TinyKvFullCache() for _ in range(model.num_hidden_layers)]
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()

    # Prefill: process the whole prompt once and populate every layer's cache.
    offset = 0
    token = _step(model, tokens, offset, kv_cache)
    mx.eval(token)
    offset += tokens.size

    while True:
        if token.item() == tokenizer.eos_token_id:
            break
        detokenizer.add_token(token.item())
        print(detokenizer.last_segment, end="", flush=True)

        # Decode: only feed the most recently generated token.
        token = _step(model, token, offset, kv_cache)
        mx.eval(token)
        offset += 1


def speculative_generate(
    draft_model: Qwen3ModelWeek2,
    model: Qwen3ModelWeek2,
    draft_tokenizer: TokenizerWrapper,
    tokenizer: TokenizerWrapper,
    prompt: str,
) -> str:
    pass
