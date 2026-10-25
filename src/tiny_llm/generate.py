import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .qwen3_week1 import Qwen3ModelWeek1
from .qwen3_week2 import Qwen3ModelWeek2
from .qwen3_week3 import Qwen3ModelWeek3
from typing import Callable, NamedTuple


def _release_kv_cache(kv_cache):
    """Release every layer's request cache, including any owned physical pages."""
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


def _greedy_step(model, tokens, offset, kv_cache):
    """Cache all input positions and return one uncached greedy prediction.

    tokens is a one-dimensional sequence; tokens[None] adds the batch axis.
    The returned offset counts consumed inputs, not the predicted token.
    """
    logits = model(tokens[None], offset, kv_cache, logits_to_keep=1)
    # Softmax is unnecessary: it preserves the ordering of the logits.
    token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
    # Materialize MLX's lazy result before Python uses it for control flow.
    mx.eval(token)
    return token, offset + tokens.size


def _bounded_draft(model, token, offset, kv_cache, proposal_length, eos_ids):
    """Propose at most proposal_length tokens after an accepted token.

    The returned offset counts cached inputs; the final predicted token has
    not yet entered the cache. EOS is included so the target can verify it.
    """
    proposal = []
    # For x -> a -> b -> c, the proposal is [a, b, c], while the newly
    # cached inputs are [x, a, b]. The caller must account for this one-token gap.
    for _ in range(proposal_length):
        token, offset = _greedy_step(model, token, offset, kv_cache)
        token_id = int(token.item())
        proposal.append(token_id)
        # A draft EOS only ends the proposal; the target can still reject it.
        if token_id in eos_ids:
            break
    return mx.array(proposal, dtype=mx.int32), offset


class _DraftVerification(NamedTuple):
    """Verification results before cache reconciliation or output truncation."""

    # Number of consecutive matches, potentially including a matched EOS.
    matched_count: int
    # Shape [1]: correction on rejection, or bonus token on full acceptance.
    next_token: mx.array
    # Target cache length after consuming all inputs, even rejected positions.
    offset: int


def _verify_draft(model, token, proposal, offset, kv_cache):
    """Verify a proposal in one target call, without committing either cache.

    Logits at input position i predict proposal[i]; the final position predicts
    the bonus token. Stop at the first mismatch, even if later positions match:
    those predictions were conditioned on a rejected prefix.

    next_token is the target correction on rejection, or the bonus token on
    full acceptance. offset includes *all* verification inputs, including any
    rejected suffix. The caller must reconcile caches before the next iteration
    and stop emission at EOS or the token budget.
    """
    # Cast before concatenation: mixing uint32 and int32 can produce int64,
    # which quantized embedding lookups do not support.
    inputs = mx.concatenate(
        [token.reshape(-1).astype(mx.int32), proposal.reshape(-1).astype(mx.int32)]
    )
    # Input [x, a, b] yields predictions [a', b', bonus]. Request every
    # position, rather than only the final logits used in ordinary decoding.
    logits = model(inputs[None], offset, kv_cache, logits_to_keep=inputs.size)
    predictions = mx.argmax(logits[0], axis=-1).astype(mx.int32)
    mx.eval(predictions)
    matched_count = 0
    for proposed, predicted in zip(proposal.tolist(), predictions.tolist()):
        if proposed != predicted:
            break
        matched_count += 1
    return _DraftVerification(
        matched_count,
        # Keeping a one-element array lets the next step reuse this as input.
        predictions[matched_count : matched_count + 1],
        offset + inputs.size,
    )


def speculative_generate(
    draft_model: Qwen3ModelWeek2 | Qwen3ModelWeek3,
    model: Qwen3ModelWeek2 | Qwen3ModelWeek3,
    draft_tokenizer: TokenizerWrapper,
    tokenizer: TokenizerWrapper,
    prompt: str,
    proposal_length: int = 4,
    max_tokens: int = 256,
) -> str:
    """Bounded proposals, single-call verification and target-only generation.

    Both caches retain the same confirmed prefix before each proposal.
    No unverified draft token is emitted.

    proposal_length=0 selects ordinary target-only greedy decoding.
    max_tokens limits newly emitted non-EOS tokens, excluding the prompt.
    Returns the generated text without the prompt or EOS token.
    """
    # Validate before creating caches or running either model. Explicitly
    # reject bool because Python otherwise treats it as an integer.
    for name, value in (
        ("proposal_length", proposal_length),
        ("max_tokens", max_tokens),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if max_tokens == 0:
        # A zero budget needs neither tokenization nor model allocation.
        return ""

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("prompt must encode to at least one token")

    def eos_ids(tok):
        """Support tokenizers with either one EOS ID or a set of EOS IDs."""
        ids = getattr(tok, "eos_token_ids", None)
        return set(ids) if ids is not None else {tok.eos_token_id}

    target_eos = eos_ids(tokenizer)
    # Equal token IDs must mean the same thing in both models. Matching only
    # this prompt is insufficient, so compare EOS IDs and the full vocabulary.
    if draft_tokenizer.encode(prompt, add_special_tokens=False) != prompt_ids:
        raise ValueError("tokenizers encode the prompt differently")
    if eos_ids(draft_tokenizer) != target_eos:
        raise ValueError("tokenizers have different EOS token ids")
    target_vocab = getattr(tokenizer, "get_vocab", None)
    draft_vocab = getattr(draft_tokenizer, "get_vocab", None)
    if not callable(target_vocab) or not callable(draft_vocab):
        raise ValueError("tokenizers must expose comparable vocabularies")
    if target_vocab() != draft_vocab():
        raise ValueError("tokenizers have different token ids")

    # Use the public detokenizer and reset request-local output state.
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    cache = draft_cache = None
    prompt_tokens = mx.array(prompt_ids, dtype=mx.int32)
    try:
        cache = model.create_kv_cache()
        # Target prefill establishes the first accepted token. Delay draft
        # allocation so an immediate target EOS incurs no draft-model work.
        token, offset = _greedy_step(model, prompt_tokens, 0, cache)
        emitted = 0
        draft_offset = 0

        def emit(token_id):
            """Stream one confirmed non-EOS token and charge it to the budget."""
            nonlocal emitted
            detokenizer.add_token(token_id)
            print(detokenizer.last_segment, end="", flush=True)
            emitted += 1

        def check_offsets():
            """Check the common-prefix invariant across both models and layers."""
            assert offset == draft_offset
            assert all(layer.offset == offset for layer in cache)
            assert all(layer.offset == draft_offset for layer in draft_cache)

        def rewind_to(caches, current, retained):
            """Trim each layer to a logical length using its cache-specific rewind."""
            assert retained <= current
            if current > retained:
                for layer in caches:
                    layer.rewind(current - retained)
            return retained

        while int(token.item()) not in target_eos and emitted < max_tokens:
            # The pending token is accepted but has not entered either cache.
            emit(int(token.item()))
            if emitted == max_tokens:
                break
            if proposal_length and draft_cache is None:
                draft_cache = draft_model.create_kv_cache()
                draft_first, draft_offset = _greedy_step(
                    draft_model, prompt_tokens, 0, draft_cache
                )
                if int(draft_first.item()) in target_eos:
                    # An unhelpful draft must not terminate target generation.
                    # Disable speculation and let the target finish on its own.
                    proposal_length = 0
            if not proposal_length:
                token, offset = _greedy_step(model, token, offset, cache)
                continue

            check_offsets()
            # Save the common prefix length before speculative inputs extend it.
            base_offset = offset
            proposal, draft_offset = _bounded_draft(
                draft_model,
                token,
                draft_offset,
                draft_cache,
                min(proposal_length, max_tokens - emitted),
                target_eos,
            )
            verification = _verify_draft(model, token, proposal, offset, cache)
            offset = verification.offset
            accepted = 0
            terminal = False
            # Only emit the verified prefix, stopping before EOS or after the
            # last budgeted token. A matched EOS must never reach detokenizer.
            for token_id in proposal[: verification.matched_count].tolist():
                if token_id in target_eos:
                    terminal = True
                    break
                emit(token_id)
                accepted += 1
                if emitted == max_tokens:
                    terminal = True
                    break
            token = verification.next_token
            terminal = terminal or int(token.item()) in target_eos
            # Retain x plus the accepted proposal prefix. The correction/bonus
            # stays pending for the next iteration rather than being cached now.
            retained = base_offset + 1 + accepted
            # With k proposed tokens and m accepted non-EOS tokens, target
            # normally rewinds k-m positions; draft rewinds k-1-m on rejection.
            if terminal:
                # No catch-up model call is needed on termination. Trim to the
                # common confirmed prefix, then release both caches in finally.
                retained = min(retained, draft_offset)
            elif draft_offset < retained:
                # Full acceptance: the draft has predicted, but not consumed,
                # its final proposal token. Cache it before using the bonus.
                _, draft_offset = _greedy_step(
                    draft_model, proposal[-1:], draft_offset, draft_cache
                )
            offset = rewind_to(cache, offset, retained)
            draft_offset = rewind_to(draft_cache, draft_offset, retained)
            check_offsets()
            if terminal:
                break
        finalize = getattr(detokenizer, "finalize", None)
        if callable(finalize):
            finalize()
        return detokenizer.text
    finally:
        # Also run on model/verification errors so paged requests do not leak pages.
        _release_kv_cache(draft_cache)
        _release_kv_cache(cache)
