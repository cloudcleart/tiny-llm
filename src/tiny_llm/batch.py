import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from datetime import datetime

from .kv_cache import BatchingKvCache


def _step(model, y, offsets, kv_cache):
    logits = model(y, offsets, kv_cache, logits_to_keep=1)
    logits = logits[:, -1, :]
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    y = mx.argmax(logprobs, axis=-1)
    return y


class Request:
    def __init__(
        self,
        model: any,
        tokenizer: TokenizerWrapper,
        prompt: str,
        prefill_max_step: int = 128,
        prompt_idx: int = 0,
        max_seq_len: int = 512,
    ):
        self.prompt = prompt
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokens) > max_seq_len:
            raise ValueError("prompt exceeds max_seq_len")
        if not tokens:
            raise ValueError("prompt must contain at least one token")
        self.max_seq_len = max_seq_len
        self.kv_cache = model.create_kv_cache()
        self.model = model
        self.detokenizer = tokenizer.detokenizer.__class__(tokenizer._tokenizer)
        self.prefill_tokens = mx.array(tokens, dtype=mx.int32)
        self.prefill_max_step = prefill_max_step
        self.is_done = False
        self.is_prefill_done = False
        self.eos_token_id = tokenizer.eos_token_id
        self.next_token = None
        self.offset = 0
        self.prompt_idx = prompt_idx

    def try_prefill(self):
        """Prefill the complete prompt and emit its first token (Day 1)."""
        if self.is_prefill_done:
            raise ValueError("prefill called after done")
        # Day 1 consumes the full prompt, regardless of the future chunk budget.
        if self.prefill_tokens.size < self.max_seq_len:
            token = _step(self.model, self.prefill_tokens[None, :], [0], self.kv_cache)
            self.offset = self.prefill_tokens.size
            self.decode_done(token.item(), update_offset=False)
        else:
            self.offset = self.prefill_tokens.size
            self.is_done = True
        self.is_prefill_done = True

    def decode_done(self, token, update_offset=True):
        # offset counts cached tokens; the newly predicted token is not cached yet.
        if self.is_done:
            raise ValueError("decode called after done")
        if update_offset:
            self.offset += 1
        if token == self.eos_token_id or self.offset >= self.max_seq_len:
            self.is_done = True
            return
        self.next_token = token
        self.detokenizer.add_token(token)

    def text(self):
        if self.is_done and hasattr(self.detokenizer, "finalize"):
            self.detokenizer.finalize()
        return self.detokenizer.text


def _print_progress(
    requests: list[Request | None],
    pending_prefill_request: Request | None,
    queue_size: int,
    progress_cnt: int,
    start_time: datetime,
):
    print(f"  --- {datetime.now() - start_time}")
    animation_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    animation_frame = animation_frames[progress_cnt % len(animation_frames)]
    for i, request in enumerate(requests):
        if request is None:
            print(f"  Decode #{i}: idle", flush=True)
        else:
            text_preview = request.text()[-80:].replace("\n", " ")
            print(
                f"{animation_frame} Decode [req {request.prompt_idx}, {request.offset}]: {text_preview}",
                flush=True,
            )
    if pending_prefill_request is not None:
        if pending_prefill_request.is_prefill_done:
            print(
                f"  Prefill [req {pending_prefill_request.prompt_idx}]: done, waiting for slot, {queue_size} requests in queue",
                flush=True,
            )
            return
        percentage = (
            pending_prefill_request.offset / pending_prefill_request.prefill_tokens.size
        ) * 100
        print(
            f"{animation_frame} Prefill [req {pending_prefill_request.prompt_idx}]: {percentage:.2f}% ({pending_prefill_request.prefill_tokens.size - pending_prefill_request.offset} remaining tokens)",
            flush=True,
        )
    else:
        print(f"  Prefill: idle, {queue_size} requests in queue", flush=True)


def batch_generate(
    model: any,
    tokenizer: TokenizerWrapper,
    prompts: list[str],
    max_seq_len=512,
    batch_size=5,
    prefill_step=128,
):
    if batch_size <= 0 or max_seq_len <= 0:
        raise ValueError("batch_size and max_seq_len must be positive")
    prompts = list(prompts)
    decode_requests: list[Request | None] = [None] * batch_size
    kv_cache = [
        BatchingKvCache(max_active_requests=batch_size, max_seq_len=max_seq_len)
        for _ in range(model.num_hidden_layers)
    ]
    result = []
    pending_prefill_request = None
    next_request_idx = 0
    progress_cnt = 0
    start_time = datetime.now()

    while True:
        if not prompts and pending_prefill_request is None and not any(decode_requests):
            break
        # prefill until no idle slots
        if len(prompts) > 0 and pending_prefill_request is None:
            prompt = prompts.pop(0)
            pending_prefill_request = Request(
                model, tokenizer, prompt, prefill_step, next_request_idx, max_seq_len
            )
            next_request_idx += 1

        # In every iteration, we do a prefill first
        if pending_prefill_request is not None:
            made_progress = False
            if not pending_prefill_request.is_prefill_done:
                pending_prefill_request.try_prefill()
                made_progress = True
            if pending_prefill_request.is_prefill_done:
                request = pending_prefill_request
                if request.is_done:
                    result.append((request.prompt_idx, request.text()))
                    for cache in request.kv_cache:
                        cache.release()
                    pending_prefill_request = None
                elif None in decode_requests:
                    slot = decode_requests.index(None)
                    for cache, request_cache in zip(kv_cache, request.kv_cache):
                        cache.add_request(request_cache, slot)
                    decode_requests[slot] = request
                    pending_prefill_request = None
            if made_progress:
                _print_progress(
                    decode_requests,
                    pending_prefill_request,
                    len(prompts),
                    progress_cnt,
                    start_time,
                )
                progress_cnt += 1

        # After the prefill request moves forward one step, we do the decode
        if any(req is not None for req in decode_requests):
            next_tokens = []
            offsets = []
            for request in decode_requests:
                next_tokens.append(0 if request is None else request.next_token)
                offsets.append(0 if request is None else request.offset)
            next_tokens = mx.array(next_tokens, dtype=mx.int32)
            next_tokens = _step(model, next_tokens.reshape(-1, 1), offsets, kv_cache)
            for i in range(batch_size):
                request = decode_requests[i]
                if request is None:
                    continue
                request.decode_done(next_tokens[i].item())
                if request.is_done:
                    result.append((request.prompt_idx, request.text()))
                    for cache in kv_cache:
                        cache.remove_request(i)
                    decode_requests[i] = None
            _print_progress(
                decode_requests,
                pending_prefill_request,
                len(prompts),
                progress_cnt,
                start_time,
            )
            progress_cnt += 1
    return result
