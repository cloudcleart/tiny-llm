import mlx.core as mx
from extensions import tiny_llm_ext


class FastRMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim = dim
        self.weight = weight
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"FastRMSNorm expected final dimension {self.dim}, got {x.shape[-1]}"
            )
        return tiny_llm_ext.rms_norm(
            mx.contiguous(x),
            mx.contiguous(self.weight.astype(x.dtype)),
            self.eps,
        )


class FastRoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        self.dims = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional

    def __call__(self, x: mx.array, offset: int | list[int] | mx.array = 0) -> mx.array:
        if x.ndim != 4:
            raise ValueError("FastRoPE expects input with shape [B, L, H, D]")

        batch_size = x.shape[0]
        if isinstance(offset, int):
            offset = mx.full((batch_size,), offset, dtype=mx.int32)
        elif isinstance(offset, list):
            if len(offset) != batch_size:
                raise ValueError("FastRoPE needs one offset per batch row")
            offset = mx.array(offset, dtype=mx.int32)
        elif offset.ndim == 0:
            offset = mx.broadcast_to(offset.astype(mx.int32), (batch_size,))
        elif offset.shape != (batch_size,):
            raise ValueError("FastRoPE needs one offset per batch row")

        return tiny_llm_ext.rope(
            mx.contiguous(x),
            mx.contiguous(offset.astype(mx.int32)),
            self.dims,
            self.base,
            self.traditional,
        )


def swiglu(gate: mx.array, up: mx.array) -> mx.array:
    return tiny_llm_ext.swiglu(mx.contiguous(gate), mx.contiguous(up))


def scaled_dot_product_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float,
    mask: mx.array | str | None = None,
) -> mx.array:
    """Readable GQA oracle: [B, H_q, L, D] queries attend to [B, H_kv, S, D].

    Masks are additive and broadcastable to [B, H_q, L, S], or "causal".
    Accumulate in float32 and return the query dtype.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("attention expects query, key, and value with four dimensions")

    batch_size, num_heads, query_length, head_dim = query.shape
    key_batch_size, num_kv_heads, context_length, key_head_dim = key.shape
    if batch_size != key_batch_size or key.shape != value.shape:
        raise ValueError(
            "query, key, and value batch dimensions and K/V shapes must match"
        )
    if head_dim != key_head_dim or num_kv_heads == 0 or num_heads % num_kv_heads != 0:
        raise ValueError("incompatible grouped-query attention shapes")
    if isinstance(mask, str) and mask != "causal":
        raise ValueError(f"unsupported attention mask: {mask}")

    repeats = num_heads // num_kv_heads
    # H_q = H_kv * repeats: each consecutive group of query heads shares one KV head.
    # Reshape Q to [B, H_kv, repeats, L, D]; insert a singleton dimension in K/V
    # for broadcasting without duplication. Use FP32 intermediates in this oracle.
    grouped_query = query.astype(mx.float32).reshape(
        batch_size, num_kv_heads, repeats, query_length, head_dim
    )
    grouped_key = key.astype(mx.float32).reshape(
        batch_size, num_kv_heads, 1, context_length, head_dim
    )
    grouped_value = value.astype(mx.float32).reshape(
        batch_size, num_kv_heads, 1, context_length, head_dim
    )
    # [L, D] @ [D, S] -> [L, S]: scaled dot products of each query with all keys.
    scores = mx.matmul(grouped_query, grouped_key.swapaxes(-2, -1)) * scale

    if isinstance(mask, str):
        # Q covers the final L positions of the cache; S - L is the history length.
        # Query i can attend only to keys with key_position <= S - L + i.
        query_positions = mx.arange(query_length) + context_length - query_length
        allowed = mx.arange(context_length)[None, :] <= query_positions[:, None]
        scores = mx.where(allowed, scores, -mx.inf)
    elif mask is not None:
        # Broadcast to the model-facing shape, then split heads to match grouped scores.
        grouped_mask = mx.broadcast_to(
            mask, (batch_size, num_heads, query_length, context_length)
        ).reshape(batch_size, num_kv_heads, repeats, query_length, context_length)
        scores = scores + grouped_mask.astype(mx.float32)

    # Normalize over S and weight V, then merge head groups and restore the input dtype.
    probabilities = mx.softmax(scores, axis=-1)
    output = mx.matmul(probabilities, grouped_value)
    return output.reshape(query.shape).astype(query.dtype)


# The extension always takes a mask array; reuse this unread placeholder when absent.
_NO_ATTENTION_MASK = mx.array(0.0, dtype=mx.float32)


def decode_attention_custom(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float,
    mask: mx.array | str | None = None,
) -> mx.array:
    """Run the Metal online-softmax primitive without changing Q/K/V dtypes."""
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("attention expects query, key, and value with four dimensions")
    batch, heads, length, dim = query.shape
    key_batch, kv_heads, context, key_dim = key.shape
    if batch != key_batch or key.shape != value.shape:
        raise ValueError("query batch and K/V shapes must match")
    if kv_heads <= 0 or heads <= 0 or heads % kv_heads or dim != key_dim:
        raise ValueError("incompatible grouped-query attention shapes")
    if not 1 <= dim <= 256 or batch <= 0 or length <= 0 or context <= 0:
        raise ValueError("decode attention requires nonempty inputs and 1 <= D <= 256")
    if isinstance(mask, str) and mask != "causal":
        raise ValueError(f"unsupported attention mask: {mask}")
    # Unknown strings were rejected above. Pass a causal flag instead of building a mask.
    is_causal = isinstance(mask, str)
    has_mask = mask is not None and not is_causal
    if has_mask:
        # Normalize additive masks to contiguous FP32 [B*H_q, L, S] for row indexing.
        mask = mx.contiguous(
            mx.broadcast_to(mask, (batch, heads, length, context))
            .astype(mx.float32)
            .reshape(batch * heads, length, context)
        )
    else:
        mask = _NO_ATTENTION_MASK

    # Flatten batch/head, retaining H_q/H_kv so the kernel can recover the GQA mapping.
    # Materialize noncontiguous views, such as transposes, without changing Q/K/V dtypes.
    # BF16/FP16 values are converted to FP32 inside Metal registers.
    result = tiny_llm_ext.decode_attention(
        mx.contiguous(query.reshape(batch * heads, length, dim)),
        mx.contiguous(key.reshape(batch * kv_heads, context, dim)),
        mx.contiguous(value.reshape(batch * kv_heads, context, dim)),
        mask,
        scale,
        is_causal,
        has_mask,
        heads,
        kv_heads,
    )
    return result.reshape(query.shape)
