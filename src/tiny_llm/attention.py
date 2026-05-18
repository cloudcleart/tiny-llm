import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    """
    Args:
        query: Query vectors with shape (..., L, D).
        key: Key vectors with shape (..., L, D).
        value: Value vectors with shape (..., L, D).
        scale: Optional multiplier for attention scores. Defaults to 1 / sqrt(D).
        mask: Optional additive mask with shape (..., L, L), added before softmax.

    Returns:
        Attention output with shape (..., L, D).
    """
    # Use the standard attention scale unless the caller provides one.
    factor = mx.rsqrt(query.shape[-1]) if scale is None else scale

    # Each query attends over every key position along the sequence dimension.
    scores = mx.matmul(query, key.swapaxes(-2, -1)) * factor
    if mask is not None:
        scores = scores + mask

    # Normalize attention scores over keys, then mix the value vectors.
    return mx.matmul(softmax(scores, axis=-1), value)


class SimpleMultiHeadAttention:
    """
    A minimal multi-head attention layer without bias terms.

    It projects query/key/value from hidden size E into H heads of dimension D,
    runs scaled dot-product attention per head, then projects the concatenated
    heads back to hidden size E.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        """
        Args:
            hidden_size: Model hidden size E. Also equals H * D.
            num_heads: Number of attention heads H.
            wq: Query projection weight with shape (E, E).
            wk: Key projection weight with shape (E, E).
            wv: Value projection weight with shape (E, E).
            wo: Output projection weight with shape (E, E).
        """
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0
        self.head_dim = hidden_size // num_heads
        self.scale = mx.rsqrt(self.head_dim)
        assert wq.shape == (hidden_size, hidden_size)
        assert wk.shape == (hidden_size, hidden_size)
        assert wv.shape == (hidden_size, hidden_size)
        assert wo.shape == (hidden_size, hidden_size)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        """
        Args:
            query: Query input with shape (N, L, E).
            key: Key input with shape (N, L, E).
            value: Value input with shape (N, L, E).
            mask: Optional additive attention mask with shape broadcastable to
                (N, H, L, L).

        Returns:
            Multi-head attention output with shape (N, L, E).
        """
        batch_size, sequence_length, _ = query.shape
        assert query.shape == key.shape == value.shape

        # Project each token from E to H * D, then split the last dimension
        # into num_heads independent heads.
        q = linear(query, self.wq).reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )
        k = linear(key, self.wk).reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )
        v = linear(value, self.wv).reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )

        # Move heads before the sequence dimension: (N, L, H, D) -> (N, H, L, D).
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Run attention independently for each head.
        x = scaled_dot_product_attention_simple(
            q,
            k,
            v,
            scale=self.scale,
            mask=mask,
        )

        # Restore token-first layout and concatenate all heads back into E.
        x = x.transpose(0, 2, 1, 3).reshape(
            batch_size, sequence_length, self.hidden_size
        )

        # Mix information across heads with the output projection.
        return linear(x, self.wo)


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    # When S > L, the extra key/value positions are already-existing context.
    # Shift the causal diagonal right so queries can attend to that history.
    mask = mx.tril(mx.ones((L, S)), k=S - L)
    return mx.where(mask, mx.array(0), mx.array(-mx.inf)).astype(dtype)


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    """
    Args:
        query: Query vectors with shape (..., H_q, L, D).
        key: Key vectors with shape (..., H, S, D).
        value: Value vectors with shape (..., H, S, D).
        scale: Optional multiplier for attention scores. Defaults to 1 / sqrt(D).
        mask: Optional additive mask with shape broadcastable to (..., H_q, L, S),
            or the string "causal".

    Returns:
        Attention output with shape (..., H_q, L, D).
    """
    assert query.ndim >= 3
    assert key.ndim == query.ndim
    assert value.ndim == query.ndim
    assert key.shape == value.shape

    # Treat every leading dimension as batch; Task 1 tests 0, 1, and 2 batch dims.
    batch_shape = query.shape[:-3]
    H_q, L, D = query.shape[-3:]
    H, S, D_k = key.shape[-3:]

    assert key.shape[:-3] == batch_shape
    assert D == D_k
    assert H_q % H == 0

    # Split query heads into KV groups. Each KV head serves n_repeats query heads.
    n_repeats = H_q // H
    expected_shape = query.shape
    factor = mx.rsqrt(D) if scale is None else mx.array(scale)
    factor = factor.astype(query.dtype)

    # Insert the group dimension explicitly; K/V use size 1 so matmul broadcasts
    # the shared KV head across all query heads in the same group.
    query = query.reshape(*batch_shape, H, n_repeats, L, D)
    key = key.reshape(*batch_shape, H, 1, S, D)
    value = value.reshape(*batch_shape, H, 1, S, D)

    scores = mx.matmul(query, key.swapaxes(-2, -1)) * factor
    if mask is not None:
        if mask == "causal":
            scores = scores + causal_mask(L, S, scores.dtype)
        else:
            # Match the mask layout to grouped scores: (..., H, n_repeats, L, S).
            mask = mx.broadcast_to(mask, (*batch_shape, H_q, L, S))
            mask = mask.reshape(*batch_shape, H, n_repeats, L, S)
            scores = scores + mask

    # Collapse H and n_repeats back into the original H_q dimension.
    output = mx.matmul(softmax(scores, axis=-1), value)
    return output.reshape(expected_shape)


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass
