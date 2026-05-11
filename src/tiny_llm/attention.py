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
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass
