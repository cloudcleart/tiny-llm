import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        """
        Args:
            dims: Head dimension size. Must be even because RoPE rotates pairs.
            seq_len: Maximum sequence length to precompute frequencies for.
            base: Frequency base used by RoPE, usually 10000 or the model's theta.
            traditional: Whether to use adjacent-dimension pairing for rotation.
                If false, pair the first half of the head dimension with the
                second half, matching Qwen2-style RoPE.
        """
        assert dims % 2 == 0, "dims must be even"
        self.dims = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional
        self.half_dims = dims // 2

        # Precompute one RoPE angle per position and per dimension pair.
        inner = mx.arange(self.half_dims, dtype=mx.float32) / self.half_dims
        freqs = mx.power(base, -inner)
        positions = mx.arange(seq_len)
        freqs = mx.outer(positions, freqs)
        self.cos_freqs = mx.cos(freqs)
        self.sin_freqs = mx.sin(freqs)

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        """
        Args:
            x: Input tensor with shape (batch, seq_len, num_heads, head_dim).
            offset: Shared absolute position range or one range per batch row.
                None means positions start at 0.

        Returns:
            x with RoPE applied, preserving the original shape and dtype.
        """
        N, L, H, D = x.shape
        orig_dtype = x.dtype

        if offset is None:
            cos_freqs = self.cos_freqs[:L]
            sin_freqs = self.sin_freqs[:L]
        elif isinstance(offset, list):
            assert len(offset) == N, "offset must contain one slice per batch row"
            for position_range in offset:
                assert isinstance(position_range, slice), (
                    "offset entries must be slices"
                )
                assert position_range.stop - position_range.start == L, (
                    f"offset must be of length {L}"
                )
            cos_freqs = mx.stack(
                [self.cos_freqs[position_range] for position_range in offset]
            )
            sin_freqs = mx.stack(
                [self.sin_freqs[position_range] for position_range in offset]
            )
        else:
            # During generation, the input may start at a later absolute position.
            assert isinstance(offset, slice), "offset must be None or a slice"
            assert offset.stop - offset.start == L, f"offset must be of length {L}"
            cos_freqs = self.cos_freqs[offset]
            sin_freqs = self.sin_freqs[offset]

        # Broadcast position frequencies over the batch and attention heads.
        cos_freqs = cos_freqs.reshape(-1, L, 1, self.half_dims)
        sin_freqs = sin_freqs.reshape(-1, L, 1, self.half_dims)

        if self.traditional:
            # Traditional RoPE treats adjacent values as complex pairs: (0, 1), (2, 3), ...
            x = x.reshape(N, L, H, self.half_dims, 2)
            x1 = x[..., 0]
            x2 = x[..., 1]
        else:
            # Qwen2-style RoPE pairs the first half with the second half.
            x1 = x[..., : self.half_dims]
            x2 = x[..., self.half_dims :]

        # Complex multiplication by cos(theta) + i sin(theta).
        real = x1 * cos_freqs - x2 * sin_freqs
        imag = x1 * sin_freqs + x2 * cos_freqs
        if self.traditional:
            y = mx.stack([real, imag], axis=-1).reshape(N, L, H, D)
        else:
            y = mx.concat([real, imag], axis=-1)
        return y.astype(orig_dtype)
