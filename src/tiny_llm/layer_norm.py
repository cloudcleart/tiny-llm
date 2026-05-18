import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        """
        Args:
            dim: Size of the last input dimension to normalize.
            weight: Learnable scale parameter with shape (dim,).
            eps: Small value added before rsqrt for numerical stability.
        """
        self.dim = dim
        self.weight = weight.astype(mx.float32)
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: Input tensor with shape (..., dim).

        Returns:
            RMS-normalized tensor with the same shape and dtype as x.
        """
        orig_dtype = x.dtype

        # Accumulate in float32 so low-precision inputs keep enough accuracy.
        x = x.astype(mx.float32)

        # Normalize independently over each hidden vector, i.e. the last axis.
        output = self.weight * x * mx.rsqrt(
            mx.mean(mx.square(x), axis=-1, keepdims=True) + self.eps
        )
        return output.astype(orig_dtype)
