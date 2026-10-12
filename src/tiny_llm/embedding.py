import mlx.core as mx

from extensions import tiny_llm_ext

from .basics import linear
from .quantize import QuantizedWeights, dequantize_weights, quantized_linear


class Embedding:
    def __init__(self, vocab_size: int, embedding_dim: int, weight: mx.array):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = weight

    def __call__(self, x: mx.array) -> mx.array:
        return self.weight[x, :]

    def as_linear(self, x: mx.array) -> mx.array:
        return linear(x, self.weight)


class QuantizedEmbedding:
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        weight: QuantizedWeights,
        use_custom_kernel: bool = False,
    ):
        if weight.bits != 4:
            raise ValueError(
                f"only 4-bit embedding weights are supported, got bits={weight.bits}"
            )
        values_per_pack = 32 // weight.bits
        if weight.weight.ndim != 2:
            raise ValueError(
                f"embedding weight must be a matrix, got shape {weight.weight.shape}"
            )
        if weight.weight.shape[0] != vocab_size:
            raise ValueError(
                f"expected {vocab_size} embedding rows, got {weight.weight.shape[0]}"
            )
        if weight.weight.shape[1] * values_per_pack != embedding_dim:
            raise ValueError(
                f"expected embedding dimension {embedding_dim}, got "
                f"{weight.weight.shape[1] * values_per_pack}"
            )

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = weight
        self.use_custom_kernel = use_custom_kernel

    def __call__(self, x: mx.array) -> mx.array:
        # Sampling may return uint32 token IDs, while MLX advanced indexing is
        # consistently supported with signed integer indices.
        indices = x.astype(mx.int32)

        if self.use_custom_kernel:
            # Fuse token lookup, W4 unpacking, and dequantization into one kernel.
            # Avoid gathering separate packed-weight, scale, and bias intermediates.
            if self.weight.group_size != 128:
                raise ValueError("custom embedding requires group_size=128")
            if x.dtype not in (mx.int32, mx.uint32):
                raise ValueError("custom embedding indices must be int32 or uint32")
            biases = self.weight.biases
            if biases is None:
                biases = mx.zeros_like(self.weight.scales)
            return tiny_llm_ext.quantized_embedding(
                mx.contiguous(indices),
                mx.contiguous(self.weight.scales.astype(mx.bfloat16)),
                mx.contiguous(biases.astype(mx.bfloat16)),
                mx.contiguous(self.weight.weight),
                self.weight.group_size,
                self.weight.bits,
            )

        # Gather packed rows and their group metadata first. This avoids
        # materializing the full dequantized vocabulary table for every lookup.
        biases = None if self.weight.biases is None else self.weight.biases[indices]
        return dequantize_weights(
            weight=self.weight.weight[indices],
            scales=self.weight.scales[indices],
            biases=biases,
            group_size=self.weight.group_size,
            bits=self.weight.bits,
        )

    def as_linear(self, x: mx.array) -> mx.array:
        # A tied output head must consume the packed table through the custom
        # quantized projection instead of dequantizing the whole vocabulary.
        return quantized_linear(x, self.weight)
