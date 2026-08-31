from math import prod
from typing import Any

import mlx.core as mx

from extensions import tiny_llm_ext


def dequantize_linear(mx_layer: Any) -> mx.array:
    w = mx.dequantize(
        mx_layer.weight,
        mx_layer.scales,
        mx_layer.biases,
        mx_layer.group_size,
        mx_layer.bits,
    )
    return w.astype(mx.bfloat16)


class QuantizedWeights:
    def __init__(
        self,
        scales: mx.array,
        biases: mx.array | None,
        group_size: int,
        bits: int,
        weight: mx.array,
        use_simdgroup_matmul: bool = False,
        use_simdgroup_matvec: bool = True,
        use_split_k_matmul: bool = False,
        use_mlx_quantized_linear: bool = False,
    ):
        self.scales = scales
        self.biases = biases
        self.group_size = group_size
        self.bits = bits
        self.weight = weight
        self.use_simdgroup_matmul = use_simdgroup_matmul
        self.use_simdgroup_matvec = use_simdgroup_matvec
        self.use_split_k_matmul = use_split_k_matmul
        self.use_mlx_quantized_linear = use_mlx_quantized_linear

    @staticmethod
    def from_mlx_layer(
        mlx_layer: Any,
        use_simdgroup_matmul: bool = False,
        use_simdgroup_matvec: bool = True,
        use_split_k_matmul: bool = False,
        use_mlx_quantized_linear: bool = False,
    ) -> "QuantizedWeights":
        biases = mlx_layer.biases
        return QuantizedWeights(
            scales=mlx_layer.scales.astype(mx.bfloat16),
            biases=None if biases is None else biases.astype(mx.bfloat16),
            group_size=mlx_layer.group_size,
            bits=mlx_layer.bits,
            weight=mlx_layer.weight,
            use_simdgroup_matmul=use_simdgroup_matmul,
            use_simdgroup_matvec=use_simdgroup_matvec,
            use_split_k_matmul=use_split_k_matmul,
            use_mlx_quantized_linear=use_mlx_quantized_linear,
        )


def mlx_quantized_linear(
    x: mx.array,
    w: QuantizedWeights,
    bias: mx.array | None = None,
) -> mx.array:
    pass


def quantized_matmul(
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    a: mx.array,
    b: mx.array,
    transpose_b: bool = False,
    use_simdgroup: bool = False,
    use_split_k: bool = False,
) -> mx.array:
    if a.ndim == 0:
        raise ValueError("quantized_matmul expects at least one activation dimension")

    # The C++ primitive consumes a two-dimensional M x N activation matrix.
    # Flatten all leading dimensions here and restore them on the lazy output.
    input_shape = a.shape
    flat_a = mx.contiguous(a.reshape(-1, input_shape[-1]))
    result = tiny_llm_ext.quantized_matmul(
        mx.contiguous(scales),
        mx.contiguous(biases),
        group_size,
        bits,
        flat_a,
        mx.contiguous(b),
        transpose_b=transpose_b,
        use_simdgroup=use_simdgroup,
        use_split_k=use_split_k,
    )
    return result.reshape(*input_shape[:-1], result.shape[-1])


def dequantize_weights(
    weight: mx.array,
    scales: mx.array,
    biases: mx.array | None,
    group_size: int,
    bits: int,
) -> mx.array:
    if bits != 4:
        raise ValueError(f"only 4-bit weights are supported, got bits={bits}")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if weight.ndim == 0 or scales.ndim == 0:
        raise ValueError("weight and scales must have at least one dimension")
    if weight.dtype != mx.uint32:
        raise TypeError(f"packed weights must be uint32, got {weight.dtype}")

    # MLX packs eight consecutive W4 codes into one uint32. The least
    # significant nibble is the first logical weight in each pack.
    values_per_pack = 32 // bits
    logical_width = weight.shape[-1] * values_per_pack
    if logical_width % group_size != 0:
        raise ValueError(
            f"unpacked width {logical_width} must be divisible by group_size "
            f"{group_size}"
        )

    expected_scale_shape = (*weight.shape[:-1], logical_width // group_size)
    if scales.shape != expected_scale_shape:
        raise ValueError(
            f"expected scales shape {expected_scale_shape}, got {scales.shape}"
        )
    if biases is not None and biases.shape != expected_scale_shape:
        raise ValueError(
            f"expected biases shape {expected_scale_shape}, got {biases.shape}"
        )

    # Add a trailing lane for the eight nibbles, then collapse the pack and
    # nibble dimensions back into the original logical weight dimension.
    shifts = mx.arange(0, 32, bits, dtype=mx.uint32)
    mask = mx.array((1 << bits) - 1, dtype=mx.uint32)
    unpacked = ((weight[..., None] >> shifts) & mask).reshape(
        *weight.shape[:-1], logical_width
    )

    # A scale and bias are shared by every logical weight in a quantization
    # group. Keep the stored sign of the scale: both code orientations occur
    # in the supplied checkpoints.
    scales = mx.repeat(scales.astype(mx.bfloat16), group_size, axis=-1)
    result = unpacked.astype(mx.bfloat16) * scales
    if biases is not None:
        result = result + mx.repeat(biases.astype(mx.bfloat16), group_size, axis=-1)
    return result.astype(mx.bfloat16)


def quantized_matvec_custom(
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    a: mx.array,
    b: mx.array,
    transpose_b: bool = False,
) -> mx.array:
    return quantized_matmul(
        scales=scales,
        biases=biases,
        group_size=group_size,
        bits=bits,
        a=a,
        b=b,
        transpose_b=transpose_b,
        use_simdgroup=True,
    )


def quantized_matmul_vanilla(
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    a: mx.array,
    b: mx.array,
    transpose_b: bool = False,
) -> mx.array:
    return quantized_matmul(
        scales=scales,
        biases=biases,
        group_size=group_size,
        bits=bits,
        a=a,
        b=b,
        transpose_b=transpose_b,
        use_simdgroup=False,
    )


def quantized_linear(
    x: mx.array,
    w: QuantizedWeights,
    bias: mx.array | None = None,
) -> mx.array:
    if x.ndim == 0:
        raise ValueError("quantized_linear expects at least one input dimension")
    if w.bits != 4:
        raise ValueError(f"only 4-bit weights are supported, got bits={w.bits}")

    expected_input_dim = w.weight.shape[-1] * (32 // w.bits)
    if x.shape[-1] != expected_input_dim:
        raise ValueError(
            f"input dimension {x.shape[-1]} does not match packed weight "
            f"dimension {expected_input_dim}"
        )

    if w.biases is None:
        raise ValueError("quantized_linear requires affine quantization biases")

    num_rows = prod(x.shape[:-1])

    # Decode and other tiny inputs benefit from the cooperative matvec kernel;
    # matrix-shaped prefill stays on the matmul schedule.
    if w.use_simdgroup_matvec and num_rows <= 8:
        result = quantized_matvec_custom(
            scales=w.scales,
            biases=w.biases,
            group_size=w.group_size,
            bits=w.bits,
            a=x,
            b=w.weight,
            transpose_b=True,
        )
    else:
        result = quantized_matmul(
            scales=w.scales,
            biases=w.biases,
            group_size=w.group_size,
            bits=w.bits,
            a=x,
            b=w.weight,
            transpose_b=True,
            use_simdgroup=w.use_simdgroup_matmul,
            use_split_k=w.use_split_k_matmul,
        )

    if bias is not None:
        result = result + bias
    return result
