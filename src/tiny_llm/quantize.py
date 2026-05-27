import mlx.core as mx
from typing import Any
from pathlib import Path
import sys


def dequantize_linear(mx_layer: Any) -> mx.array:
    """
    Converts an MLX quantized linear layer back to a dense weight matrix.

    Args:
        mx_layer: MLX quantized linear-like layer with `weight`, `scales`,
            `biases`, `group_size`, and `bits` attributes.

    Returns:
        Dense dequantized weight with shape (out_features, in_features).
    """
    w = mx.dequantize(
        mx_layer.weight,
        mx_layer.scales,
        mx_layer.biases,
        mx_layer.group_size,
        mx_layer.bits,
    )
    return w


class QuantizedWeights:
    """
    Container for one quantized linear weight matrix.

    Args:
        scales: Per-group dequantization scales, shape
            (out_features, in_features / group_size).
        biases: Per-group dequantization biases, same shape as `scales`.
        group_size: Number of consecutive input features sharing one
            scale/bias pair.
        bits: Quantization bit width. This chapter uses 4-bit weights.
        weight: Packed quantized weights, shape (out_features, in_features / 8)
            for int4 weights packed 8 values per uint32.
    """

    def __init__(
        self,
        scales: mx.array,
        biases: mx.array,
        group_size: int,
        bits: int,
        weight: mx.array,
    ):
        self.scales = scales
        self.biases = biases
        self.group_size = group_size
        self.bits = bits
        self.weight = weight

    @staticmethod
    def from_mlx_layer(mlx_layer: Any) -> "QuantizedWeights":
        """
        Extracts quantized weight fields from an MLX quantized linear layer.

        Args:
            mlx_layer: MLX layer exposing `weight`, `scales`, `biases`,
                `group_size`, and `bits`.

        Returns:
            A `QuantizedWeights` object that can be passed to
            `quantized_linear`.
        """
        return QuantizedWeights(
            scales=mlx_layer.scales,
            biases=mlx_layer.biases,
            group_size=mlx_layer.group_size,
            bits=mlx_layer.bits,
            weight=mlx_layer.weight,
        )


def quantized_matmul(
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    a: mx.array,
    b: mx.array,
    transpose_b: bool = False,
) -> mx.array:
    """
    Matrix multiply with a quantized right-hand-side matrix.

    Computes `a @ dequantize(b).T` when `transpose_b=True`, matching the
    storage layout used by linear weights.

    Args:
        scales: Per-group scales for `b`, shape
            (out_features, in_features / group_size).
        biases: Per-group biases for `b`, same shape as `scales`.
        group_size: Number of unquantized values covered by one scale/bias.
        bits: Quantization bit width. Expected to be 4 for this task.
        a: Dense activation matrix, shape (batch_items, in_features).
        b: Packed quantized weight matrix. For int4 packed into uint32, shape is
            (out_features, in_features / 8).
        transpose_b: Whether to treat dequantized `b` as transposed. Linear
            layers should pass `True` because weights are stored as
            (out_features, in_features).

    Returns:
        Output matrix with shape (batch_items, out_features) when
        `transpose_b=True`.
    """
    try:
        # The extension package lives outside the normal `src` import root.
        # Add it lazily so Task 1 can run before the extension is installed.
        extension_dir = Path(__file__).resolve().parents[1] / "extensions"
        if str(extension_dir) not in sys.path:
            sys.path.insert(0, str(extension_dir))

        from tiny_llm_ext import quantized_matmul as ext_quantized_matmul

        return ext_quantized_matmul(
            scales=scales,
            biases=biases,
            group_size=group_size,
            bits=bits,
            a=a,
            b=b,
            transpose_b=transpose_b,
        )
    except (ImportError, AttributeError):
        # Until Task 2 exports the custom extension op, use MLX's reference op
        # so the Python-level wrapper remains usable and testable.
        return mx.quantized_matmul(
            a,
            b,
            scales,
            biases,
            group_size=group_size,
            bits=bits,
            transpose=transpose_b,
        )


def quantized_linear(
    x: mx.array,
    w: QuantizedWeights,
    bias: mx.array | None = None,
) -> mx.array:
    """
    Applies a quantized linear layer to the last dimension of `x`.

    This mirrors `linear(x, w, bias)` but keeps `w` in packed quantized form.

    Args:
        x: Dense input activations with shape (..., in_features).
        w: Quantized linear weight. Its original dense shape is
            (out_features, in_features).
        bias: Optional dense bias with shape (out_features,).

    Returns:
        Dense output activations with shape (..., out_features).
    """
    x_shape = x.shape
    # The extension op accepts 2D matrices, while model activations may have
    # leading batch/sequence dimensions. Flatten those and restore them after.
    x_2d = x.reshape(-1, x_shape[-1])
    output = quantized_matmul(
        scales=w.scales,
        biases=w.biases,
        group_size=w.group_size,
        bits=w.bits,
        a=x_2d,
        b=w.weight,
        transpose_b=True,
    )
    output = output.reshape(*x_shape[:-1], output.shape[-1])
    if bias is not None:
        output = output + bias
    return output
