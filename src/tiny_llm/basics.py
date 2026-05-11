import mlx.core as mx
import math


def softmax(x: mx.array, axis: int) -> mx.array:
    # TODO: manual implementation
    return mx.softmax(x, axis=axis)


def linear(
    x: mx.array,
    w: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    """
    Applies a linear transformation to the last dimension: y = x W^T + b.

    Args:
        x: Input tensor with shape (..., I), where I is the input dimension.
        w: Weight tensor with shape (O, I), mapping I input features to O outputs.
        bias: Optional bias tensor with shape (O,), added to each output vector.

    Returns:
        Output tensor with shape (..., O)
    """
    output = mx.matmul(x, w.T)
    if bias is not None:
        output = output + bias
    return output


def silu(x: mx.array) -> mx.array:
    pass
