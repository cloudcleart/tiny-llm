import mlx.core as mx

from .basics import silu
from .quantize import QuantizedWeights, quantized_linear


def grouped_expert_linear(
    x: mx.array,
    w_experts: QuantizedWeights,
    expert_ids: mx.array,
) -> mx.array:
    """Apply each row's selected quantized expert, preserving input order."""
    leading_shape = x.shape[:-1]
    flat_x = x.reshape(-1, x.shape[-1])
    flat_expert_ids = expert_ids.reshape(-1)

    # Keep jobs for the same expert contiguous, retaining the inverse mapping.
    order = mx.argsort(flat_expert_ids)
    inverse_order = mx.argsort(order)
    grouped_x = flat_x[order]
    grouped_ids = flat_expert_ids[order]

    # Each activation row is a separate 1 x D matrix in the gathered batch.
    out = mx.gather_qmm(
        mx.expand_dims(grouped_x, -2),
        w_experts.weight,
        scales=w_experts.scales,
        biases=w_experts.biases,
        lhs_indices=mx.arange(grouped_x.shape[0], dtype=mx.uint32),
        rhs_indices=grouped_ids,
        transpose=True,
        group_size=w_experts.group_size,
        bits=w_experts.bits,
        sorted_indices=True,
    ).squeeze(-2)
    return out[inverse_order].reshape(*leading_shape, out.shape[-1])


def route_topk(
    x: mx.array,
    w_router: QuantizedWeights,
    top_k: int,
    norm_topk_prob: bool = False,
) -> tuple[mx.array, mx.array, mx.array]:
    """Return all router probabilities and the selected experts and scores."""
    router_logits = quantized_linear(x, w_router)
    router_probs = mx.softmax(router_logits, axis=-1, precise=True)
    expert_ids = mx.argpartition(-router_probs, kth=top_k - 1, axis=-1)[..., :top_k]
    expert_scores = mx.take_along_axis(router_probs, expert_ids, axis=-1)
    if norm_topk_prob:
        expert_scores = expert_scores / expert_scores.sum(axis=-1, keepdims=True)
    return router_probs, expert_ids, expert_scores


class Moe:
    def __init__(
        self,
        w_router: QuantizedWeights,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
        num_experts_per_tok: int,
        norm_topk_prob: bool = False,
    ):
        self.w_router = w_router
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob

    def __call__(self, x: mx.array) -> mx.array:
        _, expert_ids, expert_scores = route_topk(
            x, self.w_router, self.num_experts_per_tok, self.norm_topk_prob
        )
        # Create one input row per selected expert: (..., D) -> (..., K, D).
        expert_inputs = mx.broadcast_to(
            mx.expand_dims(x, -2), (*expert_ids.shape, x.shape[-1])
        )
        gate = grouped_expert_linear(expert_inputs, self.w_gate, expert_ids)
        up = grouped_expert_linear(expert_inputs, self.w_up, expert_ids)
        hidden = silu(gate) * up
        expert_outputs = grouped_expert_linear(hidden, self.w_down, expert_ids)
        # The grouped helper restores token/top-k order before this reduction.
        return (expert_outputs * mx.expand_dims(expert_scores, -1)).sum(axis=-2)
