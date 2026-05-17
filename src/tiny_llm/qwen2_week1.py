import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        assert hidden_size % num_heads == 0
        # Qwen2 uses GQA, so several query heads may share one KV head.
        assert num_heads % num_kv_heads == 0

        self.head_dim = hidden_size // num_heads
        self.scale = mx.rsqrt(self.head_dim)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv
        self.rope = RoPE(self.head_dim, max_seq_len, theta)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        """
        Args:
            x: Input hidden states with shape (B, L, E), where B is batch size,
                L is sequence length, and E is hidden_size.
            mask: Optional attention mask. Use None for no masking, "causal" for
                autoregressive masking, or an additive mask broadcastable to
                (B, num_heads, L, L).

        Returns:
            Output hidden states with shape (B, L, E), matching the input dtype.
        """
        B, L, _ = x.shape

        # Q has num_heads heads, while K/V have fewer num_kv_heads heads for GQA.
        q = linear(x, self.wq, bias=self.bq).reshape(
            B, L, self.num_heads, self.head_dim
        )
        k = linear(x, self.wk, bias=self.bk).reshape(
            B, L, self.num_kv_heads, self.head_dim
        )
        v = linear(x, self.wv, bias=self.bv).reshape(
            B, L, self.num_kv_heads, self.head_dim
        )

        # RoPE changes attention scores, so it applies to Q/K but not V.
        q = self.rope(q, offset=slice(0, L))
        k = self.rope(k, offset=slice(0, L))

        # grouped attention expects head-first layout: B x H x L x D.
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Run attention in float32 for stable score/softmax computation.
        x = scaled_dot_product_attention_grouped(
            q.astype(mx.float32),
            k.astype(mx.float32),
            v.astype(mx.float32),
            scale=self.scale,
            mask=mask,
        ).astype(x.dtype)

        # Return to token-first layout and concatenate query heads back to hidden size.
        x = x.transpose(0, 2, 1, 3).reshape(B, L, self.hidden_size)
        return linear(x, self.wo)


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        """
        Args:
            dim: Input and output hidden size E.
            hidden_dim: Intermediate MLP size I.
            w_gate: Gate projection weight with shape (I, E).
            w_up: Up projection weight with shape (I, E).
            w_down: Down projection weight with shape (E, I).
        """
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: Input hidden states with shape (..., E).

        Returns:
            Output hidden states with shape (..., E).
        """
        # Qwen2 uses a gated MLP: SiLU(gate_proj(x)) gates up_proj(x).
        gated = silu(linear(x, self.w_gate)) * linear(x, self.w_up)
        return linear(gated, self.w_down)


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        pass

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        pass


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        pass

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        pass
