import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen3MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        q_norm: mx.array,
        k_norm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        rms_norm_eps: float = 1e-5,
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


class Qwen3MLP:
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


class Qwen3TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        q_norm: mx.array,
        k_norm: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        """
        Args:
            num_attention_heads: Number of query attention heads.
            num_kv_heads: Number of key/value heads used by grouped-query attention.
            hidden_size: Hidden dimension E of each token representation.
            intermediate_size: MLP intermediate dimension I.
            rms_norm_eps: Epsilon used by both RMSNorm layers.
            wq, wk, wv, wo: Attention projection weights.
            bq, bk, bv: Attention projection biases.
            w_gate, w_up, w_down: Gated MLP projection weights.
            w_input_layernorm: RMSNorm weight before self-attention, shape (E,).
            w_post_attention_layernorm: RMSNorm weight before MLP, shape (E,).
            max_seq_len: Maximum sequence length for RoPE position cache.
            theta: RoPE base frequency.
        """
        # Qwen2 blocks use pre-norm: normalize before each sublayer.
        self.input_layernorm = RMSNorm(
            hidden_size, w_input_layernorm, eps=rms_norm_eps
        )
        # Self-attention handles Qwen2's grouped-query attention and RoPE.
        self.self_attn = Qwen2MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            bq=bq,
            bk=bk,
            bv=bv,
            max_seq_len=max_seq_len,
            theta=theta,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size, w_post_attention_layernorm, eps=rms_norm_eps
        )
        # Qwen2's feed-forward network is a gated MLP: SiLU(gate) * up.
        self.mlp = Qwen2MLP(
            dim=hidden_size,
            hidden_dim=intermediate_size,
            w_gate=w_gate,
            w_up=w_up,
            w_down=w_down,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        """
        Args:
            x: Input hidden states with shape (B, L, E), where B is batch size,
                L is sequence length, and E is hidden_size.
            mask: Optional attention mask. Use None for full attention, "causal"
                for autoregressive masking, or an additive mask broadcastable to
                attention scores.

        Returns:
            Output hidden states with shape (B, L, E).
        """
        # First residual branch: attention(input_layernorm(x)).
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, mask=mask)
        x = residual + x

        # Second residual branch: mlp(post_attention_layernorm(x)).
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return residual + x


class Qwen3ModelWeek1:
    def __init__(self, mlx_model: Any):
        """
        Args:
            mlx_model: Qwen2 model loaded by mlx_lm.load. Its args describe the
                architecture, and its modules hold quantized MLX weights.
        """
        args = mlx_model.args
        model = mlx_model.model
        precision = mx.float16

        self.num_hidden_layers = args.num_hidden_layers
        self.hidden_size = args.hidden_size
        self.vocab_size = args.vocab_size
        self.tie_word_embeddings = args.tie_word_embeddings

        # Convert the quantized input embedding table into a dense matrix.
        self.embedding = Embedding(
            vocab_size=self.vocab_size,
            embedding_dim=self.hidden_size,
            weight=dequantize_linear(model.embed_tokens).astype(precision),
        )

        # Build each Qwen2 block from the corresponding MLX layer weights.
        self.layers = []
        for layer in model.layers:
            self.layers.append(
                Qwen2TransformerBlock(
                    num_attention_heads=args.num_attention_heads,
                    num_kv_heads=args.num_key_value_heads,
                    hidden_size=args.hidden_size,
                    intermediate_size=args.intermediate_size,
                    rms_norm_eps=args.rms_norm_eps,
                    wq=dequantize_linear(layer.self_attn.q_proj).astype(precision),
                    wk=dequantize_linear(layer.self_attn.k_proj).astype(precision),
                    wv=dequantize_linear(layer.self_attn.v_proj).astype(precision),
                    wo=dequantize_linear(layer.self_attn.o_proj).astype(precision),
                    bq=layer.self_attn.q_proj.bias.astype(precision),
                    bk=layer.self_attn.k_proj.bias.astype(precision),
                    bv=layer.self_attn.v_proj.bias.astype(precision),
                    w_gate=dequantize_linear(layer.mlp.gate_proj).astype(precision),
                    w_up=dequantize_linear(layer.mlp.up_proj).astype(precision),
                    w_down=dequantize_linear(layer.mlp.down_proj).astype(precision),
                    w_input_layernorm=layer.input_layernorm.weight.astype(precision),
                    w_post_attention_layernorm=layer.post_attention_layernorm.weight.astype(
                        precision
                    ),
                    max_seq_len=args.max_position_embeddings,
                    theta=args.rope_theta,
                )
            )

        self.norm = RMSNorm(
            args.hidden_size,
            weight=model.norm.weight.astype(precision),
            eps=args.rms_norm_eps,
        )

        # If weights are not tied, Qwen2 uses a separate language-model head.
        if self.tie_word_embeddings:
            self.w_lm_head = None
        else:
            self.w_lm_head = dequantize_linear(mlx_model.lm_head).astype(precision)

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        """
        Args:
            inputs: Token ids with shape (B, L).

        Returns:
            Vocabulary logits with shape (B, L, vocab_size).
        """
        x = self.embedding(inputs)
        mask = "causal" if inputs.shape[1] > 1 else None

        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.norm(x)
        if self.w_lm_head is not None:
            return linear(x, self.w_lm_head)
        return self.embedding.as_linear(x)
