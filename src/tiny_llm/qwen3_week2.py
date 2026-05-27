import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear, QuantizedWeights, quantized_linear
from .kv_cache import TinyKvCache


class Qwen3MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        q_norm: mx.array,
        k_norm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        rms_norm_eps: float = 1e-5,
        use_flash_attention: bool = False,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        assert num_heads % num_kv_heads == 0

        self.head_dim = head_dim
        self.scale = mx.rsqrt(self.head_dim)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.q_norm = RMSNorm(head_dim, q_norm, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, k_norm, eps=rms_norm_eps)
        self.rope = RoPE(self.head_dim, max_seq_len, theta)
        self.use_flash_attention = use_flash_attention

    def __call__(
        self,
        x: mx.array,
        offset: int,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        """
        Args:
            x: New token hidden states with shape (B, L_new, E).
            offset: Absolute position of the first new token. It should match
                the number of tokens already stored in the cache.
            cache: Layer-local key-value cache.
            mask: Optional attention mask.

        Returns:
            Output hidden states for the new tokens with shape (B, L_new, E).
        """
        B, L_new, _ = x.shape
        assert cache.offset == offset

        # Step 1: project the new tokens into Q/K/V head states.
        q = quantized_linear(x, self.wq).reshape(
            B, L_new, self.num_heads, self.head_dim
        )
        k = quantized_linear(x, self.wk).reshape(
            B, L_new, self.num_kv_heads, self.head_dim
        )
        v = quantized_linear(x, self.wv).reshape(
            B, L_new, self.num_kv_heads, self.head_dim
        )

        # Step 2: normalize Q/K per head before applying RoPE.
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Step 3: use absolute positions so one-token decode does not restart at 0.
        position_slice = slice(offset, offset + L_new)
        q = self.rope(q, offset=position_slice)
        k = self.rope(k, offset=position_slice)

        # Step 4: append the new K/V states and fetch the full cached history.
        k, v, _, mask = cache.update_and_fetch(k, v, mask=mask)

        # Step 5: grouped attention expects head-first layout: B x H x L x D.
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Step 6: attend from new queries over all cached keys/values.
        x = scaled_dot_product_attention_grouped(
            q.astype(mx.float32),
            k.astype(mx.float32),
            v.astype(mx.float32),
            scale=self.scale,
            mask=mask,
        ).astype(x.dtype)

        # Step 7: restore token-first layout and project back to hidden size.
        x = x.transpose(0, 2, 1, 3).reshape(
            B, L_new, self.num_heads * self.head_dim
        )
        return quantized_linear(x, self.wo)


class Qwen3MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
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
        gated = silu(quantized_linear(x, self.w_gate)) * quantized_linear(
            x, self.w_up
        )
        return quantized_linear(gated, self.w_down)


class Qwen3TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        q_norm: mx.array,
        k_norm: mx.array,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_flash_attention: bool = False,
    ):
        """
        Args:
            num_attention_heads: Number of query attention heads.
            num_kv_heads: Number of key/value heads used by grouped-query attention.
            hidden_size: Hidden dimension E of each token representation.
            intermediate_size: MLP intermediate dimension I.
            rms_norm_eps: Epsilon used by both RMSNorm layers.
            wq, wk, wv, wo: Attention projection weights.
            q_norm, k_norm: Per-head RMSNorm weights for query and key states.
            w_gate, w_up, w_down: Gated MLP projection weights.
            w_input_layernorm: RMSNorm weight before self-attention, shape (E,).
            w_post_attention_layernorm: RMSNorm weight before MLP, shape (E,).
            max_seq_len: Maximum sequence length for RoPE position cache.
            theta: RoPE base frequency.
        """
        self.input_layernorm = RMSNorm(
            hidden_size, w_input_layernorm, eps=rms_norm_eps
        )
        self.self_attn = Qwen3MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            q_norm=q_norm,
            k_norm=k_norm,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            theta=theta,
            rms_norm_eps=rms_norm_eps,
            use_flash_attention=use_flash_attention,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size, w_post_attention_layernorm, eps=rms_norm_eps
        )
        self.mlp = Qwen3MLP(
            dim=hidden_size,
            hidden_dim=intermediate_size,
            w_gate=w_gate,
            w_up=w_up,
            w_down=w_down,
        )

    def __call__(
        self,
        x: mx.array,
        offset: int,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        """
        Args:
            x: Input hidden states with shape (B, L_new, E).
            offset: Absolute position of the first new token.
            cache: This layer's key-value cache.
            mask: Optional attention mask.

        Returns:
            Output hidden states with shape (B, L_new, E).
        """
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, offset=offset, cache=cache, mask=mask)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return residual + x


class Qwen3ModelWeek2:
    def __init__(
        self,
        mlx_model: Any,
        enable_flash_attn: bool = False,
    ):
        """
        Args:
            mlx_model: Qwen3 model loaded by mlx_lm.load. Its args describe the
                architecture, and its modules hold quantized MLX weights.
            enable_flash_attn: Whether to enable the later flash-attention path.
        """
        args = mlx_model.args
        model = mlx_model.model
        self.num_hidden_layers = args.num_hidden_layers
        self.hidden_size = args.hidden_size
        self.vocab_size = args.vocab_size
        self.tie_word_embeddings = args.tie_word_embeddings

        self.embedding = Embedding(
            vocab_size=self.vocab_size,
            embedding_dim=self.hidden_size,
            weight=dequantize_linear(model.embed_tokens),
        )

        self.layers = []
        for layer in model.layers:
            self.layers.append(
                Qwen3TransformerBlock(
                    num_attention_heads=args.num_attention_heads,
                    num_kv_heads=args.num_key_value_heads,
                    hidden_size=args.hidden_size,
                    head_dim=args.head_dim,
                    intermediate_size=args.intermediate_size,
                    rms_norm_eps=args.rms_norm_eps,
                    wq=QuantizedWeights.from_mlx_layer(layer.self_attn.q_proj),
                    wk=QuantizedWeights.from_mlx_layer(layer.self_attn.k_proj),
                    wv=QuantizedWeights.from_mlx_layer(layer.self_attn.v_proj),
                    wo=QuantizedWeights.from_mlx_layer(layer.self_attn.o_proj),
                    q_norm=layer.self_attn.q_norm.weight,
                    k_norm=layer.self_attn.k_norm.weight,
                    w_gate=QuantizedWeights.from_mlx_layer(layer.mlp.gate_proj),
                    w_up=QuantizedWeights.from_mlx_layer(layer.mlp.up_proj),
                    w_down=QuantizedWeights.from_mlx_layer(layer.mlp.down_proj),
                    w_input_layernorm=layer.input_layernorm.weight,
                    w_post_attention_layernorm=layer.post_attention_layernorm.weight,
                    max_seq_len=args.max_position_embeddings,
                    theta=args.rope_theta,
                    use_flash_attention=enable_flash_attn,
                )
            )

        self.norm = RMSNorm(
            args.hidden_size,
            weight=model.norm.weight,
            eps=args.rms_norm_eps,
        )
        if self.tie_word_embeddings:
            self.w_lm_head = None
        else:
            self.w_lm_head = dequantize_linear(mlx_model.lm_head)

    def __call__(
        self,
        inputs: mx.array,
        offset: int,
        cache: list[TinyKvCache],
    ) -> mx.array:
        """
        Args:
            inputs: Token ids for new tokens with shape (B, L_new).
            offset: Absolute position of the first new token.
            cache: One key-value cache per transformer layer.

        Returns:
            Vocabulary logits with shape (B, L_new, vocab_size).
        """
        assert len(cache) == self.num_hidden_layers

        x = self.embedding(inputs)
        mask = "causal" if inputs.shape[1] > 1 else None

        for layer, layer_cache in zip(self.layers, cache):
            x = layer(x, offset=offset, cache=layer_cache, mask=mask)

        x = self.norm(x)
        if self.w_lm_head is not None:
            return linear(x, self.w_lm_head)
        return self.embedding.as_linear(x)
