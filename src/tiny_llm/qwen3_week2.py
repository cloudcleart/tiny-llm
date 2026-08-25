from typing import Any

import mlx.core as mx

from .attention import scaled_dot_product_attention_grouped
from .basics import linear, silu
from .embedding import Embedding, QuantizedEmbedding
from .kv_cache import TinyKvCache
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from .quantize import QuantizedWeights, dequantize_linear, quantized_linear
from .week2_kernels import (
    FastRMSNorm,
    FastRoPE,
    swiglu,
)

WEEK2_CHECKPOINTS = (
    "kv-cache",
    "quantized-matvec",
    "rmsnorm",
    "rope",
    "swiglu",
    "decode-attention",
    "simd-matmul",
    "split-k",
)

DECODE_ATTENTION_MAX_CONTEXT = 256
DECODE_ATTENTION_MAX_QUERY = 2


def _linear(x: mx.array, weight: mx.array | QuantizedWeights) -> mx.array:
    if isinstance(weight, QuantizedWeights):
        return quantized_linear(x, weight)
    return linear(x, weight)


def _readable_rope_offset(
    offset: int | list[int] | mx.array, sequence_length: int
) -> slice | list[slice]:
    """Convert absolute token positions to the slices used by readable RoPE."""
    if isinstance(offset, int):
        return slice(offset, offset + sequence_length)
    if isinstance(offset, list):
        return [slice(value, value + sequence_length) for value in offset]

    values = offset.tolist()
    if not isinstance(values, list):
        values = [values]
    return [slice(value, value + sequence_length) for value in values]


class Qwen3MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: mx.array | QuantizedWeights,
        wk: mx.array | QuantizedWeights,
        wv: mx.array | QuantizedWeights,
        wo: mx.array | QuantizedWeights,
        q_norm: mx.array,
        k_norm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        rms_norm_eps: float = 1e-5,
        use_fast_rms_norm: bool = True,
        use_fast_rope: bool = True,
        use_decode_attention: bool = True,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        assert hidden_size % num_heads == 0, (
            f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        )
        assert num_heads % num_kv_heads == 0, (
            f"num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads}"
        )

        self.head_dim = head_dim
        self.scale = self.head_dim**-0.5
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.use_fast_rope = use_fast_rope
        self.use_decode_attention = use_decode_attention

        norm_cls = FastRMSNorm if use_fast_rms_norm else RMSNorm
        rope_cls = FastRoPE if use_fast_rope else RoPE
        self.q_norm = norm_cls(head_dim, q_norm, eps=rms_norm_eps)
        self.k_norm = norm_cls(head_dim, k_norm, eps=rms_norm_eps)
        self.rope = rope_cls(head_dim, max_seq_len, theta)

    def __call__(
        self,
        x: mx.array,
        offsets: int | list[int] | mx.array,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        """
        Args:
            x: Incoming hidden states with shape (B, L, E).
            offsets: Absolute position of the first incoming token.
            cache: This Transformer's layer-specific key-value cache.
            mask: Optional causal or additive attention mask.

        Returns:
            Attention output with shape (B, L, E).
        """
        batch_size, sequence_length, _ = x.shape

        query = _linear(x, self.wq).reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )
        key = _linear(x, self.wk).reshape(
            batch_size, sequence_length, self.num_kv_heads, self.head_dim
        )
        value = _linear(x, self.wv).reshape(
            batch_size, sequence_length, self.num_kv_heads, self.head_dim
        )

        # Qwen3 normalizes Q/K before assigning their absolute RoPE positions.
        query = self.q_norm(query)
        key = self.k_norm(key)
        rope_offsets = offsets
        if not self.use_fast_rope:
            rope_offsets = _readable_rope_offset(offsets, sequence_length)
        query = self.rope(query, offset=rope_offsets)
        key = self.rope(key, offset=rope_offsets)

        # Attention and KV cache use head-first tensors: (B, H, L, D).
        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)

        # Append only this call's K/V, then attend over the complete cached prefix.
        key, value, _, mask = cache.update_and_fetch(
            key,
            value,
            mask_length=sequence_length,
            mask=mask,
        )
        output = scaled_dot_product_attention_grouped(
            query.astype(mx.float32),
            key.astype(mx.float32),
            value.astype(mx.float32),
            scale=self.scale,
            mask=mask,
        ).astype(x.dtype)

        output = output.transpose(0, 2, 1, 3).reshape(
            batch_size, sequence_length, self.num_heads * self.head_dim
        )
        return _linear(output, self.wo)


class Qwen3MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array | QuantizedWeights,
        w_up: mx.array | QuantizedWeights,
        w_down: mx.array | QuantizedWeights,
        use_fast_swiglu: bool = True,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down
        self.use_fast_swiglu = use_fast_swiglu

    def __call__(self, x: mx.array) -> mx.array:
        gate = _linear(x, self.w_gate)
        up = _linear(x, self.w_up)
        hidden = swiglu(gate, up) if self.use_fast_swiglu else silu(gate) * up
        return _linear(hidden, self.w_down)


class Qwen3TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array | QuantizedWeights,
        wk: mx.array | QuantizedWeights,
        wv: mx.array | QuantizedWeights,
        wo: mx.array | QuantizedWeights,
        q_norm: mx.array,
        k_norm: mx.array,
        w_gate: mx.array | QuantizedWeights,
        w_up: mx.array | QuantizedWeights,
        w_down: mx.array | QuantizedWeights,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_fast_rms_norm: bool = True,
        use_fast_rope: bool = True,
        use_fast_swiglu: bool = True,
        use_decode_attention: bool = True,
    ):
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size

        norm_cls = FastRMSNorm if use_fast_rms_norm else RMSNorm
        self.input_layernorm = norm_cls(
            hidden_size, w_input_layernorm, eps=rms_norm_eps
        )
        self.post_attention_layernorm = norm_cls(
            hidden_size, w_post_attention_layernorm, eps=rms_norm_eps
        )
        self.self_attn = Qwen3MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            q_norm=q_norm,
            k_norm=k_norm,
            max_seq_len=max_seq_len,
            theta=theta,
            rms_norm_eps=rms_norm_eps,
            use_fast_rms_norm=use_fast_rms_norm,
            use_fast_rope=use_fast_rope,
            use_decode_attention=use_decode_attention,
        )
        self.mlp = Qwen3MLP(
            dim=hidden_size,
            hidden_dim=intermediate_size,
            w_gate=w_gate,
            w_up=w_up,
            w_down=w_down,
            use_fast_swiglu=use_fast_swiglu,
        )

    def __call__(
        self,
        x: mx.array,
        offset: int,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        # Qwen3 uses pre-norm and one residual connection around each sublayer.
        attention_output = self.self_attn(self.input_layernorm(x), offset, cache, mask)
        hidden = x + attention_output
        mlp_output = self.mlp(self.post_attention_layernorm(hidden))
        return hidden + mlp_output


class Qwen3ModelWeek2:
    def __init__(self, mlx_model: Any, checkpoint: str = "split-k"):
        """
        Args:
            mlx_model: Qwen3 model loaded by mlx_lm.
            checkpoint: Week 2 optimization checkpoint to construct.
        """
        if checkpoint not in WEEK2_CHECKPOINTS:
            raise ValueError(
                f"unknown Week 2 checkpoint {checkpoint!r}; "
                f"choose one of {WEEK2_CHECKPOINTS}"
            )

        checkpoint_index = WEEK2_CHECKPOINTS.index(checkpoint)
        use_quantized_weights = checkpoint_index >= WEEK2_CHECKPOINTS.index(
            "quantized-matvec"
        )
        use_fast_rms_norm = checkpoint_index >= WEEK2_CHECKPOINTS.index("rmsnorm")
        use_fast_rope = checkpoint_index >= WEEK2_CHECKPOINTS.index("rope")
        use_fast_swiglu = checkpoint_index >= WEEK2_CHECKPOINTS.index("swiglu")
        use_decode_attention = checkpoint_index >= WEEK2_CHECKPOINTS.index(
            "decode-attention"
        )
        use_simdgroup_matmul = checkpoint_index >= WEEK2_CHECKPOINTS.index(
            "simd-matmul"
        )
        use_split_k_matmul = checkpoint_index >= WEEK2_CHECKPOINTS.index("split-k")

        args = mlx_model.args
        model = mlx_model.model
        self.checkpoint = checkpoint
        self.num_hidden_layers = mlx_model.args.num_hidden_layers
        self.hidden_size = args.hidden_size
        self.vocab_size = args.vocab_size
        self.precision = mx.bfloat16
        self.use_fast_rope = use_fast_rope

        def model_weight(layer: Any) -> mx.array | QuantizedWeights:
            if use_quantized_weights:
                return QuantizedWeights.from_mlx_layer(
                    layer,
                    use_simdgroup_matmul=use_simdgroup_matmul,
                    use_split_k_matmul=use_split_k_matmul,
                )
            return dequantize_linear(layer).astype(self.precision)

        embedding_weight = model_weight(model.embed_tokens)
        if isinstance(embedding_weight, QuantizedWeights):
            self.embedding = QuantizedEmbedding(
                vocab_size=self.vocab_size,
                embedding_dim=self.hidden_size,
                weight=embedding_weight,
            )
        else:
            self.embedding = Embedding(
                vocab_size=self.vocab_size,
                embedding_dim=self.hidden_size,
                weight=embedding_weight,
            )

        # Each block receives the cache entry at the same layer index in __call__.
        self.layers_inner = []
        for layer in model.layers:
            self.layers_inner.append(
                Qwen3TransformerBlock(
                    num_attention_heads=args.num_attention_heads,
                    num_kv_heads=args.num_key_value_heads,
                    hidden_size=args.hidden_size,
                    head_dim=args.head_dim,
                    intermediate_size=args.intermediate_size,
                    rms_norm_eps=args.rms_norm_eps,
                    wq=model_weight(layer.self_attn.q_proj),
                    wk=model_weight(layer.self_attn.k_proj),
                    wv=model_weight(layer.self_attn.v_proj),
                    wo=model_weight(layer.self_attn.o_proj),
                    q_norm=layer.self_attn.q_norm.weight,
                    k_norm=layer.self_attn.k_norm.weight,
                    w_gate=model_weight(layer.mlp.gate_proj),
                    w_up=model_weight(layer.mlp.up_proj),
                    w_down=model_weight(layer.mlp.down_proj),
                    w_input_layernorm=layer.input_layernorm.weight,
                    w_post_attention_layernorm=layer.post_attention_layernorm.weight,
                    max_seq_len=args.max_position_embeddings,
                    theta=args.rope_theta,
                    use_fast_rms_norm=use_fast_rms_norm,
                    use_fast_rope=use_fast_rope,
                    use_fast_swiglu=use_fast_swiglu,
                    use_decode_attention=use_decode_attention,
                )
            )

        norm_cls = FastRMSNorm if use_fast_rms_norm else RMSNorm
        self.norm = norm_cls(
            args.hidden_size,
            weight=model.norm.weight,
            eps=args.rms_norm_eps,
        )
        self.w_lm_head = (
            None if args.tie_word_embeddings else model_weight(mlx_model.lm_head)
        )
        self.mlx_model = mlx_model

    def create_kv_cache(self) -> list[TinyKvCache]:
        """Create one independent full KV cache for every Transformer layer."""
        from .kv_cache import TinyKvFullCache

        # A new list and new cache objects keep cache state scoped to one request.
        return [TinyKvFullCache() for _ in range(self.num_hidden_layers)]

    def __call__(
        self,
        inputs: mx.array,
        offset: int,
        cache: list[TinyKvCache],
        logits_to_keep: int | None = None,
    ) -> mx.array:
        """
        Args:
            inputs: Incoming token ids with shape (B, L).
            offset: Number of tokens already stored in each layer cache.
            cache: One key-value cache per Transformer layer.
            logits_to_keep: Optional number of trailing token logits to return.

        Returns:
            Vocabulary logits for the incoming tokens, or their requested suffix.
        """
        if len(cache) != self.num_hidden_layers:
            raise ValueError(
                f"expected {self.num_hidden_layers} layer caches, got {len(cache)}"
            )

        # Validate every layer before updating any cache to avoid partial writes.
        if isinstance(offset, int):
            for layer_index, layer_cache in enumerate(cache):
                cache_offset = getattr(layer_cache, "offset", None)
                if cache_offset is not None and cache_offset != offset:
                    raise ValueError(
                        f"layer {layer_index} cache offset {cache_offset} "
                        f"does not match model offset {offset}"
                    )

        hidden = self.embedding(inputs)
        mask = None if inputs.shape[1] == 1 else "causal"

        # Fast RoPE consumes one scalar offset per batch row; readable RoPE uses slices.
        if not self.use_fast_rope:
            rope_offsets = offset
        elif isinstance(offset, int):
            rope_offsets = mx.full((inputs.shape[0],), offset, dtype=mx.int32)
        else:
            rope_offsets = offset

        for layer_index, layer in enumerate(self.layers_inner):
            hidden = layer(
                hidden,
                rope_offsets,
                cache[layer_index],
                mask=mask,
            )

        if logits_to_keep is not None:
            if logits_to_keep <= 0:
                raise ValueError("logits_to_keep must be positive")
            hidden = hidden[:, -logits_to_keep:, :]

        hidden = self.norm(hidden)
        if self.w_lm_head is not None:
            return _linear(hidden, self.w_lm_head)
        return self.embedding.as_linear(hidden)
