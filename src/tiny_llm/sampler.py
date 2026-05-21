import mlx.core as mx


def make_sampler(temp: float, top_p: float, top_k: int | None):
    """Create a sampler that maps next-token log-probabilities to token ids.

    Args:
        temp: Sampling temperature. ``0`` means greedy decoding; larger values
            make the distribution flatter before random sampling.
        top_p: Nucleus sampling threshold. When greater than ``0``, keep the
            highest-probability tokens whose cumulative probability is below
            this threshold.
        top_k: Top-k sampling limit. When positive, keep only the ``top_k``
            most likely tokens.

    Returns:
        A function that accepts log-probabilities with shape
        ``batch_size x vocab_size`` and returns one sampled token id per batch.
    """

    def sample(logprobs: mx.array):
        """Sample token ids from a batch of next-token log-probabilities."""
        if temp == 0:
            return mx.argmax(logprobs, axis=-1)

        # Filter to the k most likely tokens before temperature scaling.
        if top_k is not None and top_k > 0:
            top_k_indices = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[
                ..., :top_k
            ]
            top_k_mask = mx.put_along_axis(
                mx.zeros(logprobs.shape, dtype=mx.bool_),
                top_k_indices,
                mx.ones(top_k_indices.shape, dtype=mx.bool_),
                axis=-1,
            )
            logprobs = mx.where(top_k_mask, logprobs, -mx.inf)

        # Filter to the smallest high-probability set under the top-p threshold.
        if top_p is not None and top_p > 0:
            sorted_indices = mx.argsort(-logprobs, axis=-1)
            sorted_logprobs = mx.take_along_axis(logprobs, sorted_indices, axis=-1)
            keep_mask = mx.cumsum(mx.exp(sorted_logprobs), axis=-1) < top_p
            keep_mask[..., 0] = True
            sorted_logprobs = mx.where(keep_mask, sorted_logprobs, -mx.inf)
            logprobs = mx.put_along_axis(
                logprobs, sorted_indices, sorted_logprobs, axis=-1
            )

        # Temperature rescales the remaining logits before categorical sampling.
        logprobs = logprobs / temp
        return mx.random.categorical(logprobs, axis=-1)

    return sample
