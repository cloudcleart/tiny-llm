import mlx.core as mx
from .basics import linear


class Embedding:
    def __init__(self, vocab_size: int, embedding_dim: int, weight: mx.array):
        """
        Args:
            vocab_size: Number of tokens in the vocabulary.
            embedding_dim: Hidden dimension E for each token embedding.
            weight: Embedding table with shape (vocab_size, embedding_dim).
        """
        assert weight.shape == (vocab_size, embedding_dim)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = weight

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: Token ids with shape (...).

        Returns:
            Embedded token vectors with shape (..., embedding_dim).
        """
        # Embedding lookup: replace each token id with the corresponding row.
        return self.weight[x]

    def as_linear(self, x: mx.array) -> mx.array:
        """
        Args:
            x: Hidden states with shape (..., embedding_dim).

        Returns:
            Vocabulary logits with shape (..., vocab_size).
        """
        # Reuse the embedding table as an output projection: x @ weight.T.
        return linear(x, self.weight)
