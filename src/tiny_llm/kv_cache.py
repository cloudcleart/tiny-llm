from abc import ABC, abstractmethod
from typing import Optional

import mlx.core as mx


class TinyKvCache(ABC):
    @abstractmethod
    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        """
        Update the key-value cache and fetch the updated key-value cache.

        Args:
            key: The key to update the cache with.
            value: The value to update the cache with.
            mask_length: The length of the mask (only used in batching mode)
            mask: The mask to use (only used in batching mode)

        Returns:
            A tuple of the updated key-value cache, the updated value, the sequence length, and the mask.
            In week 2 day 1, we only need to return the updated key-value cache, the updated value.
            In week 2 day 6/7, we need to return the updated key-value cache, the updated value, the sequence length, and the mask.
            so that the batching kv cache can use this information to generate the mask.
        """


class BatchingKvCache(TinyKvCache):
    def __init__(self, max_active_requests: int, max_seq_len: int):
        self.max_active_requests = max_active_requests
        self.max_seq_len = max_seq_len

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        pass

    def add_request(self, prefilled: TinyKvCache, id: int):
        pass

    def remove_request(self, id: int):
        pass


class TinyKvFullCache(TinyKvCache):
    def __init__(self):
        self.key_values = None
        self.offset = 0

    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        """
        Update the full key-value cache and fetch the updated cache.

        Args:
            key: The key to update the cache with. Shape: (B, L_new, H_kv, D).
            value: The value to update the cache with. Shape: (B, L_new, H_kv, D).
            mask_length: The length of the mask (only used in batching mode).
            mask: The mask to use (only used in batching mode).

        Returns:
            A tuple of the updated key cache, the updated value cache, the sequence length, and the mask.
        """
        # Step 1: initialize the cache on the first update.
        if self.key_values is None:
            self.key_values = (key, value)
        else:
            # Step 2: append new K/V states to the cached history along the
            # sequence-length dimension.
            cached_key, cached_value = self.key_values
            self.key_values = (
                mx.concatenate([cached_key, key], axis=1),
                mx.concatenate([cached_value, value], axis=1),
            )

        # Step 3: track the total number of cached tokens.
        self.offset += key.shape[1]

        # Step 4: return the full K/V cache and metadata expected by callers.
        key_cache, value_cache = self.key_values
        return key_cache, value_cache, self.offset, mask

    def rewind(self, n: int):
        if n <= 0 or self.key_values is None:
            return

        keep_len = max(self.offset - n, 0)
        key, value = self.key_values
        self.key_values = (key[:, :keep_len], value[:, :keep_len])
        self.offset = keep_len
